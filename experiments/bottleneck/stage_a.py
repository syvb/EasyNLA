"""Stage A: teacher-forced domain map of codec fidelity + substitution damage.

Per domain: one clean forward collecting the layer-24 stream at every position;
verbalize+reconstruct every position (batched — this is >95% of the cost); then
THREE cheap patched forwards reusing the same ĥ:

  1. all positions replaced ("full") — teacher-forced C1: position i's metrics
     include reconstructed history at all j<=i through layers 25-35;
  2. all-but-sinks replaced ("nosink") — position 0 and top-1%-‖h‖ positions
     (Qwen massive-activation sinks) kept clean. This is the real robustness
     check: dropping sink ROWS from the summary doesn't help when a mangled
     sink direction contaminates every other position through attention;
  3. one-position-only calibration (first batch per domain, strided positions)
     — single-step damage with clean history, decomposing "full" into
     per-step vs compounded damage.

Metrics per position vs the clean forward: ΔNLL of the true next token,
KL(clean‖patched), top-1 flip, plus the roundtrip cosine (free byproduct).

    python -m experiments.bottleneck.stage_a --config experiments/bottleneck/config.yaml

Input: {domains_parquet} with columns (domain, doc_id, text) from
prep_domains.py. The special domain "onpolicy" is generated here (M's own
greedy non-thinking outputs on benchmark-style prompts — open-ended, math,
code, QA — then teacher-forced).

Output: {out_dir}/stage_a/{domain}.parquet (one row per verbalized position)
and {domain}_onestep.parquet (the calibration subset).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from experiments.bottleneck.codec import NLACodec
from experiments.bottleneck.patched import BottleneckModel
from experiments.bottleneck.tasks import FLUENCY_PROMPTS, GSM_INSTR, SHORT_INSTR


def build_onpolicy_seqs(m: BottleneckModel, n_seqs: int, seq_len: int,
                        max_gen: int = 192) -> list[dict]:
    """M's own greedy outputs on benchmark-style prompts, for teacher-forcing.

    Generation-time activations come from the model's own distribution — if the
    codec was trained mostly on pretraining-like text (it was: finefineweb),
    this shift is part of what Stage B hits, so it gets its own Stage A domain.
    Prompt mix spans open-ended / math / code / QA so the domain isn't just
    fluent prose. (Only TRAIN splits — Stage B evaluates on test splits.)
    """
    import datasets as hfd
    quota = max(n_seqs // 4, 1)
    prompts = [{"q": p, "kind": "openended"} for p in FLUENCY_PROMPTS[:quota]]
    gsm = hfd.load_dataset("openai/gsm8k", "main", split="train")
    prompts += [{"q": f"{GSM_INSTR}\n\n{gsm[i]['question']}", "kind": "gsm8k_train"}
                for i in range(quota)]
    mbpp = hfd.load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    prompts += [{"q": f"{row['prompt']}\nOutput only a ```python code block.",
                 "kind": "mbpp_train"} for row in list(mbpp)[:quota]]
    tqa = hfd.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    prompts += [{"q": f"{SHORT_INSTR}\n\nQ: {tqa[i]['question']}\nA:", "kind": "triviaqa_train"}
                for i in range(quota)]
    prompts = prompts[:n_seqs]

    out = []
    bs = 32
    for cs in range(0, len(prompts), bs):
        chunk = prompts[cs:cs + bs]
        pid_lists = [m.build_chat_ids([{"role": "user", "content": c["q"]}]) for c in chunk]
        gens = m.generate(pid_lists, max_gen, condition="clean")
        for j, c in enumerate(chunk):
            ids = (pid_lists[j] + gens[j])[:seq_len + 1]
            out.append({
                "doc_id": f"onpolicy/{cs + j}/{c['kind']}",
                "ids": ids,
                "response_start": min(len(pid_lists[j]), len(ids)),
            })
    return out


def _row_metrics(logits_row: torch.Tensor, lp_clean: torch.Tensor,
                 ids_row: torch.Tensor, L: int):
    """(kl [L], flip [L], nll [L-1]) of a patched row vs precomputed clean lp."""
    lp_p = F.log_softmax(logits_row[:L].float(), dim=-1)
    kl = (lp_clean.exp() * (lp_clean - lp_p)).sum(-1)
    flip = lp_clean.argmax(-1) != lp_p.argmax(-1)
    tgt = ids_row[1:L]
    nll = -lp_p[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return kl, flip, nll


@torch.no_grad()
def process_batch(m: BottleneckModel, codec: NLACodec, batch: list[dict],
                  domain: str, store_z: bool, identity_check: bool = False,
                  calibrate: bool = False) -> tuple[list[dict], list[dict]]:
    device = m.device
    B = len(batch)
    S = max(len(b["ids"]) for b in batch)
    ids = torch.full((B, S), m.pad_id, dtype=torch.long)
    mask = torch.zeros((B, S), dtype=torch.long)
    for r, b in enumerate(batch):                      # right-pad (teacher-forced)
        L = len(b["ids"])
        ids[r, :L] = torch.tensor(b["ids"], dtype=torch.long)
        mask[r, :L] = 1
    ids, mask = ids.to(device), mask.to(device)

    logits_clean, h = m.forward_teacher_forced(ids, mask, capture=True)
    assert h is not None and h.shape[:2] == ids.shape

    if identity_check:
        logits_ident, _ = m.forward_teacher_forced(
            ids, mask, replacement=h.clone(), replace_mask=mask.bool())
        assert torch.equal(logits_clean, logits_ident), (
            "identity patched forward diverged from clean — replace_all path is buggy")
        print("[stage_a] identity patched-forward check PASSED (bitwise)", flush=True)

    # Flatten real positions → one big roundtrip (codec chunks internally).
    flat_idx = mask.bool().nonzero(as_tuple=False)     # [N, 2] (row, pos)
    h_flat = h[flat_idx[:, 0], flat_idx[:, 1]]          # [N, d]
    t0 = time.time()
    h_hat, recs = codec.roundtrip(h_flat)
    print(f"[stage_a {domain}] verbalized {h_flat.shape[0]} positions "
          f"in {time.time()-t0:.0f}s", flush=True)

    replacement = h.clone()
    replacement[flat_idx[:, 0], flat_idx[:, 1]] = h_hat.to(h.dtype)
    logits_patch, _ = m.forward_teacher_forced(
        ids, mask, replacement=replacement, replace_mask=mask.bool())

    # nosink: same ĥ, but position 0 and top-1%-norm positions stay clean.
    h_norms = h_flat.float().norm(dim=-1)
    norm_cut = torch.quantile(h_norms, 0.99)
    is_sink_flat = (h_norms >= norm_cut) | (flat_idx[:, 1] == 0)
    nosink_mask = mask.bool().clone()
    sink_pos = flat_idx[is_sink_flat]
    nosink_mask[sink_pos[:, 0], sink_pos[:, 1]] = False
    logits_nosink, _ = m.forward_teacher_forced(
        ids, mask, replacement=replacement, replace_mask=nosink_mask)

    rows = []
    rec_ptr = 0
    for r, b in enumerate(batch):
        L = int(mask[r].sum())
        lp_c = F.log_softmax(logits_clean[r, :L].float(), dim=-1)   # [L, V]
        kl, flip, nll_p = _row_metrics(logits_patch[r], lp_c, ids[r], L)
        kl_ns, flip_ns, nll_ns = _row_metrics(logits_nosink[r], lp_c, ids[r], L)
        tgt = ids[r, 1:L]
        nll_c = -lp_c[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [L-1]
        resp_start = b.get("response_start")
        for i in range(L):
            rec = recs[rec_ptr]
            rows.append({
                "domain": domain, "doc_id": b["doc_id"], "pos": i,
                "token_id": int(ids[r, i]),
                "token": m.tokenizer.decode([int(ids[r, i])]),
                "is_response": bool(resp_start is not None and i >= resp_start),
                "is_sink": bool(is_sink_flat[rec_ptr]),
                "cosine": rec.cosine, "h_norm": rec.h_norm, "pred_norm": rec.pred_norm,
                "z_len": rec.verb.n_tokens,
                "z_text": rec.verb.text if store_z else None,
                "truncated": rec.verb.truncated,
                "extract_failed": rec.verb.extract_failed,
                "steer_verified": rec.verb.steer_verified,
                "nll_clean": float(nll_c[i]) if i < L - 1 else None,
                "nll_patched": float(nll_p[i]) if i < L - 1 else None,
                "nll_patched_nosink": float(nll_ns[i]) if i < L - 1 else None,
                "kl": float(kl[i]),
                "kl_nosink": float(kl_ns[i]),
                "top1_flip": bool(flip[i]),
                "top1_flip_nosink": bool(flip_ns[i]),
            })
            rec_ptr += 1
    assert rec_ptr == len(recs)

    # One-position-only calibration: single-step damage with CLEAN history.
    # One batched forward per strided position — codec-free (ĥ already known).
    onestep_rows = []
    if calibrate:
        min_L = int(mask.sum(-1).min())
        stride_positions = [p for p in range(8, min_L - 1, max((min_L - 9) // 8, 1))][:8]
        for p in stride_positions:
            one_mask = torch.zeros_like(mask, dtype=torch.bool)
            one_mask[:, p] = mask.bool()[:, p]
            lg, _ = m.forward_teacher_forced(
                ids, mask, replacement=replacement, replace_mask=one_mask)
            for r, b in enumerate(batch):
                L = int(mask[r].sum())
                if p >= L - 1:
                    continue
                lp_c = F.log_softmax(logits_clean[r, p].float(), dim=-1)
                lp_p = F.log_softmax(lg[r, p].float(), dim=-1)
                onestep_rows.append({
                    "domain": domain, "doc_id": b["doc_id"], "pos": p,
                    "kl_onestep": float((lp_c.exp() * (lp_c - lp_p)).sum()),
                    "flip_onestep": bool(lp_c.argmax() != lp_p.argmax()),
                    "nll_clean": float(-lp_c[ids[r, p + 1]]),
                    "nll_onestep": float(-lp_p[ids[r, p + 1]]),
                })
        print(f"[stage_a {domain}] one-step calibration: {len(onestep_rows)} rows "
              f"at positions {stride_positions}", flush=True)

    del logits_clean, logits_patch, logits_nosink, h
    torch.cuda.empty_cache()
    return rows, onestep_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--domains", default="all", help="comma list or 'all'")
    p.add_argument("--max-seqs", type=int, default=None, help="override seqs/domain (smoke)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"]) / "stage_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = cfg.get("stage_a_seq_len", 256)
    n_seqs = args.max_seqs or cfg.get("stage_a_seqs_per_domain", 64)
    batch_size = cfg.get("stage_a_batch_size", 8)
    store_z = cfg.get("store_z", True)

    m = BottleneckModel(cfg.get("m_ckpt", "Qwen/Qwen3-8B"),
                        layer_index=cfg.get("layer_index", 24))
    codec = NLACodec(
        av_merged_dir=cfg["av_merged_dir"], ar_dir=cfg["ar_dir"],
        vllm_gpu_mem=cfg.get("vllm_gpu_mem", 0.35),
        vllm_max_len=cfg.get("vllm_max_len", 1024),
        av_max_tokens=cfg.get("av_max_tokens", 150),
        av_temperature=cfg.get("av_temperature", 1.0),
        seed=cfg.get("stage_a_seed", 0),
    )

    table = pq.read_table(cfg["domains_parquet"])
    all_domains = sorted(set(table.column("domain").to_pylist())) + ["onpolicy"]
    domains = all_domains if args.domains == "all" else args.domains.split(",")
    unknown = set(domains) - set(all_domains)
    assert not unknown, f"unknown domains {unknown}; available: {all_domains}"

    first = True
    for domain in domains:
        out_path = out_dir / f"{domain}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists")
            continue
        if domain == "onpolicy":
            seqs = build_onpolicy_seqs(m, n_seqs, seq_len)
        else:
            dmask = pa.compute.equal(table.column("domain"), domain)
            texts = table.filter(dmask).column("text").to_pylist()[: 2 * n_seqs]
            docids = table.filter(dmask).column("doc_id").to_pylist()[: 2 * n_seqs]
            seqs = []
            for t, d in zip(texts, docids):
                enc = m.tokenizer.encode(t, add_special_tokens=False)[:seq_len + 1]
                if len(enc) >= 64:
                    seqs.append({"doc_id": d, "ids": enc})
                if len(seqs) >= n_seqs:
                    break
            print(f"[stage_a {domain}] {len(seqs)} seqs usable (>=64 tok)")
        assert seqs, f"no usable sequences for domain {domain}"

        rows, onestep = [], []
        for cs in range(0, len(seqs), batch_size):
            r_, o_ = process_batch(
                m, codec, seqs[cs:cs + batch_size], domain, store_z,
                identity_check=first and cs == 0, calibrate=cs == 0)
            rows.extend(r_)
            onestep.extend(o_)
            first = False
            print(f"[stage_a {domain}] {min(cs + batch_size, len(seqs))}/{len(seqs)} seqs",
                  flush=True)
        pq.write_table(pa.Table.from_pylist(rows), out_path)
        if onestep:
            pq.write_table(pa.Table.from_pylist(onestep),
                           out_dir / f"{domain}_onestep.parquet")
        print(f"[out] wrote {len(rows)} rows -> {out_path}", flush=True)

    print(f"[codec stats FINAL] {codec.report_stats()}", flush=True)
    print("STAGE_A_DONE", flush=True)


if __name__ == "__main__":
    main()
