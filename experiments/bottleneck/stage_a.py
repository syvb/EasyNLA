"""Stage A: teacher-forced domain map of codec fidelity + one-step substitution.

Per domain: one clean forward collecting the layer-24 stream at every position;
verbalize+reconstruct every position (batched — this is >95% of the cost); one
patched forward with ALL positions' layer-24 streams replaced at once; compare
next-token distributions. Caveat (by design): this verbalizes clean-history
activations — compounding under substituted history is Stage B's job.

    python -m experiments.bottleneck.stage_a --config experiments/bottleneck/config.yaml

Input: {domains_parquet} with columns (domain, doc_id, text) from
prep_domains.py. The special domain "onpolicy" is generated here (M's own
greedy non-thinking outputs on benchmark-style prompts, then teacher-forced).

Output: {out_dir}/stage_a/{domain}.parquet — one row per verbalized position.
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
from experiments.bottleneck.tasks import FLUENCY_PROMPTS, GSM_INSTR


def build_onpolicy_seqs(m: BottleneckModel, n_seqs: int, seq_len: int,
                        max_gen: int = 192) -> list[dict]:
    """M's own greedy outputs on benchmark-style prompts, for teacher-forcing.

    Generation-time activations come from the model's own distribution — if the
    codec was trained mostly on pretraining-like text (it was: finefineweb),
    this shift is part of what Stage B hits, so it gets its own Stage A domain.
    """
    import datasets as hfd
    prompts = [{"q": p, "kind": "openended"} for p in FLUENCY_PROMPTS]
    gsm = hfd.load_dataset("openai/gsm8k", "main", split="train")
    for i in range(min(n_seqs, len(gsm))):
        prompts.append({"q": f"{GSM_INSTR}\n\n{gsm[i]['question']}", "kind": "gsm8k_train"})
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


@torch.no_grad()
def process_batch(m: BottleneckModel, codec: NLACodec, batch: list[dict],
                  domain: str, store_z: bool, identity_check: bool = False) -> list[dict]:
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

    rows = []
    rec_ptr = 0
    for r, b in enumerate(batch):
        L = int(mask[r].sum())
        lp_c = F.log_softmax(logits_clean[r, :L].float(), dim=-1)   # [L, V]
        lp_p = F.log_softmax(logits_patch[r, :L].float(), dim=-1)
        kl = (lp_c.exp() * (lp_c - lp_p)).sum(-1)                    # [L]
        flip = lp_c.argmax(-1) != lp_p.argmax(-1)                    # [L]
        tgt = ids[r, 1:L]
        nll_c = -lp_c[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [L-1]
        nll_p = -lp_p[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        resp_start = b.get("response_start")
        for i in range(L):
            rec = recs[rec_ptr]
            rows.append({
                "domain": domain, "doc_id": b["doc_id"], "pos": i,
                "token_id": int(ids[r, i]),
                "token": m.tokenizer.decode([int(ids[r, i])]),
                "is_response": bool(resp_start is not None and i >= resp_start),
                "cosine": rec.cosine, "h_norm": rec.h_norm, "pred_norm": rec.pred_norm,
                "z_len": rec.verb.n_tokens,
                "z_text": rec.verb.text if store_z else None,
                "truncated": rec.verb.truncated,
                "extract_failed": rec.verb.extract_failed,
                "steer_verified": rec.verb.steer_verified,
                "nll_clean": float(nll_c[i]) if i < L - 1 else None,
                "nll_patched": float(nll_p[i]) if i < L - 1 else None,
                "kl": float(kl[i]),
                "top1_flip": bool(flip[i]),
            })
            rec_ptr += 1
    assert rec_ptr == len(recs)
    del logits_clean, logits_patch, h
    torch.cuda.empty_cache()
    return rows


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
            texts = table.filter(dmask).column("text").to_pylist()[:n_seqs]
            docids = table.filter(dmask).column("doc_id").to_pylist()[:n_seqs]
            seqs = []
            for t, d in zip(texts, docids):
                enc = m.tokenizer.encode(t, add_special_tokens=False)[:seq_len + 1]
                if len(enc) >= 64:
                    seqs.append({"doc_id": d, "ids": enc})
            print(f"[stage_a {domain}] {len(seqs)}/{len(texts)} seqs usable (≥64 tok)")

        rows = []
        for cs in range(0, len(seqs), batch_size):
            rows.extend(process_batch(
                m, codec, seqs[cs:cs + batch_size], domain, store_z,
                identity_check=first and cs == 0))
            first = False
            print(f"[stage_a {domain}] {min(cs + batch_size, len(seqs))}/{len(seqs)} seqs",
                  flush=True)
        pq.write_table(pa.Table.from_pylist(rows), out_path)
        print(f"[out] wrote {len(rows)} rows -> {out_path}", flush=True)

    print(f"[codec stats FINAL] {codec.report_stats()}", flush=True)
    print("STAGE_A_DONE", flush=True)


if __name__ == "__main__":
    main()
