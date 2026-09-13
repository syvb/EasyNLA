"""Controlled evaluation of a future-lens decoder.

For every (layer, N) and every condition, greedy-decode K = N+1 tokens and score
against the stored ground truth:

    p1          readout token N == x_{t+1+N}
    p5          readout token N in the target's teacher-forced top-5 at that offset
    exact       all K readout tokens match
    surprisal   -mean log p_target(readout | true prefix)   (needs docs.parquet)
    p1_conf_*   p1 bucketed by the target's confidence at t (0-30/30-60/60-90/90-100)

Conditions (spec Phase 3):
    real         the row's own activation
    shuffled     another row's activation from the same layer (context-free prior)
    none         marker left as its own embedding (no injection)
    wrong_layer  the layer-`--wrong-layer` activation of the SAME position, prompt unchanged

Output: JSON lines {layer, N, k, condition, injection, metric, value, n, seed, checkpoint}
so the final plots are one script (nla/future_lens/plots.py).

    python -m nla.future_lens.eval --base-ckpt Qwen/Qwen3-8B --adapter <sft_or_rl_dir> \
        --parquet <data>/eval.parquet --out evals/sft_seed0.jsonl --conditions real,shuffled,none
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nla.config import load_nla_config
from nla.future_lens.data import (
    build_prompt_messages, encode_prompt, load_docs, load_fl_meta, load_fl_rows, resolve_docs_path,
)
from nla.future_lens.inject import INJECTION_MODES, prepare_vectors, register_injection
from nla.future_lens.rewards import target_logp_reward, truncate_readout

CONDITIONS = ("real", "shuffled", "none", "wrong_layer")
CONF_BUCKETS = ((0.0, 0.3, "conf_00_30"), (0.3, 0.6, "conf_30_60"), (0.6, 0.9, "conf_60_90"),
                (0.9, 1.01, "conf_90_100"))


def stop_ids(tokenizer, model=None) -> set[int]:
    ids = {tokenizer.eos_token_id}
    gc = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc is not None:
        ids.update(gc if isinstance(gc, (list, tuple)) else [gc])
    ids.discard(None)
    return ids


@torch.no_grad()
def generate_readouts(model, tokenizer, jobs: list[dict], *, inject_char: str, vectors_ref,
                      injection_mode: str, device, eos_ids: set[int], batch_size: int = 32,
                      max_new_tokens: int | None = None, do_sample: bool = False,
                      temperature: float = 1.0) -> list[list[int]]:
    """jobs: dicts with `prompt` (messages), `k`, `vector` (raw np/tensor or None for
    the no-injection condition), `alpha`. Returns EOS-stripped readouts truncated to k.
    Left-padded batched generation; the injection hook scans for the marker so
    padding is harmless."""
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outs: list[list[int]] = [None] * len(jobs)   # type: ignore[list-item]
    # group by whether a vector is present (the hook needs one vector per sequence)
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i]["vector"] is None, jobs[i]["k"]))
    for cs in range(0, len(order), batch_size):
        idx = order[cs: cs + batch_size]
        chunk = [jobs[i] for i in idx]
        has_vec = [j["vector"] is not None for j in chunk]
        if any(has_vec) and not all(has_vec):
            # split mixed chunks (cheap; only at the boundary)
            for sub in ([i for i, h in zip(idx, has_vec) if h], [i for i, h in zip(idx, has_vec) if not h]):
                if sub:
                    for i, r in zip(sub, generate_readouts(
                            model, tokenizer, [jobs[i] for i in sub], inject_char=inject_char,
                            vectors_ref=vectors_ref, injection_mode=injection_mode, device=device,
                            eos_ids=eos_ids, batch_size=batch_size, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, temperature=temperature)):
                        outs[i] = r
            continue
        enc = [encode_prompt(tokenizer, j["prompt"], inject_char) for j in chunk]
        L = max(len(e) for e in enc)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, L - len(e):] = torch.tensor(e, dtype=torch.long)
            attn[r, L - len(e):] = 1
        kmax = max(j["k"] for j in chunk)
        n_new = max_new_tokens or (kmax + 3)
        if all(has_vec):
            raw = torch.tensor(np.stack([np.asarray(j["vector"], dtype=np.float32) for j in chunk]))
            alphas = torch.tensor([float(j["alpha"]) for j in chunk])
            vectors_ref[0] = prepare_vectors(raw, alphas, injection_mode).to(device)
        else:
            vectors_ref[0] = None
        try:
            gen_kwargs = dict(max_new_tokens=n_new, pad_token_id=pad_id, do_sample=do_sample)
            if do_sample:
                gen_kwargs.update(temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0)
            gen = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device), **gen_kwargs)
        finally:
            vectors_ref[0] = None
        for r, i in enumerate(idx):
            resp = gen[r, L:].tolist()
            # cut at the first stop id, then truncate to k
            cut = next((p for p, t in enumerate(resp) if t in eos_ids), len(resp))
            outs[i], _ = truncate_readout(resp[:cut], chunk[r]["k"], eos_ids)
    if was_training:
        model.train()
    return outs


def score_readout(readout: list[int], row: dict, k: int) -> dict:
    """Per-offset hit/top-5 flags for one readout against the row's targets."""
    tgt = np.asarray(row["target_ids"])
    top5 = np.asarray(row["target_top5"]).reshape(-1, 5)
    p1 = [int(j < len(readout) and int(readout[j]) == int(tgt[j])) for j in range(k)]
    p5 = [int(j < len(readout) and int(readout[j]) in set(int(x) for x in top5[j])) for j in range(k)]
    return {"p1": p1, "p5": p5, "exact": int(all(p1)), "len_ok": int(len(readout) == k)}


def summarize_by_offset(rows: list[dict], readouts: list[list[int]], ks: list[int]) -> dict:
    """Aggregate p1/p5 per offset over rows whose k > offset (mixed-K monitor)."""
    hits = defaultdict(list)
    for row, ro, k in zip(rows, readouts, ks):
        s = score_readout(ro, row, k)
        for j in range(k):
            hits[("p1", j)].append(s["p1"][j])
            hits[("p5", j)].append(s["p5"][j])
        hits[("exact", -1)].append(s["exact"])
        hits[("len_ok", -1)].append(s["len_ok"])
    out = {}
    for (m, j), v in hits.items():
        key = f"{m}_off{j}" if j >= 0 else m
        out[key] = float(np.mean(v))
        out[f"n_{key}"] = len(v)
    return out


def _wrong_layer_lookup(rows: list[dict], wrong_layer: int) -> dict[tuple[int, int], np.ndarray]:
    return {(int(r["doc_idx"]), int(r["t"])): r["activation_vector"]
            for r in rows if int(r["activation_layer"]) == wrong_layer}


def evaluate(model, tokenizer, rows: list[dict], fl, cfg, *, injection_mode: str, vectors_ref,
             device, conditions: list[str], ks: list[int], layers: list[int] | None = None,
             alpha_mult: float = 1.0, wrong_layer: int = 4, docs: dict | None = None,
             batch_size: int = 32, seed: int = 0, tag: dict | None = None,
             eos_ids: set[int] | None = None, verbose: bool = True) -> list[dict]:
    """Full controlled eval -> list of JSON records."""
    eos_ids = eos_ids or stop_ids(tokenizer, model)
    layers = layers or fl.layer_indices
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    tag = tag or {}
    wl = _wrong_layer_lookup(rows, wrong_layer) if "wrong_layer" in conditions else {}
    by_layer = {l: [r for r in rows if int(r["activation_layer"]) == l] for l in layers}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for cond in conditions:
        assert cond in CONDITIONS, f"unknown condition {cond!r}; choices {CONDITIONS}"
        for layer in layers:
            lrows = by_layer[layer]
            if not lrows:
                continue
            if cond == "shuffled":
                perm = rng.permutation(len(lrows))
                perm = np.roll(perm, 1) if len(lrows) > 1 else perm
                vec_src = [lrows[i]["activation_vector"] for i in perm]
            elif cond == "wrong_layer":
                vec_src = [wl.get((int(r["doc_idx"]), int(r["t"]))) for r in lrows]
            elif cond == "none":
                vec_src = [None] * len(lrows)
            else:
                vec_src = [r["activation_vector"] for r in lrows]
            for k in ks:
                t0 = time.time()
                keep = [i for i, v in enumerate(vec_src) if cond in ("none",) or v is not None]
                jobs = [{"prompt": build_prompt_messages(fl.template, layer, k), "k": k,
                         "vector": vec_src[i], "alpha": fl.alpha(layer, alpha_mult)} for i in keep]
                if not jobs:
                    continue
                ro = generate_readouts(model, tokenizer, jobs, inject_char=cfg.injection_char,
                                       vectors_ref=vectors_ref, injection_mode=injection_mode,
                                       device=device, eos_ids=eos_ids, batch_size=batch_size)
                srows = [lrows[i] for i in keep]
                scores = [score_readout(r, row, k) for r, row in zip(ro, srows)]
                N = k - 1
                base = {"layer": layer, "N": N, "k": k, "condition": cond,
                        "injection": injection_mode, "seed": seed, **tag}
                n = len(scores)
                p1 = [s["p1"][N] for s in scores]
                records.append({**base, "metric": "p1", "value": float(np.mean(p1)), "n": n})
                records.append({**base, "metric": "p5", "value": float(np.mean([s["p5"][N] for s in scores])), "n": n})
                records.append({**base, "metric": "exact", "value": float(np.mean([s["exact"] for s in scores])), "n": n})
                records.append({**base, "metric": "len_ok", "value": float(np.mean([s["len_ok"] for s in scores])), "n": n})
                for lo, hi, name in CONF_BUCKETS:
                    sel = [p for p, row in zip(p1, srows) if lo <= float(row["p_top1"]) < hi]
                    if sel:
                        records.append({**base, "metric": f"p1_{name}", "value": float(np.mean(sel)), "n": len(sel)})
                if docs is not None:
                    prefixes = [docs[int(row["doc_idx"])][: int(row["t"]) + 1] for row in srows]
                    lp = target_logp_reward(model, prefixes, ro, device, pad_id=pad_id,
                                            micro_batch=max(1, batch_size // 4))
                    vals = [-v for v in lp if v is not None]
                    if vals:
                        records.append({**base, "metric": "surprisal", "value": float(np.mean(vals)), "n": len(vals)})
                if verbose:
                    print(f"[eval] {cond:11s} L{layer:2d} N={N} p1={float(np.mean(p1)):.3f} "
                          f"exact={float(np.mean([s['exact'] for s in scores])):.3f} n={n} ({time.time() - t0:.0f}s)", flush=True)
    return records


def write_records(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def load_decoder(base_ckpt: str, adapter: str | None, device: str, dtype, *, affine_path=None):
    """Base model (+ LoRA adapter) ready for generation."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_ckpt, torch_dtype=dtype,
                                                 attn_implementation="sdpa").to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, adapter_name="default")
    model.eval()
    affine = None
    if affine_path and Path(affine_path).exists():
        from nla.future_lens.inject import AffineInjector
        sd = torch.load(affine_path, map_location="cpu")
        affine = AffineInjector(sd["proj.weight"].shape[0])
        affine.load_state_dict(sd)
        affine = affine.to(device)
    return model, affine


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--adapter", default=None, help="LoRA dir (SFT iter_* or RL iter_*); omit for the bare base")
    p.add_argument("--parquet", required=True, help="eval.parquet from nla.future_lens.collect")
    p.add_argument("--sidecar", default=None)
    p.add_argument("--out", required=True, help="JSONL to append records to")
    p.add_argument("--injection", choices=INJECTION_MODES, default="replace_embed")
    p.add_argument("--alpha-mult", type=float, default=1.0)
    p.add_argument("--conditions", default="real,shuffled,none")
    p.add_argument("--ks", default=None, help="readout lengths (default: sidecar k_choices)")
    p.add_argument("--layers", default=None)
    p.add_argument("--wrong-layer", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None, help="cap rows PER LAYER")
    p.add_argument("--surprisal", action=argparse.BooleanOptionalAction, default=True,
                   help="score readouts under the frozen target (needs docs.parquet)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="JSON dict merged into every record (e.g. checkpoint name)")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    sidecar = args.sidecar or args.parquet
    cfg = load_nla_config(sidecar, tokenizer)
    fl = load_fl_meta(sidecar)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else fl.layer_indices
    ks = [int(x) for x in args.ks.split(",")] if args.ks else fl.k_choices
    conditions = args.conditions.split(",")
    need_layers = set(layers) | ({args.wrong_layer} if "wrong_layer" in conditions else set())
    rows = load_fl_rows(args.parquet, layers=sorted(need_layers))
    if args.max_rows:
        per: dict[int, int] = defaultdict(int)
        kept = []
        for r in rows:
            l = int(r["activation_layer"])
            if per[l] < args.max_rows:
                kept.append(r); per[l] += 1
        rows = kept
    docs = None
    if args.surprisal:
        dp = resolve_docs_path(args.parquet, fl)
        docs = load_docs(dp) if dp else None
        if docs is None:
            print("[eval] docs.parquet not found — skipping surprisal")
    affine_path = Path(args.adapter) / "affine.pt" if args.adapter else None
    model, affine = load_decoder(args.base_ckpt, args.adapter, device, dtype, affine_path=affine_path)
    vectors_ref = [None]
    register_injection(model, args.injection, vectors_ref, cfg.injection_token_id,
                       cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, affine)
    tag = json.loads(args.tag) if args.tag else {}
    tag.setdefault("checkpoint", args.adapter or "base")
    recs = evaluate(model, tokenizer, rows, fl, cfg, injection_mode=args.injection, vectors_ref=vectors_ref,
                    device=device, conditions=conditions, ks=ks, layers=layers, alpha_mult=args.alpha_mult,
                    wrong_layer=args.wrong_layer, docs=docs, batch_size=args.batch_size, seed=args.seed, tag=tag)
    write_records(recs, args.out)
    print(f"[eval] wrote {len(recs)} records -> {args.out}")


if __name__ == "__main__":
    main()
