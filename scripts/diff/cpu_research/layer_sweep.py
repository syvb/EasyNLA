"""Gate 0's auditing-SNR statistic at every layer, not only the planned one.

One whole-model forward pass per (model, wrapper) captures the residual stream at
`asst_last` after every decoder block. For each layer this reports, per task and
Base wrapper:

  * Fisher SNR of raw h_target and of unit(naive diff - mean), and their ratio
    (the Gate 0 statistic, with identity alignment instead of the fitted ridge),
  * how much of the class signal the Base model already carries,
  * how much of h_target's prompt-to-prompt variance the Base model explains.

The ridge alignment is fitted on web tokens for one layer only, so the sweep uses
the naive diff h_target - h_base. `gate_controls.py` shows the two differ little at
the planned layer.

    python layer_sweep.py --out sweep.json --cache-dir /home/user/act_cache/sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from gate_controls import TOPIC_SPLITS, cu, few_shot_acc
from jit_full import load_full_jit

from nla.diff.benchmarks import (
    flatten_binary_pairs,
    load_eval_awareness_pairs,
    load_historical_period_items,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.metrics import compute_fisher_snr
from nla.diff.wrappers import encode_aligned_pair


@torch.no_grad()
def capture_all_layers(model_name: str, seqs: list[list[int]], pos: list[int], pad_id: int, *,
                       n_resident: int, token_budget: int = 2048) -> np.ndarray:
    """Return fp16 [n_seqs, n_layers, d]: the output of every decoder block at one position per sequence."""
    model = load_full_jit(model_name, n_resident=n_resident, keep_head=False)
    layers = model.model.layers
    cur: dict = {}
    grabbed: list[torch.Tensor | None] = [None] * len(layers)

    def make_hook(li: int):
        def hook(_m, _a, out) -> None:
            h = out[0] if isinstance(out, tuple) else out
            grabbed[li] = h[cur["rows"], cur["pos"]].float().clone()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    d = int(model.config.hidden_size)
    out = np.zeros((len(seqs), len(layers), d), dtype=np.float16)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    i = 0
    t0 = time.time()
    n_tok = sum(len(s) for s in seqs)
    done = 0
    while i < len(order):
        max_len = len(seqs[order[i]])
        j = i + max(1, min(64, token_budget // max_len))
        idx = order[i:j]
        ids = torch.full((len(idx), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), max_len), dtype=torch.long)
        for r, q in enumerate(idx):
            ids[r, : len(seqs[q])] = torch.tensor(seqs[q], dtype=torch.long)
            mask[r, : len(seqs[q])] = 1
        cur["rows"] = torch.arange(len(idx))
        cur["pos"] = torch.tensor([pos[q] for q in idx], dtype=torch.long)
        model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        stacked = torch.stack([g for g in grabbed], dim=1).numpy()  # [b, n_layers, d]
        assert np.isfinite(stacked).all()
        out[idx] = stacked.astype(np.float16)
        done += sum(len(seqs[q]) for q in idx)
        i = j
        print(f"[sweep] {model_name}: {done}/{n_tok} tok ({done / (time.time() - t0):.0f} tok/s)", flush=True)
    for h in handles:
        h.remove()
    del model
    return out


def stats(ht: torch.Tensor, hb: torch.Tensor, y: np.ndarray, groups: np.ndarray) -> dict:
    def parts(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        x = x.double()
        m1, m0 = x[y == 1].mean(0), x[y == 0].mean(0)
        within = 0.5 * ((x[y == 1] - m1).pow(2).sum(1).mean() + (x[y == 0] - m0).pow(2).sum(1).mean())
        return m1 - m0, float((m1 - m0).pow(2).sum()), float(within)

    dt, bt, wt = parts(ht)
    db, bb, _ = parts(hb)
    _, bd, wd = parts(ht - hb)
    d_cu = cu(ht - hb)
    snr_t = compute_fisher_snr(ht, y)
    htc = (ht - ht.mean(0)).double()
    dc = ((ht - hb) - (ht - hb).mean(0)).double()
    return {
        "snr_h_target": snr_t,
        "ratio_h_target_cu": compute_fisher_snr(cu(ht), y) / max(snr_t, 1e-12),
        "ratio_h_base": compute_fisher_snr(hb, y) / max(snr_t, 1e-12),
        "ratio_naive_diff_cu": compute_fisher_snr(d_cu, y) / max(snr_t, 1e-12),
        "between_base_over_target": bb / max(bt, 1e-12),
        "between_diff_over_target": bd / max(bt, 1e-12),
        "within_diff_over_target": wd / max(wt, 1e-12),
        "cos_class_direction_base_vs_target": float(torch.nn.functional.cosine_similarity(dt, db, dim=0)),
        "base_explains_target_fve": float(1.0 - dc.pow(2).sum() / htc.pow(2).sum()),
        "few_shot_k4_h_target": few_shot_acc(ht, y, groups),
        "few_shot_k4_naive_diff_cu": few_shot_acc(d_cu, y, groups),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--target-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-resident", type=int, default=22)
    ap.add_argument("--check-layer", type=int, default=24, help="layer to compare against the cached_io cache")
    ap.add_argument("--position", choices=("asst_last", "user_last"), default="asst_last",
                    help="asst_last is what Gate 0 scores; user_last is the last user-content token")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target_model)
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    tasks: dict[str, dict] = {}
    for name, pairs in (
        ("refusal", load_refusal_pairs(args.n_pairs, source="auto", seed=0)),
        ("eval_awareness", load_eval_awareness_pairs(args.n_pairs)),
        ("sycophancy", load_sycophancy_pairs(args.n_pairs)),
    ):
        prompts, labels, meta = flatten_binary_pairs(pairs)
        tasks[name] = {"prompts": prompts, "y": np.asarray(labels), "meta": meta,
                       "groups": np.asarray([m["pair_id"] for m in meta])}
    hist = load_historical_period_items(60)
    tasks["historical_era"] = {
        "prompts": [f"Read the following historical excerpt and identify its historical era:\n\n{it['text']}" for it in hist],
        "y": np.asarray([int(it["binary_label"]) for it in hist]),
        "meta": [{"topic": it["era"]} for it in hist],
        "groups": np.asarray([it["item_id"] for it in hist]),
    }

    seqs: dict[str, list[list[int]]] = {"inst": [], "plain": []}
    pos: dict[str, list[int]] = {"inst": [], "plain": []}
    spans: dict[str, tuple[int, int]] = {}
    for name, t in tasks.items():
        start = len(seqs["inst"])
        for txt in t["prompts"]:
            ie, bp = encode_aligned_pair(tok, txt, base_wrapper="plain", enable_thinking=False, max_total_tokens=512)
            seqs["inst"].append(ie.input_ids)
            pos["inst"].append(ie.asst_last_idx if args.position == "asst_last" else ie.user_content_positions[-1])
            seqs["plain"].append(bp.input_ids)
            pos["plain"].append(bp.asst_last_idx if args.position == "asst_last" else bp.user_content_positions[-1])
        spans[name] = (start, len(seqs["inst"]))

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    runs = {"target": (args.target_model, "inst"), "base_chatml": (args.base_model, "inst"),
            "base_plain": (args.base_model, "plain")}
    acts: dict[str, np.ndarray] = {}
    for rname, (model_name, key) in runs.items():
        tag = "" if args.position == "asst_last" else f"_{args.position}"
        f = cache / f"{rname}{tag}_n{args.n_pairs}.npy"
        if not f.exists():
            arr = capture_all_layers(model_name, seqs[key], pos[key], pad, n_resident=args.n_resident)
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(f)
        acts[rname] = np.load(f, mmap_mode="r")
    n_layers = acts["target"].shape[1]

    # The JIT whole-model pass must agree with the split loader that produced every other result.
    check: dict = {}
    for rname, (model_name, key) in runs.items():
        ref = cached_io.cached_extract(cached_io.LazyLM(model_name), seqs[key][:64], [[p] for p in pos[key][:64]],
                                       layer_index=args.check_layer, pad_token_id=pad)
        ref_h = torch.cat([e.hidden for e in ref]).double()
        got = torch.from_numpy(np.asarray(acts[rname][:64, args.check_layer], dtype=np.float32)).double()
        check[rname] = float(((ref_h - got).norm(dim=1) / ref_h.norm(dim=1)).max())
    print(f"[sweep] max relative difference vs cached_io at L{args.check_layer}: {check}", flush=True)

    report: dict = {"n_pairs": args.n_pairs, "n_layers": n_layers, "position": args.position,
                    "agreement_with_cached_io": check, "layers": {}}
    for li in range(n_layers):
        ht_all = torch.from_numpy(np.asarray(acts["target"][:, li], dtype=np.float32))
        row: dict = {}
        for w in ("chatml", "plain"):
            hb_all = torch.from_numpy(np.asarray(acts[f"base_{w}"][:, li], dtype=np.float32))
            rw: dict = {}
            for name, t in tasks.items():
                a, b = spans[name]
                rw[name] = stats(ht_all[a:b], hb_all[a:b], t["y"], t["groups"])
                if name in TOPIC_SPLITS:
                    y_topic = np.asarray([int(m["topic"] in TOPIC_SPLITS[name]) for m in t["meta"]])
                    rw[f"{name}__topic"] = stats(ht_all[a:b], hb_all[a:b], y_topic, t["groups"])
            audit = [rw[n]["ratio_naive_diff_cu"] for n in ("refusal", "eval_awareness", "sycophancy")]
            content = [rw[n]["ratio_naive_diff_cu"] for n in rw if n == "historical_era" or n.endswith("__topic")]
            rw["mean_audit_ratio"] = float(np.mean(audit))
            rw["min_audit_ratio"] = float(np.min(audit))
            rw["mean_content_ratio"] = float(np.mean(content))
            row[w] = rw
        report["layers"][str(li)] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    for w in ("chatml", "plain"):
        print(f"\n=== Base wrapper = {w}, {args.position}: SNR ratio unit(naive diff - mean) / raw h_target, by layer ===")
        print(f"{'L':>3s} {'refusal':>8s} {'eval_aw':>8s} {'sycoph':>8s} {'mean':>6s} {'content':>8s} | "
              f"{'base FVE(ref)':>13s} {'base FVE(ea)':>12s} {'base FVE(syc)':>13s} | few-shot h_t -> diff (mean of 3)")
        for li in range(n_layers):
            rw = report["layers"][str(li)][w]
            fs_t = np.mean([rw[n]["few_shot_k4_h_target"] for n in ("refusal", "eval_awareness", "sycophancy")])
            fs_d = np.mean([rw[n]["few_shot_k4_naive_diff_cu"] for n in ("refusal", "eval_awareness", "sycophancy")])
            print(f"{li:3d} {rw['refusal']['ratio_naive_diff_cu']:8.2f} {rw['eval_awareness']['ratio_naive_diff_cu']:8.2f} "
                  f"{rw['sycophancy']['ratio_naive_diff_cu']:8.2f} {rw['mean_audit_ratio']:6.2f} {rw['mean_content_ratio']:8.2f} | "
                  f"{rw['refusal']['base_explains_target_fve']:13.2f} {rw['eval_awareness']['base_explains_target_fve']:12.2f} "
                  f"{rw['sycophancy']['base_explains_target_fve']:13.2f} | {fs_t:.3f} -> {fs_d:.3f}")


if __name__ == "__main__":
    main()
