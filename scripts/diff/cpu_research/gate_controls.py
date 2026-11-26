"""Controls for the Gate 0 "auditing SNR boost" criterion.

Gate 0 passes when Fisher SNR of unit(Delta - mean) over Fisher SNR of raw
h_target, at `asst_last`, averages >= 1.25 across refusal / eval-awareness /
sycophancy. This script recomputes that statistic on every available pair with
pair-level bootstrap CIs, and adds the controls the gate lacks:

  * the same centering + unit-normalisation applied to h_target and to h_base
    (does differencing matter, or only the normalisation?)
  * h_base alone (is the label already linearly present in the Base model?)
  * topic labels on the same prompts and the historical-era set (does the
    ratio favour audit concepts over plain content?)

    python gate_controls.py --phase0-dir <dir with align_L24_*.pt> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.benchmarks import (
    flatten_binary_pairs,
    load_eval_awareness_pairs,
    load_historical_period_items,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.metrics import compute_fisher_snr, unit_normalize
from nla.diff.wrappers import encode_aligned_pair

TOPIC_SPLITS = {
    "eval_awareness": {"cybersecurity", "coding_agents", "tool_use_agents", "science_bio", "medical_health"},
    "sycophancy": {"physics", "biology", "math_prob", "computer_science", "astronomy_earth"},
}


def cu(x: torch.Tensor) -> torch.Tensor:
    return unit_normalize(x - x.double().mean(dim=0, keepdim=True).float())


def raw_mats(ht: torch.Tensor, hb: torch.Tensor, art_id: dict, art_std: dict) -> dict[str, torch.Tensor]:
    """Row-wise quantities, computed once; centring/unit-normalisation is redone per resample."""
    return {"ht": ht, "hb": hb, "d_id": apply_alignment(hb, ht, art_id), "d_std": apply_alignment(hb, ht, art_std)}


def reps_for(raw: dict[str, torch.Tensor], idx: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    sel = raw if idx is None else {k: v[torch.from_numpy(idx)] for k, v in raw.items()}
    ht, hb, d_id, d_std = sel["ht"], sel["hb"], sel["d_id"], sel["d_std"]
    return {
        "h_target": ht,
        "h_target_cu": cu(ht),
        "h_base": hb,
        "h_base_cu": cu(hb),
        "delta_idanc": d_id,
        "delta_idanc_cu": cu(d_id),  # the Gate 0 numerator (Arm 2 geometry)
        "delta_naive_cu": cu(ht - hb),
        "delta_std_cu": cu(d_std),
    }


def grouped_auroc(x: torch.Tensor, y: np.ndarray, groups: np.ndarray, *, seed: int = 0) -> float:
    """Group-held-out dual-ridge probe AUROC (pairs never straddle train/test)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    arr = x.double().numpy()
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    best = 0.0
    for alpha in (1.0, 100.0, 10000.0):
        oof = np.zeros(len(y))
        for tr, va in gkf.split(arr, y, groups):
            mu = arr[tr].mean(axis=0, keepdims=True)
            xt, xv = arr[tr] - mu, arr[va] - mu
            yt = 2.0 * y[tr] - 1.0
            k = xt @ xt.T
            k[np.diag_indices_from(k)] += alpha * np.trace(k) / len(k) * 1e-3
            oof[va] = (xv @ xt.T) @ np.linalg.solve(k, yt - yt.mean())
        best = max(best, float(roc_auc_score(y, oof)))
    return best


def few_shot_acc(x: torch.Tensor, y: np.ndarray, groups: np.ndarray, *, k_groups: int = 4,
                 n_rep: int = 200, seed: int = 0) -> float:
    """Difference-of-means classifier trained on k groups, tested on the rest (balanced acc)."""
    arr = x.double().numpy()
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    accs = []
    for _ in range(n_rep):
        tr_g = rng.choice(ug, size=k_groups, replace=False)
        tr = np.isin(groups, tr_g)
        if len(np.unique(y[tr])) < 2:
            continue
        m1, m0 = arr[tr & (y == 1)].mean(0), arr[tr & (y == 0)].mean(0)
        w = m1 - m0
        s = (arr[~tr] - 0.5 * (m1 + m0)) @ w
        yt = y[~tr]
        accs.append(0.5 * ((s[yt == 1] > 0).mean() + (s[yt == 0] <= 0).mean()))
    return float(np.mean(accs))


def snr_table(ht: torch.Tensor, hb: torch.Tensor, y: np.ndarray, groups: np.ndarray,
              art_id: dict, art_std: dict, *, n_boot: int, seed: int) -> dict:
    """Point estimates plus group-bootstrap CIs for SNR and SNR ratios vs raw h_target."""
    raw = raw_mats(ht, hb, art_id, art_std)

    def snrs(idx: np.ndarray) -> dict[str, float]:
        r = reps_for(raw, idx)
        return {k: compute_fisher_snr(v, y[idx]) for k, v in r.items()}

    full_idx = np.arange(len(y))
    point = snrs(full_idx)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    boots: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if min((y[idx] == 0).sum(), (y[idx] == 1).sum()) < 2:
            continue
        s = snrs(idx)
        for k, v in s.items():
            boots[k].append(v)
    out: dict = {"snr": point, "ratio_vs_h_target": {}, "ratio_ci95": {}, "p_ratio_ge_1_25": {}}
    base = np.asarray(boots["h_target"])
    for k in point:
        r = np.asarray(boots[k]) / np.maximum(base, 1e-12)
        out["ratio_vs_h_target"][k] = point[k] / max(point["h_target"], 1e-12)
        out["ratio_ci95"][k] = [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]
        out["p_ratio_ge_1_25"][k] = float((r >= 1.25).mean())
    reps = reps_for(raw)
    out["grouped_auroc"] = {k: grouped_auroc(v, y, groups, seed=seed) for k, v in reps.items()}
    out["few_shot_bal_acc_k4"] = {k: few_shot_acc(v, y, groups, seed=seed) for k, v in reps.items()}
    return out


def load_tasks(n_pairs: int, seed: int = 0) -> dict[str, dict]:
    """The three audit pair sets and the historical-era control: prompts, labels, pair ids, metadata."""
    tasks: dict[str, dict] = {}
    for name, pairs in (
        ("refusal", load_refusal_pairs(n_pairs, source="auto", seed=seed)),
        ("eval_awareness", load_eval_awareness_pairs(n_pairs)),
        ("sycophancy", load_sycophancy_pairs(n_pairs)),
    ):
        prompts, labels, meta = flatten_binary_pairs(pairs)
        tasks[name] = {"prompts": prompts, "y": np.asarray(labels), "meta": meta, "kind": "audit",
                       "groups": np.asarray([m["pair_id"] for m in meta])}
    hist = load_historical_period_items(60)
    tasks["historical_era"] = {
        "prompts": [f"Read the following historical excerpt and identify its historical era:\n\n{it['text']}" for it in hist],
        "y": np.asarray([int(it["binary_label"]) for it in hist]),
        "meta": [{"topic": it["era"]} for it in hist],
        "kind": "content_control",
        "groups": np.asarray([it["item_id"] for it in hist]),
    }
    return tasks


def topic_labels(tasks: dict[str, dict]) -> dict[str, np.ndarray]:
    """Content controls on the audit prompts themselves: label = topic half, pairs kept together."""
    out = {name: np.asarray([int(m["topic"] in topic_set) for m in tasks[name]["meta"]])
           for name, topic_set in TOPIC_SPLITS.items()}
    meta = tasks["refusal"]["meta"]
    cats = sorted({m["topic"] for m in meta})
    y = np.asarray([int(m["topic"] in set(cats[: len(cats) // 2])) for m in meta])
    if 0 < y.sum() < len(y):
        out["refusal"] = y
    return out


def labelings(tasks: dict[str, dict]) -> list[tuple[str, str, np.ndarray, str]]:
    """(report name, task whose prompts it uses, labels, kind) for every audit label and content control."""
    out = [(name, name, t["y"], t["kind"]) for name, t in tasks.items()]
    out += [(f"{name}__topic", name, y, "content_control") for name, y in topic_labels(tasks).items()]
    return out


def asst_last_activations(tasks: dict[str, dict], tok: Any, *, target_model: str, base_model: str,
                          layer: int) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Layer activations at `asst_last` for every task prompt: Target, and Base under both wrappers."""
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    encs: dict[str, dict[str, list]] = {}
    for name, t in tasks.items():
        e: dict[str, list] = {"inst": [], "chatml": [], "plain": []}
        for txt in t["prompts"]:
            ie, bc = encode_aligned_pair(tok, txt, base_wrapper="chatml", enable_thinking=False, max_total_tokens=512)
            _, bp = encode_aligned_pair(tok, txt, base_wrapper="plain", enable_thinking=False, max_total_tokens=512)
            e["inst"].append(ie)
            e["chatml"].append(bc)
            e["plain"].append(bp)
        encs[name] = e

    def grab(model: cached_io.LazyLM, enc_list: list) -> torch.Tensor:
        exs = cached_io.cached_extract(model, [e.input_ids for e in enc_list], [[e.asst_last_idx] for e in enc_list],
                                       layer_index=layer, pad_token_id=pad)
        return torch.cat([x.hidden for x in exs], dim=0)

    tgt = cached_io.lazy_load_causal_lm(target_model)
    ht = {name: grab(tgt, encs[name]["inst"]) for name in tasks}
    tgt.free()
    base = cached_io.lazy_load_causal_lm(base_model)
    hb = {w: {name: grab(base, encs[name][w]) for name in tasks} for w in ("chatml", "plain")}
    base.free()
    return ht, hb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--target-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer-index", type=int, default=24)
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    k = args.layer_index
    p0 = Path(args.phase0_dir)
    tok = AutoTokenizer.from_pretrained(args.target_model)
    tasks = load_tasks(args.n_pairs, args.seed)
    ht, hb = asst_last_activations(tasks, tok, target_model=args.target_model, base_model=args.base_model, layer=k)
    if cached_io.RECORD_PATH:
        print("[record] gate_controls requests recorded")
        return
    art_id = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    art_std = load_alignment(p0 / f"align_L{k}_std_ridge.pt")

    report: dict = {"n_pairs": args.n_pairs, "layer": k, "wrappers": {}}
    for w in ("chatml", "plain"):
        rep_w: dict = {}
        for name, src, y, kind in labelings(tasks):
            rep_w[name] = snr_table(ht[src], hb[w][src], y, tasks[src]["groups"], art_id, art_std,
                                    n_boot=args.n_boot, seed=args.seed)
            rep_w[name]["kind"] = kind
        report["wrappers"][w] = rep_w

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    cols = ["h_target_cu", "h_base", "h_base_cu", "delta_idanc", "delta_idanc_cu", "delta_naive_cu", "delta_std_cu"]
    for w in ("chatml", "plain"):
        print(f"\n=== Base wrapper = {w}: Fisher SNR ratio vs raw h_target (95% pair-bootstrap CI) ===")
        print(f"{'task':26s} {'SNR(h_t)':>9s} " + " ".join(f"{c:>22s}" for c in cols))
        for name, r in report["wrappers"][w].items():
            cells = []
            for c in cols:
                lo, hi = r["ratio_ci95"][c]
                cells.append(f"{r['ratio_vs_h_target'][c]:5.2f} [{lo:4.2f},{hi:5.2f}]".rjust(22))
            print(f"{name:26s} {r['snr']['h_target']:9.3f} " + " ".join(cells))
        print(f"--- grouped-CV probe AUROC / 4-pair few-shot balanced accuracy ({w}) ---")
        for name, r in report["wrappers"][w].items():
            a, f = r["grouped_auroc"], r["few_shot_bal_acc_k4"]
            print(f"{name:26s} AUROC h_t={a['h_target']:.3f} h_b={a['h_base']:.3f} d_cu={a['delta_idanc_cu']:.3f} | "
                  f"few-shot h_t={f['h_target']:.3f} h_t_cu={f['h_target_cu']:.3f} h_b={f['h_base']:.3f} "
                  f"d_cu={f['delta_idanc_cu']:.3f}")


if __name__ == "__main__":
    main()
