"""Aggregate future-lens eval JSONL records into tables (and PNGs if matplotlib is present).

Every eval / baseline run appends records {layer, N, condition, metric, value, seed,
checkpoint, group?, ...}. This script is the ONLY consumer: it groups by (group or
checkpoint, condition, layer, N), keeps the LAST record per seed (re-running an eval
appends, it must not count twice), averages over seeds (mean +- std), and prints

  * the headline table: p1 per (layer, N) for real vs shuffled and their gap,
    per checkpoint (SFT, GRPO seeds...), with the n-gram and linear-probe baselines
  * the success-criteria check from the spec: GRPO - SFT gain at N=2 or N=3 (our N is
    the spec's N: N=1 <-> x_{t+2}, so N=2,3 <-> x_{t+3}, x_{t+4}; K=4 is in k_choices so
    N=3 is a first-class prompt) on real activations vs the same gain on shuffled ones

    python -m nla.future_lens.plots evals/*.jsonl --out plots/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(paths: list[str]) -> list[dict]:
    recs = []
    for pth in paths:
        with open(pth) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def key_of(r: dict) -> tuple:
    name = r.get("group") or r.get("checkpoint")
    return (str(name), str(r.get("condition")), int(r.get("layer", -1)), int(r["N"]), r["metric"])


def aggregate_per_seed(recs: list[dict]) -> dict[tuple, dict]:
    """key -> {seed: value}; the last record per (key, seed) wins."""
    per_seed: dict[tuple, dict] = defaultdict(dict)
    for r in recs:
        per_seed[key_of(r)][r.get("seed", 0)] = float(r["value"])
    return per_seed


def aggregate(recs: list[dict]) -> dict[tuple, dict]:
    """One value per (key, seed) — the last record wins — then mean/std over seeds."""
    per_seed = aggregate_per_seed(recs)
    return {k: {"mean": float(np.mean(list(v.values()))), "std": float(np.std(list(v.values()))),
                "n_seeds": len(v)} for k, v in per_seed.items()}


def partition(recs: list[dict]) -> dict[tuple, list[dict]]:
    """Never pool across labels or eval sets: records are grouped by (label, evalset) first."""
    parts: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        parts[(str(r.get("label", "text")), str(r.get("evalset", "")))].append(r)
    return parts


def table(agg: dict, metric: str = "p1") -> str:
    ckpts = sorted({k[0] for k in agg if k[4] == metric})
    layers = sorted({k[2] for k in agg if k[4] == metric})
    Ns = sorted({k[3] for k in agg if k[4] == metric})
    lines = []
    for ck in ckpts:
        conds = sorted({k[1] for k in agg if k[0] == ck and k[4] == metric})
        lines.append(f"\n### {ck}  ({metric})")
        hdr = "| layer | cond | " + " | ".join(f"N={N}" for N in Ns) + " |"
        lines.append(hdr); lines.append("|" + "---|" * (len(Ns) + 2))
        for layer in layers:
            for cond in conds:
                cells = []
                for N in Ns:
                    a = agg.get((ck, cond, layer, N, metric))
                    cells.append("" if a is None else (f"{a['mean']:.3f}" + (f"±{a['std']:.3f}" if a["n_seeds"] > 1 else "")))
                if any(cells):
                    lines.append(f"| {layer} | {cond} | " + " | ".join(cells) + " |")
            a_r = [agg.get((ck, "real", layer, N, metric)) for N in Ns]
            a_s = [agg.get((ck, "shuffled", layer, N, metric)) for N in Ns]
            if any(a_r) and any(a_s):
                cells = ["" if (r is None or s is None) else f"{r['mean'] - s['mean']:+.3f}" for r, s in zip(a_r, a_s)]
                lines.append(f"| {layer} | **gap** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def success_check(agg: dict, sft: str, rl: str, offsets=(2, 3), metric="p1", per_seed: dict | None = None) -> str:
    """Spec: GRPO-SFT gain > 5pp at N=2 or N=3 on real activations, shuffled gains < 1/3 of it.
    Reported per layer AND pooled over layers (the pre-declared primary cell); with `per_seed`
    (from aggregate_per_seed) the pooled line also states whether every RL seed beats SFT."""
    out = []
    layers = sorted({k[2] for k in agg if k[0] == rl and k[2] >= 0})
    for N in offsets:   # pooled over layers first
        r_sft = [agg.get((sft, "real", l, N, metric)) for l in layers]
        r_rl = [agg.get((rl, "real", l, N, metric)) for l in layers]
        s_sft = [agg.get((sft, "shuffled", l, N, metric)) for l in layers]
        s_rl = [agg.get((rl, "shuffled", l, N, metric)) for l in layers]
        if not (all(r_sft) and all(r_rl)):
            continue
        gain = float(np.mean([x["mean"] for x in r_rl]) - np.mean([x["mean"] for x in r_sft]))
        sgain = (float(np.mean([x["mean"] for x in s_rl]) - np.mean([x["mean"] for x in s_sft]))
                 if all(s_sft) and all(s_rl) else float("nan"))
        verdict = ("POSITIVE" if gain > 0.05 and (np.isnan(sgain) or sgain < gain / 3)
                   else "priors-only" if gain > 0.05 else "no gain")
        seeds_txt = ""
        if per_seed:
            sft_pool = np.mean([np.mean(list(per_seed[(sft, "real", l, N, metric)].values())) for l in layers])
            rl_seeds = {}
            for l in layers:
                for sd, v in per_seed.get((rl, "real", l, N, metric), {}).items():
                    rl_seeds.setdefault(sd, []).append(v)
            gains = {sd: float(np.mean(v) - sft_pool) for sd, v in rl_seeds.items()}
            seeds_txt = f"; per-seed gains {', '.join(f'{sd}:{g:+.3f}' for sd, g in sorted(gains.items()))}" + \
                        (" (all > 0)" if gains and all(g > 0 for g in gains.values()) else " (NOT all > 0)")
        out.append(f"POOLED layers {layers} N={N} [{metric}]: real gain {gain:+.3f}, shuffled gain {sgain:+.3f} -> {verdict}{seeds_txt}")
    for layer in layers:
        for N in offsets:
            r_sft, r_rl = agg.get((sft, "real", layer, N, metric)), agg.get((rl, "real", layer, N, metric))
            s_sft, s_rl = agg.get((sft, "shuffled", layer, N, metric)), agg.get((rl, "shuffled", layer, N, metric))
            if not (r_sft and r_rl):
                continue
            gain = r_rl["mean"] - r_sft["mean"]
            sgain = (s_rl["mean"] - s_sft["mean"]) if (s_sft and s_rl) else float("nan")
            verdict = ("POSITIVE" if gain > 0.05 and (np.isnan(sgain) or sgain < gain / 3)
                       else "priors-only" if gain > 0.05 else "no gain")
            out.append(f"L{layer} N={N}: real gain {gain:+.3f} (±{r_rl['std']:.3f} over {r_rl['n_seeds']} seeds), "
                       f"shuffled gain {sgain:+.3f} -> {verdict}")
    return "\n".join(out)


def maybe_plot(agg: dict, out_dir: Path, metric="p1"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed; tables only")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted({k[0] for k in agg if k[4] == metric})
    layers = sorted({k[2] for k in agg if k[4] == metric and k[2] >= 0})
    Ns = sorted({k[3] for k in agg if k[4] == metric and k[3] >= 0})
    fig, axes = plt.subplots(1, max(1, len(layers)), figsize=(4 * max(1, len(layers)), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, layer in zip(axes, layers):
        for ck in ckpts:
            for cond, ls in (("real", "-"), ("shuffled", "--")):
                ys = [agg.get((ck, cond, layer, N, metric)) for N in Ns]
                if not any(ys):
                    continue
                ax.plot(Ns, [np.nan if y is None else y["mean"] for y in ys], ls, marker="o", label=f"{ck} {cond}")
        for ck in ckpts:      # layer-agnostic baselines (layer -1)
            ys = [agg.get((ck, "baseline", -1, N, metric)) for N in Ns]
            if any(ys):
                ax.plot(Ns, [np.nan if y is None else y["mean"] for y in ys], ":", label=ck)
        ax.set_title(f"layer {layer}"); ax.set_xlabel("N (offset)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_by_layer.png", dpi=130)
    print(f"[plots] wrote {out_dir / f'{metric}_by_layer.png'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default=None, help="directory for PNGs + summary.md")
    p.add_argument("--metric", default="p1")
    p.add_argument("--sft", default=None, help="group/checkpoint name of the SFT eval for the success check")
    p.add_argument("--rl", default=None, help="group/checkpoint name of the GRPO eval for the success check")
    p.add_argument("--offsets", default="2,3", help="N values for the success check (spec: 2 or 3)")
    args = p.parse_args(argv)
    parts = partition(load(args.jsonl))
    txt_all = []
    for (label, evalset), recs in sorted(parts.items()):
        agg = aggregate(recs)
        txt = f"## label={label}" + (f" evalset={evalset}" if evalset else "") + "\n\n" + table(agg, args.metric)
        if args.sft and args.rl:
            offs = tuple(int(x) for x in args.offsets.split(","))
            for metric in dict.fromkeys([args.metric, "p1", "tf_p1"]):   # primary first, then both conventions
                txt += f"\n\n### success criteria [{metric}]\n" + success_check(
                    agg, args.sft, args.rl, offsets=offs, metric=metric, per_seed=aggregate_per_seed(recs))
        txt_all.append(txt)
        if args.out:
            out = Path(args.out) / (f"{label}_{evalset}" if evalset else label); out.mkdir(parents=True, exist_ok=True)
            (out / "summary.md").write_text(txt)
            maybe_plot(agg, out, args.metric)
    print("\n\n".join(txt_all))


if __name__ == "__main__":
    main()
