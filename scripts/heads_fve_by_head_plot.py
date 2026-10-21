"""Distribution of mean FVE across attention heads (held-out eval rows, T=1): one labelled block per head.

    python scripts/heads_fve_by_head_plot.py --results ~/l24_std/fetched/results --out docs/head_nla_l24_std_fve_by_head.png

FVE per row is direction-space with the head's mean write as baseline (as in heads_sft_checks.py); each
block is a head placed by its mean over its ~200 eval rows (RL pair); the SFT pair's range is given in the subtitle.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    R = Path(a.results)
    zr, zs = (np.load(R / f"checks_{k}_t1" / "row_errors_direction.npz") for k in ("rl", "sft"))
    assert (zr["head"] == zs["head"]).all()
    d = pd.DataFrame({"head": zr["head"], "rl": 100 * (1 - zr["av__true"] / zr["base"]),
                      "sft": 100 * (1 - zs["av__true"] / zs["base"])})
    g = d.groupby("head")[["rl", "sft"]].mean()
    n_rows = d.groupby("head").size()

    pct = lambda x: f"{x:+.0f}%".replace("-", "−")  # noqa: E731
    surf, ink, ink2, grid, col = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#eb6834"
    w, lo, hi = 10, -70, 70
    bins = np.arange(lo, hi + w, w)
    vals = g.rl.sort_values()
    assert vals.min() >= lo and vals.max() < hi
    fig, ax = plt.subplots(figsize=(10.5, 4.9), dpi=160)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    level: dict[int, int] = {}
    for h, v in vals.items():
        b = int(np.digitize(v, bins) - 1)
        y = level.get(b, 0)
        level[b] = y + 1
        ax.add_patch(FancyBboxPatch((bins[b] + 0.6, y + 0.08), w - 1.2, 0.84, boxstyle="round,pad=0,rounding_size=1.2",
                                    mutation_aspect=0.12, fc=col, ec=surf, lw=1.5, zorder=2))
        ax.text(bins[b] + w / 2, y + 0.5, f"H{h}", ha="center", va="center", fontsize=8, color=ink, zorder=3)
    top = max(level.values())
    m = vals.mean()
    ax.axvline(0, color=ink2, lw=1, zorder=1)
    ax.text(-1, top + 0.55, "0% = as good as predicting\nthe head's mean write", ha="right", va="top", fontsize=7.8, color=ink2)
    ax.plot([m, m], [0, top + 0.25], color=ink, lw=1.2, ls=(0, (3, 2)), zorder=1)
    ax.text(m, top + 0.32, f"mean over heads {pct(m)}", ha="center", va="bottom", fontsize=8.5, color=ink)
    ax.set_ylim(0, top + 1.0)
    ax.set_yticks(range(0, top + 1))
    ax.set_ylabel("number of heads", color=ink2, fontsize=9)
    ax.grid(axis="y", color=grid, lw=.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    ticks = range(lo, hi + 1, 10)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:+d}%".replace("-", "−") if t else "0%" for t in ticks])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("mean FVE of the head over its held-out examples  (10-point bins)", color=ink2, fontsize=9, labelpad=8)
    fig.suptitle("Head NLA, Qwen3-8B layer 24: average FVE by attention head (RL step 400)", x=0.07, ha="left",
                 fontsize=12.5, color=ink, y=0.975)
    fig.text(0.07, 0.885, f"Each block is one head, placed by its mean FVE over ~{int(n_rows.median())} held-out examples "
             f"(T=1). {int((vals > 0).sum())} of 32 heads beat the predict-the-mean baseline.\n"
             f"For comparison, the SFT warm start puts every head between {pct(g.sft.min())} and {pct(g.sft.max())} "
             f"(mean {pct(g.sft.mean())}).", fontsize=8.5, color=ink2, va="top", linespacing=1.5)
    fig.text(0.07, 0.015, "FVE in direction space, baseline = the head's mean write; at most +100%, a random direction scores "
             "≈ −100%. H16–19 share one KV head.", fontsize=7.6, color=ink2)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.77, bottom=0.16)
    plt.savefig(a.out, facecolor=surf)
    print(f"wrote {a.out}")
    print(g.round(1).sort_values("rl", ascending=False).to_string())


if __name__ == "__main__":
    main()
