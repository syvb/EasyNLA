"""Figure for heads_quote_topk_shuffle.py: share of quoted terms that name one of the explaining head's top-k
attended tokens, against a shuffled explanation, the other heads, and k random positions.

    python scripts/heads_quote_topk_plot.py --json docs/head_nla_quote_topk_shuffle.json \\
        --out docs/head_nla_quote_topk_shuffle.png [--rule strict]

Top row: all quotes. Bottom row: quotes that appear in the context (the true and the control quotes each
conditioned on being in the context at the site they are checked against).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SRCS = [("std RL", "Standard RL (step 400)"), ("matryoshka", "Matryoshka (step 640)"), ("matryoshka, first 16 tokens", "Matryoshka, first 16 tokens")]


def single(a, r: dict, title: str, ks: list[int], series: list, colours: tuple) -> None:
    """One source, quotes that appear in the text only."""
    surf, ink, ink2, grid = colours
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=160)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    for key, ckey, lab, col, mk, ls in series[::-1]:
        y = [100 * r["k"][str(k)][ckey] for k in ks]
        ax.plot(ks, y, color=col, lw=2 if key != "uniform" else 1.4, ls=ls, marker=mk, ms=6.5, mec=surf, mew=1.2,
                zorder=3 if key == "true" else 2, solid_capstyle="round", label=lab)
    for key, ckey, _, col, *_ in series[:3]:                                    # value labels at k = 10
        v = 100 * r["k"]["10"][ckey]
        ax.annotate(f"{v:.0f}%", (10, v), xytext=(0, 8 if key != "shuffled" else -9), textcoords="offset points",
                    ha="center", va="bottom" if key != "shuffled" else "top", fontsize=9,
                    color=ink if key == "true" else ink2, fontweight="bold" if key == "true" else "normal")
    ax.grid(axis="y", color=grid, lw=.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    ax.set_xticks(ks)
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_xlabel("k (the head's most-attended tokens, sink excluded)", fontsize=8.8, color=ink2)
    ax.set_ylabel("quotes naming a top-k token", fontsize=8.8, color=ink2)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="upper left", frameon=False, fontsize=8.2, labelcolor=ink)
    ax.set_title(f"{title}: quotes that appear in the text", loc="left", fontsize=11.5, color=ink, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, facecolor=surf)
    print(f"wrote {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rule", default="strict", choices=["strict", "loose"])
    p.add_argument("--single", default=None, choices=[k for k, _ in SRCS],
                   help="draw only this source's in-text panel (quotes that appear in the text)")
    a = p.parse_args()
    J = json.load(open(a.json))
    R = J["rules"][a.rule]
    ks = J["ks"]

    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    series = [  # (key, key given in context, label, colour, marker, linestyle)
        ("true", "true_given_in_context", "This head's top-k (the actual explanation)", "#2a78d6", "o", "-"),
        ("other_heads", "other_heads_given_in_context", "Other 31 heads' top-k, same quotes", "#eb6834", "s", "-"),
        ("shuffled", "shuffled_given_in_context", "Shuffled explanation (same head, same text)", "#1baf7a", "^", "-"),
        ("uniform", "uniform_given_in_context", "k random positions", "#8f8e89", None, (0, (4, 3))),
    ]
    if a.single:
        single(a, R[a.single], dict(SRCS)[a.single].split(" (")[0].replace("Standard RL", "Standard NLA"), ks, series,
               (surf, ink, ink2, grid))
        return
    rows = [("All quotes", 0), ("Quotes that appear in the text", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), dpi=160, sharex=True, sharey="row")
    fig.patch.set_facecolor(surf)
    for (row_title, ri), axr in zip(rows, axes):
        for (src, title), ax in zip(SRCS, axr):
            r = R[src]
            ax.set_facecolor(surf)
            for key, ckey, _, col, mk, ls in series[::-1]:
                y = [100 * r["k"][str(k)][ckey if ri else key] for k in ks]
                ax.plot(ks, y, color=col, lw=2 if key != "uniform" else 1.4, ls=ls, marker=mk, ms=6.5,
                        mec=surf, mew=1.2, zorder=3 if key == "true" else 2, solid_capstyle="round")
            # label the headline pair at k = 10, above the explaining head's point
            k10 = r["k"]["10"]
            t, sh = (100 * k10["true_given_in_context"], 100 * k10["shuffled_given_in_context"]) if ri else \
                (100 * k10["true"], 100 * k10["shuffled"])
            fmt = lambda x: f"{x:.0f}%" if x >= 10 else f"{x:.1f}%"  # noqa: E731
            ax.annotate(f"top-10: {fmt(t)} vs {fmt(sh)} shuffled", (10, t), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=ink)
            ax.grid(axis="y", color=grid, lw=.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(grid)
            ax.tick_params(colors=ink2, labelsize=8.5, length=0)
            ax.set_xticks(ks)
            ax.set_xlim(0, 21.5)
            if ri == 0:
                n = r["n_quotes"]
                ax.set_title(f"{title}\n", fontsize=10.5, color=ink, loc="left", fontweight="bold")
                ax.text(0, 1.03, f"{n / 1000:.0f}k quotes" if n >= 10000 else f"{n / 1000:.1f}k quotes",
                        transform=ax.transAxes, fontsize=8.5, color=ink2, va="bottom")
            elif src == "matryoshka":
                ax.set_xlabel("k (the head's most-attended positions, sink excluded)", fontsize=8.5, color=ink2)
        axr[0].set_ylabel(f"{row_title}\n% naming a top-k token", fontsize=9, color=ink2)
    axes[1, 0].set_ylim(0, 100)
    axes[0, 0].set_ylim(0, 19)
    axes[0, 0].set_yticks([0, 5, 10, 15])
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ms=6.5, mec=surf, lw=2 if k != "uniform" else 1.4, ls=ls,
                               label=lab) for k, _, lab, c, m, ls in series],
               loc="upper left", bbox_to_anchor=(0.006, 0.955), ncol=4, frameon=False, fontsize=8.5, labelcolor=ink,
               handlelength=2.6, columnspacing=1.6)
    fig.text(0.012, 0.985, "Quoted terms name the head's top-k attended tokens", fontsize=13, color=ink,
             fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.6, w_pad=1.2)
    fig.savefig(a.out, facecolor=surf)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
