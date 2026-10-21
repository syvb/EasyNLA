"""Readable outputs for the L24 standard-recipe RL results: 100 held-out examples and the FVE distribution chart.

    python scripts/heads_std_rl_report.py --results <fetched std-rl results dir> --out-dir <dir>

<results> needs checks_rl_t1/ and checks_sft_t1/ (heads_sft_checks.py --temperature 1, row-aligned) and
eval_ctx.parquet (each eval row's context and current token). Writes examples_100.md (≈3 random rows per head:
context, gold / SFT AV / RL AV explanations with cos and FVE) and fve_distribution.png (per-row FVE histograms).
FVE is in direction space with the head's mean write as baseline (≤ +100%; below 0 when cos < ≈0.5).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    R, out = Path(args.results), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gr = pd.read_parquet(R / "checks_rl_t1" / "generations.parquet")
    gs = pd.read_parquet(R / "checks_sft_t1" / "generations.parquet")
    assert (gr.example_idx.values == gs.example_idx.values).all()
    zr, zs = np.load(R / "checks_rl_t1" / "row_errors_direction.npz"), np.load(R / "checks_sft_t1" / "row_errors_direction.npz")
    cos = lambda z, k: 1 - z[k] / (2 * 64 ** 2)
    fve = lambda z, k: 1 - z[k] / z["base"]
    c_rl, c_sft, c_grl, c_gsft = cos(zr, "av__true"), cos(zs, "av__true"), cos(zr, "ar__gold"), cos(zs, "ar__gold")
    f_rl, f_sft, f_grl, f_gsft = fve(zr, "av__true"), fve(zs, "av__true"), fve(zr, "ar__gold"), fve(zs, "ar__gold")
    jr = json.loads((R / "checks_rl_t1" / "checks.json").read_text())["direction"]["fve"]
    js = json.loads((R / "checks_sft_t1" / "checks.json").read_text())["direction"]["fve"]
    ctx = pd.read_parquet(R / "eval_ctx.parquet").set_index("example_idx")
    rng = np.random.default_rng(args.seed)
    h = gr["head"].values
    pick = [i for hh in range(32) for i in rng.choice(np.where(h == hh)[0], 3, replace=False)]
    rest = [i for i in rng.permutation(len(gr)) if i not in set(pick)][:4]
    pick = sorted(pick + rest, key=lambda i: (h[i], i))
    q = re.compile(r'["“”]')
    clean = lambda s: s.replace("\n", " ⏎ ").replace("|", "\\|").strip()
    pct = lambda x: f"{100 * x:+.0f}%"
    L = ["# Head-contribution NLA (Qwen3-8B, layer 24): 100 held-out examples", "",
         "Each example is one attention head's write at one token of a held-out document (documents never seen in SFT or RL). The **AV** reads only the write vector and explains it; the **AR** reads the explanation (plus the head's name) and reconstructs the write's direction.", "",
         "Two scores per explanation, both in the direction-only space the models are trained in (write and reconstruction each scaled to norm √d):",
         "- `cos`: cosine between the AR's reconstruction and the true write. 0 ≈ no information; each head's mean direction gets ≈ 0.06.",
         "- `FVE`: 1 − error / error of predicting the head's mean write (the standard NLA definition). Because the reconstruction is normalized but the mean isn't, FVE = 2·cos − 1 roughly: it is **negative below cos ≈ 0.5** even when the explanation carries real information (see `docs/head_nla_l24_std_plan.md`).", "",
         "- **Gold**: Haiku 5.5 explanation written from the write's soft-token readouts (the SFT target), scored by the RL AR (and the SFT AR).",
         "- **SFT AV**: the warm-start verbalizer (standard NLA SFT recipe), scored by the SFT AR.",
         "- **RL AV**: after 400 GRPO steps, scored by the co-trained RL AR. Both AVs sampled at T=1 (as in RL).", "",
         "Across all 6,360 eval rows (pooled):", "", "| | cos | FVE |", "|---|---|---|",
         f"| SFT AV → SFT AR | {c_sft.mean():.3f} | {pct(js['av/true'])} |", f"| RL AV → RL AR | {c_rl.mean():.3f} | {pct(jr['av/true'])} |",
         f"| Gold → SFT AR | {c_gsft.mean():.3f} | {pct(js['ar/gold'])} |", f"| Gold → RL AR | {c_grl.mean():.3f} | {pct(jr['ar/gold'])} |",
         f"| AV given another write (RL) | {cos(zr, 'av__shuf').mean():.3f} | {pct(jr['av/shuf'])} |", "",
         f"RL explanations quote tokens in {100 * gr.av_true.str.contains(q).mean():.0f}% of cases (SFT {100 * gs.av_true.str.contains(q).mean():.0f}%). The RL gain is not a private AV↔AR code: a paraphrase of the RL text keeps 97% of what the RL AR recovers, while the text's form alone keeps 2%. But a blinded judge rates RL and SFT explanations equally faithful to the write's readouts. Read by the frozen SFT AR, the RL AV's explanations score cos 0.178 (greedy). Details: `docs/head_nla_l24_std_plan.md` on [`sv/head-nla`](https://github.com/syvb/EasyNLA/tree/sv/head-nla).", "",
         "Context: the document text just before the token, with the token itself in **[[ ]]**. The AV never sees this text.", ""]
    for n, i in enumerate(pick, 1):
        e = gr.example_idx.iloc[i]
        r = ctx.loc[e]
        before = tok.decode(r.ctx_ids[-41:-1]).replace("\n", " ⏎ ")
        cur = tok.decode([r.tok_cur]).replace("\n", "⏎")
        L += [f"## {n}. L24H{h[i]} · example {e}", "", f"> …{before}**[[{cur}]]**", "",
              f"- **Gold** (RL AR: cos {c_grl[i]:.2f}, FVE {pct(f_grl[i])}; SFT AR: cos {c_gsft[i]:.2f}, FVE {pct(f_gsft[i])}): {clean(gr.gold.iloc[i])}",
              f"- **SFT AV** (cos {c_sft[i]:.2f}, FVE {pct(f_sft[i])}): {clean(gs.av_true.iloc[i])}",
              f"- **RL AV** (cos {c_rl[i]:.2f}, FVE {pct(f_rl[i])}): {clean(gr.av_true.iloc[i])}", ""]
    (out / "examples_100.md").write_text("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = [("SFT AV → SFT AR", 100 * f_sft, "#2a78d6", "-", None), ("RL AV → RL AR (step 400)", 100 * f_rl, "#eb6834", "-", None),
         ("Control: RL AV given another write", 100 * fve(zr, "av__shuf"), "#8a8984", (0, (4, 3)), (-182, 4.6))]
    lo, hi, w = -250, 100, 5
    bins = np.arange(lo, hi + w, w)
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    H = []
    for name, x, c, ls, _ in S:
        hh, _ = np.histogram(np.clip(x, lo, hi - 1e-9), bins=bins)
        hh = 100 * hh / len(x)
        H.append(hh)
        ax.stairs(hh, bins, color=c, lw=2, ls=ls, label=name, baseline=None)
    ymax = max(x.max() for x in H) * 1.12
    ax.set_ylim(0, ymax)
    ax.axvline(0, color=ink2, lw=1)
    ax.text(2, ymax * 0.985, "0% = as good as predicting\nthe head's mean write", color=ink2, fontsize=8, va="top")
    for (name, x, c, ls, posn), hh in zip(S, H):
        med = np.median(x)
        ax.plot([med], [hh[np.searchsorted(bins, med) - 1]], marker="o", ms=7, color=c, mec=surf, mew=2, zorder=5)
        lab = f"{name.split(' (')[0].replace('Control: ', '')}\nmedian {med:+.0f}%"
        if posn:
            ax.text(posn[0], posn[1], lab, ha="center", fontsize=8.5, color=ink)
        else:
            k = np.argmax(hh)
            ax.annotate(lab, xy=(bins[k] + w / 2, hh[k]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=ink)
    ax.set_xlim(lo, hi)
    ax.set_xticks(range(-250, 101, 50))
    ax.set_xticklabels([f"{t:+d}%".replace("-", "−") if t else "0%" for t in range(-250, 101, 50)])
    ax.set_xlabel("FVE per held-out example  (direction space; baseline = the head's mean write)", color=ink2, fontsize=9.5, labelpad=8)
    ax.set_ylabel("share of examples (%, 5-point bins)", color=ink2, fontsize=9.5)
    ax.set_title(f"Head NLA, Qwen3-8B layer 24: FVE on {len(f_rl):,} held-out examples", loc="left", color=ink, fontsize=12.5, pad=24)
    ax.text(0, 1.02, f"RL moves most examples above the predict-the-mean baseline ({100 * (f_rl > 0).mean():.0f}% > 0, median "
            f"{100 * np.median(f_rl):+.0f}%); the SFT pair and the shuffled control sit near −90% to −100%.", transform=ax.transAxes, color=ink2, fontsize=9)
    ax.grid(axis="y", color=grid, lw=.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=ink, handlelength=3)
    fig.text(0.012, 0.012, "Both AVs sampled at T=1. Dots mark medians. Values outside the axis are clipped into the end bins (≤1% of any series). "
             "Below cos ≈ 0.5 a normalized reconstruction scores FVE < 0 even when informative.", fontsize=7.3, color=ink2)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig(out / "fve_distribution.png", facecolor=surf)
    print(f"wrote {out / 'examples_100.md'} and {out / 'fve_distribution.png'}")


if __name__ == "__main__":
    main()
