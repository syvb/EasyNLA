"""Reconstruction-quality figures for the matryoshka head NLA (step 640) and the standard one (step 400), from the
public per-row files of syvb/head-nla-qwen3-8b (downloaded anonymously). CPU only.

    python scripts/heads_mtry_quality_plots.py --out-dir docs [--n-boot 1000]

Inputs:
- l24/mtry/sweep/row_cos.npz + generations.parquet: 2,000 held-out writes; per-row cosine when the AR reads the
  first L explanation tokens (L = 0 [empty], 1, 2, 4, …, 120, whole) for each (AV text, AR) pair;
- l24/{mtry/checks_t1, std_rl/results/checks_rl_t1}/row_errors_direction.npz: all 6,360 held-out writes, whole
  explanations (T=1), per-row errors, and the per-row baseline error (the head's mean write) used for both.
FVE = 1 − Σ error / Σ baseline error (direction space, both vectors at norm √d), pooled over rows as in the reports;
it reproduces l24/mtry/report/summary.json. CIs are 95% bootstraps over writes (within a head for per-head values).

Writes to --out-dir (prefix head_nla_l24_):
  mtry_marginal_fve.png        FVE gained per added token at each truncation step, and per step
  mtry_quality_by_prefix.png   per-write cosine (median, IQR), share of writes beating the head's mean write, and
                               share of the empty → whole gain reached, against prefix length
  fve_distribution_std_vs_mtry.png   per-write cosine and FVE of whole explanations, std vs matryoshka vs shuffled
  fve_per_head_std_vs_mtry.png       per-head FVE: whole explanations (std vs matryoshka), first 16 tokens
                                     (fair baseline vs matryoshka)
  mtry_prefix_by_head.png      per-head FVE at every prefix length, matryoshka and fair baseline
  mtry_marginal_by_head.png    per-head FVE gained at each truncation step, matryoshka
  mtry_quality.json            every plotted number
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm  # noqa: E402

HF_REPO = "syvb/head-nla-qwen3-8b"
S2 = 4096.0                                                   # |v|² at norm √d (d = 4096)
LS = ["0", "1", "2", "4", "8", "16", "32", "64", "120", "full"]
LLAB = ["empty", "1", "2", "4", "8", "16", "32", "64", "120", "whole"]
STEPS = [f"{a}→{b}" for a, b in zip(LLAB[:-1], LLAB[1:])]
# (key in row_cos, label, colour, line style, AV text's length column)
LINES = [("mtry|mtry", "Matryoshka (step-640 AV + AR)", "#eb6834", "-", "ntok__mtry"),
         ("rl400|mtry_w", "Fair baseline: std AV + warm-up AR", "#2a78d6", "-", "ntok__rl400"),
         ("rl400|rl400", "Std pair, no prefix training (step 400)", "#7d7c77", (0, (4, 3)), "ntok__rl400")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STD_C, MTRY_C, SHUF_C = "#2a78d6", "#eb6834", "#8f8e89"
DIV = LinearSegmentedColormap.from_list("fve", ["#8c1f22", "#e34948", "#f0efec", "#2a78d6", "#0d366b"])


def dl(path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, path, repo_type="dataset", token=False)


def pct(x: float, signed: bool = True) -> str:
    return (f"{100 * x:+.0f}%" if signed else f"{100 * x:.0f}%").replace("-", "−")


def style(ax, ygrid: bool = True) -> None:
    ax.set_facecolor(SURF)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def heading(fig, title: str, sub: str, y0: float = 0.975) -> None:
    fig.text(0.012, y0, title, fontsize=13, color=INK, fontweight="bold", va="top")
    fig.text(0.012, y0 - 0.045 * (6.0 / fig.get_figheight()), sub, fontsize=8.8, color=INK2, va="top", linespacing=1.45)


def fig_legend(fig, handles, y: float, ncol: int) -> None:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, y), ncol=ncol, frameon=False, fontsize=8.5,
               labelcolor=INK, handlelength=2.8, columnspacing=1.8)


def line_handle(col, ls, lab, marker="o"):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=col, ls=ls, lw=2, marker=marker, ms=5.5, mec=SURF, label=lab)


EMPTY_NOTE = ("“empty” = the AR reads only “Attention head L24Hh:”. It never trained on empty text and scores about "
              "−100%,\nbelow the −89% of the best head-only guess (the head's mean direction), so about 11 points of the "
              "first step only bring it up to that guess.")


def pcty(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:+.0f}%".replace("-", "−") if y else "0%"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out_dir)
    rng = np.random.default_rng(a.seed)

    rc = np.load(dl("l24/mtry/sweep/row_cos.npz"))
    gen = pq.read_table(dl("l24/mtry/sweep/generations.parquet"),
                        columns=["eval_row", "head", "ntok__mtry", "ntok__rl400"]).to_pandas()
    cm = np.load(dl("l24/mtry/checks_t1/row_errors_direction.npz"))
    cs = np.load(dl("l24/std_rl/results/checks_rl_t1/row_errors_direction.npz"))
    assert np.allclose(cm["base"], cs["base"]) and (cm["head"] == cs["head"]).all()
    heads = gen["head"].to_numpy()
    base = cm["base"][gen.eval_row.to_numpy()].astype(np.float64)           # per-row baseline error, sweep rows
    N = len(gen)
    summ = json.loads(Path(dl("l24/mtry/report/summary.json")).read_text())

    def err(key: str, L: str) -> np.ndarray:
        return 2 * S2 * (1 - rc[f"{key}|{L}"].astype(np.float64))

    E = {k: np.stack([err(k, L) for L in LS]) for k, *_ in LINES}           # [len(LS), N]
    fve = {k: 1 - e.sum(1) / base.sum() for k, e in E.items()}
    assert abs(fve["mtry|mtry"][LS.index("16")] - summ["lines"][LINES[0][1]]["16"]["fve"]) < 1e-6
    boot = rng.integers(0, N, size=(a.n_boot, N))
    bsum = base[boot].sum(1)                                                  # [B]

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    res: dict = {"n_sweep_rows": N, "lengths": LS, "lines": {}}
    for k, lab, _, _, nt in LINES:
        e = E[k]
        gain = (e[:-1] - e[1:]).sum(1) / base.sum()                            # FVE gained per step
        gain_b = (e[:-1, boot] - e[1:, boot]).sum(2) / bsum                    # [steps, B]
        # tokens added by each step, 0→1 … 120→whole (the last: mean tokens past 120, 0 for shorter explanations)
        width = np.array([1, 1, 2, 4, 8, 16, 32, 56, np.maximum(gen[nt].to_numpy() - 120, 0).mean()], float)
        fv_b = 1 - e[:, boot].sum(2) / bsum                                    # [L, B]
        rf = 1 - e / base                                                      # per-row FVE
        cosr = np.stack([rc[f"{k}|{L}"] for L in LS]).astype(np.float64)
        tot = fve[k][-1] - fve[k][0]
        res["lines"][lab] = {
            "fve": fve[k].tolist(), "fve_ci": [ci(x) for x in fv_b],
            "step_gain": gain.tolist(), "step_gain_ci": [ci(x) for x in gain_b],
            "step_tokens": width.tolist(),
            "per_token": (gain / width).tolist(), "per_token_ci": [ci(x / w) for x, w in zip(gain_b, width)],
            "share_of_gain": ((fve[k] - fve[k][0]) / tot).tolist(),
            "share_of_gain_ci": [ci((fv_b[i] - fv_b[0]) / (fv_b[-1] - fv_b[0])) for i in range(len(LS))],
            "rows_beating_mean": (rf > 0).mean(1).tolist(),
            "rows_beating_mean_ci": [ci((rf[i][boot] > 0).mean(1)) for i in range(len(LS))],
            "cos_median": np.median(cosr, 1).tolist(), "cos_q25": np.quantile(cosr, 0.25, 1).tolist(),
            "cos_q75": np.quantile(cosr, 0.75, 1).tolist(), "cos_mean": cosr.mean(1).tolist()}
    R = res["lines"]
    x = np.arange(len(LS))

    # ---- 1. marginal FVE at each truncation step ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.1), dpi=160)
    fig.patch.set_facecolor(SURF)
    xs = np.arange(len(STEPS))
    for k, lab, col, ls, _ in LINES:
        r = R[lab]
        y = 100 * np.array(r["per_token"])
        lo, hi = (100 * np.array([c[i] for c in r["per_token_ci"]]) for i in (0, 1))
        a1.errorbar(xs, y, yerr=[y - np.maximum(lo, 1e-4), hi - y], color=col, ls=ls, lw=2, marker="o", ms=5.5,
                    mec=SURF, mew=1.2, elinewidth=1.2, capsize=0, label=lab, zorder=3)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    a1.set_ylabel("FVE points gained per added token (log scale)", color=INK2, fontsize=9)
    a1.set_title("Per token", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    w = 0.38
    for j, (k, lab, col, ls, _) in enumerate(LINES[:2]):
        r = R[lab]
        y = 100 * np.array(r["step_gain"])
        lo, hi = (100 * np.array([c[i] for c in r["step_gain_ci"]]) for i in (0, 1))
        a2.bar(xs + (j - 0.5) * w, y, w * 0.92, color=col, label=lab, zorder=2)
        a2.errorbar(xs + (j - 0.5) * w, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK2, elinewidth=1, zorder=3)
    v = R[LINES[0][1]]["step_gain"][0]
    a2.annotate(f"+{100 * v:.0f} points from\nthe first token", (xs[0] - 0.5 * w, 100 * v), xytext=(14, -6),
                textcoords="offset points", fontsize=8.5, color=INK, va="top")
    a2.set_ylabel("FVE points gained in the step", color=INK2, fontsize=9)
    a2.set_title("Per truncation step (each step doubles the prefix)", loc="left", fontsize=10.5, color=INK,
                 fontweight="bold")
    for ax in (a1, a2):
        style(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(STEPS, rotation=40, ha="right", fontsize=8)
        ax.set_xlabel("explanation tokens the AR reads: from → to", color=INK2, fontsize=8.8)
    a1.legend(loc="upper right", frameon=False, fontsize=8.3, labelcolor=INK)
    a2.legend(loc="upper right", frameon=False, fontsize=8.3, labelcolor=INK)
    heading(fig, "Marginal FVE at each truncation point: matryoshka puts the information first",
            f"{N:,} held-out writes, one T=1 explanation each; the AR reads the first L tokens. FVE pooled over writes; "
            "bars = 95% bootstrap over writes.\nThe last step averages "
            f"{R[LINES[0][1]]['step_tokens'][-1]:.0f} extra tokens (matryoshka) and {R[LINES[1][1]]['step_tokens'][-1]:.0f} "
            "(std); explanations shorter than 120 tokens add nothing there.\n" + EMPTY_NOTE)
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=2.5)
    fig.savefig(out / "head_nla_l24_mtry_marginal_fve.png", facecolor=SURF)
    plt.close(fig)

    # ---- 2. per-write quality by prefix length ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.6), dpi=160)
    fig.patch.set_facecolor(SURF)
    for k, lab, col, ls, _ in LINES:
        r = R[lab]
        axs[0].fill_between(x, r["cos_q25"], r["cos_q75"], color=col, alpha=0.12, lw=0)
        axs[0].plot(x, r["cos_median"], color=col, ls=ls, lw=2, marker="o", ms=5, mec=SURF, mew=1.2)
        y = 100 * np.array(r["rows_beating_mean"])
        lo, hi = (100 * np.array([c[i] for c in r["rows_beating_mean_ci"]]) for i in (0, 1))
        axs[1].fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        axs[1].plot(x, y, color=col, ls=ls, lw=2, marker="o", ms=5, mec=SURF, mew=1.2)
        y = 100 * np.array(r["share_of_gain"])
        axs[2].plot(x, y, color=col, ls=ls, lw=2, marker="o", ms=5, mec=SURF, mew=1.2)
    axs[0].set_title("Cosine with the true write, per write", loc="left", fontsize=10, color=INK, fontweight="bold")
    axs[0].set_ylabel("median (band: middle 50% of writes)", color=INK2, fontsize=8.8)
    axs[1].set_title("Writes reconstructed better than\nthe head's mean write (FVE > 0)", loc="left", fontsize=10,
                     color=INK, fontweight="bold")
    axs[1].set_ylim(0, 100)
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    axs[2].set_title("Share of the empty → whole FVE gain\nreached by the first L tokens", loc="left", fontsize=10,
                     color=INK, fontweight="bold")
    axs[2].set_ylim(0, 105)
    axs[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    m, b = R[LINES[0][1]], R[LINES[1][1]]
    i16 = LS.index("16")
    axs[2].text(0.98, 0.04, f"Matryoshka: {100 * m['share_of_gain'][1]:.0f}% at token 1,\n"
                f"fair baseline {100 * b['share_of_gain'][1]:.0f}%", transform=axs[2].transAxes, fontsize=8.3,
                color=INK, ha="right", va="bottom")
    axs[1].text(0.98, 0.04, f"At 16 tokens:\nmatryoshka {100 * m['rows_beating_mean'][i16]:.0f}%,\n"
                f"fair baseline {100 * b['rows_beating_mean'][i16]:.0f}%", transform=axs[1].transAxes, fontsize=8.3,
                color=INK, ha="right", va="bottom")
    for ax in axs:
        style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(LLAB, fontsize=8)
        ax.set_xlabel("explanation tokens the AR reads", color=INK2, fontsize=8.8)
    heading(fig, "How reconstruction builds up as the AR reads more of the explanation",
            f"{N:,} held-out writes (all 32 heads). Middle: bands = 95% bootstrap over writes. Right: (FVE(L) − FVE(empty)) / "
            "(FVE(whole) − FVE(empty)), pooled FVE.\n" + EMPTY_NOTE)
    fig_legend(fig, [line_handle(col, ls, lab) for _, lab, col, ls, _ in LINES], 0.845, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=2.2)
    fig.savefig(out / "head_nla_l24_mtry_quality_by_prefix.png", facecolor=SURF)
    plt.close(fig)

    # ---- 3. whole-explanation distributions, std vs matryoshka (all 6,360 held-out writes) ----
    bfull = cm["base"].astype(np.float64)
    dist = [("Std pair (step 400)", cs["av__true"], STD_C, "-"), ("Matryoshka pair (step 640)", cm["av__true"], MTRY_C, "-"),
            ("Shuffled control (matryoshka)", cm["av__shuf"], SHUF_C, (0, (4, 3)))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=160)
    fig.patch.set_facecolor(SURF)
    res["whole"] = {"n_rows": int(len(bfull))}
    for lab, e, col, ls in dist:
        e = e.astype(np.float64)
        c = 1 - e / (2 * S2)
        f = 100 * (1 - e / bfull)
        a1.hist(c, bins=np.linspace(-0.4, 1, 57), histtype="step", color=col, ls=ls, lw=2)
        a2.hist(f[f >= -150], bins=np.linspace(-150, 100, 51), histtype="step", color=col, ls=ls, lw=2)
        res["whole"][lab] = {"cos_mean": float(c.mean()), "cos_median": float(np.median(c)),
                             "fve_pooled": float(1 - e.sum() / bfull.sum()), "rows_fve_gt0": float((e < bfull).mean()),
                             "rows_fve_below_-150": float((f < -150).mean())}
    for ax, key, fmt in ((a1, "cos_mean", lambda v: f"{v:.2f}"), (a2, "fve_pooled", lambda v: pct(v))):
        style(ax)
        ax.set_ylabel("writes", color=INK2, fontsize=8.8)
        for lab, _, col, _ in dist[:2]:
            v = res["whole"][lab][key] * (100 if key == "fve_pooled" else 1)
            ax.axvline(v, color=col, lw=1, ls=(0, (1, 2)))
    a1.set_title("Cosine with the true write", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    a1.set_xlabel("cosine (dotted: mean)", color=INK2, fontsize=8.8)
    a2.set_title("Per-write FVE", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    a2.set_xlabel("FVE vs the head's mean write (dotted: pooled FVE)", color=INK2, fontsize=8.8)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:+.0f}%".replace("-", "−") if y else "0%"))
    W = res["whole"]
    a2.text(0.02, 0.97, "Below −150%, not shown: " + ", ".join(
        f"{lab.split(' (')[0].lower()} {100 * W[lab]['rows_fve_below_-150']:.1f}%" for lab, *_ in dist),
        transform=a2.transAxes, fontsize=8, color=INK2, va="top")
    heading(fig, "Whole explanations: how well each write is reconstructed",
            f"All {len(bfull):,} held-out writes, one T=1 explanation each, read by its own AR. Mean cosine "
            f"{W[dist[0][0]]['cos_mean']:.2f} (std) and {W[dist[1][0]]['cos_mean']:.2f} (matryoshka); pooled FVE "
            f"{pct(W[dist[0][0]]['fve_pooled'])} and {pct(W[dist[1][0]]['fve_pooled'])}.\nShuffled control: the matryoshka "
            "AV describes a different write of the same head, scored against this one.")
    fig_legend(fig, [line_handle(col, ls, lab, marker=None) for lab, _, col, ls in dist], 0.85, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=2.5)
    fig.savefig(out / "head_nla_l24_fve_distribution_std_vs_mtry.png", facecolor=SURF)
    plt.close(fig)

    # ---- 4. per-head FVE ----
    H = 32
    hfull = cm["head"]

    def head_fve(e: np.ndarray, b: np.ndarray, hh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f, c = np.zeros(H), np.zeros((H, 2))
        for h in range(H):
            idx = np.flatnonzero(hh == h)
            f[h] = 1 - e[idx].sum() / b[idx].sum()
            bi = idx[rng.integers(0, len(idx), size=(a.n_boot, len(idx)))]
            c[h] = np.quantile(1 - e[bi].sum(1) / b[bi].sum(1), [0.025, 0.975])
        return f, c

    ph = {"whole: std pair (step 400)": head_fve(cs["av__true"].astype(np.float64), bfull, hfull),
          "whole: matryoshka pair (step 640)": head_fve(cm["av__true"].astype(np.float64), bfull, hfull),
          "first 16: fair baseline (std AV + warm-up AR)": head_fve(E["rl400|mtry_w"][i16], base, heads),
          "first 16: matryoshka": head_fve(E["mtry|mtry"][i16], base, heads)}
    res["per_head"] = {k: {"fve": v[0].tolist(), "ci": v[1].tolist()} for k, v in ph.items()}
    res["per_head"]["rows_whole"] = np.bincount(hfull, minlength=H).tolist()
    res["per_head"]["rows_first16"] = np.bincount(heads, minlength=H).tolist()
    order = np.argsort(ph["whole: matryoshka pair (step 640)"][0])               # bottom → top = worst → best
    ypos = {h: i for i, h in enumerate(order)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 8.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURF)
    panels = [(a1, "Whole explanation", ["whole: std pair (step 400)", "whole: matryoshka pair (step 640)"],
               ["Std pair (step 400)", "Matryoshka pair (step 640)"]),
              (a2, "First 16 explanation tokens", ["first 16: fair baseline (std AV + warm-up AR)", "first 16: matryoshka"],
               ["Std AV + warm-up AR", "Matryoshka (step 640)"])]
    for ax, title, keys, labs in panels:
        (f0, c0), (f1, c1) = ph[keys[0]], ph[keys[1]]
        for h in range(H):
            y = ypos[h]
            ax.plot([100 * f0[h], 100 * f1[h]], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        for (f, c), col, lab, dy in (((f0, c0), STD_C, labs[0], -0.17), ((f1, c1), MTRY_C, labs[1], 0.17)):
            yy = np.array([ypos[h] for h in range(H)]) + dy
            ax.errorbar(100 * f, yy, xerr=[100 * (f - c[:, 0]), 100 * (c[:, 1] - f)], fmt="o", color=col, ms=5.5,
                        mec=SURF, mew=1, elinewidth=1, capsize=0, label=lab, zorder=3)
        ax.axvline(0, color=INK2, lw=1, zorder=0)
        style(ax, ygrid=False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:+.0f}%".replace("-", "−") if y else "0%"))
        ax.set_xlabel("FVE vs the head's mean write (95% bootstrap CI)", color=INK2, fontsize=8.8)
        ax.legend(loc="lower right", frameon=False, fontsize=8.3, labelcolor=INK)
    lo = min(100 * min(v[1][:, 0].min() for v in ph.values()), -60)
    hi = 100 * max(v[1][:, 1].max() for v in ph.values())
    a1.set_xlim(lo - 5, hi + 5)
    a2.set_xlim(lo - 5, hi + 5)
    a1.set_yticks(range(H))
    a1.set_yticklabels([f"H{h}  (KV{h // 4})" for h in order], fontsize=8)
    a1.set_ylim(-0.7, H - 0.3)
    nw, n16 = np.bincount(hfull).mean(), np.bincount(heads).mean()
    heading(fig, "FVE per attention head, standard and matryoshka NLA",
            f"Held-out writes: about {nw:.0f} per head for whole explanations (all {len(bfull):,}), about {n16:.0f} per "
            f"head for the first 16 tokens (prefix sweep, {N:,}).\nHeads sorted by the matryoshka pair's whole-explanation "
            "FVE. Left: each pair's own AR. Right: the std AV's text read by the warm-up AR (trained on prefixes),\nthe "
            "fair comparison.", y0=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.905), w_pad=1.5)
    fig.savefig(out / "head_nla_l24_fve_per_head_std_vs_mtry.png", facecolor=SURF)
    plt.close(fig)

    # ---- 5. per-head FVE at every prefix length (heatmaps) ----
    def head_by_L(e: np.ndarray) -> np.ndarray:                                 # [H, L]
        return np.stack([[1 - e[i][heads == h].sum() / base[heads == h].sum() for i in range(len(LS))]
                         for h in range(H)])
    HM = {"Matryoshka (step-640 AV + AR)": head_by_L(E["mtry|mtry"]),
          "Fair baseline: std AV + warm-up AR": head_by_L(E["rl400|mtry_w"])}
    res["per_head_by_prefix"] = {k: v.tolist() for k, v in HM.items()}
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    fig, axs = plt.subplots(1, 2, figsize=(11, 8.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (lab, M) in zip(axs, HM.items()):
        im = ax.imshow(np.clip(100 * M[order], -100, 100), cmap=DIV, norm=norm, aspect="auto", origin="lower",
                       interpolation="nearest")
        for (yy, xx), v in np.ndenumerate(100 * M[order]):
            ax.text(xx, yy, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                    color="#fcfcfb" if abs(v) > 55 else INK)
        ax.set_title(lab, loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(LLAB, fontsize=8)
        ax.set_xlabel("explanation tokens the AR reads", color=INK2, fontsize=8.8)
        ax.tick_params(colors=INK2, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_yticks(range(H))
    axs[0].set_yticklabels([f"H{h}" for h in order], fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_axes((0.925, 0.25, 0.015, 0.5)))
    cb.set_label("FVE (%), clipped at ±100", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cb.outline.set_visible(False)
    heading(fig, "FVE per head at every prefix length",
            f"Prefix sweep: {N:,} held-out writes, about {n16:.0f} per head. Cells: pooled FVE (%) of that head's writes "
            "when the AR reads the first L tokens.\nHeads in the per-head figure's order (the matryoshka pair's "
            "whole-explanation FVE, best at the top).", y0=0.985)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.07, wspace=0.06)
    fig.savefig(out / "head_nla_l24_mtry_prefix_by_head.png", facecolor=SURF)
    plt.close(fig)

    # ---- 6. per-head marginal gains, matryoshka ----
    M = HM["Matryoshka (step-640 AV + AR)"]
    G = 100 * (M[:, 1:] - M[:, :-1])                                            # [H, steps]
    res["per_head_step_gain_mtry"] = G.tolist()
    fig, ax = plt.subplots(figsize=(9.2, 9.4), dpi=160)
    fig.patch.set_facecolor(SURF)
    seq = LinearSegmentedColormap.from_list("gain", ["#f4f3f0", "#86b6ef", "#2a78d6", "#104281"])
    gn = PowerNorm(gamma=0.5, vmin=0, vmax=100)
    im = ax.imshow(np.clip(G[order], 0, 100), cmap=seq, norm=gn, aspect="auto", origin="lower", interpolation="nearest")
    for (yy, xx), v in np.ndenumerate(G[order]):
        ax.text(xx, yy, f"{v:+.0f}".replace("-", "−") if abs(v) >= 0.5 else "0", ha="center", va="center",
                fontsize=6.3, color="#fcfcfb" if v > 30 else INK)
    tot = M[:, -1] - M[:, 0]
    first_share = G[:, 0] / (100 * tot)
    late = G[:, LS.index("32"):]                                               # steps from 32 on
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels(STEPS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(H))
    ax.set_yticklabels([f"H{h}" for h in order], fontsize=8)
    ax.set_xlabel("explanation tokens the AR reads: from → to", color=INK2, fontsize=8.8)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("FVE points gained in the step (square-root scale)", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cb.outline.set_visible(False)
    heading(fig, "Matryoshka: FVE gained at each truncation step, per head",
            f"About {n16:.0f} held-out writes per head, so single steps are noisy (a few points). Heads in the per-head "
            f"figure's order.\nThe first step is more than half of the empty → whole gain for {(first_share > 0.5).sum()} of "
            f"{H} heads (see the note on “empty”\nin the marginal figure). From 32 tokens on, no step adds more than "
            f"{late.max():.0f} points for any head (median {np.median(late):.0f}).\nColour: square-root scale; "
            f"{(G < 0).sum()} negative cell(s) shown as 0.", y0=0.985)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.85, bottom=0.1)
    fig.savefig(out / "head_nla_l24_mtry_marginal_by_head.png", facecolor=SURF)
    plt.close(fig)

    (out / "head_nla_l24_mtry_quality.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote 6 figures and head_nla_l24_mtry_quality.json to {out}")


if __name__ == "__main__":
    main()
