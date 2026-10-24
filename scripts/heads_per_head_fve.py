"""Per-head FVE of the L24 head NLA, plus how much GQA siblings (heads sharing a KV head) resemble each other.

    python scripts/heads_per_head_fve.py --out-dir <dir> [--repo syvb/head-nla-qwen3-8b]

Everything is read from the public HF dataset (cached by huggingface_hub); no GPU. Writes to --out-dir:
  fve_per_head.png   per head, 95% bootstrap CIs (rows resampled within the head):
                     - whole explanation, held-out eval rows at T=1: SFT, RL (step 400), matryoshka (step 640),
                       each pair read by its own AR (l24/std_rl/results/checks_*_t1, l24/mtry/checks_t1);
                     - first 16 explanation tokens (l24/mtry/sweep): matryoshka pair vs the fair baseline
                       (std AV + warm-up AR);
                     - per-head matryoshka FVE vs mean attention to the first token (doc viewer, every token).
  fve_per_head.csv   the plotted per-head values, plus doc-viewer per-head means (every token of 4 held-out docs)
  summary.json       KV-group share of between-head variance, and sibling vs non-sibling similarity at the same
                     token: |cos| of the writes, correlation of non-sink attention rows, word-bag cosine of the
                     matryoshka explanations
FVE is direction-space with the head's mean AR-training write as baseline: per head 1 − mean(err) / mean(base),
which reproduces checks.json's fve_head_avg and sweep.json exactly. Eval rows were sampled weighted toward
larger-than-typical writes; the doc-viewer columns cover every token.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression",
        "04_law_gainsborough_fraud"]
CHECKS = {"SFT": "l24/std_rl/results/checks_sft_t1", "RL (step 400)": "l24/std_rl/results/checks_rl_t1",
          "Matryoshka (step 640)": "l24/mtry/checks_t1"}
PREFIX = {"Matryoshka, first 16 tokens": "mtry|mtry|16", "Std AV + warm-up AR, first 16 tokens": "rl400|mtry_w|16"}
COLOR = {"SFT": "#9a9a9a", "RL (step 400)": "#d9822b", "Matryoshka (step 640)": "#2a9d8f",
         "Matryoshka, first 16 tokens": "#2a9d8f", "Std AV + warm-up AR, first 16 tokens": "#d9822b"}
N_HEADS, GROUP = 32, 4   # Qwen3-8B: 32 query heads, 8 KV heads


def fve_ci(err, base, rng, n=2000):
    f = 1 - err.mean() / base.mean()
    idx = rng.integers(0, len(err), (n, len(err)))
    b = 1 - err[idx].mean(1) / base[idx].mean(1)
    return f, *np.percentile(b, [2.5, 97.5])


def group_r2(x: pd.Series) -> float:
    g = x.groupby(x.index // GROUP).transform("mean")
    return float(((g - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo", default="syvb/head-nla-qwen3-8b")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    get = lambda f: hf_hub_download(a.repo, f, repo_type="dataset")  # noqa: E731
    rng = np.random.default_rng(0)

    rows = []
    Z = {k: np.load(get(f"{v}/row_errors_direction.npz")) for k, v in CHECKS.items()}
    for name, z in Z.items():
        for h in range(N_HEADS):
            m = z["head"] == h
            rows.append((name, h, *fve_ci(z["av__true"][m], z["ar__head_mean"][m], rng)))
    G = pd.read_parquet(get("l24/mtry/sweep/generations.parquet"), columns=["eval_row", "head"])
    R = np.load(get("l24/mtry/sweep/row_cos.npz"))
    base = Z["SFT"]["ar__head_mean"][G.eval_row.values]   # the same eval rows; base is identical across runs
    assert (Z["SFT"]["head"][G.eval_row.values] == G["head"].values).all()
    for name, key in PREFIX.items():
        err = 2 * 64 ** 2 * (1 - R[key])                  # direction space: cos = 1 − err / (2·64²)
        for h in range(N_HEADS):
            m = G["head"].values == h
            rows.append((name, h, *fve_ci(err[m], base[m], rng)))
    T = pd.DataFrame(rows, columns=["series", "head", "fve", "lo", "hi"])

    # Doc viewer: every token × head of 4 held-out documents; sibling similarity at the same token.
    dv, sim = [], {k: {True: [], False: []} for k in ("write_abs_cos", "attn_corr", "expl_word_cos")}
    iu = np.triu_indices(N_HEADS, 1)
    sib = (iu[0] // GROUP) == (iu[1] // GROUP)
    for d in DOCS:
        S = pd.read_parquet(get(f"l24/std_rl/doc_viewer/{d}/sites.parquet"), columns=["pos", "head", "first_w", "v_norm", "rl_fve"])
        S = S.merge(pd.read_parquet(get(f"l24/mtry/doc_viewer/{d}/mtry.parquet"), columns=["pos", "head", "mtry_fve", "mtry_text"]),
                    on=["pos", "head"], validate="one_to_one")
        dv.append(S.drop(columns="mtry_text"))
        v = np.load(get(f"l24/std_rl/doc_viewer/{d}/writes.npz"))["v"].astype(np.float32)   # [T, 32, 4096]
        u = v / np.linalg.norm(v, axis=-1, keepdims=True)
        att = np.load(get(f"l24/std_rl/doc_viewer/{d}/attn.npz"))["attn"].astype(np.float32)   # [32, T, T]
        W = CountVectorizer(token_pattern=r"[a-z]{3,}").fit_transform(S.sort_values(["pos", "head"]).mtry_text.str.lower())
        Wn = W.multiply(1 / np.sqrt(W.multiply(W).sum(1).A + 1e-12)).tocsr()
        for t in range(5, v.shape[0]):
            c = np.abs(u[t] @ u[t].T)[iu]
            e = (Wn[t * N_HEADS:(t + 1) * N_HEADS] @ Wn[t * N_HEADS:(t + 1) * N_HEADS].T).toarray()[iu]
            r = np.corrcoef(att[:, t, 1:t + 1])[iu]        # non-sink part of each head's attention row
            for k, x in (("write_abs_cos", c), ("expl_word_cos", e), ("attn_corr", r)):
                ok = np.isfinite(x)
                sim[k][True].extend(x[ok & sib]); sim[k][False].extend(x[ok & ~sib])
    DV = pd.concat(dv)
    ph = DV.groupby("head").agg(sink=("first_w", "mean"), write_norm=("v_norm", "mean"),
                                dv_rl_fve=("rl_fve", "mean"), dv_mtry_fve=("mtry_fve", "mean"))
    W = T.pivot(index="head", columns="series", values="fve").join(ph)
    W.round(4).to_csv(out / "fve_per_head.csv")
    mt = W["Matryoshka (step 640)"]
    summary = {
        "kv_group_r2_between_heads": {"chance": (N_HEADS // GROUP - 1) / (N_HEADS - 1),
                                      **{c: group_r2(W[c]) for c in ["RL (step 400)", "Matryoshka (step 640)", "sink", "write_norm"]}},
        "sibling_vs_nonsibling_same_token": {k: {"sibling": float(np.mean(v[True])), "non_sibling": float(np.mean(v[False]))}
                                             for k, v in sim.items()},
        "spearman": {"eval_vs_docviewer_mtry": spearmanr(mt, W.dv_mtry_fve).statistic,
                     "sink_vs_eval_mtry": spearmanr(W.sink, mt).statistic, "sink_vs_docviewer_mtry": spearmanr(W.sink, W.dv_mtry_fve).statistic},
        "heads_mtry_beats_rl_whole": int((mt > W["RL (step 400)"]).sum()),
        "heads_mtry_beats_baseline_16": int((W["Matryoshka, first 16 tokens"] > W["Std AV + warm-up AR, first 16 tokens"]).sum()),
        "per_head_mean_fve": {c: float(W[c].mean()) for c in CHECKS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1.05], hspace=0.42, wspace=0.22)
    x = np.arange(N_HEADS)

    def panel(ax, series, offs, title):
        lo, hi = -1.5, 1.0
        for q in range(1, N_HEADS // GROUP):
            ax.axvline(GROUP * q - 0.5, color="#dddddd", lw=0.8, zorder=0)
        ax.axhline(0, color="#555555", lw=0.8)
        for s, o in zip(series, offs):
            d = T[T.series == s].sort_values("head")
            ax.errorbar(x + o, d.fve, yerr=[d.fve - d.lo, d.hi - d.fve], fmt="o", ms=4.5, color=C(s), elinewidth=1,
                        capsize=0, label=s, mfc="white" if s.startswith("Std AV") else C(s))
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in x], fontsize=8, rotation=90)
        ax.set_xlim(-0.7, N_HEADS - 0.3)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("direction FVE")
        for q in range(N_HEADS // GROUP):
            ax.text(GROUP * q + 1.5, hi - 0.04 * (hi - lo), f"KV{q}", ha="center", va="top", fontsize=8, color="#888888")
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=3)

    C = COLOR.get
    panel(fig.add_subplot(gs[0, :]), list(CHECKS), [-0.22, 0, 0.22],
          "Whole explanation, held-out eval rows (≈199 per head, T=1; 95% bootstrap CI)")
    panel(fig.add_subplot(gs[1, :]), ["Std AV + warm-up AR, first 16 tokens", "Matryoshka, first 16 tokens"], [-0.15, 0.15],
          "First 16 explanation tokens only (prefix sweep, ≈62 rows per head)")
    kvc = plt.cm.tab10(np.arange(N_HEADS // GROUP))
    for j, (col, lab) in enumerate([("Matryoshka (step 640)", "Matryoshka FVE (eval rows)"),
                                    ("dv_mtry_fve", "Matryoshka FVE (every token, doc viewer)")]):
        ax = fig.add_subplot(gs[2, j])
        for h in range(N_HEADS):
            ax.scatter(W.sink[h], W[col][h], color=kvc[h // GROUP], s=28, zorder=3)
            ax.annotate(f"H{h}", (W.sink[h], W[col][h]), fontsize=7, xytext=(3, 2), textcoords="offset points", color="#444444")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xlabel("mean attention to the first token (doc viewer, every token)")
        ax.set_ylabel(lab, fontsize=9.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"Per head vs sink attention (Spearman ρ = {spearmanr(W.sink, W[col]).statistic:+.2f})", loc="left", fontsize=10.5)
    fig.suptitle("Head NLA, Qwen3-8B layer 24: FVE per attention head (scatter colour = KV group)", x=0.065, ha="left",
                 fontsize=14, y=0.925)
    fig.savefig(out / "fve_per_head.png", dpi=150, bbox_inches="tight")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
