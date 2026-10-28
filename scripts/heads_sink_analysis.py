"""How much of the head NLA's reconstruction quality is driven by attention sinks?

    python scripts/heads_sink_analysis.py --out-dir <dir>

"Sink" = the head's attention to the first token (Qwen adds no BOS; the first real token is the sink). Two data
sets, both held out, read from the public HF dataset (no GPU):
  eval  the 6,360 eval rows behind the headline numbers (T=1 checks of the RL and matryoshka pairs). Sink weight
        from l24/std_rl/results/eval_ctx.parquet (the top-16 attention weights; position 0 is almost always among
        them, else its weight is below the 16th and is taken as 0). Sampling: one site per document per head,
        weighted toward larger-than-typical writes.
  doc   every token × head of the 4 doc-viewer documents (stored per-site cos / FVE for both pairs).
Attention is used only to evaluate here; it never enters an explanation.

For each: pooled FVE by sink-weight bin, each bin's share of sites and of the explained variance Σ(base − err),
and FVE with sink-dominated sites (weight ≥ 0.9) removed. Against two sink-only readers that know the head and
its sink weight and nothing else (fit per head, leaving each document out in turn):
  sink_linear   the target direction as a linear function of the sink weight
  sink_binned   the mean target direction of the site's sink-weight bin
and, on the doc viewer only, an exact one, sink_exact: the write the head would make if its non-sink attention
wrote exactly its average, a_t0 · write(first token) − mean write (the first token's write is its value write,
since the first token attends only to itself).
Writes summary.json and sink_fve.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "syvb/head-nla-qwen3-8b"
DOCS = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression",
        "04_law_gainsborough_fraud"]
BINS = [0, 0.25, 0.5, 0.75, 0.9, 1.0001]
LABELS = ["<0.25", "0.25–0.5", "0.5–0.75", "0.75–0.9", "≥0.9"]
S2 = 2 * 64 ** 2      # direction space: err = 2·64²·(1 − cos) for a unit prediction; base in the same units


def get(f: str) -> str:
    return hf_hub_download(REPO, f, repo_type="dataset")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fve(err, base) -> float:
    return float(1 - np.sum(err) / np.sum(base))


def sink_readers(g: np.ndarray, a: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-group-out predictions of unit targets g [n, d] from the sink weight a [n] alone."""
    lin, binned = np.zeros_like(g), np.zeros_like(g)
    b = np.digitize(a, BINS) - 1
    for grp in np.unique(groups):
        tr, te = groups != grp, groups == grp
        X = np.stack([np.ones(tr.sum()), a[tr]], 1)
        coef, *_ = np.linalg.lstsq(X, g[tr], rcond=None)                     # [2, d]
        lin[te] = np.stack([np.ones(te.sum()), a[te]], 1) @ coef
        mean_all = g[tr].mean(0)
        for k in range(len(LABELS)):
            m = tr & (b == k)
            binned[te & (b == k)] = g[m].mean(0) if m.sum() >= 3 else mean_all
    return {"sink_linear": unit(lin), "sink_binned": unit(binned)}


def by_bin(a, errs: dict, base) -> dict:
    b = np.digitize(a, BINS) - 1
    out = {"share_sites": [float((b == k).mean()) for k in range(len(LABELS))]}
    for name, e in errs.items():
        tot = np.sum(base - e)
        out[name] = {"fve": [fve(e[b == k], base[b == k]) if (b == k).any() else None for k in range(len(LABELS))],
                     "share_explained": [float(np.sum((base - e)[b == k]) / tot) for k in range(len(LABELS))],
                     "fve_all": fve(e, base), "fve_without_sink_ge_0.9": fve(e[a < 0.9], base[a < 0.9])}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"bins": LABELS}

    # ---- eval rows ----
    ctx = pq.read_table(get("l24/std_rl/results/eval_ctx.parquet"), columns=["example_idx", "att_pos", "att_w"]).to_pandas()
    ctx["a0"] = [float(w[list(ps).index(0)]) if 0 in list(ps) else 0.0 for ps, w in zip(ctx.att_pos, ctx.att_w)]
    gen = pq.read_table(get("l24/mtry/checks_t1/generations.parquet"), columns=["row", "example_idx", "head"]).to_pandas()
    ev = pq.read_table(get("l24/sft_data/eval.parquet"), columns=["activation_vector", "example_idx", "doc_id", "head"]).to_pandas()
    ev = gen.merge(ev, on=["example_idx", "head"], validate="one_to_one").merge(ctx[["example_idx", "a0"]], on="example_idx", validate="one_to_one")
    ev = ev.sort_values("row")
    assert (ev.row.values == np.arange(len(ev))).all()
    Z = {"RL": np.load(get("l24/std_rl/results/checks_rl_t1/row_errors_direction.npz")),
         "Matryoshka": np.load(get("l24/mtry/checks_t1/row_errors_direction.npz"))}
    base = Z["RL"]["ar__head_mean"]
    assert (Z["RL"]["head"] == ev["head"].values).all()
    g = unit(np.stack(ev.activation_vector.values).astype(np.float32))
    a, heads, docs = ev.a0.values, ev["head"].values, ev.doc_id.values
    errs = {k: z["av__true"] for k, z in Z.items()}
    oracle = {"sink_linear": np.zeros(len(g)), "sink_binned": np.zeros(len(g))}
    for h in range(32):
        m = heads == h
        groups = pd.factorize(docs[m])[0] % 5                                  # 5 folds, documents kept together
        for k, pred in sink_readers(g[m], a[m], groups).items():
            oracle[k][m] = S2 * (1 - (pred * g[m]).sum(1))
    errs.update(oracle)
    summary["eval"] = {"n": len(a), "sink_weight_missing_from_top16": float((~ctx.att_pos.map(lambda x: 0 in list(x))).mean()),
                       "mean_sink_weight": float(a.mean()), **by_bin(a, errs, base)}
    del g

    # ---- doc viewer: every token ----
    st = load_file(get("l24/meta/head_stats.safetensors"))
    rms, mw = st["dev_rms"][0], st["mean_write"][0]
    rows = []
    for di, d in enumerate(DOCS):
        S = pd.read_parquet(get(f"l24/std_rl/doc_viewer/{d}/sites.parquet"), columns=["pos", "head", "first_w", "rl_cos", "rl_fve"])
        S = S.merge(pd.read_parquet(get(f"l24/mtry/doc_viewer/{d}/mtry.parquet"), columns=["pos", "head", "mtry_cos", "mtry_fve"]),
                    on=["pos", "head"], validate="one_to_one").sort_values(["head", "pos"])
        V = np.load(get(f"l24/std_rl/doc_viewer/{d}/writes.npz"))["v"].astype(np.float32)        # [T, 32, 4096]
        raw0 = V[0] * rms[:, None] + mw                                                          # first token's write
        ex = np.zeros(len(S))
        for h in range(32):
            sh = S["head"].values == h
            t = S.pos.values[sh]
            pred = unit(S.first_w.values[sh, None] * raw0[h] - mw[h])
            ex[sh] = 2 * (1 - (pred * unit(V[t, h])).sum(1))
        S["err_exact"] = ex
        S["doc"] = di
        S["g"] = list(unit(V[S.pos.values, S["head"].values]).astype(np.float16))
        rows.append(S)
        del V
    D = pd.concat(rows, ignore_index=True)
    D = D[D.pos > 0]                                   # the first token itself is trivially sink-only
    base_d = 2 * (1 - D.rl_cos.values) / np.maximum(1 - D.rl_fve.values, 1e-6)   # per-site baseline, unit units
    errs_d = {"RL": 2 * (1 - D.rl_cos.values), "Matryoshka": 2 * (1 - D.mtry_cos.values), "sink_exact": D.err_exact.values,
              "sink_linear": np.zeros(len(D)), "sink_binned": np.zeros(len(D))}
    for h in range(32):
        m = (D["head"] == h).values
        gh = np.stack(D.g.values[m]).astype(np.float32)
        for k, pred in sink_readers(gh, D.first_w.values[m], D.doc.values[m]).items():
            errs_d[k][m] = 2 * (1 - (pred * gh).sum(1))
    summary["doc"] = {"n": int(len(D)), "mean_sink_weight": float(D.first_w.mean()), **by_bin(D.first_w.values, errs_d, base_d)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    C = {"RL": "#d9822b", "Matryoshka": "#2a9d8f", "sink_binned": "#7a5195", "sink_exact": "#444444"}
    for ax, key, title in [(axes[0], "eval", "Held-out eval rows (headline metric)"), (axes[1], "doc", "Every token of 4 held-out docs")]:
        s = summary[key]
        x = np.arange(len(LABELS))
        ax2 = ax.twinx()
        ax2.bar(x, s["share_sites"], color="#e6e6e6", width=0.6, zorder=0)
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax2.set_ylabel("share of sites (bars)", color="#999999")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        for name, lab in [("RL", "RL NLA"), ("Matryoshka", "Matryoshka NLA"), ("sink_binned", "sink-only reader"), ("sink_exact", "exact sink-only write")]:
            if name in s:
                ax.plot(x, s[name]["fve"], "o-", color=C[name], label=f"{lab} (all: {s[name]['fve_all']:+.0%})", lw=2 if name in ("RL", "Matryoshka") else 1.4,
                        ls="-" if name in ("RL", "Matryoshka") else "--")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("the head's attention to the first token (sink)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(fontsize=8.5, frameon=False, loc="center left")
    axes[0].set_ylabel("direction FVE (pooled within bin)")
    fig.suptitle("Head NLA, Qwen3-8B L24: reconstruction quality vs attention-sink weight", x=0.06, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "sink_fve.png", dpi=150, bbox_inches="tight")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
