"""Figure + examples for the matryoshka prefix sweep (scripts/heads_matryoshka_sweep.py output).

    python scripts/heads_matryoshka_report.py --sweep-dir <results/sweep> --sft-dir <l24 sft_data> --out-dir <dir>

Writes to --out-dir:
  fve_by_prefix.png   held-out FVE vs how many explanation tokens the AR reads, with 95% bootstrap bands
                      (rows resampled): the matryoshka pair, the fair baseline (std AV + warm-up AR), the
                      std pair, and the matryoshka text with its sentences reversed (the ordering check)
  examples.md         a table of the sweep, then side-by-side examples (one per head, seeded) of the std
                      and matryoshka explanations of the same write, each with its first 16 tokens and scores
  summary.json        every plotted value and its CI
FVE is the sweep's (direction space; baseline = the head's mean normalized AR-training write), recomputed per
row from row_cos.npz so it can be bootstrapped; it is checked against sweep.json.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, AV text, AR, colour, line style) — colours in fixed order, text in ink, never the series colour
LINES = [("Matryoshka (step-640 AV + AR)", "mtry", "mtry", "#eb6834", "-"),
         ("Fair baseline: std AV + warm-up AR", "rl400", "mtry_w", "#2a78d6", "-"),
         ("Std pair, no prefix training (step 400)", "rl400", "rl400", "#7d7c77", (0, (4, 3))),
         ("Matryoshka text, sentences reversed", "mtry_rev", "mtry", "#eb6834", (0, (1, 2)))]


def pct(x: float) -> str:
    return f"{100 * x:+.0f}%".replace("-", "−")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-examples", type=int, default=24)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)

    sw, sft, out = Path(a.sweep_dir), Path(a.sft_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((sw / "sweep.json").read_text())
    gen = pq.read_table(sw / "generations.parquet").to_pandas()
    rc = np.load(sw / "row_cos.npz")
    Ls = [str(x) for x in res["lengths"]]

    # per-row baseline error, as in the sweep: both vectors at norm s = √d, baseline = head-mean direction
    ev = pq.read_table(sft / "eval.parquet", columns=["activation_vector", "head"])
    v = np.asarray(ev.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(ev.num_rows, -1)
    v = v[gen.eval_row.values]
    s = float(np.sqrt(v.shape[1]))

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * s
    tr = pq.read_table(sft / "ar_sft.parquet", columns=["activation_vector", "head"])
    tr_u = unit(np.asarray(tr.column("activation_vector").combine_chunks().flatten(), dtype=np.float32)
                .reshape(tr.num_rows, -1))
    by = defaultdict(list)
    for i, h in enumerate(tr.column("head").to_pylist()):
        by[h].append(i)
    heads = gen["head"].values
    base = np.stack([tr_u[by[h]].mean(0) for h in heads])
    base_err = ((base - unit(v)) ** 2).sum(-1)

    def err(av: str, ar: str, L: str) -> np.ndarray:
        return 2 * s * s * (1 - rc[f"{av}|{ar}|{L}"].astype(np.float64))

    # consistency with the sweep's own pooled FVE
    for av, ar in (("mtry", "mtry"), ("rl400", "rl400")):
        mine = 1 - err(av, ar, "full").sum() / base_err.sum()
        assert abs(mine - res["sweep"][av][ar]["full"]["fve"]) < 2e-3, (av, ar, mine)

    rng = np.random.default_rng(a.seed)
    boot = rng.integers(0, len(gen), size=(a.n_boot, len(gen)))
    summ: dict = {"n_rows": len(gen), "lines": {}}
    for label, av, ar, *_ in LINES:
        row = {}
        for L in Ls:
            e = err(av, ar, L)
            f = 1 - e.sum() / base_err.sum()
            fb = 1 - e[boot].sum(1) / base_err[boot].sum(1)
            row[L] = {"fve": float(f), "ci": [float(np.quantile(fb, 0.025)), float(np.quantile(fb, 0.975))],
                      "cos": float(rc[f"{av}|{ar}|{L}"].mean())}
        summ["lines"][label] = row
    # paired differences at each L: matryoshka − fair baseline, and forward − reversed
    summ["diffs"] = {}
    for name, (a1, r1), (a2, r2) in (("mtry_minus_baseline", ("mtry", "mtry"), ("rl400", "mtry_w")),
                                     ("forward_minus_reversed_mtry", ("mtry", "mtry"), ("mtry_rev", "mtry")),
                                     ("forward_minus_reversed_rl400", ("rl400", "mtry_w"), ("rl400_rev", "mtry_w"))):
        summ["diffs"][name] = {}
        for L in Ls:
            d = (err(a2, r2, L) - err(a1, r1, L))                      # > 0: the first beats the second
            db = d[boot].sum(1) / base_err[boot].sum(1)
            summ["diffs"][name][L] = {"fve": float(d.sum() / base_err.sum()),
                                      "ci": [float(np.quantile(db, 0.025)), float(np.quantile(db, 0.975))]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))

    # ---- figure ----
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    num = [L for L in Ls if L not in ("0", "full")]
    xs = [int(L) for L in num]
    xfull = xs[-1] * 2.2
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    for j, (label, av, ar, col, ls) in enumerate(LINES):
        r = summ["lines"][label]
        y = [100 * r[L]["fve"] for L in num]
        lo = [100 * r[L]["ci"][0] for L in num]
        hi = [100 * r[L]["ci"][1] for L in num]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13 if ls != (0, (1, 2)) else 0.07, lw=0, zorder=1)
        ax.plot(xs, y, color=col, ls=ls, lw=2, zorder=3, label=label)
        ax.plot(xs, y, "o", color=col, ms=4.5, mec=surf, mew=1.5, zorder=4)
        fv = 100 * r["full"]["fve"]
        ax.errorbar([xfull * (1 + 0.05 * (j - 1.5))], [fv], yerr=[[fv - 100 * r["full"]["ci"][0]], [100 * r["full"]["ci"][1] - fv]],
                    fmt="o", color=col, ms=6, mec=surf, mew=1.5, elinewidth=1.5, capsize=0, zorder=4)
    ax.axhline(0, color=ink2, lw=1, zorder=0)
    ax.text(1.05, 1.5, "0% = as good as predicting\nthe head's mean write", color=ink2, fontsize=7.8, va="bottom")
    ax.axvline(xs[-1] * 1.48, color=grid, lw=1, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs + [xfull])
    ax.set_xticklabels([str(x) for x in xs] + ["whole"])
    ax.set_xlim(0.8, xfull * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:+.0f}%".replace("-", "−") if y else "0%"))
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5, length=0)
    ax.set_xlabel("explanation tokens the AR reads (first N; “whole” = the full explanation)", color=ink2, fontsize=9,
                  labelpad=8)
    ax.set_ylabel("held-out FVE", color=ink2, fontsize=9)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=3.2)
    for t in leg.get_texts():
        t.set_color(ink)
    m16 = summ["lines"][LINES[0][0]]["16"]["fve"]
    b16 = summ["lines"][LINES[1][0]]["16"]["fve"]
    fig.suptitle("Head NLA, Qwen3-8B layer 24: FVE from the first N explanation tokens", x=0.07, ha="left",
                 fontsize=12.5, color=ink, y=0.975)
    fig.text(0.07, 0.895, f"{len(gen):,} held-out writes (all 32 heads), one T=1 explanation each; bands = 95% bootstrap "
             f"over writes. At 16 tokens: matryoshka {pct(m16)} vs baseline {pct(b16)}.\nMatryoshka = the step-400 std "
             "pair trained 240 more steps (40 AR-only, 200 matryoshka) on explanations cut to their first 1–120 tokens.",
             fontsize=8.5, color=ink2, va="top", linespacing=1.5)
    fig.text(0.07, 0.015, "FVE in direction space (≤ +100%; a random direction ≈ −100%); baseline = the head's mean write. "
             "Warm-up AR = the AR after 40 AR-only steps on prefixes of the std AV's text.", fontsize=7.4, color=ink2)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.15)
    plt.savefig(out / "fve_by_prefix.png", facecolor=surf)

    # ---- examples ----
    def first(t: str, n: int = 16) -> str:
        return tok.decode(tok.encode(t, add_special_tokens=False)[:n]).strip()

    def row_fve(av: str, ar: str, L: str, i: int) -> float:
        return 1 - err(av, ar, L)[i] / base_err[i]

    def md(t: str) -> str:
        return t.replace("\n", " ⏎ ").replace("|", "\\|").replace("<", "&lt;").strip()

    lines = ["# Head NLA L24: matryoshka vs std explanations", "",
             f"{len(gen):,} held-out writes (documents neither model trained on), all 32 heads, one T=1 explanation "
             "per AV. **Std** = the step-400 RL pair. **Matryoshka** = that pair after 40 AR-only warm-up steps and "
             "200 matryoshka steps (training explanations cut to their first 1–120 tokens before the AR reads them).",
             "", "## FVE by how many explanation tokens the AR reads", "",
             "| text · reader | " + " | ".join(Ls) + " |", "|---|" + "---:|" * len(Ls)]
    for label, *_ in LINES:
        lines.append(f"| {label} | " + " | ".join(pct(summ["lines"][label][L]["fve"]) for L in Ls) + " |")
    lines += ["", "Paired differences (FVE points, 95% bootstrap CI over writes):", "",
              "| comparison | 4 | 16 | 64 | whole |", "|---|---:|---:|---:|---:|"]
    names = {"mtry_minus_baseline": "matryoshka − fair baseline",
             "forward_minus_reversed_mtry": "matryoshka text: forward − reversed (matryoshka AR)",
             "forward_minus_reversed_rl400": "std text: forward − reversed (warm-up AR)"}
    for k, nm in names.items():
        dd = summ["diffs"][k]
        lines.append(f"| {nm} | " + " | ".join(f"{pct(dd[L]['fve'])} [{pct(dd[L]['ci'][0])}, {pct(dd[L]['ci'][1])}]"
                                                for L in ("4", "16", "64", "full")) + " |")
    ntok = {k: gen[f"ntok__{k}"] for k in ("gold", "rl400", "mtry")}
    lines += ["", "Explanation length (tokens, median): " + ", ".join(f"{k} {int(v.median())}" for k, v in ntok.items()),
              "", "## Examples", "",
              f"One write per head for {a.n_examples} heads (seeded). Scores: FVE of the whole explanation (each read by "
              "its own AR) and of its first 16 tokens (std: warm-up AR; matryoshka: matryoshka AR).", ""]
    ex_rng = np.random.default_rng(a.seed)
    picked_heads = sorted(ex_rng.choice(sorted(set(heads)), size=min(a.n_examples, len(set(heads))), replace=False))
    for h in picked_heads:
        i = int(ex_rng.choice(np.flatnonzero(heads == h)))
        lines += [f"### L24H{h} · eval row {int(gen.eval_row[i])}", "",
                  f"- **Gold (Haiku 5.5):** {md(gen.av__gold[i])}", "",
                  f"- **Std** — whole {pct(row_fve('rl400', 'rl400', 'full', i))}, first 16 tokens "
                  f"{pct(row_fve('rl400', 'mtry_w', '16', i))}  ",
                  f"  first 16 tokens: *{md(first(gen.av__rl400[i]))}*  ",
                  f"  {md(gen.av__rl400[i])}", "",
                  f"- **Matryoshka** — whole {pct(row_fve('mtry', 'mtry', 'full', i))}, first 16 tokens "
                  f"{pct(row_fve('mtry', 'mtry', '16', i))}  ",
                  f"  first 16 tokens: *{md(first(gen.av__mtry[i]))}*  ",
                  f"  {md(gen.av__mtry[i])}", ""]
    (out / "examples.md").write_text("\n".join(lines))
    print(f"wrote {out / 'fve_by_prefix.png'}, {out / 'examples.md'}, {out / 'summary.json'}")
    print(pd.DataFrame({lab: {L: round(100 * summ['lines'][lab][L]['fve'], 1) for L in Ls}
                        for lab, *_ in LINES}).T.to_string())


if __name__ == "__main__":
    main()
