"""Paired comparison of two heads_sft_checks.py runs on the eval rows they share.

    python scripts/heads_compare_checks.py --a <checks dir> --b <checks dir> [--label-a SFT --label-b RL]

Each dir needs generations.parquet + row_errors.npz (row-aligned). Rows are matched on
example_idx, so a --max-rows run compares against the same rows of a full run. For every condition
both runs have (ar/gold, av/true, ...), prints FVE in each run on the shared rows (pooled, baseline =
the per-head AR-training mean, as heads_sft_checks.py) and the difference b − a with a 95% paired
bootstrap CI over rows. With --texts (heads_soft_token_readouts.py's texts.parquet) it also reports,
per run, the AV's quote rate and copying (share of an explanation's content words that are one of
that write's own readout tokens): the token-listing drift the reward could favour. Writes nothing
unless --out is given.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
_QUOTE = re.compile(r'["“”]')


def text_stats(d: Path, texts: Path) -> dict[str, float]:
    from heads_explain_probe import _STOP
    g = pq.read_table(d / "generations.parquet", columns=["example_idx", "av_true"]).to_pydict()
    t = pq.read_table(texts, columns=["example_idx", "avg16", "lens"]).to_pydict()
    at = {e: i for i, e in enumerate(t["example_idx"])}

    def stems(xs):
        return {w.lower().rstrip("s") for x in xs for w in re.findall(r"[A-Za-z]{4,}", x)}
    ov = []
    for text, e in zip(g["av_true"], g["example_idx"], strict=True):
        i = at[e]
        ev = stems(ast.literal_eval(f"[{t['avg16'][i]}]") + ast.literal_eval(f"[{t['lens'][i]}]"))
        ws = [w.rstrip("s") for w in (w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)) if w not in _STOP]
        ov.append(sum(w in ev for w in ws) / max(1, len(ws)))
    return {"quote_rate": float(np.mean([bool(_QUOTE.search(x)) for x in g["av_true"]])),
            "copying_mean": float(np.mean(ov)), "words_mean": float(np.mean([len(x.split()) for x in g["av_true"]]))}


def load(d: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    ex = pq.read_table(d / "generations.parquet", columns=["example_idx"]).column("example_idx").to_pylist()
    z = np.load(d / "row_errors.npz")
    return ex, {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="b")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--texts", default=None, help="readouts (texts.parquet) for the quote / copying measures")
    p.add_argument("--out", default=None, help="write the table as JSON here")
    args = p.parse_args()

    ex_a, za = load(Path(args.a))
    ex_b, zb = load(Path(args.b))
    common = sorted(set(ex_a) & set(ex_b))
    ia = np.array([{e: i for i, e in enumerate(ex_a)}[e] for e in common])
    ib = np.array([{e: i for i, e in enumerate(ex_b)}[e] for e in common])
    base_a, base_b = za["base"][ia], zb["base"][ib]
    assert np.allclose(base_a, base_b, rtol=1e-4), "the two runs' baselines differ: different ar_sft data?"
    conds = [k for k in za if k not in ("base", "layer", "head") and k in zb]
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(common), size=(args.boot, len(common)))
    print(f"{len(common)} shared eval rows ({len(ex_a)} in {args.label_a}, {len(ex_b)} in {args.label_b})")
    print(f"{'condition':<16}{args.label_a:>9}{args.label_b:>9}{'b − a':>9}  95% CI")
    table = {}
    for k in conds:
        ea, eb = za[k][ia], zb[k][ib]
        fa = 1 - ea.sum() / base_a.sum()
        fb = 1 - eb.sum() / base_a.sum()
        boot = (ea[idx].sum(1) - eb[idx].sum(1)) / base_a[idx].sum(1)   # = FVE_b − FVE_a per resample
        lo, hi = np.quantile(boot, [0.025, 0.975])
        name = k.replace("__", "/")
        table[name] = {"a": float(fa), "b": float(fb), "diff": float(fb - fa), "ci95": [float(lo), float(hi)]}
        print(f"{name:<16}{fa:>9.3f}{fb:>9.3f}{fb - fa:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]")
    stats = {}
    if args.texts:
        for lab, d in ((args.label_a, args.a), (args.label_b, args.b)):
            stats[lab] = text_stats(Path(d), Path(args.texts))
            print(f"AV text, {lab}: quote rate {stats[lab]['quote_rate']:.3f}, copying {stats[lab]['copying_mean']:.3f}, "
                  f"{stats[lab]['words_mean']:.1f} words")
    if args.out:
        Path(args.out).write_text(json.dumps({"n_rows": len(common), "a": args.a, "b": args.b, "conditions": table,
                                              "av_text": stats}, indent=1))


if __name__ == "__main__":
    main()
