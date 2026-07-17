"""Aggregate Stage A per-position logs into the domain map (CPU-ok, offline).

Per domain: mean/median ΔNLL, KL, top-1 flip rate, cosine (+ the same stats
with high-norm outlier positions excluded, since Qwen massive-activation
positions can dominate means). Token-class bins: numerals, non-ASCII,
whitespace/punct, other. Also the cross-domain cosine↔ΔNLL correlation —
domains far off that trend line are the interesting ones.

    python -m experiments.bottleneck.analysis_stage_a --results <out_dir>/stage_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def token_class(tok: str) -> str:
    s = tok.strip()
    if not s:
        return "space"
    if any(c.isdigit() for c in s):
        return "numeral"
    if any(ord(c) > 127 for c in s):
        return "non_ascii"
    if all(not c.isalnum() for c in s):
        return "punct"
    return "word"


def summarize(name: str, d: dict) -> dict:
    ok = ~np.isnan(d["nll_clean"])
    dnll = d["nll_patched"][ok] - d["nll_clean"][ok]
    hi = d["h_norm"] > np.percentile(d["h_norm"], 99)
    out = {
        "domain": name,
        "n": int(len(d["kl"])),
        "cosine_mean": float(d["cosine"].mean()),
        "cosine_p10": float(np.percentile(d["cosine"], 10)),
        "dnll_mean": float(dnll.mean()),
        "dnll_median": float(np.median(dnll)),
        "kl_mean": float(d["kl"].mean()),
        "kl_median": float(np.median(d["kl"])),
        "flip_rate": float(d["flip"].mean()),
        "nll_clean_mean": float(d["nll_clean"][ok].mean()),
        "trunc_rate": float(d["truncated"].mean()),
        "extract_fail_rate": float(d["extract_failed"].mean()),
        "z_len_mean": float(d["z_len"].mean()),
        "h_norm_median": float(np.median(d["h_norm"])),
        # robustness: drop top-1% norm positions
        "kl_mean_lo_norm": float(d["kl"][~hi].mean()),
        "flip_rate_lo_norm": float(d["flip"][~hi].mean()),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    args = p.parse_args()
    res = Path(args.results)

    summaries, class_rows = [], []
    for f in sorted(res.glob("*.parquet")):
        t = pq.read_table(f, columns=["domain", "pos", "token", "cosine", "h_norm",
                                      "z_len", "truncated", "extract_failed",
                                      "nll_clean", "nll_patched", "kl", "top1_flip",
                                      "is_response"])
        d = {
            "cosine": np.array(t["cosine"].to_pylist(), dtype=float),
            "h_norm": np.array(t["h_norm"].to_pylist(), dtype=float),
            "z_len": np.array(t["z_len"].to_pylist(), dtype=float),
            "truncated": np.array(t["truncated"].to_pylist(), dtype=float),
            "extract_failed": np.array(t["extract_failed"].to_pylist(), dtype=float),
            "nll_clean": np.array([x if x is not None else np.nan
                                   for x in t["nll_clean"].to_pylist()], dtype=float),
            "nll_patched": np.array([x if x is not None else np.nan
                                     for x in t["nll_patched"].to_pylist()], dtype=float),
            "kl": np.array(t["kl"].to_pylist(), dtype=float),
            "flip": np.array(t["top1_flip"].to_pylist(), dtype=float),
        }
        name = f.stem
        summaries.append(summarize(name, d))

        toks = t["token"].to_pylist()
        classes = np.array([token_class(x) for x in toks])
        for c in sorted(set(classes)):
            m = classes == c
            ok = m & ~np.isnan(d["nll_clean"])
            if ok.sum() < 20:
                continue
            class_rows.append({
                "domain": name, "class": c, "n": int(m.sum()),
                "cosine": float(d["cosine"][m].mean()),
                "kl": float(d["kl"][m].mean()),
                "flip": float(d["flip"][m].mean()),
                "dnll": float((d["nll_patched"][ok] - d["nll_clean"][ok]).mean()),
            })

    cols = list(summaries[0])
    print("\n== Stage A domain map ==")
    print("  ".join(f"{c:>16s}" for c in cols))
    for s in summaries:
        print("  ".join(f"{s[c]:>16.4f}" if isinstance(s[c], float) else f"{s[c]!s:>16s}"
                        for c in cols))

    if len(summaries) >= 3:
        cosv = np.array([s["cosine_mean"] for s in summaries])
        dnll = np.array([s["dnll_mean"] for s in summaries])
        r = np.corrcoef(cosv, dnll)[0, 1]
        print(f"\ncross-domain corr(cosine, ΔNLL) = {r:.3f} "
              f"(hypothesis: strongly negative; off-trend domains are the story)")

    print("\n== token-class bins ==")
    for r_ in class_rows:
        print(f"{r_['domain']:>12s} {r_['class']:>10s} n={r_['n']:>6d} "
              f"cos={r_['cosine']:.3f} kl={r_['kl']:.3f} flip={r_['flip']:.3f} "
              f"dnll={r_['dnll']:+.3f}")

    import csv
    with (res / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summaries)
    with (res / "token_classes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(class_rows[0]))
        w.writeheader()
        w.writerows(class_rows)
    print(f"\nwrote {res/'summary.csv'} and {res/'token_classes.csv'}")


if __name__ == "__main__":
    main()
