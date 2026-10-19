"""Probe test: do the explanations still carry the write?

    python scripts/heads_explain_probe.py --work-dir <dir> [--embed-model Qwen/Qwen3-8B]

Reads heads_soft_token_readouts.py's texts.parquet / targets.npz / hidden.npy and
heads_soft_token_explain.py's explain.parquet from --work-dir. Per head, a 2-fold cross-fitted ridge
(nla.datagen.heads.probe.crossfit) from each feature set to the write's 128 coordinates in the head's
write basis; held-out FVE at several examples-per-head sizes, so the curve shows whether more data
would lift a feature.

  avg16, lens      the readouts the explanation was written from (TF-IDF)
  explain          the explanation (TF-IDF)
  explain_embed    the explanation, mean-pooled hidden states of --embed-model (without it, read from
                   <work-dir>/explain_embed.npz if an earlier run saved one; else skipped)
  hidden           the injected prompt's final hidden state (dense ceiling)
  explain_shuf     explanations permuted within the head (control; ~0 or below)
  empty            no features (control: the predict-the-mean baseline, ≈ 0 by construction)

Only examples with an explanation are used, for every feature, so all rows of a curve share examples.
Writes probe.json and prints the mean over heads and the explain / avg16 ratio.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.datagen.heads.probe import crossfit, lm_embedder  # noqa: E402

# function words long enough to pass the 4-letter filter; they say nothing about copying
_STOP = {
    "that", "this", "with", "from", "into", "than", "then", "them", "they", "their", "there", "what", "when",
    "where", "which", "while", "whose", "about", "after", "before", "being", "both", "each", "more", "most",
    "much", "other", "over", "same", "some", "such", "very", "also", "just", "only", "like", "toward",
    "towards", "often", "tends", "tend", "sense", "kind", "sort", "word", "words", "thing", "things",
    "something", "someone"
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--explain", default=None, help="default <work-dir>/explain.parquet")
    p.add_argument("--sizes", default="200,400,800,1500")
    p.add_argument("--embed-model", default=None, help="e.g. Qwen/Qwen3-8B (GPU); omit to skip explain_embed")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    work = Path(args.work_dir)
    t = pq.read_table(work / "texts.parquet").to_pydict()
    tg = np.load(work / "targets.npz")
    assert (tg["example_idx"] == np.array(t["example_idx"])).all(), "texts / targets rows differ"
    hidden = np.load(work / "hidden.npy").astype(np.float32)
    ex = pq.read_table(args.explain or work / "explain.parquet").to_pydict()
    ok = ex["ok"] if "ok" in ex else [True] * len(ex["explain"])
    explain = {e: x for e, x, k in zip(ex["example_idx"], ex["explain"], ok, strict=True) if k}
    keep = np.array([bool(explain.get(e)) for e in t["example_idx"]])
    print(f"{keep.sum()}/{len(keep)} examples have a usable explanation (ok=True)", flush=True)

    rows = np.nonzero(keep)[0]
    feats: dict[str, list | np.ndarray] = {
        "avg16": [t["avg16"][i] for i in rows],
        "lens": [t["lens"][i] for i in rows],
        "explain": [explain[t["example_idx"][i]] for i in rows],
        "hidden": hidden[rows],
    }
    cached = work / "explain_embed.npz"
    if not args.embed_model and cached.exists():    # embeddings saved by an earlier run (same rows)
        z = np.load(cached)
        at_row = {e: j for j, e in enumerate(z["example_idx"].tolist())}
        missing = [t["example_idx"][i] for i in rows if t["example_idx"][i] not in at_row]
        assert not missing, f"{len(missing)} explained examples have no cached embedding"
        feats["explain_embed"] = z["x"][[at_row[t["example_idx"][i]] for i in rows]].astype(np.float32)
        print(f"explain_embed from {cached}", flush=True)
    if args.embed_model:
        emb = lm_embedder(args.embed_model, args.device_map)
        feats["explain_embed"] = emb(feats["explain"])
        np.savez(work / "explain_embed.npz", example_idx=np.array(t["example_idx"])[rows],
                 x=feats["explain_embed"].astype(np.float16))
    y_all = tg["y"][rows]
    heads = [f"L{t['layer'][i]}H{t['head'][i]}" for i in rows]
    sizes = [int(s) for s in args.sizes.split(",")]

    res: dict[str, dict[str, dict[str, float]]] = {}
    for hd in sorted(set(heads), key=lambda s: tuple(int(x) for x in s[1:].split("H"))):
        idx = np.array([i for i, h in enumerate(heads) if h == hd])
        res[hd] = {}
        for n in [*sizes, None]:          # None: every example the head has ("all")
            m = len(idx) if n is None else n
            if m > len(idx) or (n is None and m in sizes):   # "all" == a listed size: same fit
                continue
            key = "all" if n is None else str(n)
            sub = idx[np.random.default_rng([args.seed, m]).permutation(len(idx))[:m]]
            y = y_all[sub]
            for name, f in feats.items():
                x = [f[i] for i in sub] if isinstance(f, list) else f[sub]
                res[hd].setdefault(name, {})[key] = crossfit(x, y, seed=args.seed)[0]
            shuf = np.random.default_rng([args.seed, m, 1]).permutation(len(sub))
            res[hd].setdefault("explain_shuf", {})[key] = crossfit([feats["explain"][sub[j]] for j in shuf], y,
                                                                   seed=args.seed)[0]
            res[hd].setdefault("empty", {})[key] = crossfit([""] * len(sub), y, seed=args.seed)[0]
        print(hd, len(idx), json.dumps({k: {n: round(v, 3) for n, v in c.items()} for k, c in res[hd].items()}),
              flush=True)
    (work / "probe.json").write_text(json.dumps(res, indent=1))

    def at(c: dict[str, float], n: str) -> float | None:
        """FVE at size n; "all" is the head's largest fit when it equals a listed size."""
        if n in c:
            return c[n]
        sized = [k for k in c if k != "all"]
        return c[max(sized, key=int)] if n == "all" and sized else None

    names = [*feats, "explain_shuf", "empty"]
    print(f"\nmean held-out FVE over {len(res)} heads, by examples per head (2-fold: trained on half; "
          f"'all' = every example of the head)")
    cols = [*[str(n) for n in sizes], "all"]
    n_heads = {n: sum(at(res[h]["avg16"], n) is not None for h in res) for n in cols}
    cols = [n for n in cols if n == "all" or n_heads[n] == len(res)]   # a size only some heads reach is biased
    print(f"{'features':<14}" + "".join(f"{n:>8}" for n in cols))
    mean = {}
    for name in names:
        mean[name] = {n: float(np.mean(v)) if (v := [at(res[h][name], n) for h in res if at(res[h][name], n)
                                                     is not None]) else float("nan") for n in cols}
        print(f"{name:<14}" + "".join(f"{mean[name][n]:>8.3f}" for n in cols))
    top = "all"
    base = mean["avg16"][top]
    if base > 0.01:
        print(f"\nexplain / avg16 at '{top}': {mean['explain'][top] / base:.2f}"
              + (f"   explain_embed / avg16: {mean['explain_embed'][top] / base:.2f}" if "explain_embed" in mean else ""))
    else:
        print(f"\navg16 FVE at '{top}' is {base:.3f}: too small for an explain / avg16 ratio")
    for name in [n for n in ("explain", "explain_embed") if n in mean]:
        d = np.array([at(res[h][name], top) - at(res[h]["avg16"], top) for h in res])
        print(f"{name} − avg16 per head at '{top}': {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE over "
              f"{len(d)} heads; {int((d > 0).sum())} heads above)")

    # Did the rewriting rewrite? Share of each explanation's content words that are literally one of
    # its own avg16 / lens tokens (case-folded, plural-stripped). High overlap would mean near-lists.
    def stems(xs):
        return {w.lower().rstrip("s") for x in xs for w in re.findall(r"[A-Za-z]{4,}", x)}
    over = []
    for i in rows:
        ev = stems(ast.literal_eval(f"[{t['avg16'][i]}]") + ast.literal_eval(f"[{t['lens'][i]}]"))
        ws = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", explain[t["example_idx"][i]])]
        ws = [w.rstrip("s") for w in ws if w not in _STOP]
        over.append(sum(w in ev for w in ws) / max(1, len(ws)))
    over = np.array(over)
    q = np.quantile(over, [0.5, 0.9, 0.99])
    print(f"explanation content words that are one of its own readout tokens: mean {over.mean():.2f}, "
          f"median {q[0]:.2f}, p90 {q[1]:.2f}, p99 {q[2]:.2f}")
    np.savez(work / "overlap.npz", example_idx=np.array(t["example_idx"])[rows], overlap=over)

if __name__ == "__main__":
    main()
