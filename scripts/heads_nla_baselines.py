"""How much of each head's NLA score is reachable without any explanation? (doc-viewer data, CPU only)

Per site, FVE = 1 - err/base with err = 2*64^2*(1-cos); base is recovered from the stored RL cos/FVE, so any
unit-vector guess p can be scored on the same scale. Guesses (none reads an explanation):
  sink_const  the head's position-0 write (= the direction of its first-token "sink" contribution; write-only)
  doc_mean    mean direction of the head's other writes in the same doc (leave-one-out)
  run_mean    mean direction of the head's writes at earlier tokens (causal)
  prev_site   the head's write at the previous token
sink_cos2 = mean cos^2 between a write and the head's position-0 write.

    python scripts/heads_nla_baselines.py --viewer-dir ~/l24_docs/hf/l24/std_rl/doc_viewer
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer-dir", required=True)
    a = p.parse_args()
    docs = sorted(glob.glob(os.path.join(a.viewer_dir, "0*")))
    W = [unit(np.load(f"{d}/writes.npz")["v"].astype(np.float32)) for d in docs]
    sink = unit(np.mean([w[0] for w in W], 0))
    rows = []
    for di, d in enumerate(docs):
        S = pd.read_parquet(f"{d}/sites.parquet")
        for h in range(W[di].shape[1]):
            sh = S[(S["head"] == h) & (S.pos > 0)].sort_values("pos")
            u = W[di][1:, h]
            base = 2 * (1 - sh.rl_cos.values) / (1 - sh.rl_fve.values)
            fve = lambda g: (1 - 2 * (1 - (unit(g) * u).sum(1)) / base).mean()  # noqa: E731
            run = np.vstack([sink[h][None], np.cumsum(u, 0)[:-1]])
            rows.append(dict(head=h, rl=sh.rl_fve.mean(), sink_const=fve(np.broadcast_to(sink[h], u.shape)),
                             doc_mean=fve(u.sum(0)[None] - u), run_mean=fve(run),
                             prev_site=fve(np.vstack([sink[h][None], u[:-1]])), sink_cos2=((u @ sink[h]) ** 2).mean()))
    g = pd.DataFrame(rows).groupby("head").mean()
    g["rl_minus_doc_mean"] = g.rl - g.doc_mean
    pd.set_option("display.width", 200)
    print(g.round(2).sort_values("rl_minus_doc_mean", ascending=False).to_string())
    print("\nmean over heads:\n" + g.mean().round(3).to_string())
    for c in ["sink_const", "doc_mean", "run_mean", "prev_site"]:
        print(f"heads where {c} beats RL: {(g[c] > g.rl).sum()}/{len(g)}")


if __name__ == "__main__":
    main()
