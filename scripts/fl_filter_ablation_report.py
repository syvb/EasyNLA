"""Report for the top-1-filter ablation: precision@1 by offset for each adapter x eval set x
condition, with the unfiltered eval set split into positions where the target's next-token
prediction was correct ("top1_ok") and where it was not ("top1_wrong").

    python scripts/fl_filter_ablation_report.py --evals-dir <dir with readouts_ablation_*.jsonl> \
        --filt-parquet data/eval.parquet --unf-parquet data_unf/eval.parquet

Offsets follow the eval convention: p@1 at N uses rows with K = N+1 (the last readout token).
The `all-K` columns use every readout that is long enough, i.e. offset j of every row with K > j."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--evals-dir", required=True)
p.add_argument("--filt-parquet", required=True)
p.add_argument("--unf-parquet", required=True)
p.add_argument("--offsets", default="0,1,2,3")
a = p.parse_args()
offs = [int(x) for x in a.offsets.split(",")]


def load_positions(path):
    t = pq.read_table(path, columns=["doc_idx", "t", "activation_layer", "target_ids", "target_top5", "p_top1", "greedy_ids"]).to_pydict()
    out = {}
    for d, tt, tid, top5, pt, gid in zip(t["doc_idx"], t["t"], t["target_ids"], t["target_top5"], t["p_top1"], t["greedy_ids"]):
        out[(int(d), int(tt))] = {"text": list(tid), "greedy": list(gid), "top1_ok": int(top5[0]) == int(tid[0]), "p_top1": float(pt)}
    return out


pos = {"filt": load_positions(a.filt_parquet), "unf": load_positions(a.unf_parquet)}
hits = defaultdict(list)     # (run, evalset, cond, subset, N, mode) -> [0/1]
for path in sorted(glob.glob(os.path.join(a.evals_dir, "readouts_ablation_*.jsonl"))):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            info = pos[r["evalset"]][(r["doc_idx"], r["t"])]
            subsets = ["all"] + (["top1_ok" if info["top1_ok"] else "top1_wrong"] if r["evalset"] == "unf" else [])
            k, ro, tgt = r["k"], r["readout"], info[r.get("label", "text")]   # score against the label the eval used
            for sub in subsets:
                N = k - 1
                if N in offs:
                    hits[(r["checkpoint"], r["evalset"], r["condition"], sub, N, "eval-K")].append(int(len(ro) > N and ro[N] == tgt[N]))
                for j in offs:
                    if j < k:
                        hits[(r["checkpoint"], r["evalset"], r["condition"], sub, j, "all-K")].append(int(len(ro) > j and ro[j] == tgt[j]))

runs = sorted({k[0] for k in hits}); conds = ["real", "shuffled", "none"]
for mode in ("eval-K", "all-K"):
    print(f"\n=== p@1 by offset N ({mode}) ===")
    print(f'{"adapter":<14}{"evalset":<9}{"subset":<12}{"cond":<10}' + "".join(f"{'N='+str(n):>10}" for n in offs) + f"{'n(N=0)':>9}")
    for run in runs:
        for es in ("filt", "unf"):
            for sub in (["all"] if es == "filt" else ["all", "top1_ok", "top1_wrong"]):
                for c in conds:
                    vals = [hits.get((run, es, c, sub, n, mode), []) for n in offs]
                    if not any(vals):
                        continue
                    print(f"{run:<14}{es:<9}{sub:<12}{c:<10}" + "".join(f"{np.mean(v) if v else float('nan'):>10.3f}" for v in vals) + f"{len(vals[0]):>9}")
frac = np.mean([v["top1_ok"] for v in pos["unf"].values()])
print(f"\nunfiltered eval: {len(pos['unf'])} positions, top-1 correct fraction = {frac:.3f}; mean p_top1 = {np.mean([v['p_top1'] for v in pos['unf'].values()]):.3f}")
print(f"filtered eval:   {len(pos['filt'])} positions; mean p_top1 = {np.mean([v['p_top1'] for v in pos['filt'].values()]):.3f}")
