"""How good is the head NLA beyond knowing which head it reads? Per-head FVE against head-only controls.

    python scripts/heads_head_identity_controls.py --out <json>

Head identity is nearly free: the AV can read it off the write, and the AR is told it ("Attention head L24Hh:").
FVE's baseline is already each head's mean write, so 0% means "knows the head". This script compares the NLA
against every head-only control on the held-out eval rows (T=1 checks of the RL and matryoshka pairs):
  same_head_other_write  the AR reads the AV's explanation of ANOTHER write from the SAME head (av__shuf)
  empty_text             the AR reads only the head tag (ar__empty)
  head_only_floor        the best constant a normalized reader can output knowing the head: its mean direction
and, for scale, re-scores the head-only floor and the NLA against ONE mean over all heads (a baseline that ignores
head identity). Per head: true − the best head-only control, and the share of the floor→perfect gap closed.
Also how a head's writes spread in its output space (effective directions, pairwise cosines): why identity buys little.
Reads only public HF data; no GPU.
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True)
a = p.parse_args()
get = lambda f: hf_hub_download("syvb/head-nla-qwen3-8b", f, repo_type="dataset")
S2 = 2 * 64 ** 2                       # direction space: err = 2·64²·(1 − cos) for a unit prediction
runs = {"RL": "l24/std_rl/results/checks_rl_t1", "Matryoshka": "l24/mtry/checks_t1", "SFT": "l24/std_rl/results/checks_sft_t1"}
Z = {k: np.load(get(f"{v}/row_errors_direction.npz")) for k, v in runs.items()}
h = Z["RL"]["head"]; base = Z["RL"]["ar__head_mean"]
# head-only floor: the per-head mean direction (best constant normalized guess), from the AR-train split
def load(f):
    t = pq.read_table(get(f), columns=["activation_vector", "head"]).to_pandas()
    v = np.stack(t.activation_vector.values).astype(np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True), t["head"].values
Utr, Htr = load("l24/sft_data/ar_sft.parquet"); Ute, Hte = load("l24/sft_data/eval.parquet")
assert (Hte == h).all()
mdir = {k: (lambda m: m / np.linalg.norm(m))(Utr[Htr == k].mean(0)) for k in range(32)}
floor_err = np.array([S2 * (1 - Ute[i] @ mdir[h[i]]) for i in range(len(h))])
gmean = Utr.mean(0)                                    # ONE mean over all heads: a baseline that ignores head identity
gbase = (64 ** 2) * ((Ute - gmean) ** 2).sum(1)
assert np.allclose((64 ** 2) * ((Ute - np.stack([Utr[Htr == k].mean(0) for k in range(32)])[h]) ** 2).sum(1), base, rtol=1e-3, atol=1e-2)
f = lambda e, b, m=slice(None): 1 - e[m].mean() / b[m].mean()
out = {"pooled": {}, "per_head": {}}
for name, z in [("RL", Z["RL"]), ("Matryoshka", Z["Matryoshka"])]:
    out["pooled"][name] = {"true": f(z["av__true"], base), "same_head_other_write": f(z["av__shuf"], base),
                           "empty_text": f(z["ar__empty"], base), "head_only_floor": f(floor_err, base),
                           "vs_global_mean__head_only": f(floor_err, gbase), "vs_global_mean__true": f(z["av__true"], gbase)}
    rows = []
    for k in range(32):
        m = h == k
        t, s, e, fl = f(z["av__true"], base, m), f(z["av__shuf"], base, m), f(z["ar__empty"], base, m), f(floor_err, base, m)
        rows.append({"head": k, "true": t, "shuf": s, "empty": e, "floor": fl, "beyond_head_only": t - max(s, e, fl),
                     "share_of_gap_closed": (t - fl) / (1 - fl)})
    out["per_head"][name] = rows
for name in ("RL", "Matryoshka"):
    p = out["pooled"][name]; r = out["per_head"][name]
    print(f"{name}: FVE true {p['true']:+.1%} | same head, other write {p['same_head_other_write']:+.1%} | empty text {p['empty_text']:+.1%} | head-only floor {p['head_only_floor']:+.1%}")
    print(f"   vs ONE global mean instead: head identity alone {p['vs_global_mean__head_only']:+.1%}, NLA {p['vs_global_mean__true']:+.1%}")
    b = np.array([x["beyond_head_only"] for x in r]); g = np.array([x["share_of_gap_closed"] for x in r])
    print(f"   per head: true − best head-only control: median {np.median(b):+.0%} (min {b.min():+.0%} H{int(np.argmin(b))}, max {b.max():+.0%} H{int(np.argmax(b))}); heads where true beats every head-only control: {(b > 0).sum()}/32")
    print(f"   per head: share of the floor→perfect gap closed: median {np.median(g):.0%} (range {g.min():.0%}–{g.max():.0%})")
# Why head identity buys so little: how a head's unit writes spread within its 128-d output space.
def spread(u, rng):
    i, j = rng.integers(0, len(u), 4000), rng.integers(0, len(u), 4000); k = i != j
    c = (u[i[k]] * u[j[k]]).sum(1)
    ev = np.linalg.eigvalsh(u.T @ u / len(u))[::-1]          # second moment E[uuᵀ]
    ev = ev[ev > 0]
    return {"rho": float(np.linalg.norm(u.mean(0))), "mean_cos_pair": float(c.mean()), "mean_abs_cos_pair": float(np.abs(c).mean()),
            "top1_share": float(ev[0] / ev.sum()), "top10_share": float(ev[:10].sum() / ev.sum()), "effective_dims": float(ev.sum() ** 2 / (ev ** 2).sum())}
rng = np.random.default_rng(0)
sp = [spread(Utr[Htr == k], rng) for k in range(32)]
i, j = rng.integers(0, len(Utr), 20000), rng.integers(0, len(Utr), 20000); k = Htr[i] != Htr[j]
out["spread_ar_train"] = {key: float(np.median([s[key] for s in sp])) for key in sp[0]}
out["spread_ar_train"]["random_128d_mean_abs_cos"] = float(np.sqrt(2 / np.pi / 128))
out["spread_ar_train"]["different_heads_mean_abs_cos"] = float(np.abs((Utr[i[k]] * Utr[j[k]]).sum(1)).mean())
print("spread (median over heads, AR-train sites):", {k: round(v, 3) for k, v in out["spread_ar_train"].items()})
json.dump(out, open(a.out, "w"), indent=1, default=float)
