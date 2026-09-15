"""Cross-size report for the single-layer Future Oracle scan (0.6B / 1.7B / 8B).

    python scripts/fl_size_scan_report.py --root <dir with evals_{size}_L{L}/ and data_*/> \
        --sizes 0.6b:19:data_0.6b_L19,1.7b:19:data_1.7b_L19,8b:24:data_8b_evalu --out <dir>

Per size, from the pipeline outputs:
  own set     p1 / tf_p1 at N=0..3 for real + controls, target_window m=1..32, n-gram, probe
              (each model's own top-1-correct positions of the SHARED unfiltered eval split)
  m*          effective context window: log2-interpolated m at which the frozen target given the
              last m tokens matches the oracle's free-running p1 (size-free "worth m tokens")
  intersection  positions where ALL sizes are top-1 correct; p1 per N recomputed from the dumped
              readouts vs each model's own greedy labels, with binomial SE
  trajectory  own-set p1 per saved checkpoint
Writes size_scan.md (+ size_scan.json, size_scan.png)."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

N_MAX = 3
COND = ("real", "shuffled", "none", "wrong_layer")


def load_agg(path: str) -> dict:
    """aggregate eval JSONL -> {(condition, N, metric): (value, n)}"""
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("N") is not None and r["N"] <= N_MAX:
            out[(r["condition"], r["N"], r["metric"])] = (r["value"], r.get("n"))
    return out


def load_baselines(path: str) -> dict:
    """baselines.jsonl -> {('target_window', m) | ('ngram', order) | ('probe', layer): {N: p1}}"""
    out = defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r.get("metric") != "p1" or r.get("N") is None or not (0 <= r["N"] <= N_MAX):
            continue
        b = r.get("baseline") or ""
        if "window" in r:
            out[("target_window", int(r["window"]))][r["N"]] = r["value"]
        elif "order" in r:
            out[("ngram", int(r["order"]))][r["N"]] = r["value"]
        elif "layer" in r and (r.get("kind") == "probe" or "probe" in b):
            out[("probe", int(r["layer"]))][r["N"]] = r["value"]
        else:
            out[(b or "other", r.get("layer"))][r["N"]] = r["value"]
    return out


def effective_window(real: dict, tw: dict) -> dict:
    """per N: m at which target_window p1 == real p1, log2-interpolated (None if outside range)."""
    ms = sorted(m for (_, m) in [k for k in tw if k[0] == "target_window"])
    out = {}
    for N in range(N_MAX + 1):
        y = real.get(N)
        xs = [(m, tw[("target_window", m)][N]) for m in ms if N in tw[("target_window", m)]]
        if y is None or len(xs) < 2:
            out[N] = None; continue
        if y <= xs[0][1]:
            out[N] = xs[0][0]; continue
        if y >= xs[-1][1]:
            out[N] = float("inf"); continue
        for (m0, p0), (m1, p1) in zip(xs, xs[1:]):
            if p0 <= y <= p1:
                f = (y - p0) / max(1e-9, p1 - p0)
                out[N] = 2 ** (math.log2(m0) + f * (math.log2(m1) - math.log2(m0)))
                break
    return out


def doc_ids(data_dir: str) -> dict[int, str]:
    """doc_idx is a per-collection counter (a skipped doc shifts it; an eval-only split starts at 0):
    join across collections on the corpus doc_id string from docs.parquet instead."""
    t = pq.read_table(f"{data_dir}/docs.parquet", columns=["doc_idx", "doc_id"]).to_pandas()
    return {int(i): str(d) for i, d in zip(t.doc_idx, t.doc_id)}


def labels_from_parquet(path: str, layer: int, ids: dict[int, str]) -> dict[tuple[str, int], list[int]]:
    """(doc_id, t) -> greedy_ids for rows of `layer` (unfiltered split: every stored position)."""
    t = pq.read_table(path, columns=["doc_idx", "t", "activation_layer", "greedy_ids"]).to_pandas()
    t = t[t.activation_layer == layer]
    return {(ids[int(d)], int(p)): list(g) for d, p, g in zip(t.doc_idx, t.t, t.greedy_ids)}


def per_position_hits(readouts_path: str, labels: dict, ids: dict[int, str], condition: str = "real") -> dict[tuple[str, int], list[bool]]:
    """(doc_id, t) -> [hit at N=0..N_MAX] using the k=N+1 readout row and the greedy label."""
    hits: dict[tuple[str, int], list] = defaultdict(lambda: [None] * (N_MAX + 1))
    for l in open(readouts_path):
        r = json.loads(l)
        if r["condition"] != condition or r["k"] > N_MAX + 1:
            continue
        key = (ids[int(r["doc_idx"])], int(r["t"]))
        N = r["k"] - 1
        lab = labels.get(key)
        if lab is None:
            continue
        ro = r["readout"]
        hits[key][N] = (len(ro) > N and len(lab) > N and int(ro[N]) == int(lab[N]))
    return hits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--sizes", required=True, help="size:layer:datadir,... e.g. 0.6b:19:data_0.6b_L19")
    p.add_argument("--run-fmt", default="sft_replace_embed_a2.0_greedy_distill_{size}_L{layer}")
    p.add_argument("--step", type=int, default=8000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sizes = [s.split(":") for s in a.sizes.split(",")]
    R = {}
    for size, layer, ddir in sizes:
        layer = int(layer)
        run = a.run_fmt.format(size=size, layer=layer)
        E = f"{a.root}/evals_{size}_L{layer}"
        final = f"{E}/{run}_iter_{a.step:07d}.jsonl"
        if not os.path.exists(final):
            print(f"[skip] {size}: {final} missing"); continue
        agg = load_agg(final)
        real = {N: agg[("real", N, "p1")][0] for N in range(N_MAX + 1) if ("real", N, "p1") in agg}
        bl = load_baselines(f"{E}/baselines.jsonl") if os.path.exists(f"{E}/baselines.jsonl") else {}
        traj = {}
        for f in sorted(glob.glob(f"{E}/{run}_iter_*.jsonl")):
            step = int(f.rsplit("iter_", 1)[1][:7])
            g = load_agg(f)
            traj[step] = {N: g[("real", N, "p1")][0] for N in range(N_MAX + 1) if ("real", N, "p1") in g}
        ids = doc_ids(f"{a.root}/{ddir}")
        labels = labels_from_parquet(f"{a.root}/{ddir}/eval.parquet", layer, ids)
        hits = per_position_hits(f"{E}/readouts_{run}_iter_{a.step:07d}.jsonl", labels, ids)
        R[size] = dict(layer=layer, run=run, agg=agg, real=real, baselines=bl, mstar=effective_window(real, bl),
                       traj=traj, hits=hits, n_own=agg.get(("real", 1, "p1"), (None, None))[1])
        print(f"[{size}] L{layer} own n={R[size]['n_own']} real p1={[round(real[N],3) for N in sorted(real)]} "
              f"m*={ {N: (round(m,1) if m not in (None, float('inf')) else m) for N, m in R[size]['mstar'].items()} }")
    if not R:
        raise SystemExit("nothing to report")
    # ---- shared docs: an eval-only collection can start a few corpus docs earlier than the others'
    # eval split (those docs are then TRAINING docs of that model) -> own-set p1 is recomputed from
    # the per-position hits on the documents every size has in its eval split
    shared_docs = None
    for r in R.values():
        ds = {k[0] for k in r["hits"]}
        shared_docs = ds if shared_docs is None else shared_docs & ds
    for size, r in R.items():
        H = np.array([h for k, h in r["hits"].items() if k[0] in shared_docs and all(x is not None for x in h)], dtype=float)
        r["own_shared"] = {N: float(H[:, N].mean()) for N in range(N_MAX + 1)} if len(H) else {}
        r["n_own_shared"] = len(H)
        r["mstar"] = effective_window(r["own_shared"] or r["real"], r["baselines"])
        print(f"[{size}] own set on shared docs: n={len(H)} p1={[round(v, 3) for v in r['own_shared'].values()]}")
    # ---- intersection ----
    keys = None
    for size, r in R.items():
        ks = {k for k, h in r["hits"].items() if k[0] in shared_docs and all(x is not None for x in h)}
        keys = ks if keys is None else keys & ks
    inter = {}
    for size, r in R.items():
        H = np.array([r["hits"][k] for k in sorted(keys)], dtype=float) if keys else np.zeros((0, N_MAX + 1))
        m = H.mean(0) if len(H) else [float("nan")] * (N_MAX + 1)
        inter[size] = {N: (float(m[N]), float(math.sqrt(m[N] * (1 - m[N]) / max(1, len(H))))) for N in range(N_MAX + 1)}
    n_inter = len(keys or [])
    # ---- markdown ----
    L = ["# Single-layer Future Oracle size scan", "",
         "Same 300 held-out fineweb docs, 40 unfiltered positions/doc, for every size; each oracle is scored on the positions "
         "where its own model is top-1 correct (own set) and on the positions where ALL sizes are (intersection). "
         "p1 = free-running precision at offset N against the model's own greedy continuation; tf = teacher-forced.", "",
         f"## Own set (final checkpoint; {len(shared_docs)} documents shared by every size)", "",
         "p1 recomputed per position on the shared documents; tf (teacher-forced) from the pipeline aggregate over the model's full own set.", "",
         "| size | layer | n | " + " | ".join(f"N={N} p1 / tf" for N in range(N_MAX + 1)) + " |",
         "|---|---|---|" + "---|" * (N_MAX + 1)]
    for size, r in R.items():
        cells = [f"{r['own_shared'].get(N, float('nan')):.3f} / {r['agg'].get(('real', N, 'tf_p1'), (float('nan'),))[0]:.3f}" for N in range(N_MAX + 1)]
        L.append(f"| {size} | {r['layer']} | {r['n_own_shared']} | " + " | ".join(cells) + " |")
    L += ["", "## Controls (own set, p1)", "", "| size | cond | " + " | ".join(f"N={N}" for N in range(N_MAX + 1)) + " |",
          "|---|---|" + "---|" * (N_MAX + 1)]
    for size, r in R.items():
        for c in COND[1:]:
            L.append(f"| {size} | {c} | " + " | ".join(f"{r['agg'].get((c, N, 'p1'), (float('nan'),))[0]:.3f}" for N in range(N_MAX + 1)) + " |")
    L += ["", "## Baselines (own set, p1) and effective window m*", "",
          "| size | baseline | " + " | ".join(f"N={N}" for N in range(N_MAX + 1)) + " |", "|---|---|" + "---|" * (N_MAX + 1)]
    for size, r in R.items():
        for key in sorted(r["baselines"], key=str):
            L.append(f"| {size} | {key[0]} {key[1]} | " + " | ".join(f"{r['baselines'][key].get(N, float('nan')):.3f}" for N in range(N_MAX + 1)) + " |")
        L.append(f"| {size} | **oracle L{r['layer']}** | " + " | ".join(f"**{r['own_shared'].get(N, r['real'][N]):.3f}**" for N in range(N_MAX + 1)) + " |")
        L.append(f"| {size} | m* (tokens of context the vector is worth) | " +
                 " | ".join(("≥32" if m == float("inf") else ("n/a" if m is None else f"{m:.1f}")) for m in r["mstar"].values()) + " |")
    L += ["", f"## Intersection (positions top-1 correct for every size; n = {n_inter})", "",
          "| size | " + " | ".join(f"N={N} p1 ± SE" for N in range(N_MAX + 1)) + " |", "|---|" + "---|" * (N_MAX + 1)]
    for size in R:
        L.append(f"| {size} | " + " | ".join(f"{inter[size][N][0]:.3f} ± {inter[size][N][1]:.3f}" for N in range(N_MAX + 1)) + " |")
    L += ["", "## Trajectory (own set, real p1 by checkpoint)", "",
          "| size | step | " + " | ".join(f"N={N}" for N in range(N_MAX + 1)) + " |", "|---|---|" + "---|" * (N_MAX + 1)]
    for size, r in R.items():
        for step, v in sorted(r["traj"].items()):
            L.append(f"| {size} | {step} | " + " | ".join(f"{v.get(N, float('nan')):.3f}" for N in range(N_MAX + 1)) + " |")
    open(f"{a.out}/size_scan.md", "w").write("\n".join(L) + "\n")
    json.dump({s: {"layer": r["layer"], "run": r["run"], "n_own": r["n_own"], "real": r["real"], "mstar": r["mstar"],
                   "own_shared": r["own_shared"], "n_own_shared": r["n_own_shared"], "shared_docs": len(shared_docs),
                   "traj": r["traj"], "baselines": {f"{k[0]}_{k[1]}": v for k, v in r["baselines"].items()},
                   "intersection": inter[s], "n_intersection": n_inter}
               for s, r in R.items()}, open(f"{a.out}/size_scan.json", "w"), indent=1, default=str)
    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        Ns = list(range(N_MAX + 1))
        for size, r in R.items():
            ax[0].plot(Ns, [r["own_shared"].get(N, r["real"][N]) for N in Ns], "o-", label=f"{size} L{r['layer']} (own set)")
            ax[1].errorbar(Ns, [inter[size][N][0] for N in Ns], yerr=[inter[size][N][1] for N in Ns], fmt="o-", capsize=3, label=size)
            tw = sorted((m, [r["baselines"][("target_window", m)].get(N) for N in Ns]) for (b, m) in r["baselines"] if b == "target_window")
            if tw:
                ax[2].plot([m for m, _ in tw], [v[1] for _, v in tw], "s--", label=f"{size}: target given last m tokens, N=1")
                ax[2].axhline(r["own_shared"].get(1, r["real"][1]), ls=":", color=ax[2].lines[-1].get_color())
        ax[0].set(title="free-running p1 vs offset (own set)", xlabel="N (tokens ahead)", ylabel="p1"); ax[0].legend(fontsize=8)
        ax[1].set(title=f"intersection positions (n={n_inter})", xlabel="N"); ax[1].legend(fontsize=8)
        ax[2].set(title="effective window: oracle p1 at N=1 (dotted) vs context baseline", xlabel="m (tokens of context)", xscale="log", ylabel="p1 at N=1")
        ax[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{a.out}/size_scan.png", dpi=130)
    except Exception as e:      # matplotlib optional
        print("[plot] skipped:", e)
    print(open(f"{a.out}/size_scan.md").read())


if __name__ == "__main__":
    main()
