"""Paired comparison of the two RL pilot arms' checkpoint curves (docs/kl_nla_phase2.md).

Both arms' kl_curve_eval runs score the SAME 500 audit rows (same seed) with the same frozen
judges, so each checkpoint can be compared row by row: KL arm minus MSE arm, per step and
pooled over steps 20-100, with a 95% bootstrap over documents (resampling documents jointly
across steps). Metrics: KL recovered under A_cal and A_kl (from rows.npz), and names-the-next-
token (recomputed per row from explanations.json + the target's greedy next token).
FVE is reported per arm from curves.json (per-row MSE is not saved).

    python scripts/kl_rl_compare.py <curves_rl_kl dir> <curves_rl_mse dir> <out dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_audit import names_token  # noqa: E402


def load(d):
    z = np.load(os.path.join(d, "rows.npz"))
    e = json.load(open(os.path.join(d, "explanations.json")))
    c = json.load(open(os.path.join(d, "curves.json")))
    return z, e, c


def main():
    dk, dm, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    zk, ek, ck = load(dk)
    zm, em, cm = load(dm)
    assert (zk["doc_id"] == zm["doc_id"]).all() and (zk["top_orig"] == zm["top_orig"]).all(), "row sets differ"
    docs = zk["doc_id"]
    u, inv = np.unique(docs, return_inverse=True)
    base = zk["kl/mean/None"]
    assert np.allclose(base, zm["kl/mean/None"], rtol=1e-3, atol=1e-4), "baselines differ"
    steps = sorted(int(s) for s in ek if s.isdigit())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ts = [tok.decode([t]).strip() for t in zk["top_orig"]]
    content = np.array([len(x) >= 3 and any(c.isalpha() for c in x) for x in ts])

    def named(e, s):
        ex = e[str(s)]["explanation"]
        return np.array([names_token(ex[i], ts[i]) if content[i] else np.nan for i in range(len(ts))])

    rng = np.random.default_rng(0)
    J = [rng.integers(0, len(u), len(u)) for _ in range(4000)]
    bsum = lambda x: np.bincount(inv, weights=x, minlength=len(u))           # noqa: E731

    def rec_diff(sel_steps, judge):
        """pooled over sel_steps: KL recovered(kl arm) - KL recovered(mse arm), rows valid in both."""
        K, M, D = 0, 0, 0
        for s in sel_steps:
            a, b = zk[f"kl/{judge}/{s}"], zm[f"kl/{judge}/{s}"]
            ok = ~(np.isnan(a) | np.isnan(b))
            K = K + bsum(np.where(ok, a, 0)); M = M + bsum(np.where(ok, b, 0)); D = D + bsum(np.where(ok, base, 0))
        f = lambda j: (M[j].sum() - K[j].sum()) / D[j].sum()                 # noqa: E731
        allj = np.arange(len(u))
        bs = [f(j) for j in J]
        return [float(f(allj)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def mean_diff(sel_steps, fn):
        S, C = 0, 0
        for s in sel_steps:
            a, b = fn(ek, s), fn(em, s)
            ok = ~(np.isnan(a) | np.isnan(b))
            S = S + bsum(np.where(ok, a - b, 0)); C = C + bsum(ok.astype(float))
        f = lambda j: S[j].sum() / C[j].sum()                                  # noqa: E731
        bs = [f(j) for j in J]
        return [float(f(np.arange(len(u)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    res = {"per_step": {}, "pooled_20_100": {}}
    for s in steps:
        res["per_step"][s] = {
            "kl_recovered/A_cal": rec_diff([s], "cal"),
            "kl_recovered/A_kl": rec_diff([s], "kl"),
            "names_next_token": mean_diff([s], named),
            "fve/A_mse (kl, mse)": [next(r["fve/A_mse"] for r in ck["curves"] if r["step"] == s),
                                    next(r["fve/A_mse"] for r in cm["curves"] if r["step"] == s)],
        }
    late = [s for s in steps if s >= 20]
    res["pooled_20_100"] = {"kl_recovered/A_cal": rec_diff(late, "cal"), "kl_recovered/A_kl": rec_diff(late, "kl"),
                            "names_next_token": mean_diff(late, named)}
    # each arm's own change from step 0 (same rows), pooled over late steps: did RL move the judges?
    for arm, z in [("kl", zk), ("mse", zm)]:
        for judge in ["cal", "kl"]:
            num = 0; den = 0
            for s in late:
                a, b = z[f"kl/{judge}/{s}"], z[f"kl/{judge}/0"]
                ok = ~(np.isnan(a) | np.isnan(b))
                num = num + bsum(np.where(ok, b - a, 0)); den = den + bsum(np.where(ok, base, 0))
            f = lambda j, n=num, d=den: n[j].sum() / d[j].sum()              # noqa: E731
            bs = [f(j) for j in J]
            res["pooled_20_100"][f"{arm}_arm_gain_vs_step0/A_{judge}"] = [
                float(f(np.arange(len(u)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    json.dump(res, open(os.path.join(out, "compare.json"), "w"), indent=1)
    f3 = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"                  # noqa: E731
    L = ["# RL pilot: KL arm minus MSE arm (paired, 500 rows / %d docs, 95%% doc bootstrap)" % len(u), "",
         "| step | KL rec. A_cal | KL rec. A_kl | names next token | FVE A_mse (kl / mse) |", "|---|---|---|---|---|"]
    for s, v in res["per_step"].items():
        fk, fm = v["fve/A_mse (kl, mse)"]
        L.append(f"| {s} | {f3(v['kl_recovered/A_cal'])} | {f3(v['kl_recovered/A_kl'])} | "
                 f"{f3(v['names_next_token'])} | {fk[0]:.3f} / {fm[0]:.3f} |")
    L += ["", "Pooled over steps 20-100:", ""]
    for k, v in res["pooled_20_100"].items():
        L.append(f"- {k}: {f3(v)}")
    open(os.path.join(out, "compare.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
