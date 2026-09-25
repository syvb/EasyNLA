"""Paired Phase 1 contrasts from a kl_audit rows.npz (docs/kl_nla.md).

For AR pairs (kl vs mse, mse vs sft), per text condition: difference in pooled
KL recovered and in top-1 agreement, 95% paired bootstrap over documents.
Also: where the kl-AR gain sits (quartiles of the mse-AR's own per-row KL), and
the hedging check (how often a WRONG document's explanation beats the mean-
direction baseline, and the wrong-minus-matched KL gap).

    python scripts/kl_phase1_contrasts.py <audit_dir>     # writes <audit_dir>/contrasts.{json,md}
"""
import json
import os
import sys

import numpy as np

TEXT = ["gold", "av_greedy", "av_sample", "rl_greedy", "rl_sample", "quote", "wrong"]


def main():
    out = sys.argv[1]
    z = np.load(os.path.join(out, "rows.npz"))
    docs, base = z["doc_id"], z["kl/mean_dir"]
    u, inv = np.unique(docs, return_inverse=True)
    rng = np.random.default_rng(0)

    def paired(a, b, kind, B=4000):
        ok = ~(np.isnan(a) | np.isnan(b))
        A = np.bincount(inv[ok], weights=a[ok], minlength=len(u))
        Bv = np.bincount(inv[ok], weights=b[ok], minlength=len(u))
        D = np.bincount(inv[ok], weights=base[ok], minlength=len(u))
        C = np.bincount(inv[ok], minlength=len(u))
        # KL recovered difference = (KL_b - KL_a) / sum(base); mean difference for top-1
        f = ((lambda A, Bv, D, C: (Bv.sum() - A.sum()) / D.sum()) if kind == "kl_recovered"
             else (lambda A, Bv, D, C: (A.sum() - Bv.sum()) / C.sum()))
        bs = []
        for _ in range(B):
            j = rng.integers(0, len(u), len(u))
            bs.append(f(A[j], Bv[j], D[j], C[j]))
        return [float(f(A, Bv, D, C)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    res = {"paired": {}, "tail": {}, "hedging": {}}
    for x, y in [("kl", "mse"), ("mse", "sft"), ("kl", "sft")]:
        for c in TEXT:
            res["paired"][f"{x}-{y}/{c}"] = {
                "kl_recovered": paired(z[f"kl/{x}/{c}"], z[f"kl/{y}/{c}"], "kl_recovered"),
                "top1": paired(z[f"top1/{x}/{c}"], z[f"top1/{y}/{c}"], "top1"),
            }
    k, m = z["kl/kl/av_greedy"], z["kl/mse/av_greedy"]
    ok = ~(np.isnan(k) | np.isnan(m))
    edges = [-np.inf, *np.percentile(m[ok], [25, 50, 75]), np.inf]
    for q in range(4):
        s = ok & (m > edges[q]) & (m <= edges[q + 1])
        res["tail"][f"Q{q + 1}"] = {"mse_kl": float(m[s].mean()), "kl_kl": float(k[s].mean()),
                                    "kl_better_frac": float(np.mean(k[s] < m[s]))}
    for ar in ["sft", "kl", "mse"]:
        res["hedging"][ar] = {
            "wrong_minus_matched_kl": float(np.nanmean(z[f"kl/{ar}/wrong"] - z[f"kl/{ar}/av_greedy"])),
            "matched_beats_mean_frac": float(np.nanmean(z[f"kl/{ar}/av_greedy"] < base)),
            "wrong_beats_mean_frac": float(np.nanmean(z[f"kl/{ar}/wrong"] < base)),
        }
    json.dump(res, open(os.path.join(out, "contrasts.json"), "w"), indent=1)

    f3 = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"               # noqa: E731
    L = ["# Phase 1 paired contrasts (95% doc bootstrap)", "",
         "| contrast | KL recovered diff | top-1 agreement diff |", "|---|---|---|"]
    for key, v in res["paired"].items():
        L.append(f"| {key} | {f3(v['kl_recovered'])} | {f3(v['top1'])} |")
    L += ["", "Tail (av_greedy; quartiles of the mse-AR's per-row KL):", "",
          "| quartile | mse-AR KL | kl-AR KL | kl-AR better on |", "|---|---|---|---|"]
    for q, v in res["tail"].items():
        L.append(f"| {q} | {v['mse_kl']:.3f} | {v['kl_kl']:.3f} | {v['kl_better_frac']:.0%} |")
    L += ["", "Hedging check (av_greedy vs wrong document):", "",
          "| AR | KL(wrong) - KL(matched) | matched beats mean dir | wrong beats mean dir |", "|---|---|---|---|"]
    for ar, v in res["hedging"].items():
        L.append(f"| {ar} | {v['wrong_minus_matched_kl']:.2f} | {v['matched_beats_mean_frac']:.1%} | "
                 f"{v['wrong_beats_mean_frac']:.1%} |")
    open(os.path.join(out, "contrasts.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
