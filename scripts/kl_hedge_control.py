"""Hedging control for the Phase 1 result (docs/kl_nla.md).

The KL-trained AR beats the MSE-trained AR by +0.139 KL recovered on the SFT
AV's explanations, but it also makes a WRONG document's explanation far less
harmful. Is its gain extra information read from the explanation, or just
less committal predictions? Two tests on the same 2,844 rows:

1. No-information baselines. An explanation-free input (empty / generic text)
   gives an AR the same prompt for every row, so its prediction is ONE constant
   vector: that AR's learned prior. Its KL is the AR's own no-info baseline.
2. Shrinkage. Blend an AR's predicted direction with a prior direction,
       v = normalize(alpha * pred_hat + (1 - alpha) * prior_hat),
   alpha on a grid, chosen on half the documents and scored on the other half
   (2-fold cross-fit, so no row is scored with an alpha fit on its own doc).
   Priors: the mean activation direction, and (stronger) the KL-trained AR's
   own no-info constant. If shrinking the MSE-trained AR closes most of the
   gap, the KL-trained AR is mostly a better-calibrated MSE AR.

Pre-registered reading (docs/kl_nla.md, written before this ran):
    gap closed = (best cross-fit shrunk mse - raw mse) / (raw kl - raw mse),
    pooled KL recovered on av_greedy.
    >= 50%            -> mostly calibration (hedging)
    < 25% and raw kl - best shrunk mse has CI > 0 -> mostly content
    otherwise         -> mixed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kl_audit import ar_predict, boot_docs, boot_ratio, load_ar, load_eval_rows, log  # noqa: E402

CONDS = ["av_greedy", "gold", "wrong"]
NOINFO = {"empty": "", "generic_short": "A passage of text.",
          "generic_long": "Web text continuing a document about a general topic, written in a neutral "
                          "informative register."}
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 1.0 = raw prediction; 0 = the prior itself


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--explanations", required=True)
    ap.add_argument("--ar-kl", required=True, help="DIR[:LORA_DIR] of the KL-trained AR")
    ap.add_argument("--ar-mse", required=True, help="DIR[:LORA_DIR] of the MSE-trained AR")
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from nla.config import load_nla_config
    from nla.schema import normalize_activation
    from nla.utils.kl_splice import load_splice_kl, tokenize_prefixes

    T, keep, E, docs, gold = load_eval_rows(a.val, a.exclude, a.explanations, a.n)
    ar_dir0 = a.ar_kl.partition(":")[0]
    tok = AutoTokenizer.from_pretrained(ar_dir0)
    cfg = load_nla_config(ar_dir0, tok)
    pids = tokenize_prefixes(tok, [T.detokenized_text_truncated[i] for i in keep],
                             [T.n_raw_tokens[i] for i in keep])
    rows = [i for i in range(len(keep)) if pids[i] is not None]
    P = [pids[i] for i in rows]
    R = len(rows)
    d_rows = docs[rows]
    mean_hat = unit(normalize_activation(gold, 1.0).mean(0))

    # ---- AR predictions: text conditions (per row) and no-info constants ----
    pred, const = {}, {}
    for name, spec in [("kl", a.ar_kl), ("mse", a.ar_mse)]:
        _, critic = load_ar(f"{name}={spec}", a.device)
        for c in CONDS:
            pred[(name, c)] = ar_predict(critic, tok, cfg.critic_prompt_template, E[c],
                                         cfg.mse_scale, a.device)[rows]
        for k, txt in NOINFO.items():
            const[(name, k)] = ar_predict(critic, tok, cfg.critic_prompt_template, [txt],
                                          cfg.mse_scale, a.device)[0]
        del critic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # keep rows where every text condition has a prediction (NaN = no/too-long explanation;
    # the text is shared by both ARs, so the pattern is too). Nothing below sees a NaN.
    ok = torch.ones(R, dtype=torch.bool)
    for v in pred.values():
        ok &= ~torch.isnan(v[:, 0])
    sel = ok.nonzero().flatten().tolist()
    P = [P[i] for i in sel]
    d_rows = d_rows[sel]
    pred = {k: v[sel] for k, v in pred.items()}
    log(f"AR predictions done; {len(sel)}/{R} rows have all of {CONDS}")
    R = len(sel)

    # ---- the vectors to splice (all per row; constants broadcast) ----
    V = {"mean_dir": mean_hat.expand(R, -1)}
    for (name, k), v in const.items():
        V[f"noinfo/{name}/{k}"] = v.expand(R, -1)
    # the KL-trained AR's prior: its best no-info constant, picked below after scoring;
    # shrink toward all three kl constants so the pick costs no extra pass
    priors = {"mean": mean_hat}
    for k in NOINFO:
        priors[f"klprior_{k}"] = unit(const[("kl", k)])
    for name in ["kl", "mse"]:
        for c in CONDS:
            p_hat = unit(pred[(name, c)])
            for pr_name, pr in priors.items():
                if name == "kl" and pr_name != "mean":
                    continue            # the KL AR is only shrunk toward the mean (its own prior is itself)
                for al in ALPHAS:
                    V[f"{name}/{c}/{pr_name}/{al}"] = unit(al * p_hat + (1 - al) * pr.expand_as(p_hat))
    conds = list(V)
    log(f"{len(conds)} vectors per row x {R} rows")

    splice = load_splice_kl(a.target, ar_dir0, cfg.extraction_layer_index, a.device,
                            a.micro_batch, dtype=getattr(torch, a.target_dtype))
    KL = {c: np.full(R, np.nan) for c in conds}
    t0 = time.time()
    chunk = 64
    for cs in range(0, R, chunk):
        idx = list(range(cs, min(cs + chunk, R)))
        pref, vecs, who = [], [], []
        for c in conds:
            v = V[c][idx]
            for k, i in enumerate(idx):
                pref.append(P[i]); vecs.append(v[k]); who.append((c, i))
        with torch.no_grad():
            kl = splice.kl(pref, torch.stack(vecs).to(a.device)).float().cpu().numpy()
        for j, (c, i) in enumerate(who):
            KL[c][i] = kl[j]
        log(f"  splice {min(cs + chunk, R)}/{R} rows ({time.time() - t0:.0f}s)")

    base = KL["mean_dir"]
    def rec(k, m=None):
        m = np.ones(R, dtype=bool) if m is None else m
        return float(1 - k[m].sum() / base[m].sum())

    # ---- 2-fold cross-fit of alpha, by document ----
    u = np.unique(d_rows)
    perm = np.random.default_rng(0).permutation(len(u))
    fold_of_doc = {u[j]: (r % 2) for r, j in enumerate(perm)}
    fold = np.array([fold_of_doc[d] for d in d_rows])

    def crossfit(prefix):
        """Per-row KL at the alpha fit on the OTHER fold (on av_greedy, pooled KL recovered),
        applied to every condition. Returns ({cond: kl array}, {fold: alpha})."""
        best = {}
        for f in (0, 1):
            m = (fold != f)
            scores = {al: rec(KL[f"{prefix.format(c='av_greedy')}/{al}"], m) for al in ALPHAS}
            best[f] = max(scores, key=scores.get)
        out = {}
        for c in CONDS:
            arr = np.full(R, np.nan)
            for f in (0, 1):
                m = fold == f
                arr[m] = KL[f"{prefix.format(c=c)}/{best[f]}"][m]
            out[c] = arr
        return out, best

    shr = {}
    for name, pr in [("mse", "mean"), ("kl", "mean")] + [("mse", f"klprior_{k}") for k in NOINFO]:
        shr[(name, pr)] = crossfit(f"{name}/{{c}}/{pr}")
    raw = {(n, c): KL[f"{n}/{c}/mean/1.0"] for n in ["kl", "mse"] for c in CONDS}

    # paired doc bootstrap for differences of pooled KL recovered, and for the gap-closed ratio
    uu, inv = np.unique(d_rows, return_inverse=True)
    bsum = lambda x: np.bincount(inv, weights=x, minlength=len(uu))  # noqa: E731
    Bd = bsum(base)
    rng = np.random.default_rng(1)
    J = [rng.integers(0, len(uu), len(uu)) for _ in range(4000)]

    def diff(x, y):   # rec(x) - rec(y) = (sum y - sum x) / sum base
        X, Y = bsum(x), bsum(y)
        pt = (Y.sum() - X.sum()) / Bd.sum()
        bs = [(Y[j].sum() - X[j].sum()) / Bd[j].sum() for j in J]
        return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def gap_closed(shrunk):
        K, M, S = bsum(raw[("kl", "av_greedy")]), bsum(raw[("mse", "av_greedy")]), bsum(shrunk)
        f = lambda j: (M[j].sum() - S[j].sum()) / (M[j].sum() - K[j].sum())   # noqa: E731
        allj = np.arange(len(uu))
        bs = [f(j) for j in J]
        return [float(f(allj)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    summ = {"n_rows": R, "n_docs": int(len(u)), "alphas": ALPHAS,
            "constants": {c: boot_ratio(KL[c], base, d_rows) for c in conds if c.startswith("noinfo/")},
            "constants_kl_nats": {c: boot_docs(KL[c], d_rows) for c in conds if c.startswith("noinfo/")},
            "raw": {f"{n}/{c}": boot_ratio(raw[(n, c)], base, d_rows) for (n, c) in raw},
            "shrunk": {}, "alpha_curves": {}, "contrasts": {}}
    for (name, pr), (arrs, best) in shr.items():
        key = f"{name}->{pr}"
        summ["shrunk"][key] = {"alpha_by_fold": best,
                               **{c: boot_ratio(arrs[c], base, d_rows) for c in CONDS}}
        summ["alpha_curves"][key] = {str(al): rec(KL[f"{name}/av_greedy/{pr}/{al}"]) for al in ALPHAS}
    # the strongest shrunk MSE AR: best of the priors, chosen on pooled av_greedy (reported as such)
    mse_keys = [k for k in shr if k[0] == "mse"]
    best_key = max(mse_keys, key=lambda k: rec(shr[k][0]["av_greedy"]))
    best_mse = shr[best_key][0]
    summ["best_mse_shrink"] = f"mse->{best_key[1]}"
    C = summ["contrasts"]
    C["kl_raw - mse_raw (av_greedy)"] = diff(raw[("kl", "av_greedy")], raw[("mse", "av_greedy")])
    for k in mse_keys:
        C[f"kl_raw - mse->{k[1]} (av_greedy)"] = diff(raw[("kl", "av_greedy")], shr[k][0]["av_greedy"])
        C[f"gap closed by mse->{k[1]}"] = gap_closed(shr[k][0]["av_greedy"])
    C["kl->mean - mse->mean (av_greedy)"] = diff(shr[("kl", "mean")][0]["av_greedy"],
                                                 shr[("mse", "mean")][0]["av_greedy"])
    for c in ["gold", "wrong"]:
        C[f"kl_raw - best mse shrink ({c})"] = diff(raw[("kl", c)], best_mse[c])
    # information over each AR's own prior (nats): KL(own best no-info constant) - KL(matched)
    for name in ["kl", "mse"]:
        ck = min((c for c in conds if c.startswith(f"noinfo/{name}/")), key=lambda c: np.nanmean(KL[c]))
        g = KL[ck] - raw[(name, "av_greedy")]
        summ[f"gain_over_own_prior_nats/{name}"] = {"prior": ck, **dict(zip(
            ["mean", "lo", "hi", "n_docs"], boot_docs(g, d_rows)))}

    gc = C[f"gap closed by {summ['best_mse_shrink']}"]
    lo_ci = C[f"kl_raw - {summ['best_mse_shrink']} (av_greedy)"][1]
    verdict = ("mostly calibration (hedging)" if gc[0] >= 0.5 else
               "mostly content" if gc[0] < 0.25 and lo_ci > 0 else "mixed")
    summ["verdict"] = verdict
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "rows.npz"), doc_id=d_rows, fold=fold,
                        **{f"kl/{c}": KL[c] for c in conds})

    f3 = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"                   # noqa: E731
    L = [f"# Hedging control ({R} rows / {len(u)} docs)", "",
         f"**Verdict (pre-registered rule): {verdict}**. Best shrunk MSE AR: {summ['best_mse_shrink']}, "
         f"gap closed {f3(gc)}.", "", "No-info constants (KL recovered vs mean direction):", ""]
    for c, v in summ["constants"].items():
        L.append(f"- {c}: {f3(v)}  (KL {summ['constants_kl_nats'][c][0]:.3f} nats)")
    L += ["", "| AR | av_greedy | gold | wrong | alpha (fold 0/1) |", "|---|---|---|---|---|"]
    for n in ["kl", "mse"]:
        L.append(f"| {n} raw | {f3(summ['raw'][f'{n}/av_greedy'])} | {f3(summ['raw'][f'{n}/gold'])} | "
                 f"{f3(summ['raw'][f'{n}/wrong'])} | 1.0 |")
    for k, v in summ["shrunk"].items():
        L.append(f"| {k} | {f3(v['av_greedy'])} | {f3(v['gold'])} | {f3(v['wrong'])} | "
                 f"{v['alpha_by_fold'][0]}/{v['alpha_by_fold'][1]} |")
    L += ["", "Paired contrasts (pooled KL recovered, 95% doc bootstrap):", ""]
    for k, v in C.items():
        L.append(f"- {k}: {f3(v)}")
    for n in ["kl", "mse"]:
        g = summ[f"gain_over_own_prior_nats/{n}"]
        L.append(f"- {n}: KL gain over own prior ({g['prior']}): {g['mean']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}] nats")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
