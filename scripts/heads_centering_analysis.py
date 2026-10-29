"""How much did centering the head writes matter? (CPU only, public HF data)

    python scripts/heads_centering_analysis.py --out <json>

Every target is v = W_h (z_h − z̄_h) / dev_rms_h (capped), with z̄_h one global mean per head. The raw write is
w = rms_h · v + mean_write_h (exact except for capped rows, which are left out of the raw statistics). The models only
see and predict directions, so everything here compares unit vectors.

  rotation    per write, cos(ŵ, v̂): how far centering turns the direction the AV sees and the AR predicts
  free_cos    per head, |mean of the unit targets| = the mean cos a constant, head-only prediction gets (the best
              one for a reader that outputs unit vectors), raw vs centered; plus the mean pairwise cos
  offsets     per head, |mean v| (rms units) of a set of tokens: what the global z̄ leaves uncentered, against
              the removed global mean |mean_write| / rms (web documents, early positions, chats)
  variance    shares of E|v|² (web documents, positions ≥ 50): the leftover global mean, between documents, within
  quality     pooled FVE (std RL, matryoshka) by |v| bin and by rotation bin; and FVE weighted by |v|²
  identity    W_O's conditioning: whether a write's head is recoverable from the vector, centered or not
  source      where the mean comes from: (a) per KV group, the cos between the 4 query heads' mean outputs z̄ (they share
              one value space, so a mean made by the shared values points the same way whatever each head attends to);
              (b) web, positions ≥ 50: the mean raw write split into tokens parked on the sink (≥ 90% attention on the
              first token) and the rest, each part's projection on the global mean, and how much parked writes vary;
              (c) cos between one document's mean write and the global mean (could a little data, or the weights, fix it?)

Sets: eval = l24/sft_data/eval.parquet (the training selection: 200 per head, weighted toward large deviations);
web = the 4 doc-viewer documents, every token; chat = the 4 chat-viewer chats, every token.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "syvb/head-nla-qwen3-8b"
WEB = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression", "04_law_gainsborough_fraud"]
CHAT = ["01_ipv4_code", "02_lighthouse_story", "03_train_problem", "04_ada_lovelace"]
H = 32


def get(f: str) -> str:
    return hf_hub_download(REPO, f, repo_type="dataset")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def r(x, k=3):
    return round(float(x), k)


def per_head_free(vs: list[np.ndarray], m: np.ndarray, rms: np.ndarray, cap: np.ndarray) -> dict:
    """vs[h]: [n, d] targets of head h → rotation, free cos and pairwise cos, raw vs centered."""
    rot, rho_c, rho_r, pw_c, pw_r = [], [], [], [], []
    for h, v in enumerate(vs):
        keep = np.linalg.norm(v, axis=1) < cap[h] - 1e-3          # capped rows: raw write not recoverable
        v = v[keep]
        w = rms[h] * v + m[h]
        uc, ur = unit(v), unit(w)
        rot.append(float(np.median((uc * ur).sum(1))))
        n = len(v)
        for u, rho, pw in ((uc, rho_c, pw_c), (ur, rho_r, pw_r)):
            s = u.sum(0)
            rho.append(float(np.linalg.norm(s) / n))
            pw.append(float((s @ s - n) / (n * (n - 1))))
    a = {k: np.array(x) for k, x in dict(rot=rot, rho_c=rho_c, rho_r=rho_r, pw_c=pw_c, pw_r=pw_r).items()}
    return {"rotation_median_cos": {"median_head": r(np.median(a["rot"])), "min_head": r(a["rot"].min()),
                                    "per_head": [r(x) for x in a["rot"]]},
            "free_cos_centered": {"median_head": r(np.median(a["rho_c"])), "max_head": r(a["rho_c"].max())},
            "free_cos_raw": {"median_head": r(np.median(a["rho_r"])), "max_head": r(a["rho_r"].max()),
                             "per_head": [r(x) for x in a["rho_r"]]},
            "pairwise_cos_centered": {"median_head": r(np.median(a["pw_c"]), 4), "max_head": r(a["pw_c"].max())},
            "pairwise_cos_raw": {"median_head": r(np.median(a["pw_r"]), 4), "max_head": r(a["pw_r"].max()),
                                 "per_head": [r(x) for x in a["pw_r"]]}}


def load_doc(kind: str, name: str):
    base = f"l24/chat_viewer/{name}" if kind == "chat" else f"l24/std_rl/doc_viewer/{name}"
    mt = f"{base}/mtry.parquet" if kind == "chat" else f"l24/mtry/doc_viewer/{name}/mtry.parquet"
    S = pd.read_parquet(get(f"{base}/sites.parquet"), columns=["pos", "head", "v_norm", "first_w", "rl_cos", "rl_fve"])
    S = S.merge(pd.read_parquet(get(mt), columns=["pos", "head", "mtry_cos"]), on=["pos", "head"], validate="one_to_one")
    S["base"] = 2 * (1 - S.rl_cos) / np.maximum(1 - S.rl_fve, 1e-6)
    S["doc"], S["kind"] = name, kind
    V = np.load(get(f"{base}/writes.npz"))["v"].astype(np.float32)           # [T, H, d]
    return S, V


def fve(S: pd.DataFrame, col: str, wt=None) -> float:
    w = np.ones(len(S)) if wt is None else wt
    return r(1 - (w * 2 * (1 - S[col])).sum() / (w * S.base).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    st = load_file(get("l24/meta/head_stats.safetensors"))
    m, rms = st["mean_write"][0].astype(np.float64), st["dev_rms"][0].astype(np.float64)
    ql = list(st["quantile_levels"])
    cap = np.minimum(st["dev_quantiles"][0][:, ql.index(min(ql, key=lambda q: abs(q - 0.999)))] / rms, 4.0)
    mean_ratio = np.linalg.norm(m, axis=1) / rms
    out = {"mean_write_over_rms": {"per_head": [r(x) for x in mean_ratio], "median": r(np.median(mean_ratio))},
           "energy_share_of_mean": {"per_head": [r(x) for x in mean_ratio ** 2 / (1 + mean_ratio ** 2)]}}

    # identity: is a write's head recoverable from the vector? (32 heads × 128 = 4096 = d_model)
    from safetensors.torch import load_file as load_torch
    W = next(iter(load_torch(get("l24/meta/wo.safetensors")).values())).double().numpy()   # [d_model, H*128], bf16 stored
    sv = np.linalg.svd(W, compute_uv=False)
    ev = pq.read_table(get("l24/sft_data/eval.parquet"), columns=["activation_vector", "head"]).to_pydict()
    E = np.array(ev["activation_vector"], dtype=np.float64)
    eh = np.array(ev["head"])
    X = np.linalg.solve(W, E.T).T.reshape(len(E), H, -1)                   # each write's per-head coordinates
    own = (X[np.arange(len(E)), eh] ** 2).sum(1) / (X ** 2).sum((1, 2))
    out["identity"] = {"wo_condition_number": r(sv[0] / sv[-1], 1),
                       "eval_share_of_coords_in_own_head": {"min": r(own.min(), 6), "median": r(np.median(own), 6)}}

    # eval rows (the training distribution: selection-weighted)
    out["eval"] = per_head_free([E[eh == h] for h in range(H)], m, rms, cap)

    # every-token documents
    frames, Vs = [], {}
    for kind, names in (("web", WEB), ("chat", CHAT)):
        for n in names:
            S, V = load_doc(kind, n)
            frames.append(S)
            Vs[(kind, n)] = V
    D = pd.concat(frames, ignore_index=True)
    vd = Vs                                                                 # float32 [T, H, d]

    def stack(kind, lo, hi=10 ** 9):
        return [np.concatenate([V[lo:hi, h] for (k, _), V in vd.items() if k == kind]) for h in range(H)]
    out["web_pos50"] = per_head_free(stack("web", 50), m, rms, cap)
    out["web_pos1_49"] = per_head_free(stack("web", 1, 50), m, rms, cap)
    out["chat_pos50"] = per_head_free(stack("chat", 50), m, rms, cap)

    # offsets: |mean v| per head of a set, in rms units (the global mean that was removed: mean_write_over_rms)
    def off(vs):
        return np.array([np.linalg.norm(v.mean(0)) for v in vs])
    sets = {f"web {n[:2]} pos≥50": [vd[("web", n)][50:, h] for h in range(H)] for n in WEB}
    sets |= {f"chat {n[:2]} pos≥50": [vd[("chat", n)][50:, h] for h in range(H)] for n in CHAT}
    sets |= {"web pos 1–49": stack("web", 1, 50), "web pos 0": [np.stack([vd[("web", n)][0, h] for n in WEB]) for h in range(H)],
             "chat pos 0": [np.stack([vd[("chat", n)][0, h] for n in CHAT]) for h in range(H)],
             "eval rows": [E[eh == h] for h in range(H)]}
    out["offsets_mean_v_norm"] = {k: {"median_head": r(np.median(off(vs))), "max_head": r(off(vs).max()),
                                      "heads_above_removed_mean": int((off(vs) > mean_ratio).sum())} for k, vs in sets.items()}

    # variance decomposition, web pos ≥ 50: global leftover / between documents / within documents
    shares = []
    for h in range(H):
        parts = [vd[("web", n)][50:, h] for n in WEB]
        allv = np.concatenate(parts)
        tot = (allv ** 2).sum(1).mean()
        g = allv.mean(0)
        between = sum(len(x) * ((x.mean(0) - g) ** 2).sum() for x in parts) / len(allv)
        within = sum(((x - x.mean(0)) ** 2).sum() for x in parts) / len(allv)
        shares.append((g @ g / tot, between / tot, within / tot, mean_ratio[h] ** 2 / (tot + mean_ratio[h] ** 2)))
    sh = np.array(shares)
    out["variance_web_pos50"] = {"leftover_global_mean": r(np.median(sh[:, 0])), "between_docs": r(np.median(sh[:, 1])),
                                 "within_docs": r(np.median(sh[:, 2])),
                                 "removed_mean_share_of_raw": r(np.median(sh[:, 3])),
                                 "between_docs_per_head": [r(x) for x in sh[:, 1]]}

    # quality vs |v| and vs rotation (sites at position ≥ 50)
    D["rot"] = np.nan
    for (kind, n), V in vd.items():
        idx = D.index[(D.kind == kind) & (D.doc == n)]
        pos, hd = D.loc[idx, "pos"].to_numpy(), D.loc[idx, "head"].to_numpy()
        v = V[pos, hd]
        D.loc[idx, "rot"] = (unit(v) * unit(rms[hd, None] * v + m[hd])).sum(1)
    q = {}
    for kind in ("web", "chat"):
        X_ = D[(D.kind == kind) & (D.pos >= 50)]
        bins = pd.qcut(X_.v_norm, 5)
        q[kind] = {"by_v_norm_quintile": {str(iv): {"v_norm_median": r(g.v_norm.median()), "rl": fve(g, "rl_cos"),
                                                    "mtry": fve(g, "mtry_cos"), "sink": r(g.first_w.mean())}
                                          for iv, g in X_.groupby(bins, observed=True)},
                   "by_rotation": {lab: {"share": r(len(g) / len(X_)), "rl": fve(g, "rl_cos"), "mtry": fve(g, "mtry_cos")}
                                   for lab, g in X_.groupby(pd.cut(X_.rot, [-1, 0.5, 0.9, 0.99, 1.0001],
                                                                    labels=["<0.5", "0.5–0.9", "0.9–0.99", "≥0.99"]), observed=True)},
                   "unweighted": {"rl": fve(X_, "rl_cos"), "mtry": fve(X_, "mtry_cos")},
                   "weighted_by_v_norm_sq": {"rl": fve(X_, "rl_cos", X_.v_norm.to_numpy() ** 2),
                                             "mtry": fve(X_, "mtry_cos", X_.v_norm.to_numpy() ** 2)}}
    out["quality_pos50"] = q

    # source (a): mean head outputs within each KV group (Qwen3: query head h reads KV head h // 4)
    z = st["mean_head_out"][0].astype(np.float64)
    uz = unit(z)
    grp = {}
    for g in range(H // 4):
        C = uz[4 * g:4 * g + 4] @ uz[4 * g:4 * g + 4].T
        grp[f"KV{g}"] = {"z_bar_norm": [r(x, 2) for x in np.linalg.norm(z[4 * g:4 * g + 4], axis=1)],
                         "pairwise_cos_median": r(np.median(C[np.triu_indices(4, 1)]), 2),
                         "pairwise_cos_min": r(C[np.triu_indices(4, 1)].min(), 2)}
    # source (b): parked vs active tokens (web, positions >= 50)
    fw = {}
    for (kind, n), V in vd.items():
        if kind == "web":
            x = D[(D.kind == kind) & (D.doc == n)]
            f_ = np.zeros(V.shape[:2])
            f_[x.pos.to_numpy(), x["head"].to_numpy()] = x.first_w.to_numpy()
            fw[n] = f_[50:]
    F = np.concatenate([fw[n] for n in WEB])
    park = []
    for h, v in enumerate(stack("web", 50)):
        ok = np.linalg.norm(v, axis=1) < cap[h] - 1e-3
        w, pk = rms[h] * v[ok] + m[h], F[ok, h] >= 0.9
        mm = m[h] @ m[h]
        row = {"head": h, "parked_share": r(pk.mean(), 2),
               "parked_part_of_mean": r((w[pk].sum(0) / len(w)) @ m[h] / mm, 2) if pk.any() else 0.0,
               "active_part_of_mean": r((w[~pk].sum(0) / len(w)) @ m[h] / mm, 2),
               "active_spread_rms": r(np.sqrt(((w[~pk] - w[~pk].mean(0)) ** 2).sum(1).mean()) / rms[h], 2)}
        if pk.sum() > 1:
            row["parked_spread_rms"] = r(np.sqrt(((w[pk] - w[pk].mean(0)) ** 2).sum(1).mean()) / rms[h], 2)
        park.append(row)
    # source (c): how well one document (positions >= 50, ~250 tokens) recovers each head's global mean write
    one = {}
    for (kind, n), V in vd.items():
        dm = rms[None, :, None] * V[50:].mean(0)[None] + m[None]                   # [1, H, d] mean raw write
        one[(kind, n)] = (dm[0] * m).sum(1) / np.linalg.norm(dm[0], axis=1) / np.linalg.norm(m, axis=1)
    sets_h = {"KV6 (H24-27)": [24, 25, 26, 27], "parking heads (H1, H6, H20, H23)": [1, 6, 20, 23],
              "other 24 heads": [h for h in range(H) if h not in (1, 6, 20, 23, 24, 25, 26, 27)]}
    single = {}
    for kind in ("web", "chat"):
        A_ = np.array([c for (k, _), c in one.items() if k == kind])                # [docs, H]
        single[kind] = {nm: {"median": r(np.median(A_[:, hs]), 2), "min": r(A_[:, hs].min(), 2)} for nm, hs in sets_h.items()}
    out["source"] = {"kv_groups_z_bar": grp, "parked_vs_active_web_pos50": park, "single_doc_mean_cos_to_global": single}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("mean_write_over_rms", "energy_share_of_mean")}, indent=1,
                     ensure_ascii=False)[:9000])


if __name__ == "__main__":
    main()
