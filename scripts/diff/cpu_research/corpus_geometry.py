"""Geometry of the three arms' training targets on a real chat corpus with on-policy rollouts.

Reads Phase 0 outputs (alignment, tau) and a Phase 2 `rollouts.parquet`, pulls
whole-sequence Layer-K activations for Target and Base (both wrappers) through
the disk cache, and reports per active role:

  A. noise floor   - ||Delta - mu_role|| against tau_web_median, Fix 3 weights and ESS
  B. composition   - mean-offset share, token-identity (tok1) share, effective rank
  C. recoverability- held-out within-role FVE / cosine when the unit target is predicted
                     from (i) an independent small reader LM on the same text,
                     (ii) h_target and (iii) h_base at the same position
  D. wrapper check - agreement of Delta under Base(ChatML) vs Base(Plain)
  E. behaviour     - does actual refusal (from the rollout text) separate at `asst_last`?

    python corpus_geometry.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from gate_controls import few_shot_acc, grouped_auroc

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.metrics import compute_fisher_snr, compute_tok1_ridge_fve
from nla.diff.wrappers import ACTIVE_ROLES, encode_aligned_pair
from nla.utils.soft_norm import soft_norm_weights

# A rollout is a refusal when it opens with an explicit decline, and a clean compliance when it has
# no hedging or safety language at all. Everything in between ("I'm sorry to hear ...", "It's important
# to clarify ...") is left out of the behaviour analysis instead of being guessed at.
REFUSAL_RE = re.compile(
    r"(?:i['’]m sorry,? but|i am sorry,? but|i apologi[sz]e,? but|\bi (?:cannot|can['’]?t|can not|won['’]t|will not)\b|"
    r"i['’]m (?:unable|not able) to|i am (?:unable|not able) to|i must decline)",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(
    r"(sorry|apologi[sz]e|cannot|can['’]?t|unable|not able|decline|illegal|unethical|inappropriate|not appropriate|"
    r"not acceptable|against the law|harmful|important to (?:approach|clarify|note|address|consider|emphasize))",
    re.IGNORECASE,
)


def refusal_label(text: str) -> str:
    if REFUSAL_RE.search(text[:160]):
        return "refuse"
    return "ambiguous" if HEDGE_RE.search(text) else "comply"


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def center_by_role(x: np.ndarray, roles: np.ndarray, train: np.ndarray | None = None) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    for r in np.unique(roles):
        m = roles == r
        ref = m if train is None else (m & train)
        out[m] -= x[ref].astype(np.float64).mean(axis=0, keepdims=True)
    return out


def ridge_cv_within(x: np.ndarray, ys: dict[str, np.ndarray], groups: np.ndarray, *, n_folds: int = 5,
                    alphas: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed: int = 0) -> dict[str, dict]:
    """Grouped K-fold ridge of each target in `ys` on x. Everything is centred on the training fold.

    Returns, per target, held-out FVE (1 - SSE / SS around the training mean) and mean cosine
    between prediction and centred target, at the alpha with the best FVE. Alphas are relative
    to the feature covariance trace, so they transfer across feature sets. The eigendecomposition
    depends only on x and the fold, so it is shared across targets.
    """
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug.tolist(), (rng.permutation(len(ug)) % n_folds).tolist()))
    folds = np.asarray([fold_of[g] for g in groups.tolist()])
    names = list(ys)
    sse = {n: np.zeros(len(alphas)) for n in names}
    cos_sum = {n: np.zeros(len(alphas)) for n in names}
    sst = {n: 0.0 for n in names}
    n_va = 0
    xt = torch.from_numpy(np.ascontiguousarray(x)).double()
    yts = {n: torch.from_numpy(np.ascontiguousarray(y)).double() for n, y in ys.items()}
    for f in range(n_folds):
        va = torch.from_numpy(folds == f)
        tr = ~va
        xm = xt[tr].mean(0)
        xtr, xva = xt[tr] - xm, xt[va] - xm
        n_tr, p = xtr.shape
        trace = float(xtr.pow(2).sum() / n_tr)
        n_va += int(va.sum())
        primal = p <= n_tr
        if primal:
            evals, q = torch.linalg.eigh(xtr.T @ xtr / n_tr)
            xvq = xva @ q
        else:
            evals, q = torch.linalg.eigh(xtr @ xtr.T / n_tr)
            xvq = (xva @ xtr.T / n_tr) @ q
        evals = evals.clamp_min(0)
        for n in names:
            ym = yts[n][tr].mean(0)
            ytr, yva = yts[n][tr] - ym, yts[n][va] - ym
            proj = q.T @ (xtr.T @ ytr / n_tr) if primal else q.T @ ytr
            sst[n] += float(yva.pow(2).sum())
            for ai, a in enumerate(alphas):
                pred = xvq @ (proj / (evals + a * trace).unsqueeze(1))
                sse[n][ai] += float((yva - pred).pow(2).sum())
                cos_sum[n][ai] += float(torch.nn.functional.cosine_similarity(pred, yva, dim=1).sum())
    out: dict[str, dict] = {}
    for n in names:
        fve = 1.0 - sse[n] / max(sst[n], 1e-18)
        best = int(np.argmax(fve))
        out[n] = {"fve": float(fve[best]), "cos": float(cos_sum[n][best] / max(n_va, 1)),
                  "alpha_rel": alphas[best], "n": len(groups), "p": int(x.shape[1])}
    return out


def within_fve(target: np.ndarray, resid: np.ndarray) -> float:
    """1 - E||resid - mean||^2 / E||target - mean||^2: variance of `target` explained beyond a constant."""
    t = target.astype(np.float64)
    r = resid.astype(np.float64)
    return float(1.0 - ((r - r.mean(0)) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def spectrum_stats(y_centered: np.ndarray) -> dict:
    """Participation ratio and top-k variance shares of a centred [N, d] matrix."""
    yt = torch.from_numpy(y_centered).double()
    n, d = yt.shape
    cov = (yt.T @ yt) / n if d <= n else (yt @ yt.T) / n
    ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0).numpy()
    tot = ev.sum()
    return {
        "participation_ratio": float(tot**2 / (ev**2).sum()),
        "top1_share": float(ev[0] / tot),
        "top10_share": float(ev[:10].sum() / tot),
        "top100_share": float(ev[:100].sum() / tot),
        "n": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--reader-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--reader-layer", type=int, default=18)
    ap.add_argument("--max-pos-per-role", type=int, default=12)
    ap.add_argument("--norm-mult", type=float, default=5.0, help="massive-activation cutoff, x median norm")
    ap.add_argument("--max-convs", type=int, default=0, help="use only the first N conversations (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    win = str(cfg["base_wrapper"])
    tau = float(cfg["tau_web_median"])
    art_id = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    art_std = load_alignment(p0 / f"align_L{k}_std_ridge.pt")
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    tbl = pq.read_table(args.rollouts).to_pylist()
    if args.max_convs:
        tbl = tbl[: args.max_convs]
    rng = np.random.default_rng(args.seed)
    plans = []
    for ci, row in enumerate(tbl):
        hist = json.loads(row["history_json"])
        kw = dict(history=hist, user_ids=row.get("prompt_user_ids"), rollout_ids=row["rollout_token_ids"],
                  enable_thinking=False, max_total_tokens=None, truncate_history=False)
        ie, bc = encode_aligned_pair(tok, row["user_text"], base_wrapper="chatml", **kw)
        _, bp = encode_aligned_pair(tok, row["user_text"], base_wrapper="plain", **kw)
        p_ids = row.get("prompt_token_ids")
        if p_ids is not None:
            # Same check as Phase 2's extract step: teacher-force after the exact prompt the rollout saw.
            assert ie.input_ids[: len(p_ids)] == list(p_ids), f"prompt prefix diverged for {row['doc_id']}"
        pos: list[tuple[str, int, int, int]] = []  # role, inst_pos, plain_pos, token
        for role, ipos, ppos in (
            ("user_content", ie.user_content_positions, bp.user_content_positions),
            ("asst_last", [ie.asst_last_idx], [bp.asst_last_idx]),
            ("asst_rollout", ie.asst_rollout_positions, bp.asst_rollout_positions),
        ):
            assert len(ipos) == len(ppos)
            sel = np.arange(len(ipos))
            if len(sel) > args.max_pos_per_role:
                sel = np.sort(rng.choice(len(ipos), size=args.max_pos_per_role, replace=False))
            pos.extend((role, ipos[j], ppos[j], ie.input_ids[ipos[j]]) for j in sel.tolist())
        plans.append({"ci": ci, "inst_ids": ie.input_ids, "plain_ids": bp.input_ids, "pos": pos,
                      "source": row["source_corpus"], "text": row["assistant_text"]})
    print(f"[corpus] {len(plans)} conversations, {sum(len(p['pos']) for p in plans)} sampled positions", flush=True)

    def pull(model_name: str, layer: int, ids_key: str, pos_idx: int) -> np.ndarray:
        m = cached_io.lazy_load_causal_lm(model_name)
        exs = cached_io.cached_extract(m, [p[ids_key] for p in plans], [[q[pos_idx] for q in p["pos"]] for p in plans],
                                       layer_index=layer, pad_token_id=pad)
        m.free()
        return torch.cat([e.hidden for e in exs], dim=0).numpy()

    ht = pull(cfg["target_model"], k, "inst_ids", 1)
    hb = {"chatml": pull(cfg["base_model"], k, "inst_ids", 1), "plain": pull(cfg["base_model"], k, "plain_ids", 2)}
    reader = pull(args.reader_model, args.reader_layer, "inst_ids", 1)
    if cached_io.RECORD_PATH:
        print("[record] corpus_geometry requests recorded")
        return
    roles = np.asarray([q[0] for p in plans for q in p["pos"]])
    toks = np.asarray([q[3] for p in plans for q in p["pos"]])
    conv = np.asarray([p["ci"] for p in plans for _ in p["pos"]])
    print(f"[corpus] vectors: {ht.shape}, reader: {reader.shape}", flush=True)

    def delta(wrapper: str, art: dict) -> np.ndarray:
        return apply_alignment(torch.from_numpy(hb[wrapper]), torch.from_numpy(ht), art).numpy()

    other = "plain" if win == "chatml" else "chatml"
    d_win = delta(win, art_id)
    d_other = delta(other, art_id)
    d_std = delta("chatml", art_std)  # Arm 1 always diffs against Base(ChatML) with standard ridge

    # Massive-activation rows: ||h|| far above the median in either model under either wrapper.
    # They are a fraction of a percent of rows but dominate every non-robust statistic, so they
    # are counted per role and kept out of the composition / recoverability estimates below.
    n_t = np.linalg.norm(ht, axis=1)
    massive = n_t > args.norm_mult * np.median(n_t)
    for w_ in ("chatml", "plain"):
        n_b = np.linalg.norm(hb[w_], axis=1)
        massive |= n_b > args.norm_mult * np.median(n_b)
    ok = ~massive

    d_tilde = center_by_role(d_win, roles)  # Arm 2 vector as the pipeline builds it (every row in the role mean)
    targets = {
        "arm0_h_target": unit(ht),
        "arm1_naive_std": unit(d_std),
        "arm2_cleaned": unit(d_tilde),
    }
    report: dict = {"layer": k, "winning_wrapper": win, "tau_web_median": tau, "n_convs": len(plans), "roles": {}}

    for role in ACTIVE_ROLES:
        m = roles == role
        if m.sum() < 20:
            continue
        r: dict = {"n": int(m.sum())}
        all_rows = m
        m = m & ok
        mu_all = d_win[all_rows].astype(np.float64).mean(0)
        mu_ok = d_win[m].astype(np.float64).mean(0)
        r["massive_rows"] = {
            "n": int((all_rows & massive).sum()),
            "frac": float((all_rows & massive).sum() / all_rows.sum()),
            "top_tokens": [(tok.decode([int(t)]), int(c)) for t, c in
                           collections.Counter(toks[all_rows & massive].tolist()).most_common(5)],
            "role_mean_norm_over_tau": float(np.linalg.norm(mu_ok) / tau),
            "role_mean_shift_from_massive_over_tau": float(np.linalg.norm(mu_all - mu_ok) / tau),
        }
        # How much of h_target's within-role variance does the (aligned) Base model explain?
        r["base_explains_target_within_role_fve"] = {
            f"id_anchored_{win}": within_fve(ht[m], d_win[m]),
            f"id_anchored_{other}": within_fve(ht[m], d_other[m]),
            f"identity_{win}": within_fve(ht[m], ht[m] - hb[win][m]),
            f"identity_{other}": within_fve(ht[m], ht[m] - hb[other][m]),
            "std_ridge_chatml": within_fve(ht[m], d_std[m]),
        }

        # A. noise floor
        n_ht = np.linalg.norm(ht[m], axis=1)
        n_raw = np.linalg.norm(d_win[m], axis=1)
        n_til = np.linalg.norm(d_tilde[m], axis=1)
        w = soft_norm_weights(torch.from_numpy(d_tilde[m]).float(), tau).numpy()
        r["noise_floor"] = {
            "median_norm_h_target": float(np.median(n_ht)),
            "median_norm_delta_raw": float(np.median(n_raw)),
            "median_norm_delta_centered": float(np.median(n_til)),
            "delta_centered_over_tau": {q: float(np.quantile(n_til, float(q)) / tau) for q in ("0.1", "0.5", "0.9")},
            "frac_below_tau": float((n_til < tau).mean()),
            "soft_norm_w": {"min": float(w.min()), "p10": float(np.quantile(w, 0.1)), "median": float(np.median(w)),
                            "max": float(w.max())},
            "soft_norm_ess_frac": float(w.sum() ** 2 / (len(w) * (w**2).sum())),
            "delta_over_h_target_norm": float(np.median(n_raw / n_ht)),
        }

        # B. composition
        mu = d_win[m].astype(np.float64).mean(0)
        r["composition"] = {
            "mean_offset_energy_frac": float((mu**2).sum() / (d_win[m].astype(np.float64) ** 2).sum(1).mean()),
            "cos_to_role_mean": {name: float((unit(t[m]) @ unit(t[m].mean(0, keepdims=True)).T).mean())
                                 for name, t in targets.items()},
            "spectrum": {name: spectrum_stats(center_by_role(t[m], roles[m])) for name, t in targets.items()},
        }
        if role != "asst_last":
            r["composition"]["tok1_fve"] = {name: compute_tok1_ridge_fve(t[m], toks[m], seed=args.seed)
                                            for name, t in targets.items()}

        # C. recoverability of each unit target from text-side features
        feats = {"reader_lm": reader[m], "h_target": ht[m], "h_base_win": hb[win][m]}
        ys = {tname: t[m] for tname, t in targets.items()}
        by_feat = {fname: ridge_cv_within(f, ys, conv[m], seed=args.seed) for fname, f in feats.items()}
        rec = {tname: {fname: by_feat[fname][tname] for fname in feats} for tname in targets}
        r["recoverability"] = rec

        # D. wrapper agreement of the diff at identical tokens
        a = center_by_role(d_win[m], roles[m])
        b = center_by_role(d_other[m], roles[m])
        hb_a = center_by_role(hb["chatml"][m], roles[m])
        hb_b = center_by_role(hb["plain"][m], roles[m])
        r["wrapper_agreement"] = {
            "cos_delta_chatml_vs_plain": float(np.mean(np.sum(unit(a) * unit(b), axis=1))),
            "cos_h_base_chatml_vs_plain": float(np.mean(np.sum(unit(hb_a) * unit(hb_b), axis=1))),
            "median_norm_ratio_other_over_win": float(np.median(np.linalg.norm(b, axis=1) / np.linalg.norm(a, axis=1))),
        }
        report["roles"][role] = r
        print(f"[corpus] {role}: " + json.dumps({kk: r[kk] for kk in (
            "massive_rows", "base_explains_target_within_role_fve", "noise_floor", "composition", "wrapper_agreement")}),
            flush=True)
        print(f"[corpus] {role} recoverability: " + json.dumps(rec), flush=True)

    # E. behaviour: actual refusals among safety-source prompts, read at asst_last
    m_last = (roles == "asst_last") & ok
    conv_last = conv[m_last]
    labels = np.asarray([refusal_label(plans[c]["text"]) for c in conv_last])
    is_safety = np.asarray([plans[c]["source"] != "ultrachat" for c in conv_last])
    beh: dict = {"label_counts_safety": dict(collections.Counter(labels[is_safety].tolist())),
                 "label_counts_other": dict(collections.Counter(labels[~is_safety].tolist()))}
    safety = is_safety & (labels != "ambiguous")  # safety-source prompts with an unambiguous outcome
    refused = (labels == "refuse").astype(int)
    if min(refused[safety].sum(), safety.sum() - refused[safety].sum()) >= 10:
        y = refused[safety]
        g = conv_last[safety]
        d_last = d_win[m_last][safety]
        reps = {
            "h_target": ht[m_last][safety],
            "h_target_cu": unit(center_by_role(ht[m_last][safety], roles[m_last][safety])),
            "h_base_win": hb[win][m_last][safety],
            "delta_idanc": d_last,
            "delta_idanc_cu": unit(center_by_role(d_last, roles[m_last][safety])),
            "delta_other_wrapper_cu": unit(center_by_role(d_other[m_last][safety], roles[m_last][safety])),
        }
        beh["fisher_snr"] = {n_: compute_fisher_snr(torch.from_numpy(np.ascontiguousarray(v)).float(), y)
                             for n_, v in reps.items()}
        beh["grouped_auroc"] = {n_: grouped_auroc(torch.from_numpy(np.ascontiguousarray(v)).float(), y, g)
                                for n_, v in reps.items()}
        beh["few_shot_bal_acc_k8"] = {n_: few_shot_acc(torch.from_numpy(np.ascontiguousarray(v)).float(), y, g,
                                                       k_groups=8) for n_, v in reps.items()}
        beh["snr_ratio_delta_cu_over_h_target"] = beh["fisher_snr"]["delta_idanc_cu"] / beh["fisher_snr"]["h_target"]
    report["behaviour_refusal_asst_last"] = beh
    print("[corpus] behaviour: " + json.dumps(beh), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[corpus] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
