"""Does the cleaned diff track the behavioural gap between Instruct and Base?

Arm 2's SFT labels are written from the two models' next-token behaviour at a position, and
Fix 3 treats the norm of the cleaned diff as the amount of post-training signal there. This
script computes both models' exact next-token distributions at every sampled corpus position
(top half of each model on the cached Layer-K residual stream) and reports, per active role:

  1. KL(Instruct || Base) in nats under each Base wrapper, and top-1 agreement
  2. Spearman correlation of that KL with ||cleaned diff||, with the Fix 3 weight and, as a
     reference, with ||h_target - role mean||
  3. held-out R^2 of log KL from each arm's training target (grouped by conversation)
  4. held-out FVE of each arm's target from the two models' final hidden states, which are a
     linear sufficient statistic of the two next-token distributions the explainer is shown

    python behaviour_link.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from corpus_geometry import center_by_role, ridge_cv_within, unit
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.wrappers import ACTIVE_ROLES
from nla.utils.soft_norm import soft_norm_weights


def next_token_state(cfg: dict, k: int, items: dict[str, list], starts: np.ndarray, n: int) -> tuple:
    """Final hidden states of Target and Base at every corpus row, and KL(Target || Base) per Base wrapper."""
    # Target: final hidden state at every row, plus its unembedding to rebuild logits later.
    tgt = cached_io.LazyLM(cfg["target_model"])
    fp16_check = cached_io.check_fp16_feed(tgt, layer_index=k)
    print(f"[behaviour] fp16-residual logits vs fp32-residual logits (Target): {fp16_check}", flush=True)
    d = int(tgt.ensure("top", k).lm_head.weight.shape[1])
    fin_t = torch.zeros((n, d), dtype=torch.float32)
    for i, fin in cached_io.iter_final_hidden(tgt, items["chatml"], layer_index=k):
        fin_t[starts[i] : starts[i + 1]] = fin
    w_t = tgt.ensure("top", k).lm_head.weight.detach().clone()
    tgt.free()

    # Base: same, and the KL against the Target at the aligned position, under each wrapper.
    base = cached_io.LazyLM(cfg["base_model"])
    head_b = base.ensure("top", k).lm_head
    fin_b: dict[str, torch.Tensor] = {}
    kl: dict[str, np.ndarray] = {}
    agree: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for w in ("chatml", "plain"):
            fin_b[w] = torch.zeros((n, d), dtype=torch.float32)
            kl[w] = np.zeros(n)
            agree[w] = np.zeros(n, dtype=bool)
            for i, fin in cached_io.iter_final_hidden(base, items[w], layer_index=k):
                sl = slice(int(starts[i]), int(starts[i + 1]))
                fin_b[w][sl] = fin
                lp_t = F.log_softmax(fin_t[sl] @ w_t.T, dim=-1)
                lp_b = F.log_softmax(head_b(fin).float(), dim=-1)
                kl[w][sl] = (lp_t.exp() * (lp_t - lp_b)).sum(-1).double().numpy()
                agree[w][sl] = (lp_t.argmax(-1) == lp_b.argmax(-1)).numpy()
    del w_t, head_b
    base.free()
    return fin_t.numpy(), {w: v.numpy() for w, v in fin_b.items()}, kl, agree, fp16_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--max-convs", type=int, default=0, help="use only the first N conversations (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state-cache", default="", help="npz to keep the next-token state in, so a second "
                    "Phase 0 dir (same rollouts, seed and sampling) skips the model passes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    win = str(cfg["base_wrapper"])
    other = "plain" if win == "chatml" else "chatml"
    tau = float(cfg["tau_web_median"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    rows = load_corpus_rows(cfg, tok, args.rollouts, seed=args.seed, max_convs=args.max_convs)
    n = len(rows.roles)
    starts = np.concatenate([[0], np.cumsum([len(p) for p in rows.pos["chatml"]])])
    items = {w: [(cached_io.seq_key(s), p) for s, p in zip(rows.seqs[w], rows.pos[w])] for w in ("chatml", "plain")}

    state = Path(args.state_cache) if args.state_cache else None
    if state is not None and state.exists():
        z = np.load(state)
        assert int(z["n"]) == n, f"{state} was built for {int(z['n'])} rows, this run has {n}"
        fin_t = z["fin_t"].astype(np.float32)
        fin_b = {w: z[f"fin_b_{w}"].astype(np.float32) for w in ("chatml", "plain")}
        kl = {w: z[f"kl_{w}"].astype(np.float64) for w in ("chatml", "plain")}
        agree = {w: z[f"agree_{w}"] for w in ("chatml", "plain")}
        fp16_check = json.loads(str(z["fp16_check"]))
    else:
        fin_t, fin_b, kl, agree, fp16_check = next_token_state(cfg, k, items, starts, n)
        if state is not None:
            state.parent.mkdir(parents=True, exist_ok=True)
            np.savez(state, n=n, fin_t=fin_t.astype(np.float16), fp16_check=json.dumps(fp16_check),
                     **{f"fin_b_{w}": v.astype(np.float16) for w, v in fin_b.items()},
                     **{f"kl_{w}": v for w, v in kl.items()}, **{f"agree_{w}": v for w, v in agree.items()})

    art = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    d_win = apply_alignment(torch.from_numpy(rows.hb[win]), torch.from_numpy(rows.ht), art).numpy()
    d_tilde = center_by_role(d_win, rows.roles)
    ht_c = center_by_role(rows.ht, rows.roles)
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    targets = {"arm0_h_target": unit(rows.ht), "arm0_h_target_centered": unit(ht_c), "arm2_cleaned": unit(d_tilde)}

    report: dict = {"layer": k, "winning_wrapper": win, "tau_web_median": tau, "fp16_feed_check": fp16_check, "roles": {}}
    for role in ACTIVE_ROLES:
        m = (rows.roles == role) & ok
        if m.sum() < 50:
            continue
        r: dict = {"n": int(m.sum())}
        r["kl_inst_base_nats"] = {w: {"median": float(np.median(kl[w][m])), "mean": float(kl[w][m].mean()),
                                      "p90": float(np.quantile(kl[w][m], 0.9)), "top1_agree": float(agree[w][m].mean())}
                                  for w in ("chatml", "plain")}
        r["spearman_kl_chatml_vs_plain"] = float(spearmanr(kl["chatml"][m], kl["plain"][m]).statistic)
        n_til = np.linalg.norm(d_tilde[m], axis=1)
        w_soft = soft_norm_weights(torch.from_numpy(d_tilde[m]).float(), tau).numpy()
        n_ht = np.linalg.norm(ht_c[m], axis=1)
        r["spearman_with_kl"] = {
            w: {"cleaned_diff_norm": float(spearmanr(n_til, kl[w][m]).statistic),
                "fix3_weight": float(spearmanr(w_soft, kl[w][m]).statistic),
                "h_target_centered_norm": float(spearmanr(n_ht, kl[w][m]).statistic)}
            for w in (win, other)
        }
        # Share of the Fix 3 weight mass that lands on the positions where the models differ most.
        top = kl[win][m] >= np.quantile(kl[win][m], 0.75)
        r["fix3_weight_share_on_top_quartile_kl"] = float(w_soft[top].sum() / w_soft.sum())

        y = {"log_kl": np.log(kl[win][m] + 1e-3)[:, None]}
        g = rows.conv[m]
        r["log_kl_r2_from"] = {name: ridge_cv_within(t[m], y, g, seed=args.seed)["log_kl"]["fve"]
                               for name, t in targets.items()}
        r["log_kl_r2_from"]["h_base_win"] = ridge_cv_within(rows.hb[win][m], y, g, seed=args.seed)["log_kl"]["fve"]

        feats = {
            "final_inst": fin_t[m],
            "final_base_win": fin_b[win][m],
            "final_inst+final_base_win": np.concatenate([fin_t[m], fin_b[win][m]], axis=1),
        }
        ys = {name: t[m] for name, t in targets.items() if name != "arm0_h_target_centered"}
        by_feat = {fname: ridge_cv_within(f, ys, g, seed=args.seed) for fname, f in feats.items()}
        r["target_fve_from_next_token_state"] = {t: {f: by_feat[f][t]["fve"] for f in feats} for t in ys}
        report["roles"][role] = r
        print(f"[behaviour] {role}: " + json.dumps(r), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[behaviour] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
