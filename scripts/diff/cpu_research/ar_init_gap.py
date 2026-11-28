"""How far is each arm's training target from where the reconstructor (AR) starts?

The AR is the first K+1 blocks of the Instruct model with an identity-initialised value head,
trained on MSE between unit-normalised prediction and unit-normalised target. For Arm 0 the
target is the Instruct Layer-K activation itself, so an identity read of the backbone is already
in the right space. For Arm 2 the target is the centred diff. This script asks, on rows sampled
from a Phase 2 rollout corpus:

  1. cos(unit(h_target), unit(target_arm))      what an identity head scores if the backbone
                                                reproduced h_target exactly
  2. held-out FVE of a ridge map h_target -> unit(target_arm)
                                                how much of the target a *linear* value head could
                                                reach from a perfect h_target
  3. the same from h_base (wrapper used by the arm)

FVE uses the trainer's definition: 1 - MSE / Var on unit-normalised targets, split by conversation.

    NLA_ACT_CACHE=/home/user/act_cache python ar_init_gap.py \
        --phase0-dir /home/user/research/p0_8b_L24 \
        --rollouts /home/user/research/p2_8b/corpus/rollouts.parquet \
        --out /home/user/research/ar_init_gap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpus_rows import load_corpus_rows, massive_rows  # noqa: E402

from nla.diff.alignment import apply_alignment, load_alignment  # noqa: E402

ROLES = ("user_content", "asst_last", "asst_rollout")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def ridge_fve(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor,
              alphas: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)) -> dict[str, float]:
    """Best held-out FVE over a ridge grid (dual form; features standardised by a global scale)."""
    x_tr, y_tr, x_te, y_te = x_tr.double(), y_tr.double(), x_te.double(), y_te.double()
    xm, ym = x_tr.mean(0), y_tr.mean(0)
    scale = (x_tr - xm).pow(2).sum(-1).mean().sqrt()
    a, b = (x_tr - xm) / scale, (x_te - xm) / scale
    k = a @ a.T
    evals, q = torch.linalg.eigh(k)
    qty = q.T @ (y_tr - ym)
    k_te = b @ a.T
    var = (y_te - y_te.mean(0)).pow(2).sum(-1).mean()
    best = (-1e9, 0.0)
    for al in alphas:
        dual = q @ (qty / (evals.clamp_min(0) + al).unsqueeze(1))
        pred = k_te @ dual + ym
        fve = float(1.0 - (y_te - pred).pow(2).sum(-1).mean() / var)
        if fve > best[0]:
            best = (fve, al)
    return {"fve": best[0], "alpha": best[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pos-per-role", type=int, default=12)
    ap.add_argument("--max-rows-per-role", type=int, default=6000)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    wrapper = str(cfg["base_wrapper"])
    art = load_alignment(cfg["align_id_anchored_path"])
    art_std = load_alignment(cfg["align_std_ridge_path"])
    rows = load_corpus_rows(cfg, tok, args.rollouts, max_pos_per_role=args.max_pos_per_role)
    keep = ~massive_rows(rows.ht, rows.hb["plain"], rows.hb["chatml"])
    rng = np.random.default_rng(0)

    out: dict[str, Any] = {"base_wrapper": wrapper, "n_rows_total": int(len(rows.roles)),
                           "n_massive_dropped": int((~keep).sum()), "roles": {}}
    for role in ROLES:
        idx = np.flatnonzero((rows.roles == role) & keep)
        if len(idx) > args.max_rows_per_role:
            idx = np.sort(rng.choice(idx, size=args.max_rows_per_role, replace=False))
        if len(idx) < 200:
            continue
        conv = rows.conv[idx]
        convs = np.unique(conv)
        rng.shuffle(convs)
        te_conv = set(convs[: max(1, len(convs) // 5)].tolist())
        te = np.array([c in te_conv for c in conv])
        tr = ~te
        ht = torch.from_numpy(rows.ht[idx]).float()
        hb_w = torch.from_numpy(rows.hb[wrapper][idx]).float()
        hb_c = torch.from_numpy(rows.hb["chatml"][idx]).float()

        d2 = apply_alignment(hb_w, ht, art)
        d2 = d2 - d2[tr].mean(0, keepdim=True)  # Fix 2 role centring, train rows only
        d1 = apply_alignment(hb_c, ht, art_std)
        targets = {"arm0_raw": ht, "arm1_naive_diff": d1, "arm2_clean_diff": d2}
        sec: dict[str, Any] = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        for name, tgt in targets.items():
            y = unit(tgt)
            base_feat = hb_c if name == "arm1_naive_diff" else hb_w
            sec[name] = {
                "cos_to_unit_h_target": float((unit(ht) * y).sum(-1).mean()),
                "cos_to_train_mean_direction": float((y[te] * unit(y[tr].mean(0, keepdim=True))).sum(-1).mean()),
                "target_unit_variance": float((y[te] - y[te].mean(0)).pow(2).sum(-1).mean()),
                "ridge_from_h_target": ridge_fve(ht[tr], y[tr], ht[te], y[te]),
                "ridge_from_h_base": ridge_fve(base_feat[tr], y[tr], base_feat[te], y[te]),
            }
        out["roles"][role] = sec
        print(role, json.dumps(sec, indent=1), flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
