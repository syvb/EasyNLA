"""Where does the Instruct-vs-Base behavioural gap live: below Layer K or above it?

The diff target is the two models' residual-stream difference at Layer K. It can only describe
the part of the behavioural gap that is already present in that residual stream. This script
measures that part by swapping whole residual streams at Layer K between the two models, on the
ChatML sequences both models were run on (identical tokens, so every position lines up):

  full gap        KL(Instruct || Base)
  top swapped     KL(Instruct || Base blocks above K, fed Instruct's Layer-K residuals)
                  = what the blocks above K still change when the residual stream is Instruct's
  bottom swapped  KL(Instruct || Instruct blocks above K, fed Base's Layer-K residuals)
                  = what the Layer-K residual difference changes on its own

and reports them per active role, with the share of positions where each swap closes most of the gap.

    python cross_patch.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
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
from corpus_geometry import center_by_role
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.wrappers import ACTIVE_ROLES


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--max-convs", type=int, default=0, help="use only the first N conversations (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    rows = load_corpus_rows(cfg, tok, args.rollouts, seed=args.seed, max_convs=args.max_convs)
    n = len(rows.roles)
    starts = np.concatenate([[0], np.cumsum([len(p) for p in rows.pos["chatml"]])])
    items = [(cached_io.seq_key(s), p) for s, p in zip(rows.seqs["chatml"], rows.pos["chatml"])]
    tgt = cached_io.LazyLM(cfg["target_model"])
    base = cached_io.LazyLM(cfg["base_model"])

    def final(model: cached_io.LazyLM, residual_from: cached_io.LazyLM | None) -> torch.Tensor:
        d = int(model.ensure("top", k).lm_head.weight.shape[1])
        out = torch.zeros((n, d), dtype=torch.float32)
        for i, fin in cached_io.iter_final_hidden(model, items, layer_index=k, residual_from=residual_from):
            out[starts[i] : starts[i + 1]] = fin
        return out

    def kl_rows(lp_ref: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
        lp = F.log_softmax(logits.float(), dim=-1)
        return (lp_ref.exp() * (lp_ref - lp)).sum(-1).double().numpy()

    # Instruct's blocks above K: on its own residuals (the reference) and on Base's residuals.
    fin_ii = final(tgt, None)
    fin_ib = final(tgt, base)
    head_i = tgt.ensure("top", k).lm_head
    w_i = head_i.weight.detach().clone()
    kl_bottom_swapped = np.zeros(n)
    for s in range(0, n, 256):
        lp_ref = F.log_softmax(fin_ii[s : s + 256] @ w_i.T, dim=-1)
        kl_bottom_swapped[s : s + 256] = kl_rows(lp_ref, fin_ib[s : s + 256] @ w_i.T)
    del fin_ib, head_i
    tgt.free()

    # Base's blocks above K: on Instruct's residuals, and on its own (the full gap).
    kl_top_swapped = np.zeros(n)
    kl_full = np.zeros(n)
    head_b = base.ensure("top", k).lm_head
    for name, src, dst in (("top", tgt, kl_top_swapped), ("full", None, kl_full)):
        for i, fin in cached_io.iter_final_hidden(base, items, layer_index=k, residual_from=src):
            sl = slice(int(starts[i]), int(starts[i + 1]))
            dst[sl] = kl_rows(F.log_softmax(fin_ii[sl] @ w_i.T, dim=-1), head_b(fin))
        print(f"[cross] {name} pass done", flush=True)
    del head_b, w_i
    base.free()

    art = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    d_chatml = center_by_role(apply_alignment(torch.from_numpy(rows.hb["chatml"]), torch.from_numpy(rows.ht), art).numpy(),
                              rows.roles)
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    report: dict = {"layer": k, "wrapper": "chatml", "roles": {}}
    for role in ACTIVE_ROLES:
        m = (rows.roles == role) & ok
        if m.sum() < 30:
            continue
        full, top, bot = kl_full[m], kl_top_swapped[m], kl_bottom_swapped[m]
        big = full >= np.quantile(full, 0.5)  # positions where the two models actually differ
        r = {
            "n": int(m.sum()),
            "kl_nats": {name: {"mean": float(v.mean()), "median": float(np.median(v))}
                        for name, v in (("full_gap", full), ("top_swapped", top), ("bottom_swapped", bot))},
            # Of the mean gap, how much remains when Base's upper blocks read Instruct's residual stream.
            "gap_left_with_instruct_residuals": float(top.mean() / full.mean()),
            # How much of the mean gap the residual difference alone produces under Instruct's upper blocks.
            "gap_made_by_residual_difference": float(bot.mean() / full.mean()),
            "frac_positions_residual_swap_closes_half_the_gap": float((top[big] < 0.5 * full[big]).mean()),
            "spearman_cleaned_diff_norm_vs": {
                "full_gap": float(spearmanr(np.linalg.norm(d_chatml[m], axis=1), full).statistic),
                "bottom_swapped": float(spearmanr(np.linalg.norm(d_chatml[m], axis=1), bot).statistic),
            },
        }
        report["roles"][role] = r
        print(f"[cross] {role}: " + json.dumps(r), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[cross] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
