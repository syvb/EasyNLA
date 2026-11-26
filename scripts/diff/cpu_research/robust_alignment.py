"""Refit the Phase 0 alignment suite with massive-activation tokens masked out.

At Qwen3-8B layer 24, 0.14% of web tokens at positions > 0 have ||h|| around 15,000
(median 264) and carry about 80% of the trace variance. Base and Instruct fire on
different tokens, so these few rows dominate the ridge objective and every web FVE.
This script masks tokens whose norm exceeds `--norm-mult` x median in either model,
refits the repo's own `evaluate_alignment_suite`, and writes a phase0-style directory
(alignment artifacts + config with the new tau) that the other scripts accept.

    python robust_alignment.py --phase0-dir P0 --out-dir P0_robust
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff import alignment
from nla.diff.alignment import (
    apply_alignment,
    compute_fve,
    evaluate_alignment_suite,
    load_alignment,
    predict_aligned_base,
    save_alignment,
)
from nla.diff.benchmarks import load_fineweb_passages

alignment._centered_covariances = cached_io.lean_centered_covariances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-web-train", type=int, default=1600)
    ap.add_argument("--n-web-val", type=int, default=400)
    ap.add_argument("--web-max-tokens", type=int, default=64)
    ap.add_argument("--norm-mult", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    docs = load_fineweb_passages(n_passages=args.n_web_train + args.n_web_val)

    def split(ds: list[dict]) -> tuple[list[list[int]], list[list[int]], list[str]]:
        seqs, pos, dids = [], [], []
        for d in ds:
            ids = tok.encode(d["text"], add_special_tokens=False)[: args.web_max_tokens]
            if len(ids) < 2:
                continue
            seqs.append(ids)
            pos.append(list(range(1, len(ids))))
            dids.extend([d["doc_id"]] * (len(ids) - 1))
        return seqs, pos, dids

    tr_s, tr_p, tr_d = split(docs[: args.n_web_train])
    va_s, va_p, _ = split(docs[args.n_web_train : args.n_web_train + args.n_web_val])

    def grab(name: str, seqs: list, pos: list) -> torch.Tensor:
        m = cached_io.LazyLM(name)
        out = torch.cat([e.hidden for e in cached_io.cached_extract(m, seqs, pos, layer_index=k, pad_token_id=pad)])
        m.free()
        return out

    ht_tr, ht_va = grab(cfg["target_model"], tr_s, tr_p), grab(cfg["target_model"], va_s, va_p)
    hb_tr, hb_va = grab(cfg["base_model"], tr_s, tr_p), grab(cfg["base_model"], va_s, va_p)

    def keep(ht: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
        nt, nb = ht.norm(dim=1), hb.norm(dim=1)
        return (nt <= args.norm_mult * nt.median()) & (nb <= args.norm_mult * nb.median())

    m_tr, m_va = keep(ht_tr, hb_tr), keep(ht_va, hb_va)
    print(f"masked massive tokens: train {int((~m_tr).sum())}/{len(m_tr)}, val {int((~m_va).sum())}/{len(m_va)}", flush=True)
    tr_d_kept = [d for d, kk in zip(tr_d, m_tr.tolist()) if kk]
    hb_tr_kept, ht_tr_kept = hb_tr[m_tr], ht_tr[m_tr]
    del hb_tr, ht_tr  # the float64 CV below needs the headroom

    res = evaluate_alignment_suite(
        hb_tr_kept, ht_tr_kept, tr_d_kept, hb_va[m_va], ht_va[m_va],
        n_folds=5, seed=args.seed, layer_index=k, device="cpu",
    )
    del hb_tr_kept, ht_tr_kept
    art_id, art_std = res["artifact_id_anchored"], res["artifact_std_ridge"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_path = save_alignment(out / f"align_L{k}_id_anchored.pt", art_id)
    std_path = save_alignment(out / f"align_L{k}_std_ridge.pt", art_std)

    stock_id = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    stock_std = load_alignment(p0 / f"align_L{k}_std_ridge.pt")

    def fves(hb: torch.Tensor, ht: torch.Tensor) -> dict[str, float]:
        return {
            "identity": compute_fve(ht, hb),
            "stock_std_ridge": compute_fve(ht, predict_aligned_base(hb, stock_std)),
            "stock_id_anchored": compute_fve(ht, predict_aligned_base(hb, stock_id)),
            "robust_std_ridge": compute_fve(ht, predict_aligned_base(hb, art_std)),
            "robust_id_anchored": compute_fve(ht, predict_aligned_base(hb, art_id)),
        }

    summary = {
        "norm_mult": args.norm_mult,
        "n_masked_train": int((~m_tr).sum()), "n_train": len(m_tr),
        "n_masked_val": int((~m_va).sum()), "n_val": len(m_va),
        "robust_suite_on_masked_val": {kk: v for kk, v in res.items() if not kk.startswith("artifact_")},
        "val_fve_all_tokens": fves(hb_va, ht_va),
        "val_fve_masked_tokens": fves(hb_va[m_va], ht_va[m_va]),
        "tau_median_masked_val": {
            "stock_id_anchored": float(apply_alignment(hb_va[m_va], ht_va[m_va], stock_id).norm(dim=1).median()),
            "robust_id_anchored": float(apply_alignment(hb_va[m_va], ht_va[m_va], art_id).norm(dim=1).median()),
            "identity": float((ht_va[m_va] - hb_va[m_va]).norm(dim=1).median()),
        },
        "median_norm_h_target": float(ht_va[m_va].norm(dim=1).median()),
    }
    new_cfg = dict(cfg)
    new_cfg.update({
        "tau_web_median": float(res["tau_web_median"]), "tau_web_p75": float(res["tau_web_p75"]),
        "chosen_alpha_id_anchored": float(res["best_alpha_id_anchored"]),
        "chosen_alpha_std_ridge": float(res["best_alpha_std_ridge"]),
        "align_id_anchored_path": str(id_path), "align_std_ridge_path": str(std_path),
        "role_means_path": "", "robust_alignment": True,
    })
    (out / "phase0_config.json").write_text(json.dumps(new_cfg, indent=2))
    (out / "robust_alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
