"""Sanity check #1+#2: reproduce the checkpoint's held-out reconstruction
quality through OUR codec plumbing, and settle the norm question empirically.

Verbalizes+reconstructs N held-out activations from the warmstart val parquet,
then reports: normalized-MSE / FVE vs both predict-the-mean baselines (compare
against the checkpoint's claimed ~78-79% held-out FVE — large disagreement
means our loading/injection is broken), cosine distribution, and the ‖AR(z)‖ vs
‖h‖ scatter (expected: uncorrelated/collapsed — the codec is direction-only,
which is why roundtrip() norm-matches).

    python -m experiments.bottleneck.norm_scatter --config ... --parquet av_sft_val.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from experiments.bottleneck.codec import NLACodec
from nla.schema import ACTIVATION_COLUMN, load_predict_mean_baselines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--parquet", required=True, help="held-out val parquet with activations")
    p.add_argument("--n", type=int, default=512)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=[ACTIVATION_COLUMN]))
    col = batch.column(ACTIVATION_COLUMN)
    acts = torch.from_numpy(
        col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    ).reshape(len(col), -1)
    print(f"[norm_scatter] {acts.shape[0]} held-out activations, d={acts.shape[1]}")
    print(f"[norm_scatter] gold norms: mean={acts.norm(dim=-1).mean():.1f} "
          f"median={acts.norm(dim=-1).median():.1f}")

    codec = NLACodec(
        av_merged_dir=cfg["av_merged_dir"], ar_dir=cfg["ar_dir"],
        vllm_gpu_mem=cfg.get("vllm_gpu_mem", 0.35),
        vllm_max_len=cfg.get("vllm_max_len", 1024),
        av_max_tokens=cfg.get("av_max_tokens", 150),
        av_temperature=cfg.get("av_temperature", 1.0),
    )
    verbs = codec.verbalize(acts)
    texts = [v.explanation if v.explanation is not None else v.text for v in verbs]
    preds = codec.reconstruct(texts)

    mse = codec.normalized_mse(preds, acts)
    base_meannorm, base_rawvar = load_predict_mean_baselines(args.parquet, codec.mse_scale_f)
    fve_meannorm = 1 - float(mse.mean()) / base_meannorm
    fve_rawvar = 1 - float(mse.mean()) / base_rawvar

    h_n = acts.norm(dim=-1).numpy()
    p_n = preds.norm(dim=-1).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(preds.cpu(), acts, dim=-1).numpy()

    print("\n================= codec reproduction report =================")
    print(f"normalized MSE     : {float(mse.mean()):.4f}")
    print(f"FVE (meannorm base): {fve_meannorm:.4f}   (ckpt claims ~0.78-0.79)")
    print(f"FVE (rawvar base)  : {fve_rawvar:.4f}")
    print(f"cosine             : mean {cos.mean():.4f}  p10 {np.percentile(cos,10):.4f} "
          f"p50 {np.percentile(cos,50):.4f}  p90 {np.percentile(cos,90):.4f}")
    print(f"‖h‖ vs ‖AR(z)‖     : gold mean {h_n.mean():.1f} / pred mean {p_n.mean():.1f}, "
          f"pearson r = {np.corrcoef(h_n, p_n)[0,1]:.3f}")
    print(f"  -> if pred norms are collapsed/uncorrelated (expected for the "
          f"direction-only codec), roundtrip()'s rescale-to-‖h‖ is REQUIRED.")
    print(f"AV stats           : {codec.report_stats()}")
    print(f"sample z           : {texts[0][:400]!r}")

    out = Path(cfg["out_dir"]) / "norm_scatter.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, h_norm=h_n, pred_norm=p_n, cosine=cos, mse=mse.cpu().numpy())
    print(f"[out] {out}")
    print("NORM_SCATTER_DONE", flush=True)


if __name__ == "__main__":
    main()
