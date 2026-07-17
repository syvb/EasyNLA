"""Compute ZCA whitening stats (μ, W = Σ^{-1/2}, W⁻¹) from a parquet's
activation_vector column and save them as .npz.

Run this on the TRAIN split only — val/heldout parquets are then whitened
with the SAME stats (scripts/whiten_dataset.py --stats), otherwise the val
transform leaks val statistics and the held-out FVE is subtly wrong.

Usage:
    python scripts/compute_whitening_stats.py \
        --parquet <data>/av_sft_train.parquet \
        --out <data>/whitening_stats.npz \
        [--shrinkage 0.01] [--max-rows N] [--batch-size 8192]

Single streaming pass; peak memory is the d² float64 accumulator
(≈134 MB at d=4096) plus one batch. CPU-only, a few minutes for ~400k rows.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.datagen.sidecar import read_sidecar_local
from nla.schema import NORM_RAW
from nla.whitening import (
    DEFAULT_SHRINKAGE,
    compute_whitening_stats,
    describe,
    save_stats,
    whiten,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="TRAIN-split parquet with activation_vector column")
    p.add_argument("--out", required=True, help="output .npz path for the stats")
    p.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE,
                   help="covariance shrinkage λ toward (trΣ/d)·I — caps amplification of near-null directions")
    p.add_argument("--max-rows", type=int, default=None, help="cap rows read (default: all)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = p.parse_args()

    out = Path(args.out)
    assert args.force or not out.exists(), f"{out} exists — pass --force to overwrite"
    assert 0.0 <= args.shrinkage < 1.0, f"shrinkage must be in [0, 1), got {args.shrinkage}"

    # The sidecar is the contract — refuse to compute stats on already-
    # normalized data (double-whitening) or on a parquet with no provenance.
    meta = read_sidecar_local(Path(args.parquet))
    assert meta.extraction.norm == NORM_RAW, (
        f"expected raw vectors (norm={NORM_RAW!r}), got norm={meta.extraction.norm!r} — "
        f"stats must come from the raw train split, not an already-whitened parquet"
    )
    print(f"input: {args.parquet}")
    print(f"  base_model={meta.extraction.base_model} layer={meta.extraction.layer_index} "
          f"d_model={meta.extraction.d_model} sidecar_row_count={meta.row_count}")

    t0 = time.monotonic()
    stats = compute_whitening_stats(
        args.parquet,
        shrinkage=args.shrinkage,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        source=args.parquet,
    )
    assert stats.d_model == meta.extraction.d_model, (
        f"parquet vectors are {stats.d_model}-wide but sidecar says d_model={meta.extraction.d_model}"
    )
    print(f"\ncomputed in {time.monotonic() - t0:.1f}s over {stats.n_samples} rows:")
    print(describe(stats))

    # Self-check on a sample of the SAME data: whitened covariance diag ≈ 1,
    # off-diag ≈ 0. Loose tolerance — it's a sanity check, not a unit test.
    from nla.whitening import iter_activation_batches
    sample = next(iter_activation_batches(args.parquet, batch_size=min(20_000, stats.n_samples)))
    xw = whiten(sample, stats)
    cov = np.cov(xw, rowvar=False)
    diag_err = float(np.abs(np.diag(cov) - 1).mean())
    off = cov - np.diag(np.diag(cov))
    print(f"\nself-check on {len(sample)} train rows:")
    print(f"  whitened cov: mean |diag−1| = {diag_err:.4f} · mean |offdiag| = {float(np.abs(off).mean()):.4f}")
    print(f"  whitened ‖x̃‖: mean {float(np.linalg.norm(xw, axis=1).mean()):.2f} "
          f"(√d = {stats.d_model ** 0.5:.2f})")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_stats(stats, str(out))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB) · sha256 {stats.sha256()[:16]}…")


if __name__ == "__main__":
    main()
