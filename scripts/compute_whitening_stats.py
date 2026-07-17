"""Compute ZCA whitening stats (μ, W = Σ^{-1/2}, W⁻¹) from a parquet's
activation_vector column and save them as .npz.

Run this on the TRAIN split only — val/heldout parquets are then whitened
with the SAME stats (scripts/whiten_dataset.py --stats), otherwise the val
transform leaks val statistics and the held-out FVE is subtly wrong.

Usage:
    python scripts/compute_whitening_stats.py \
        --parquet <data>/av_sft_train.parquet \
        --out <data>/whitening_stats.npz \
        [--floor-quantile 0.05] [--shrinkage 0.0] [--max-rows N]

Regularization default is eigenvalue flooring at the 5th percentile: the
95% of directions above the floor are whitened EXACTLY; only the noisy tail
is capped. Check the printed "whitened var" line — that's the per-direction
variance the whitened data actually gets (1.0 = perfect); if a large
fraction of directions sits below 0.9, the treatment is being attenuated
where it matters and the regularization needs rethinking.

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
    DEFAULT_FLOOR_QUANTILE,
    DEFAULT_SHRINKAGE,
    compute_whitening_stats,
    describe,
    iter_activation_batches,
    save_stats,
    whiten,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="TRAIN-split parquet with activation_vector column")
    p.add_argument("--out", required=True, help="output .npz path for the stats")
    p.add_argument("--floor-quantile", type=float, default=DEFAULT_FLOOR_QUANTILE,
                   help="floor eigenvalues at this quantile of the spectrum (0 disables)")
    p.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE,
                   help="covariance shrinkage λ toward (trΣ/d)·I, applied before flooring (0 disables)")
    p.add_argument("--max-rows", type=int, default=None, help="cap rows read (default: all)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = p.parse_args()

    # np.savez appends ".npz" when missing — pin the real path up front so the
    # post-save stat() and the path we print/record are the file that exists.
    out = Path(args.out)
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")
    assert args.force or not out.exists(), f"{out} exists — pass --force to overwrite"

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
        floor_quantile=args.floor_quantile,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        source=args.parquet,
        base_model=meta.extraction.base_model,
        layer_index=meta.extraction.layer_index,
    )
    assert stats.d_model == meta.extraction.d_model, (
        f"parquet vectors are {stats.d_model}-wide but sidecar says d_model={meta.extraction.d_model}"
    )
    print(f"\ncomputed in {time.monotonic() - t0:.1f}s over {stats.n_samples} rows:")
    print(describe(stats))

    # Self-check on a sample of the SAME data: whitened per-element variance
    # should match the expected (regularization-aware) value, mean ≈ 0.
    sample = next(iter_activation_batches(args.parquet, batch_size=min(20_000, stats.n_samples)))
    xw = whiten(sample, stats)
    var = float(xw.var(axis=0, ddof=1).mean())
    print(f"\nself-check on {len(sample)} train rows:")
    print(f"  whitened per-element var {var:.4f} (expected {stats.expected_whitened_variance():.4f})"
          f" · ‖mean‖/√d = {float(np.linalg.norm(xw.mean(axis=0))) / stats.d_model ** 0.5:.4f}")
    print(f"  whitened ‖x̃‖: mean {float(np.linalg.norm(xw, axis=1).mean()):.2f} "
          f"(√d = {stats.d_model ** 0.5:.2f})")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_stats(stats, str(out))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB) · sha256 {stats.sha256()[:16]}…")


if __name__ == "__main__":
    main()
