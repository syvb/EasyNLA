"""Rewrite an NLA parquet with ZCA-whitened activations (offline transform).

Replaces the activation_vector column with W(x − μ) using stats from
scripts/compute_whitening_stats.py; every other column (prompts, gold
explanations, provenance) is copied through byte-identical, and the sidecar
is restamped with norm=whitened_zca_v1 + a whitening provenance block.
Trainers on THIS branch then run on the whitened parquet with no code
changes. (Pre-whitening checkouts do NOT notice the tag — never point an
old trainer at a whitened dataset.)

Usage (once per split, ALWAYS with the train-split stats):
    python scripts/whiten_dataset.py \
        --parquet <data>/av_sft_train.parquet \
        --stats   <data>/whitening_stats.npz \
        --out     <data_w>/av_sft_train.parquet

Wrong-stats protection, in order of strength:
  1. provenance: stats pinned to base_model/layer/d_model of their source
     sidecar must match the target's sidecar;
  2. distribution gate: the first batch is whitened and its per-element
     variance and mean are checked against the stats' expected values —
     stats from a different distribution (wrong run, wrong layer, stale
     extraction) fail here, before any GPU-hours are spent.

Streams with bounded memory (~TRANSFORM_ROWS×d×16B) regardless of the
input's row-group layout — the real warmstart files are a SINGLE 364k-row
row group, so the output is re-chunked into ≤TRANSFORM_ROWS-row groups
(row-group layout is a streaming granularity, not part of the contract).
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.datagen.sidecar import read_sidecar_local, write_sidecar_local
from nla.schema import ACTIVATION_COLUMN, NORM_RAW, NORM_WHITENED_ZCA
from nla.whitening import load_stats, whiten

# Rows per streamed batch = rows per output row group. Peak memory ≈ this
# many rows of raw float32 + whitened float32 + one float64 temp (~2.5 GB
# at d=4096) — independent of input row-group size.
TRANSFORM_ROWS = 65_536

# Distribution-gate tolerances (first batch, same distribution as the stats'
# train split — val-split sampling noise at ≥1k rows is ≪ these):
VAR_RATIO_BOUNDS = (0.5, 2.0)  # measured / expected per-element variance
MEAN_NORM_MAX = 0.25  # ‖mean(x̃)‖ / √d — exact stats give ~1/√n


def _rebuild_activation_column(flat_f32: np.ndarray, n: int, av_type: pa.DataType) -> pa.Array:
    """Flat float32 values → an arrow array of EXACTLY the input's activation
    type (including the list child field's name — real warmstart files use
    child name 'element', from_arrays produces 'item'; the cast reconciles)."""
    values = pa.array(flat_f32, type=pa.float32())
    if pa.types.is_fixed_size_list(av_type):
        arr = pa.FixedSizeListArray.from_arrays(values, av_type.list_size)
    elif pa.types.is_list(av_type):
        d = len(flat_f32) // n
        offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32), type=pa.int32())
        arr = pa.ListArray.from_arrays(offsets, values)
    else:
        raise AssertionError(f"unsupported activation column type: {av_type}")
    return arr.cast(av_type)


def _distribution_gate(xw: np.ndarray, stats) -> None:
    """Whitened first batch must look like the stats' training distribution.
    unwhiten(whiten(x)) == x for ANY valid (μ, W) pair, so a round-trip can
    never catch wrong stats — this check does."""
    expected = stats.expected_whitened_variance()
    var = float(xw.var(axis=0, ddof=1).mean())
    mean_norm = float(np.linalg.norm(xw.mean(axis=0))) / stats.d_model**0.5
    ratio = var / expected
    assert VAR_RATIO_BOUNDS[0] < ratio < VAR_RATIO_BOUNDS[1] and mean_norm < MEAN_NORM_MAX, (
        f"whitened data does not match the stats' distribution: per-element "
        f"variance {var:.3f} vs expected {expected:.3f} (ratio {ratio:.2f}, "
        f"allowed {VAR_RATIO_BOUNDS}), ‖mean‖/√d {mean_norm:.3f} (max {MEAN_NORM_MAX}). "
        f"Wrong stats file? (different run / layer / model — provenance: "
        f"{stats.base_model or 'unset'} L{stats.layer_index}, {stats.source})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="input parquet (raw activations, with sidecar)")
    p.add_argument("--stats", required=True, help=".npz from compute_whitening_stats.py (train split!)")
    p.add_argument("--out", required=True, help="output parquet path (sidecar written alongside)")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = p.parse_args()

    assert "@[" not in args.parquet and "@[" not in args.out, (
        "@[slice] path syntax is a trainer-side loader feature — this script "
        "takes plain file paths; whiten the whole file"
    )
    in_path, out_path = Path(args.parquet), Path(args.out)
    assert in_path.resolve() != out_path.resolve(), "refusing in-place rewrite — pick a different --out"
    assert args.force or not out_path.exists(), f"{out_path} exists — pass --force to overwrite"

    stats = load_stats(args.stats)
    meta = read_sidecar_local(in_path)
    assert meta.extraction.norm == NORM_RAW, (
        f"input norm={meta.extraction.norm!r} — already transformed? Expected {NORM_RAW!r}"
    )
    assert meta.extraction.d_model == stats.d_model, (
        f"d_model mismatch: sidecar {meta.extraction.d_model} vs stats {stats.d_model}"
    )
    if stats.base_model:
        assert stats.base_model == meta.extraction.base_model, (
            f"stats were computed on {stats.base_model!r} activations, this parquet "
            f"is {meta.extraction.base_model!r}"
        )
    if stats.layer_index is not None:
        assert stats.layer_index == meta.extraction.layer_index, (
            f"stats are for layer {stats.layer_index}, this parquet is layer "
            f"{meta.extraction.layer_index}"
        )

    pf = pq.ParquetFile(str(in_path))
    schema = pf.schema_arrow
    av_idx = schema.get_field_index(ACTIVATION_COLUMN)
    assert av_idx >= 0, f"no {ACTIVATION_COLUMN!r} column in {in_path}"
    av_field = schema.field(av_idx)

    print(f"whitening {in_path} → {out_path}")
    print(f"  stats: {args.stats} (sha256 {stats.sha256()[:16]}…, n={stats.n_samples}, "
          f"shrinkage={stats.shrinkage}, floor_quantile={stats.floor_quantile})")
    print(f"  {pf.metadata.num_rows} rows · {pf.num_row_groups} input row groups · d={stats.d_model}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_rows = 0
    gated = False
    with pq.ParquetWriter(str(out_path), schema) as writer:
        for batch in pf.iter_batches(batch_size=TRANSFORM_ROWS):
            n = len(batch)
            if n == 0:
                continue
            col = batch.column(av_idx)
            assert col.null_count == 0, "null activation rows in input"
            raw = col.flatten().to_numpy(zero_copy_only=False).reshape(n, -1)
            assert raw.shape[1] == stats.d_model, (
                f"vectors are {raw.shape[1]}-wide, stats are {stats.d_model}"
            )

            out_f64 = whiten(raw, stats)
            if not gated:
                _distribution_gate(out_f64, stats)
                gated = True
            out_flat = out_f64.astype(np.float32).reshape(-1)

            new_col = _rebuild_activation_column(out_flat, n, av_field.type)
            table = pa.Table.from_batches([batch]).set_column(av_idx, av_field, new_col)
            writer.write_table(table)
            n_rows += n

    assert n_rows > 0, f"no rows in {in_path} — nothing was written"
    print(f"  wrote {n_rows} rows in {time.monotonic() - t0:.1f}s (distribution gate OK)")
    if meta.row_count != n_rows:
        print(f"  NOTE: sidecar row_count {meta.row_count} was stale — correcting to {n_rows}")

    out_meta = dataclasses.replace(
        meta,
        extraction=dataclasses.replace(
            meta.extraction,
            norm=NORM_WHITENED_ZCA,
            whitening=stats.sidecar_block(args.stats),
        ),
        dataset_id=f"{meta.dataset_id}-wzca-{stats.sha256()[:8]}",
        parent_datasets=meta.parent_datasets + [meta.dataset_id],
        row_count=n_rows,
        created_at="",   # restamp at write time
        git_commit="",
    )
    write_sidecar_local(out_path, out_meta)
    print(f"  sidecar: {out_path}.nla_meta.yaml (norm={NORM_WHITENED_ZCA}, "
          f"dataset_id={out_meta.dataset_id})")


if __name__ == "__main__":
    main()
