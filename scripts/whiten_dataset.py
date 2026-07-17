"""Rewrite an NLA parquet with ZCA-whitened activations (offline transform).

Replaces the activation_vector column with W(x − μ) using stats from
scripts/compute_whitening_stats.py; every other column (prompts, gold
explanations, provenance) is copied through byte-identical, and the sidecar
is restamped with norm=whitened_zca_v1 + a whitening provenance block.
Trainers then run on the whitened parquet with NO code changes.

Usage (once per split, ALWAYS with the train-split stats):
    python scripts/whiten_dataset.py \
        --parquet <data>/av_sft_train.parquet \
        --stats   <data>/whitening_stats.npz \
        --out     <data_whitened>/av_sft_train.parquet

Integrity: the first batch is round-tripped (whiten → unwhiten) against the
raw input and the relative error printed — catches a wrong/corrupt stats
file before you burn GPU-hours on garbage.
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
from nla.whitening import load_stats, unwhiten, whiten

# Rows transformed per float64 matmul chunk — bounds transform memory to
# ~chunk × d × 8B regardless of the input's row-group size.
TRANSFORM_CHUNK = 16_384


def _rebuild_activation_column(flat_f32: np.ndarray, n: int, av_type: pa.DataType) -> pa.Array:
    """Flat float32 values → an arrow array of the INPUT's activation type,
    so the output schema is identical to the input's."""
    values = pa.array(flat_f32, type=pa.float32())
    if pa.types.is_fixed_size_list(av_type):
        return pa.FixedSizeListArray.from_arrays(values, av_type.list_size)
    if pa.types.is_list(av_type):
        d = len(flat_f32) // n
        offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32), type=pa.int32())
        return pa.ListArray.from_arrays(offsets, values)
    raise AssertionError(f"unsupported activation column type: {av_type}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="input parquet (raw activations, with sidecar)")
    p.add_argument("--stats", required=True, help=".npz from compute_whitening_stats.py (train split!)")
    p.add_argument("--out", required=True, help="output parquet path (sidecar written alongside)")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = p.parse_args()

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

    pf = pq.ParquetFile(str(in_path))
    schema = pf.schema_arrow
    av_idx = schema.get_field_index(ACTIVATION_COLUMN)
    assert av_idx >= 0, f"no {ACTIVATION_COLUMN!r} column in {in_path}"
    av_type = schema.field(av_idx).type

    print(f"whitening {in_path} → {out_path}")
    print(f"  stats: {args.stats} (sha256 {stats.sha256()[:16]}…, n={stats.n_samples}, "
          f"shrinkage={stats.shrinkage})")
    print(f"  {pf.num_row_groups} row groups · {pf.metadata.num_rows} rows · d={stats.d_model}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_rows = 0
    roundtrip_err = None
    with pq.ParquetWriter(str(out_path), schema) as writer:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            col = table.column(av_idx).combine_chunks()
            assert col.null_count == 0, f"null activation rows in row group {rg}"
            n = len(col)
            raw = col.flatten().to_numpy(zero_copy_only=False).reshape(n, -1)
            assert raw.shape[1] == stats.d_model, (
                f"row group {rg}: vectors are {raw.shape[1]}-wide, stats are {stats.d_model}"
            )

            out_flat = np.empty(raw.shape, dtype=np.float32)
            for lo in range(0, n, TRANSFORM_CHUNK):
                chunk = raw[lo : lo + TRANSFORM_CHUNK]
                out_flat[lo : lo + len(chunk)] = whiten(chunk, stats).astype(np.float32)

            if roundtrip_err is None:
                # whiten → unwhiten must reproduce the raw input up to float32
                # rounding; a wrong stats file shows up here, not after SFT.
                back = unwhiten(out_flat[: min(n, 1024)].astype(np.float64), stats)
                ref = raw[: min(n, 1024)].astype(np.float64)
                roundtrip_err = float(
                    np.abs(back - ref).max() / max(np.abs(ref).max(), 1e-12)
                )
                assert roundtrip_err < 1e-3, (
                    f"whiten→unwhiten round-trip rel-err {roundtrip_err:.2e} ≥ 1e-3 — "
                    f"stats don't match this data (wrong d_model source? corrupt npz?)"
                )

            new_col = _rebuild_activation_column(out_flat.reshape(-1), n, av_type)
            table = table.set_column(av_idx, schema.field(av_idx), new_col)
            writer.write_table(table)
            n_rows += n

    print(f"  wrote {n_rows} rows in {time.monotonic() - t0:.1f}s · "
          f"round-trip rel-err {roundtrip_err:.2e}")
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
