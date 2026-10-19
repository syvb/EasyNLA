"""Head stage `extract`: corpus → docs/ + records/L{ll}/ (+ head_meta.yaml, wo.safetensors).

Per document, sample `positions_per_doc` token positions (same per-doc RNG as
stage0_extract, so the same doc gets the same positions), and per position
`layers_per_position` layers from `--layers`. Each (position, layer) becomes
one records row holding that layer's input residual, all heads' outputs, and
each head's attention summary — see store.py for the schema.

Crash-safe and shardable. Work is cut into parts of chunk_size ×
chunks_per_part docs keyed by absolute corpus row; a part's files are
published atomically and then marked done in _parts/. Restarts skip done
parts; `--num-shards N --shard-index i` takes every N-th part, so N processes
(one per GPU, CUDA_VISIBLE_DEVICES=i) can fill one output_dir:

    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m nla.datagen.heads.extract \\
        ... --num-shards 4 --shard-index $i & done; wait
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.datagen._common import parse_kwargs
from nla.datagen.heads.capture import HeadCaptureExtractor
from nla.datagen.heads.store import (
    WO_NAME,
    WRITE_OPTS,
    HeadDatasetMeta,
    docs_dir,
    docs_schema,
    fixed_array,
    part_name,
    parts_marker_dir,
    records_dir,
    records_schema,
    write_or_check_meta,
)
from nla.datagen.stage0_extract import _MIN_POSITION, _sample_positions


def sample_layers(layers: list[int], k: int, doc_id: str, pos: int, seed: int) -> tuple[int, ...]:
    """Layers recorded at one position — keyed on (seed, doc, pos) like positions are."""
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}|{pos}|layers".encode()).digest())
    return tuple(sorted(rng.sample(layers, k)))


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    assert layers and all(0 <= x < n_layers for x in layers), (
        f"--layers {spec!r}: expected 'all' or comma-separated ints in [0, {n_layers})"
    )
    return layers


def load_corpus(corpus: str, split: str, config: str | None, start: int, length: int,
                columns: list[str]) -> pa.Table:
    """Local .parquet (preferred — build one with build_corpus.py) or a
    non-streaming HF dataset, sliced to [start, start + length)."""
    if corpus.endswith(".parquet") and os.path.exists(corpus):
        pf = pq.ParquetFile(corpus)
        n = pf.metadata.num_rows
        assert start + length <= n, f"corpus slice [{start}, {start + length}) exceeds {n} rows in {corpus}"
        return pq.read_table(corpus, columns=columns).slice(start, length)
    from datasets import load_dataset
    ds = load_dataset(corpus, name=config, split=split)
    # flatten_indices: select() only records an index mapping, which .data.table
    # would silently ignore (returning the first `length` rows of the whole set).
    ds = ds.select(range(start, start + length)).select_columns(columns).flatten_indices()
    return ds.data.table


def _save_wo(out_dir: Path, extractor: HeadCaptureExtractor, layers: list[int]) -> None:
    """o_proj weights the data was extracted with — stats + training compute
    writes from these, never from a separately-loaded checkpoint."""
    path = out_dir / WO_NAME
    if path.exists():
        return
    from safetensors.torch import save_file
    tensors = {f"L{layer}": extractor.o_proj_weight(layer).to(torch.bfloat16).cpu().contiguous()
               for layer in layers}
    tmp = out_dir / f".{WO_NAME}.{os.getpid()}.tmp"
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def _writer(writers: dict[Path, pq.ParquetWriter], final: Path, schema: pa.Schema) -> pq.ParquetWriter:
    """Lazily open `final`'s dot-prefixed tmp file; the part loop renames it into place."""
    if final not in writers:
        final.parent.mkdir(parents=True, exist_ok=True)
        writers[final] = pq.ParquetWriter(str(final.parent / f".{final.name}.tmp"), schema, **WRITE_OPTS)
    return writers[final]


def _plan_doc(token_ids: list[int], doc_id: str, special_ids: set[int], args: argparse.Namespace,
              layers: list[int]) -> list[tuple[int, tuple[int, ...]]]:
    positions = sorted(_sample_positions(token_ids, args.positions_per_doc, special_ids, doc_id, args.seed))
    return [(p, sample_layers(layers, args.layers_per_position, doc_id, p, args.seed)) for p in positions]


def _records_table(caps: dict[str, np.ndarray], layer: int, doc_ids: list[str], token_ids: list[list[int]],
                   schema: pa.Schema) -> pa.Table:
    doc = caps["doc"]
    pos = caps["pos"]
    return pa.Table.from_arrays([
        pa.array([doc_ids[i] for i in doc], type=pa.string()),
        pa.array(pos, type=pa.int32()),
        pa.array([token_ids[i][p] for i, p in zip(doc, pos, strict=True)], type=pa.int32()),
        pa.array(np.full(len(doc), layer, dtype=np.int16), type=pa.int16()),
        fixed_array(caps["resid_pre"].astype(np.uint16)),
        fixed_array(caps["head_out"].astype(np.uint16)),
        fixed_array(caps["attn_top_idx"].astype(np.int32)),
        fixed_array(caps["attn_top_w"].astype(np.float32)),
        fixed_array(caps["attn_first_w"].astype(np.float32)),
        fixed_array(caps["attn_self_w"].astype(np.float32)),
        fixed_array(caps["attn_entropy"].astype(np.float32)),
    ], schema=schema)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--corpus", required=True, help="local .parquet (see build_corpus.py) or HF dataset name")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument("--corpus-length", type=int, required=True)
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-column", default=None,
                   help="corpus column holding a stable doc id; default '<corpus>:<split>:<row>'")
    p.add_argument("--meta-columns", default="",
                   help="comma-separated corpus columns copied (as strings) into docs/, e.g. 'domain,url'")
    p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    p.add_argument("--layers-per-position", type=int, default=4)
    p.add_argument("--positions-per-doc", type=int, default=10)
    p.add_argument("--top-k", type=int, default=16, help="attention sources kept per head")
    p.add_argument("--chunk-size", type=int, default=256, help="docs tokenized + forwarded per step")
    p.add_argument("--chunks-per-part", type=int, default=8, help="chunks per durable part file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extractor-kwargs", default=None,
                   help="JSON: batch_size, max_length, device_map, dtype")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()

    assert 0 <= args.shard_index < args.num_shards, f"--shard-index must be in [0, {args.num_shards})"
    assert args.top_k <= _MIN_POSITION, (
        f"--top-k {args.top_k} > {_MIN_POSITION}: every sampled position must have at least "
        f"top_k visible sources, or the top-k list would contain zero-weight filler"
    )
    out_dir = Path(args.output_dir)
    meta_cols = [c for c in args.meta_columns.split(",") if c]

    kwargs = parse_kwargs(args.extractor_kwargs)
    extractor = HeadCaptureExtractor(args.base_model, **kwargs)
    layers = parse_layers(args.layers, extractor.n_layers)
    assert 1 <= args.layers_per_position <= len(layers), (
        f"--layers-per-position {args.layers_per_position} must be in [1, {len(layers)}]"
    )
    write_or_check_meta(out_dir, HeadDatasetMeta(
        base_model=args.base_model,
        n_layers=extractor.n_layers,
        n_heads=extractor.n_heads,
        n_kv_heads=extractor.n_kv_heads,
        head_dim=extractor.head_dim,
        d_model=extractor.d_model,
        layers=layers,
        layers_per_position=args.layers_per_position,
        positions_per_doc=args.positions_per_doc,
        min_position=_MIN_POSITION,
        top_k=args.top_k,
        max_length=extractor.max_length,
        seed=args.seed,
        corpus=args.corpus,
        corpus_slice={"start": args.corpus_start, "length": args.corpus_length},
        text_column=args.text_column,
        id_column=args.id_column,
        chunk_size=args.chunk_size,
        chunks_per_part=args.chunks_per_part,
    ))
    _save_wo(out_dir, extractor, layers)

    columns = [args.text_column, *([args.id_column] if args.id_column else []), *meta_cols]
    corpus = load_corpus(args.corpus, args.corpus_split, args.corpus_config,
                         args.corpus_start, args.corpus_length, columns)
    special_ids = set(extractor.tokenizer.all_special_ids)
    rec_schema = records_schema(extractor.d_model, extractor.n_heads, extractor.head_dim, args.top_k)
    doc_schema = docs_schema(meta_cols)
    markers = parts_marker_dir(out_dir)
    markers.mkdir(parents=True, exist_ok=True)

    part_docs = args.chunk_size * args.chunks_per_part
    part_offsets = list(range(0, args.corpus_length, part_docs))
    mine = [o for k, o in enumerate(part_offsets) if k % args.num_shards == args.shard_index]
    n_done = 0
    for offset in tqdm(mine, desc=f"parts (shard {args.shard_index}/{args.num_shards})"):
        name = part_name(args.corpus_start + offset)
        marker = markers / f"{name}.done"
        if marker.exists():
            n_done += 1
            continue
        # Any files without a marker are leftovers of a crashed attempt.
        final_paths = [docs_dir(out_dir) / name, *(records_dir(out_dir, lay) / name for lay in layers)]
        for fp in final_paths:
            fp.unlink(missing_ok=True)

        writers: dict[Path, pq.ParquetWriter] = {}
        counts: dict[str, int] = {"docs": 0, "records": 0, "docs_skipped": 0}
        part_end = min(offset + part_docs, args.corpus_length)
        for c0 in range(offset, part_end, args.chunk_size):
            chunk = corpus.slice(c0, min(args.chunk_size, part_end - c0))
            texts = chunk.column(args.text_column).to_pylist()
            rows = [args.corpus_start + c0 + j for j in range(len(texts))]
            doc_ids = (
                [str(x) for x in chunk.column(args.id_column).to_pylist()] if args.id_column
                else [f"{args.corpus}:{args.corpus_split}:{r}" for r in rows]
            )
            token_ids = extractor.tokenize(texts)
            plan = [_plan_doc(t, d, special_ids, args, layers) for t, d in zip(token_ids, doc_ids, strict=True)]
            caps = extractor.capture(token_ids, plan, args.top_k)

            kept = [j for j in range(len(texts)) if plan[j]]
            counts["docs_skipped"] += len(texts) - len(kept)
            doc_cols: list[Any] = [
                pa.array([doc_ids[j] for j in kept], type=pa.string()),
                pa.array([rows[j] for j in kept], type=pa.int64()),
                pa.array([len(token_ids[j]) for j in kept], type=pa.int32()),
                pa.array([token_ids[j] for j in kept], type=pa.list_(pa.int32())),
                pa.array([[pos for pos, _ in plan[j]] for j in kept], type=pa.list_(pa.int32())),
            ]
            for c in meta_cols:
                vals = chunk.column(c).to_pylist()
                doc_cols.append(pa.array([None if vals[j] is None else str(vals[j]) for j in kept],
                                         type=pa.string()))
            _writer(writers, docs_dir(out_dir) / name, doc_schema).write_table(
                pa.Table.from_arrays(doc_cols, schema=doc_schema))
            counts["docs"] += len(kept)
            for lay, layer_caps in sorted(caps.items()):
                table = _records_table(layer_caps, lay, doc_ids, token_ids, rec_schema)
                _writer(writers, records_dir(out_dir, lay) / name, rec_schema).write_table(table)
                counts["records"] += table.num_rows

        for final, w in writers.items():
            w.close()
            os.replace(final.parent / f".{final.name}.tmp", final)
        tmp_marker = markers / f".{name}.done.tmp"
        tmp_marker.write_text(json.dumps(counts))
        os.replace(tmp_marker, marker)

    print(f"shard {args.shard_index}/{args.num_shards}: {len(mine) - n_done} parts written, "
          f"{n_done} already done → {out_dir}")


if __name__ == "__main__":
    main()
