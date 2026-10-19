"""On-disk layout, schemas, and codecs for the head-contribution dataset.

Vectors (layer-input residual, per-head outputs) are stored as bf16 BIT
PATTERNS (uint16): lossless relative to a bf16 forward (the model's own
compute dtype), half the size of float32, and immune to the fp16 overflow a
65504 cap would risk on massive-activation dims. Decode with `from_bf16_bits`.

Every fixed-width array column (vectors and per-head attention summaries) is a
parquet FIXED_LEN_BYTE_ARRAY holding the row's little-endian values
(`fixed_array` / `fixed_numpy`; dtype + width in FIXED_COLUMNS). Parquet has
no fixed-size list: an arrow FixedSizeList is written as a repeated field
with per-element repetition/definition levels, which measured 1.7× the raw
bytes uncompressed and 0.86× with snappy+dictionary, against 0.76× (zstd)
for fixed-size binary — ~25 GB on the full Qwen3-8B run, and 2–3× faster
to read.

Row unit of records/: one sampled (document, token position, layer). All
`n_heads` heads of that layer share the row — head h's output is
`head_out[h*head_dim:(h+1)*head_dim]`, and its write into the residual stream
is `W_O[:, h*head_dim:(h+1)*head_dim] @ that` (wo.safetensors). Storing the
whole o_proj input (not one head's write) keeps every derived quantity
recoverable: any head's write, the post-attention residual
(resid_pre + W_O @ head_out), or "residual with this head's write removed".
"""

import dataclasses
import datetime
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from nla.datagen.sidecar import _git_commit

META_NAME = "head_meta.yaml"
WO_NAME = "wo.safetensors"
STATS_NAME = "head_stats.safetensors"
SPLITS_NAME = "splits.parquet"
SCHEMA_VERSION = 1

# Fields two extraction processes writing into the same output_dir must agree
# on (multi-GPU shards, resumes). Provenance fields (created_at, git) may differ.
_CONTRACT_FIELDS = (
    "base_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_model",
    "layers", "layers_per_position", "positions_per_doc", "min_position", "top_k",
    "max_length", "seed", "corpus", "corpus_slice", "text_column", "id_column",
    "chunk_size", "chunks_per_part",
)


@dataclass
class HeadDatasetMeta:
    base_model: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_model: int
    layers: list[int]               # layers eligible for sampling
    layers_per_position: int        # layers recorded per sampled token
    positions_per_doc: int
    min_position: int
    top_k: int                      # attention sources kept per head
    max_length: int                 # tokenization truncation
    seed: int
    corpus: str
    corpus_slice: dict[str, int]
    text_column: str
    id_column: str | None
    chunk_size: int
    chunks_per_part: int
    vector_encoding: str = "bf16_bits_uint16"
    kind: str = "nla_head_dataset"
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    git_commit: str = ""


def write_or_check_meta(out_dir: Path, meta: HeadDatasetMeta) -> None:
    """First writer records the contract; every later process must match it.

    A resumed or sharded run with a different layer set / seed / top_k would
    silently interleave incompatible parts into one dataset.
    """
    path = out_dir / META_NAME
    if path.exists():
        existing = read_meta(out_dir)
        mismatches = [
            f"{k}: {getattr(existing, k)!r} on disk vs {getattr(meta, k)!r} now"
            for k in _CONTRACT_FIELDS
            if getattr(existing, k) != getattr(meta, k)
        ]
        if mismatches:
            raise RuntimeError(
                f"{path} was written by an incompatible run:\n  " + "\n  ".join(mismatches)
                + f"\nUse a fresh output_dir, or delete {out_dir} to start over."
            )
        return
    meta.created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    meta.git_commit = _git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(asdict(meta), sort_keys=False))
    os.replace(tmp, path)


def read_meta(out_dir: Path) -> HeadDatasetMeta:
    d = yaml.safe_load((Path(out_dir) / META_NAME).read_text())
    assert d.get("kind") == "nla_head_dataset", f"not a head dataset manifest: kind={d.get('kind')!r}"
    assert d["schema_version"] == SCHEMA_VERSION, (
        f"head dataset schema {d['schema_version']} != expected {SCHEMA_VERSION}"
    )
    known = {f.name for f in dataclasses.fields(HeadDatasetMeta)}
    return HeadDatasetMeta(**{k: v for k, v in d.items() if k in known})


# ── paths ────────────────────────────────────────────────────────────────

def docs_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "docs"


def records_dir(out_dir: Path, layer: int) -> Path:
    return Path(out_dir) / "records" / f"L{layer:02d}"


def devnorms_dir(out_dir: Path, layer: int) -> Path:
    return Path(out_dir) / "devnorms" / f"L{layer:02d}"


def parts_marker_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "_parts"


def part_name(doc_start: int) -> str:
    """Parts are keyed by ABSOLUTE corpus row of their first doc, so shards on
    disjoint slices (or a resume with a different shard count) never collide."""
    return f"part_{doc_start:08d}.parquet"


def part_files(directory: Path) -> list[Path]:
    """Completed part files in key order (tmp files are dot-prefixed)."""
    return sorted(Path(directory).glob("part_*.parquet"))


# ── schemas ──────────────────────────────────────────────────────────────

# Parquet writer options for every table here: zstd, dictionary only where
# values repeat (doc ids repeat across a document's records).
WRITE_OPTS: dict[str, Any] = {"compression": "zstd", "use_dictionary": ["doc_id", "split"]}


def fixed_columns(d_model: int, n_heads: int, head_dim: int, top_k: int) -> dict[str, tuple[Any, int]]:
    """name → (numpy dtype, values per row) for the fixed-width record columns."""
    return {
        "resid_pre": (np.uint16, d_model),              # bf16 bits: input to decoder layer `layer`
        "head_out": (np.uint16, n_heads * head_dim),    # bf16 bits: o_proj input, head-major
        "attn_top_idx": (np.int32, n_heads * top_k),    # [H, K] source positions, by weight desc
        "attn_top_w": (np.float32, n_heads * top_k),    # [H, K] attention weights
        "attn_first_w": (np.float32, n_heads),          # weight on position 0 (the sink)
        "attn_self_w": (np.float32, n_heads),           # weight on the current token
        "attn_entropy": (np.float32, n_heads),          # nats
        "dev_norm": (np.float32, n_heads),              # devnorms/: ||W_h (z_h - mean z_h)||
        "write_norm": (np.float32, n_heads),            # devnorms/: ||W_h z_h||
    }


def _fixed(dtype: Any, width: int) -> pa.DataType:
    return pa.binary(np.dtype(dtype).itemsize * width)


def records_schema(d_model: int, n_heads: int, head_dim: int, top_k: int) -> pa.Schema:
    fc = fixed_columns(d_model, n_heads, head_dim, top_k)
    return pa.schema([
        ("doc_id", pa.string()),
        ("pos", pa.int32()),                                      # 0-indexed current-token position
        ("token_id", pa.int32()),
        ("layer", pa.int16()),
        *[(name, _fixed(*fc[name])) for name in ("resid_pre", "head_out", "attn_top_idx", "attn_top_w",
                                                  "attn_first_w", "attn_self_w", "attn_entropy")],
    ])


def docs_schema(meta_columns: list[str]) -> pa.Schema:
    return pa.schema([
        ("doc_id", pa.string()),
        ("corpus_row", pa.int64()),
        ("n_tokens", pa.int32()),
        ("token_ids", pa.list_(pa.int32())),   # tokenization after max_length truncation
        ("positions", pa.list_(pa.int32())),   # sampled positions (records exist for these)
        *[(c, pa.string()) for c in meta_columns],
    ])


def devnorms_schema(n_heads: int) -> pa.Schema:
    return pa.schema([
        ("doc_id", pa.string()),
        ("pos", pa.int32()),
        ("dev_norm", _fixed(np.float32, n_heads)),
        ("write_norm", _fixed(np.float32, n_heads)),
    ])


# ── codecs ───────────────────────────────────────────────────────────────

def to_bf16_bits(t: torch.Tensor) -> np.ndarray:
    """Any float tensor → uint16 numpy array of its bf16 bit patterns."""
    return t.detach().to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)


def from_bf16_bits(a: np.ndarray) -> torch.Tensor:
    """uint16 bf16 bit patterns → float32 tensor (exact)."""
    # np.require copies only when needed: arrow-backed arrays are read-only,
    # which torch.from_numpy warns about.
    a = np.require(a, dtype=np.uint16, requirements=["C", "W"])
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float()


def fixed_array(values: np.ndarray) -> pa.FixedSizeBinaryArray:
    """[n, ...] numpy → one fixed-size binary value (the row's raw bytes) per row."""
    a = np.ascontiguousarray(values)
    assert a.dtype.byteorder in ("=", "|", "<"), "fixed columns are little-endian"
    row_bytes = a.itemsize * int(np.prod(a.shape[1:]))
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(row_bytes), len(a), [None, pa.py_buffer(a.tobytes())])


def fixed_numpy(col: pa.ChunkedArray | pa.Array, dtype: Any, width: int) -> np.ndarray:
    """Fixed-size binary column → [n, width] numpy of `dtype` (a copy)."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    itemsize = np.dtype(dtype).itemsize
    assert arr.type.byte_width == itemsize * width, (
        f"column holds {arr.type.byte_width} bytes/row, expected {width} × {np.dtype(dtype).name}"
    )
    buf = arr.buffers()[1]
    start = arr.offset * arr.type.byte_width
    raw = np.frombuffer(buf, dtype=np.uint8, count=len(arr) * arr.type.byte_width, offset=start)
    return raw.view(dtype).reshape(len(arr), width).copy()


# ── readers ──────────────────────────────────────────────────────────────

def read_docs(out_dir: Path, columns: list[str] | None = None) -> pa.Table:
    files = part_files(docs_dir(out_dir))
    assert files, f"no doc parts under {docs_dir(out_dir)} — run the extract stage first"
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def read_records(out_dir: Path, layer: int, columns: list[str] | None = None) -> pa.Table:
    files = part_files(records_dir(out_dir, layer))
    assert files, f"no record parts under {records_dir(out_dir, layer)}"
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def load_wo(out_dir: Path, layer: int) -> torch.Tensor:
    """o_proj weight of `layer` as float32 [d_model, n_heads * head_dim]."""
    from safetensors import safe_open
    with safe_open(str(Path(out_dir) / WO_NAME), framework="pt") as f:
        return f.get_tensor(f"L{layer}").float()


def atomic_write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    pq.write_table(table, tmp, **WRITE_OPTS)
    os.replace(tmp, path)


@dataclass
class HeadStats:
    """Per-head statistics from stats.py, indexed by position in `layers`."""
    layers: list[int]
    mean_head_out: torch.Tensor    # [L, H, D]  mean o_proj input per head
    mean_write: torch.Tensor       # [L, H, d]  W_h @ mean_head_out — "the average write"
    dev_rms: torch.Tensor          # [L, H]     RMS of ||W_h (z - mean)|| — "typical size"
    dev_quantiles: torch.Tensor    # [L, H, Q]  quantiles of the same norm
    quantile_levels: torch.Tensor  # [Q]
    count: torch.Tensor            # [L]        records per layer

    def layer_pos(self, layer: int) -> int:
        return self.layers.index(layer)


def save_stats(out_dir: Path, s: HeadStats) -> None:
    from safetensors.torch import save_file
    tensors = {
        "layers": torch.tensor(s.layers, dtype=torch.int32),
        "mean_head_out": s.mean_head_out.float().contiguous(),
        "mean_write": s.mean_write.float().contiguous(),
        "dev_rms": s.dev_rms.float().contiguous(),
        "dev_quantiles": s.dev_quantiles.float().contiguous(),
        "quantile_levels": s.quantile_levels.float().contiguous(),
        "count": s.count.to(torch.int64).contiguous(),
    }
    path = Path(out_dir) / STATS_NAME
    tmp = path.parent / f".{path.name}.tmp"
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def load_stats(out_dir: Path) -> HeadStats:
    from safetensors.torch import load_file
    t = load_file(str(Path(out_dir) / STATS_NAME))
    return HeadStats(
        layers=t["layers"].tolist(),
        mean_head_out=t["mean_head_out"],
        mean_write=t["mean_write"],
        dev_rms=t["dev_rms"],
        dev_quantiles=t["dev_quantiles"],
        quantile_levels=t["quantile_levels"],
        count=t["count"],
    )
