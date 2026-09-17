"""Prepared-position parquet: activations + the target model's own continuations.

One row per activation position:

    row_id                stable index (assigned at prep time)
    doc_id                provenance; ALSO the split key (see `split_for_doc`)
    split                 "rl" | "val" | "eval" - document-level, never row-level
    prompt                the AV chat prompt carrying the <INJECT> marker
    activation_vector     the raw layer-24 residual at this position
    prefix_text           the document text up to (and including) this position.
                          NEVER shown to the reader or the AV - kept for the
                          qualitative review and for regenerating continuations.
    cont_text[K]          K continuation branches sampled from the TARGET model
    cont_ids[K][T]        their target-model token ids
    cont_bounds[K][T+1]   character offset of each target token boundary within
                          cont_text, so any reader can be bucketed by character
                          (see nla/pred/reader.py)

Splits are by DOCUMENT hash, not row index: the source parquet is row-shuffled,
so a row-index boundary leaves essentially no document fully unseen.
"""

from __future__ import annotations

import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("rl", "val", "eval")


def split_for_doc(doc_id: str, val_permille: int = 60, eval_permille: int = 120) -> str:
    """Deterministic document-level 3-way split.

    Buckets 0..999 by crc32(doc_id): the first `val_permille` are validation, the
    next `eval_permille` are the final evaluation set, the rest are the RL pool.
    Seed-free and order-free, so every stage and every rerun agrees without
    passing a split table around.
    """
    b = zlib.crc32(str(doc_id).encode("utf-8")) % 1000
    if b < val_permille:
        return "val"
    if b < val_permille + eval_permille:
        return "eval"
    return "rl"


def positions_schema(d_model: int, n_branches: int) -> pa.Schema:
    return pa.schema([
        ("row_id", pa.int64()),
        ("doc_id", pa.large_string()),
        ("split", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("prompt", pa.large_list(pa.struct([("role", pa.string()),
                                            ("content", pa.large_string())]))),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("prefix_text", pa.large_string()),
        ("cont_text", pa.list_(pa.large_string(), n_branches)),
        ("cont_ids", pa.list_(pa.list_(pa.int32()), n_branches)),
        ("cont_bounds", pa.list_(pa.list_(pa.int32()), n_branches)),
    ])


def load_positions(
    path: str,
    split: str | None = None,
    limit: int | None = None,
    with_activations: bool = True,
    with_prefix: bool = False,
) -> list[dict]:
    """Read prepared positions into row dicts (activations as numpy float32)."""
    cols = ["row_id", "doc_id", "split", "prompt", "cont_text", "cont_ids", "cont_bounds"]
    if with_activations:
        cols.append("activation_vector")
    if with_prefix:
        cols += ["prefix_text", "n_raw_tokens"]
    pf = pq.ParquetFile(path)
    rows: list[dict] = []
    for rg in range(pf.num_row_groups):
        if limit is not None and len(rows) >= limit:
            break
        t = pf.read_row_group(rg, columns=cols)
        splits = t.column("split").to_pylist()
        keep = [i for i, s in enumerate(splits) if split is None or s == split]
        if not keep:
            continue
        prompts = t.column("prompt").to_pylist()
        ids = t.column("row_id").to_pylist()
        docs = t.column("doc_id").to_pylist()
        ctext = t.column("cont_text").to_pylist()
        cids = t.column("cont_ids").to_pylist()
        cbounds = t.column("cont_bounds").to_pylist()
        acts = None
        if with_activations:
            col = t.column("activation_vector").combine_chunks()
            acts = np.asarray(col.flatten(), dtype=np.float32).reshape(t.num_rows, -1)
        pref = t.column("prefix_text").to_pylist() if with_prefix else None
        nraw = t.column("n_raw_tokens").to_pylist() if with_prefix else None
        for i in keep:
            r = {
                "row_id": ids[i], "doc_id": docs[i], "split": splits[i],
                "prompt": prompts[i], "cont_text": ctext[i],
                "cont_ids": cids[i], "cont_bounds": cbounds[i],
            }
            if acts is not None:
                r["activation"] = acts[i]
            if pref is not None:
                r["prefix_text"] = pref[i]
                r["n_raw_tokens"] = nraw[i]
            rows.append(r)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_counts(path: str) -> dict[str, int]:
    pf = pq.ParquetFile(path)
    out: dict[str, int] = {}
    for rg in range(pf.num_row_groups):
        for s in pf.read_row_group(rg, columns=["split"]).column("split").to_pylist():
            out[s] = out.get(s, 0) + 1
    return out


def shuffled_partner(rows: list[dict], rng) -> tuple[list[int], list[bool]]:
    """A derangement over row indices, plus a validity flag per row.

    A shuffled partner must be a DIFFERENT position from a DIFFERENT document:
    pairing a row with itself makes the shuffled condition identical to the
    matched one and drags matched-minus-shuffled toward zero (i.e. toward failing
    the gate), and pairing within a document leaks real context the other way.

    When the document structure makes a clean derangement impossible - a handful
    of positions from one document, which happens in smoke configs, not on the
    real run - the offending rows come back flagged False so the caller can score
    them as missing instead of scoring a control that is not a control.

    Returns (partner, valid) with len == len(rows).
    """
    n = len(rows)
    perm = list(range(n))
    for _ in range(64):
        perm = list(rng.permutation(n))
        bad = [k for k in range(n)
               if perm[k] == k or rows[perm[k]]["doc_id"] == rows[k]["doc_id"]]
        if not bad:
            return perm, [True] * n
        for k in bad:                      # repair pass: swap offenders out
            j = int(rng.integers(0, n))
            perm[k], perm[j] = perm[j], perm[k]
        if not any(perm[k] == k or rows[perm[k]]["doc_id"] == rows[k]["doc_id"]
                   for k in range(n)):
            return perm, [True] * n
    valid = [perm[k] != k and rows[perm[k]]["doc_id"] != rows[k]["doc_id"]
             for k in range(n)]
    n_bad = valid.count(False)
    print(f"[shuffle] WARNING: {n_bad}/{n} rows have no usable shuffled partner "
          f"(same document or self) - those rows are EXCLUDED from the shuffled "
          f"condition rather than scored against themselves. Too few documents?",
          flush=True)
    return perm, valid


def same_doc_partner(rows: list[dict]) -> tuple[list[int], list[bool]]:
    """Partner each row with ANOTHER position from the SAME document.

    This is the informative mismatch: topic, genre and register are held
    constant, so matched-minus-same_doc isolates what an explanation says about
    THIS position rather than about the document. Rows whose document has no
    other position in `rows` are flagged False (load with `take_by_doc` so
    document pairs stay together).
    """
    by_doc: dict = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)
    partner = list(range(len(rows)))
    valid = [False] * len(rows)
    for idxs in by_doc.values():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            partner[i] = idxs[(k + 1) % len(idxs)]
            valid[i] = True
    return partner, valid


def take_by_doc(rows: list[dict], n: int) -> list[dict]:
    """The first ~n rows, but never splitting a document: whole documents are
    taken in order of first appearance until at least n rows are collected, so
    every position's same-document mate is present."""
    order: list = []
    by_doc: dict = {}
    for r in rows:
        if r["doc_id"] not in by_doc:
            order.append(r["doc_id"])
            by_doc[r["doc_id"]] = []
        by_doc[r["doc_id"]].append(r)
    out: list[dict] = []
    for d in order:
        if len(out) >= n:
            break
        out.extend(by_doc[d])
    return out


__all__ = [
    "SPLITS", "load_positions", "positions_schema", "same_doc_partner",
    "shuffled_partner", "split_counts", "split_for_doc", "take_by_doc",
]
