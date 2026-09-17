"""Check the pilot's positions for DOCUMENT overlap with the data the checkpoints
were trained on.

The SFT warm start and the published reconstruction-RL adapter were built from
one FineFineWeb sample; the pilot's pool comes from a second file of the same
corpus that its producer describes as fresh. That is not byte-verified anywhere,
and if the two overlap the consequences are not symmetric: the recon_rl adapter's
RL split came from the first file, so it may have been optimized on documents the
pilot evaluates on, while behavioral_rl's RL split is doc-disjoint from eval by
construction. This script streams the reference parquets, hashes the opening of
every document, and reports which pilot positions collide.

    python scripts/pred_check_overlap.py --positions <positions.parquet> \\
        --reference asher577/easynla-warmstart-data:av_sft_train.parquet \\
        --reference asher577/easynla-warmstart-data:ar_sft_train.parquet \\
        --reference syvb/nanonla-qwen3-8b-L24-data-full:rl_full.parquet \\
        --out <positions.parquet>.overlap.json

Reference parquets are read row-group by row-group through HfFileSystem (only
the text and doc_id columns), so nothing is downloaded whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

N_CHARS = 200   # opening characters compared; documents are keyed by their start


def key(text: str) -> str:
    return hashlib.sha1(" ".join(text[:N_CHARS].split()).encode()).hexdigest()[:16]


def opening(prefix_text: str, n_raw_tokens: int) -> str:
    """The document's opening is the start of the prefix, which every position
    of the same document shares."""
    return prefix_text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", required=True)
    p.add_argument("--reference", action="append", default=[],
                   help="repo_id:file.parquet (HF dataset) or a local parquet path")
    p.add_argument("--out", default=None)
    p.add_argument("--splits", nargs="*", default=["val", "eval"],
                   help="which pilot splits to check (the RL pool is not scored)")
    args = p.parse_args()

    pf = pq.ParquetFile(args.positions)
    pilot: dict[str, list] = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["row_id", "doc_id", "split", "prefix_text"])
        for rid, did, sp, tx in zip(*(t.column(c).to_pylist() for c in
                                      ("row_id", "doc_id", "split", "prefix_text"))):
            if sp in args.splits:
                pilot.setdefault(key(tx), []).append((rid, did, sp))
    print(f"[overlap] {sum(len(v) for v in pilot.values())} pilot positions in "
          f"{len(pilot)} distinct document openings ({args.splits})", flush=True)

    token = os.environ.get("HF_TOKEN") or (
        Path.home().joinpath(".hf_token").read_text().strip()
        if Path.home().joinpath(".hf_token").exists() else None)
    hits: dict[str, list] = {}
    for ref in args.reference:
        if ":" in ref and not Path(ref).exists():
            repo, fname = ref.split(":", 1)
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem(token=token)
            f = fs.open(f"datasets/{repo}/{fname}", "rb")
        else:
            repo, fname = "local", ref
            f = open(ref, "rb")
        rpf = pq.ParquetFile(f)
        cols = [c for c in ("detokenized_text_truncated", "doc_id") if c in rpf.schema_arrow.names]
        assert "detokenized_text_truncated" in cols, f"{ref} has no source text column"
        n_ref = 0
        for rg in range(rpf.num_row_groups):
            t = rpf.read_row_group(rg, columns=cols)
            for tx in t.column("detokenized_text_truncated").to_pylist():
                n_ref += 1
                k = key(tx)
                if k in pilot:
                    hits.setdefault(k, []).append(ref)
            print(f"  {ref}: row group {rg + 1}/{rpf.num_row_groups}, "
                  f"{n_ref} rows, {len(hits)} colliding openings so far", flush=True)
        f.close()

    by_split = Counter()
    rows = []
    for k, refs in hits.items():
        for rid, did, sp in pilot[k]:
            by_split[sp] += 1
            rows.append({"row_id": rid, "doc_id": did, "split": sp,
                         "references": sorted(set(refs))})
    total = Counter(sp for v in pilot.values() for _, _, sp in v)
    report = {"n_chars": N_CHARS, "references": args.reference,
              "checked": dict(total), "overlapping": dict(by_split),
              "overlapping_rows": rows}
    out = Path(args.out or (args.positions + ".overlap.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"[overlap] overlapping positions by split: {dict(by_split)} of {dict(total)} "
          f"-> {out}", flush=True)
    if rows:
        print("[overlap] WARNING: exclude these row_ids from val/eval (or report the "
              "count) - see docs/pred_nla.md, Data.", flush=True)


if __name__ == "__main__":
    main()
