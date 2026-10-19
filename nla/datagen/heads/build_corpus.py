"""Head stage `corpus`: sample a domain-mixed document set → corpus.parquet.

Domain-sharded corpora like m-a-p/FineFineWeb (66k jsonl files under one
directory per domain, ~300 MB each) can't be sliced head-first — the first
100k rows are all one domain — and can't be loaded whole. Instead: list the
repo's files, pick `num_files` of them uniformly at random (seeded; domains
come out in proportion to their size), and stream the first `docs_per_file`
qualifying documents from each over HTTP. Only the start of each file is
read, so 200 files cost a few hundred MB of transfer, not 60 GB.

Rows are shuffled before writing so any [start, start+length) slice of the
output is itself domain-mixed.
"""

import argparse
import fnmatch
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, list_repo_files
from tqdm import tqdm


def _read_file(repo: str, path: str, n_docs: int, text_column: str, id_column: str | None,
               keep: list[str], min_chars: int, attempts: int = 4) -> list[dict] | None:
    """First n_docs qualifying docs of one shard, or None if it can't be read.

    Hub reads under concurrency fail transiently (seen: FileNotFoundError for a
    file that exists), so each attempt gets its own filesystem client — they
    are not safe to share across threads — with exponential backoff.
    """
    for attempt in range(attempts):
        try:
            return _read_once(HfFileSystem(skip_instance_cache=True), repo, path, n_docs, text_column,
                              id_column, keep, min_chars)
        except (OSError, ValueError) as e:
            if attempt == attempts - 1:
                print(f"  WARNING {path}: unreadable after {attempts} attempts ({type(e).__name__}: {e})")
                return None
            time.sleep(2 ** (attempt + 1))
    return None


def _read_once(fs: HfFileSystem, repo: str, path: str, n_docs: int, text_column: str,
               id_column: str | None, keep: list[str], min_chars: int) -> list[dict]:
    out: list[dict] = []
    with fs.open(f"datasets/{repo}/{path}", "rb", block_size=1 << 20) as f:
        for line_no, line in enumerate(f):
            if len(out) >= n_docs:
                break
            row = orjson.loads(line)
            text = row.get(text_column)
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            doc_id = str(row[id_column]) if id_column else f"{repo}/{path}:{line_no}"
            out.append({"doc_id": doc_id, "text": text,
                        **{c: None if row.get(c) is None else str(row[c]) for c in keep}})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="m-a-p/FineFineWeb", help="HF dataset repo with jsonl shards")
    p.add_argument("--file-glob", default="*.jsonl", help="which repo files are data shards")
    p.add_argument("--num-files", type=int, default=200)
    p.add_argument("--docs-per-file", type=int, default=500)
    p.add_argument("--min-chars", type=int, default=1000,
                   help="skip short docs — positions start at token 50 and longer contexts give "
                        "heads more to attend to")
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-column", default="global_id", help="stable per-doc id field ('' for file:line)")
    p.add_argument("--keep-columns", default="domain,url", help="comma-separated fields copied as strings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    keep = [c for c in args.keep_columns.split(",") if c]
    files = sorted(f for f in list_repo_files(args.dataset, repo_type="dataset")
                   if fnmatch.fnmatch(f, args.file_glob))
    assert len(files) >= args.num_files, (
        f"{args.dataset}: only {len(files)} files match {args.file_glob!r}, need {args.num_files}"
    )
    # Seeded order over ALL files: the first num_files are the sample, the rest
    # are replacements (in order) for any file that can't be read.
    order = random.Random(args.seed).sample(files, len(files))
    print(f"{args.dataset}: sampling {args.docs_per_file} docs from each of {args.num_files}/{len(files)} files")

    def read(f: str) -> list[dict] | None:
        return _read_file(args.dataset, f, args.docs_per_file, args.text_column, args.id_column or None,
                          keep, args.min_chars)

    got: dict[str, list[dict]] = {}
    nxt = 0
    with ThreadPoolExecutor(args.workers) as pool, tqdm(total=args.num_files, desc="files") as bar:
        while len(got) < args.num_files:
            batch = order[nxt : nxt + args.num_files - len(got)]
            assert batch, f"ran out of readable files after {nxt} tries"
            nxt += len(batch)
            for f, docs in zip(batch, pool.map(read, batch), strict=True):
                if docs is not None:
                    got[f] = docs
                    bar.update(1)
    chosen = sorted(got)
    if nxt > args.num_files:
        print(f"  replaced {nxt - args.num_files} unreadable files with the next seeded picks")

    rows: list[dict] = []
    seen: set[str] = set()
    for f in chosen:
        docs = got[f]
        if len(docs) < args.docs_per_file:
            print(f"  WARNING {f}: only {len(docs)} docs ≥ {args.min_chars} chars")
        for d in docs:
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                rows.append(d)
    random.Random(args.seed + 1).shuffle(rows)

    schema = pa.schema([("doc_id", pa.string()), ("text", pa.string()), *[(c, pa.string()) for c in keep]])
    table = pa.Table.from_pylist(rows, schema=schema)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    print(f"wrote {table.num_rows} docs → {out}")
    if "domain" in keep:
        from collections import Counter
        top = Counter(table.column("domain").to_pylist()).most_common(8)
        print("  top domains: " + ", ".join(f"{d}={n}" for d, n in top))


if __name__ == "__main__":
    main()
