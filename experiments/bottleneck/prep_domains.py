"""Build the Stage A domain corpus (CPU-only; run on the dev box, cache on HF).

Samples ~seqs_per_domain documents per domain via `datasets` streaming, writes
one parquet {domain, doc_id, text}. Texts are stored raw; tokenization/
truncation to 256 tokens happens in stage_a.py with M's tokenizer. Documents
are oversampled (2×) so stage_a can drop <64-token stragglers.

The "onpolicy" domain is NOT built here — stage_a generates it on the GPU box.

    python -m experiments.bottleneck.prep_domains --out domains.parquet \
        [--upload syvb/nla-bottleneck-domains]
"""

from __future__ import annotations

import argparse
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq

MIN_CHARS = 800  # ~>200 tokens; stage_a still enforces >=64 tokens post-tokenize


def _stream_texts(spec_list, n, field="text", transform=None):
    """Try each (dataset, config, split) spec until one loads; take n texts."""
    import datasets as hfd
    last_err = None
    for spec in spec_list:
        try:
            ds = hfd.load_dataset(*spec[:-1], split=spec[-1], streaming=True)
            out = []
            for row in ds:
                t = transform(row) if transform else row.get(field)
                if t and len(t) >= MIN_CHARS:
                    out.append(t)
                if len(out) >= n:
                    return out, spec
            if out:
                return out, spec
        except Exception as e:  # gated repo / renamed config → try next
            last_err = e
            print(f"[prep] {spec} failed: {e}")
    raise RuntimeError(f"all specs failed for domain; last: {last_err}")


def _chat_transform(row):
    msgs = row.get("messages")
    if not msgs:
        return None
    return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs)


def _synthetic_json(n, seed=0):
    """Structured-data domain: deterministic synthetic JSON/CSV records (no
    download; a real JSON corpus adds little for a fidelity probe and gated
    code datasets are a setup hazard)."""
    r = random.Random(seed)
    first = ["Ada", "Boris", "Chen", "Divya", "Emil", "Fatima", "Goro", "Hana",
             "Ivan", "Jun", "Kwame", "Lena", "Mateo", "Nadia", "Omar", "Priya"]
    cities = ["Lagos", "Osaka", "Porto", "Quito", "Riga", "Seoul", "Tunis",
              "Ulm", "Vigo", "Wuhan", "Xalapa", "Yerevan", "Zagreb"]
    out = []
    for i in range(n):
        records = []
        for j in range(r.randint(8, 14)):
            records.append({
                "id": r.randint(10000, 99999),
                "name": f"{r.choice(first)} {chr(65 + r.randint(0, 25))}.",
                "city": r.choice(cities),
                "balance": round(r.uniform(-5000, 25000), 2),
                "active": r.random() > 0.4,
                "scores": [r.randint(0, 100) for _ in range(r.randint(2, 5))],
                "meta": {"tier": r.choice(["gold", "silver", "bronze"]),
                         "since": f"20{r.randint(10, 25)}-{r.randint(1, 12):02d}"},
            })
        csv_hdr = "id,name,city,balance,active"
        csv_rows = "\n".join(
            f"{x['id']},{x['name']},{x['city']},{x['balance']},{x['active']}"
            for x in records)
        out.append("Customer database export (JSON):\n"
                   + json.dumps(records, indent=1)
                   + f"\n\nSame data as CSV:\n{csv_hdr}\n{csv_rows}\n")
    return out


DOMAINS = {
    "fineweb":    [("HuggingFaceFW/fineweb", "sample-10BT", "train")],
    "wikipedia":  [("wikimedia/wikipedia", "20231101.en", "train")],
    "code":       [("codeparrot/github-code-clean", "Python-all", "train"),
                   ("codeparrot/codeparrot-clean-valid", "train")],
    "math":       [("open-web-math/open-web-math", "train")],
    "arxiv":      [("ccdv/arxiv-summarization", "document", "train")],
    "pubmed":     [("ccdv/pubmed-summarization", "document", "train")],
    "chat":       [("HuggingFaceH4/ultrachat_200k", "train_sft")],
    "wiki_zh":    [("wikimedia/wikipedia", "20231101.zh", "train")],
    "wiki_fr":    [("wikimedia/wikipedia", "20231101.fr", "train")],
    "wiki_hi":    [("wikimedia/wikipedia", "20231101.hi", "train")],
}
FIELDS = {"code": "code", "arxiv": "article", "pubmed": "article"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=128, help="docs/domain (2x oversample of 64)")
    p.add_argument("--domains", default="all")
    p.add_argument("--upload", default=None, help="HF dataset repo to push the parquet to")
    args = p.parse_args()

    wanted = list(DOMAINS) + ["json_struct"] if args.domains == "all" \
        else args.domains.split(",")
    rows = []
    for domain in wanted:
        if domain == "json_struct":
            texts = _synthetic_json(args.n)
            src = "synthetic"
        else:
            transform = _chat_transform if domain == "chat" else None
            texts, src = _stream_texts(DOMAINS[domain], args.n,
                                       field=FIELDS.get(domain, "text"),
                                       transform=transform)
        print(f"[prep] {domain}: {len(texts)} docs from {src}")
        for i, t in enumerate(texts):
            rows.append({"domain": domain, "doc_id": f"{domain}/{i}", "text": t[:20000]})

    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"[prep] wrote {len(rows)} rows -> {args.out}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=open(f"{__import__('os').path.expanduser('~')}/.hf_token").read().strip())
        api.create_repo(args.upload, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=args.out, path_in_repo="domains.parquet",
                        repo_id=args.upload, repo_type="dataset")
        print(f"[prep] uploaded -> hf://datasets/{args.upload}/domains.parquet")
    print("PREP_DOMAINS_DONE")


if __name__ == "__main__":
    main()
