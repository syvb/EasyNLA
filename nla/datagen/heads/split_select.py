"""Head stage `select`: doc-level split → splits.parquet; warm-start picks → warmstart/selection.parquet.

Split at the DOCUMENT level (as stage1_split does): av_sft / ar_sft / rl /
eval. The verbalizer and reconstructor warm-starts then draw from disjoint
documents, and RL + the pre-RL checks never see a warm-start document.

Warm-start selection is spread EVENLY across heads: every (layer, head) gets
exactly `per_head_av` examples from av_sft docs and `per_head_ar` from ar_sft
docs (at most one per document per head), plus `per_head_eval` from eval docs
when set — held-out examples for the pre-RL checks, described the same way — and
`per_head_rl` RL prompts from rl docs when set (no description needed; appended after
every other row so the warm-start rows keep their example_idx). Within a head, records are drawn
without replacement with weight

    w = (1 − size_weight) + size_weight · min(dev_norm / dev_rms, size_cap)

so larger-than-typical writes are sampled more often but small ones still
appear (size_weight=0 is uniform). Weighted sampling without replacement
uses Efraimidis–Spirakis keys log(u)/w, seeded per (seed, layer, head, split).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.heads.store import (
    SPLITS_NAME,
    atomic_write_table,
    devnorms_dir,
    fixed_numpy,
    load_stats,
    part_files,
    read_docs,
    read_meta,
)

SPLITS = ("av_sft", "ar_sft", "rl", "eval")
_SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}


def split_docs(doc_ids: list[str], fracs: dict[str, float], seed: int) -> dict[str, str]:
    total = sum(fracs.values())
    assert abs(total - 1.0) < 1e-6 and all(f >= 0 for f in fracs.values()), f"split fractions {fracs} must sum to 1"
    ids = sorted(set(doc_ids))  # sorted → shuffle is reproducible across Python versions
    random.Random(seed).shuffle(ids)
    out: dict[str, str] = {}
    start = 0
    for k, s in enumerate(SPLITS):
        end = len(ids) if k == len(SPLITS) - 1 else start + int(len(ids) * fracs[s])
        out.update((d, s) for d in ids[start:end])
        start = end
    return out


def weighted_pick(dev_rel: np.ndarray, docs: np.ndarray, n: int, size_weight: float, size_cap: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Indices of up to n rows, weighted toward large dev_rel, one per doc."""
    w = (1.0 - size_weight) + size_weight * np.minimum(dev_rel, size_cap)
    keys = np.log(rng.random(len(w))) / np.maximum(w, 1e-12)
    picked: list[int] = []
    seen: set = set()
    for i in np.argsort(-keys, kind="stable"):
        if docs[i] in seen:
            continue
        seen.add(docs[i])
        picked.append(int(i))
        if len(picked) == n:
            break
    return np.array(picked, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--av-sft-frac", type=float, default=0.2)
    p.add_argument("--ar-sft-frac", type=float, default=0.2)
    p.add_argument("--rl-frac", type=float, default=0.55)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--per-head-av", type=int, default=200)
    p.add_argument("--per-head-ar", type=int, default=200)
    p.add_argument("--per-head-eval", type=int, default=0, help="held-out examples per head (eval docs)")
    p.add_argument("--per-head-rl", type=int, default=0,
                   help="RL prompts per head (rl docs); appended after every other row, so adding them "
                        "leaves the example_idx of the av_sft / ar_sft / eval rows unchanged")
    p.add_argument("--size-weight", type=float, default=0.75)
    p.add_argument("--size-cap", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert 0.0 <= args.size_weight <= 1.0, "--size-weight must be in [0, 1]"
    out_dir = Path(args.output_dir)
    meta = read_meta(out_dir)
    stats = load_stats(out_dir)
    h = meta.n_heads

    docs = read_docs(out_dir, columns=["doc_id"])
    split_of = split_docs(docs.column("doc_id").to_pylist(), {
        "av_sft": args.av_sft_frac, "ar_sft": args.ar_sft_frac, "rl": args.rl_frac, "eval": args.eval_frac,
    }, args.split_seed)
    split_ids = sorted(split_of)
    atomic_write_table(pa.table({"doc_id": split_ids, "split": [split_of[d] for d in split_ids]}),
                       out_dir / SPLITS_NAME)
    for s in SPLITS:
        print(f"  {s}: {sum(v == s for v in split_of.values())} docs")

    wanted = {"av_sft": args.per_head_av, "ar_sft": args.per_head_ar}
    if args.per_head_eval:
        wanted["eval"] = args.per_head_eval
    cols: dict[str, list] = {k: [] for k in
                             ("split", "layer", "head", "rank", "doc_id", "pos", "dev_norm", "dev_rel")}
    shortfalls = 0
    # Two passes: the warm-start splits first (their order, hence example_idx, is the same as
    # before --per-head-rl existed), then the RL prompts after every layer's warm-start rows.
    passes = [wanted] + ([{"rl": args.per_head_rl}] if args.per_head_rl else [])
    for want, li, layer in [(w, li, layer) for w in passes for li, layer in enumerate(stats.layers)]:
        dn = pa.concat_tables([pq.read_table(f, columns=["doc_id", "pos", "dev_norm"])
                               for f in part_files(devnorms_dir(out_dir, layer))])
        doc_arr = np.array(dn.column("doc_id").to_pylist(), dtype=object)
        pos_arr = dn.column("pos").to_numpy()
        dev = fixed_numpy(dn.column("dev_norm"), np.float32, h)
        row_split = np.array([split_of[d] for d in doc_arr], dtype=object)
        for split, n_per in want.items():
            idx = np.nonzero(row_split == split)[0]
            for head in range(h):
                rms = float(stats.dev_rms[li, head])
                rel = dev[idx, head] / max(rms, 1e-12)
                rng = np.random.default_rng([args.seed, layer, head, _SPLIT_CODE[split]])
                pick = idx[weighted_pick(rel, doc_arr[idx], n_per, args.size_weight, args.size_cap, rng)]
                shortfalls += n_per - len(pick)
                cols["split"] += [split] * len(pick)
                cols["layer"] += [layer] * len(pick)
                cols["head"] += [head] * len(pick)
                cols["rank"] += list(range(len(pick)))
                cols["doc_id"] += doc_arr[pick].tolist()
                cols["pos"] += pos_arr[pick].tolist()
                cols["dev_norm"] += dev[pick, head].tolist()
                cols["dev_rel"] += (dev[pick, head] / max(rms, 1e-12)).tolist()

    sel = pa.table({
        "example_idx": np.arange(len(cols["split"]), dtype=np.int64),
        "split": pa.array(cols["split"], type=pa.string()),
        "layer": pa.array(cols["layer"], type=pa.int16()),
        "head": pa.array(cols["head"], type=pa.int16()),
        "rank": pa.array(cols["rank"], type=pa.int32()),
        "doc_id": pa.array(cols["doc_id"], type=pa.string()),
        "pos": pa.array(cols["pos"], type=pa.int32()),
        "dev_norm": pa.array(cols["dev_norm"], type=pa.float32()),
        "dev_rel": pa.array(cols["dev_rel"], type=pa.float32()),
    })
    atomic_write_table(sel, out_dir / "warmstart" / "selection.parquet")
    n_heads_total = len(stats.layers) * h
    print(f"selected {sel.num_rows} warm-start examples over {n_heads_total} heads "
          f"({' + '.join(f'{n} {s}' for w in passes for s, n in w.items())} per head requested)")
    if shortfalls:
        print(f"  WARNING: {shortfalls} picks short — some heads have fewer distinct docs than requested")
    rel = sel.column("dev_rel").to_numpy()
    print(f"  selected dev/rms: median {np.median(rel):.2f}  mean {rel.mean():.2f} "
          f"(population mean of (dev/rms)² is 1 by construction)")


if __name__ == "__main__":
    main()
