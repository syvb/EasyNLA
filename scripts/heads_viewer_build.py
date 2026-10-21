"""Build the data for the head-NLA viewer: every head of one layer at a few positions of held-out texts.

    # 1. sites: N eval-split docs × M of their sampled positions × all heads of --layer
    python scripts/heads_viewer_build.py select --head-dir <head dataset> --layer 24 --out-ds <viewer ds> \
        --results <dir> [--n-docs 8 --n-pos 4]
    #    then: heads_soft_token_readouts.py --output-dir <viewer ds> ...; heads_soft_token_explain.py ...
    # 2. an eval.parquet for heads_sft_checks.py (the exact v the models read; gold = the Haiku explanation)
    python scripts/heads_viewer_build.py eval --out-ds <viewer ds> --explain <explain.parquet> \
        --sft-dir <real sft dir> --out-sft <viewer sft dir>

<viewer ds> mirrors the head dataset (symlinks) with its own warmstart/selection.parquet. <results> gets
viewer_docs.parquet (token ids, domain) and viewer_attn.parquet (each head's top attended positions and
weights, self / first-token weight, entropy at each site): evaluation evidence only — the explanations
are still made from the write's readouts alone.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.datagen.heads.store import read_meta  # noqa: E402


def select(args: argparse.Namespace) -> None:
    head_dir, out_ds, res = Path(args.head_dir), Path(args.out_ds), Path(args.results)
    meta = read_meta(head_dir)
    sel = pq.read_table(head_dir / "warmstart" / "selection.parquet").to_pandas()
    eval_docs = set(sel.doc_id[sel.split == "eval"])
    docs = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(str(head_dir / "docs" / "*.parquet")))]).to_pandas()
    for c in ("domain", "url"):   # optional corpus metadata
        if c not in docs:
            docs[c] = ""
    cand = docs[docs.doc_id.isin(eval_docs) & docs.n_tokens.between(args.min_tokens, args.max_tokens)]
    cand = cand[[sum(p >= args.min_pos for p in ps) >= args.n_pos for ps in cand.positions]]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cand))
    picked, domains = [], set()
    for k in order:          # distinct domains first, then fill
        r = cand.iloc[k]
        if r["domain"] not in domains:
            picked.append(k); domains.add(r["domain"])
        if len(picked) == args.n_docs:
            break
    picked += [k for k in order if k not in picked][: args.n_docs - len(picked)]
    chosen = cand.iloc[picked]
    rows, k = [], 0
    for _, d in chosen.iterrows():
        ps = sorted(p for p in d.positions if p >= args.min_pos)
        for p in sorted({ps[int(round(i))] for i in np.linspace(0, len(ps) - 1, args.n_pos)}):
            for h in range(meta.n_heads):
                rows.append({"example_idx": args.idx_base + k, "split": "eval", "layer": args.layer, "head": h,
                             "rank": 0, "doc_id": d.doc_id, "pos": int(p), "dev_norm": 0.0, "dev_rel": 0.0})
                k += 1
    if out_ds.exists():
        shutil.rmtree(out_ds)
    out_ds.mkdir(parents=True)
    for e in head_dir.iterdir():
        if e.name != "warmstart":
            os.symlink(e.resolve(), out_ds / e.name)
    (out_ds / "warmstart").mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(head_dir / "warmstart" / "selection.parquet")),
                   out_ds / "warmstart" / "selection.parquet")
    res.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(chosen[["doc_id", "n_tokens", "token_ids", "domain", "url"]].reset_index(drop=True)),
                   res / "viewer_docs.parquet")
    # attention evidence at each site, for every head
    sites = {(r["doc_id"], r["pos"]) for r in rows}
    H, K = meta.n_heads, meta.top_k
    att = []
    for f in sorted(glob.glob(str(head_dir / "records" / f"L{args.layer:02d}" / "*.parquet"))):
        t = pq.read_table(f, columns=["doc_id", "pos", "attn_top_idx", "attn_top_w", "attn_first_w", "attn_self_w",
                                      "attn_entropy"]).to_pydict()
        for i, (d, p) in enumerate(zip(t["doc_id"], t["pos"])):
            if (d, p) not in sites:
                continue
            ti = np.frombuffer(t["attn_top_idx"][i], np.int32).reshape(H, K)
            tw = np.frombuffer(t["attn_top_w"][i], np.float32).reshape(H, K)
            fw, sw, en = (np.frombuffer(t[c][i], np.float32) for c in ("attn_first_w", "attn_self_w", "attn_entropy"))
            for h in range(H):
                att.append({"doc_id": d, "pos": int(p), "head": h, "att_pos": ti[h].tolist(), "att_w": tw[h].tolist(),
                            "first_w": float(fw[h]), "self_w": float(sw[h]), "entropy": float(en[h])})
    assert len(att) == len(rows), f"attention for {len(att)} of {len(rows)} sites — wrong records layer dir?"
    pq.write_table(pa.Table.from_pylist(att), res / "viewer_attn.parquet")
    print(f"{len(chosen)} docs ({sorted(chosen.domain)}), {len(sites)} positions, {len(rows)} head-sites")


def build_eval(args: argparse.Namespace) -> None:
    from nla.datagen.heads.export_sft import normalized_writes
    out_ds, out_sft, sft = Path(args.out_ds), Path(args.out_sft), Path(args.sft_dir)
    rows = pq.read_table(out_ds / "warmstart" / "selection.parquet").to_pylist()
    v = normalized_writes(out_ds, rows, args.cap_quantile, args.max_norm)
    ex = {}
    if args.explain and Path(args.explain).exists():
        e = pq.read_table(args.explain).to_pydict()
        ex = dict(zip(e["example_idx"], e["explain"], strict=True))
    proto = pq.read_table(sft / "eval.parquet", columns=["prompt"]).slice(0, 1).to_pylist()[0]["prompt"]
    ntok = {}
    for f in glob.glob(str(out_ds / "docs" / "*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "n_tokens"]).to_pydict()
        ntok.update(zip(t["doc_id"], t["n_tokens"]))
    desc = [ex.get(r["example_idx"], "") for r in rows]
    d_model = v.shape[1]
    tab = pa.table({
        "prompt": [proto] * len(rows),
        "response": [f"<explanation>\n{s}\n</explanation>" for s in desc],
        "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel(), pa.float32()), d_model),
        "description": desc,
        "pos": pa.array([r["pos"] for r in rows], pa.int32()),
        "dev_rel": pa.array(np.linalg.norm(v, axis=1), pa.float32()),
        "doc_id": [r["doc_id"] for r in rows],
        "n_raw_tokens": pa.array([int(ntok[r["doc_id"]]) for r in rows], pa.int64()),
        "activation_layer": pa.array([r["layer"] for r in rows], pa.int64()),
        "head": pa.array([r["head"] for r in rows], pa.int16()),
        "example_idx": pa.array([r["example_idx"] for r in rows], pa.int64()),
    })
    out_sft.mkdir(parents=True, exist_ok=True)
    pq.write_table(tab, out_sft / "eval.parquet")
    for name in ("av_sft.parquet", "ar_sft.parquet", "av_sft.parquet.nla_meta.yaml", "ar_sft.parquet.nla_meta.yaml"):
        dst = out_sft / name
        if not dst.exists():
            os.symlink((sft / name).resolve(), dst)
    shutil.copy(sft / "eval.parquet.nla_meta.yaml", out_sft / "eval.parquet.nla_meta.yaml")
    print(f"wrote {out_sft / 'eval.parquet'}: {len(rows)} rows, {sum(bool(s) for s in desc)} with gold explanations")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--head-dir", required=True)
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--out-ds", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--n-docs", type=int, default=8)
    s.add_argument("--n-pos", type=int, default=4)
    s.add_argument("--min-pos", type=int, default=16)
    s.add_argument("--min-tokens", type=int, default=150)
    s.add_argument("--max-tokens", type=int, default=700)
    s.add_argument("--idx-base", type=int, default=900_000)
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--out-ds", required=True)
    e.add_argument("--explain", default=None)
    e.add_argument("--sft-dir", required=True)
    e.add_argument("--out-sft", required=True)
    e.add_argument("--cap-quantile", type=float, default=0.999)
    e.add_argument("--max-norm", type=float, default=4.0)
    args = p.parse_args()
    select(args) if args.cmd == "select" else build_eval(args)


if __name__ == "__main__":
    main()
