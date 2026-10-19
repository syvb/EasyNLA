"""Score soft-token readout recipes (nla.datagen.heads.soft_token) as warm-start descriptions.

    python scripts/heads_soft_token_sweep.py --output-dir <head dataset> --work-dir <dir> \
        --prompts identity,meaning,assoc --betas 0.5,1,2 --readouts greedy,topk

Heads are drawn at random across the dataset's layers and examples by write size
(split_select's deviation weighting) — no attention-pattern information is read anywhere.
Every recipe is scored by the cross-fitted description → write probe (nla.datagen.heads.probe),
per head then averaged, with TF-IDF and LM-embedding features, against two controls:
"empty" (FVE ≈ 0) and "<recipe>_shuf" (descriptions permuted within each head). A recipe
that beats its _shuf carries information about the example's write.

Generations are cached per recipe in --work-dir, so the sweep can be extended and rerun.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.datagen._common import load_tokenizer  # noqa: E402
from nla.datagen.heads import soft_token  # noqa: E402
from nla.datagen.heads.probe import crossfit, lm_embedder  # noqa: E402
from nla.datagen.heads.self_describe import head_write_deviations  # noqa: E402
from nla.datagen.heads.split_select import weighted_pick  # noqa: E402
from nla.datagen.heads.store import (  # noqa: E402
    atomic_write_table,
    devnorms_dir,
    fixed_numpy,
    load_stats,
    from_bf16_bits,
    load_wo,
    part_files,
    read_docs,
    read_meta,
    read_records,
)


def pick_heads(meta, n_heads: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """n_heads spread round-robin over the layers, uniformly at random within each."""
    per_layer = {layer: list(rng.permutation(meta.n_heads)) for layer in meta.layers}
    heads: list[tuple[int, int]] = []
    while len(heads) < n_heads:
        for layer in meta.layers:
            if per_layer[layer] and len(heads) < n_heads:
                heads.append((layer, int(per_layer[layer].pop())))
    return heads


def pick_examples(out_dir: Path, heads: list[tuple[int, int]], per_head: int, seed: int) -> list[dict]:
    stats, meta = load_stats(out_dir), read_meta(out_dir)
    rows: list[dict] = []
    for layer, head in heads:
        dn = pa.concat_tables([pq.read_table(f, columns=["doc_id", "pos", "dev_norm"])
                               for f in part_files(devnorms_dir(out_dir, layer))])
        docs = np.array(dn.column("doc_id").to_pylist(), dtype=object)
        rel = fixed_numpy(dn.column("dev_norm"), np.float32, meta.n_heads)[:, head]
        rel = rel / float(stats.dev_rms[stats.layer_pos(layer), head])
        pick = weighted_pick(rel, docs, per_head, 0.75, 4.0, np.random.default_rng([seed, layer, head]))
        pos = dn.column("pos").to_numpy()
        rows += [{"layer": layer, "head": head, "doc_id": docs[i], "pos": int(pos[i]), "dev_rel": float(rel[i])}
                 for i in pick]
    for i, r in enumerate(rows):
        r["example_idx"] = i
    return rows


def normalized(out_dir: Path, rows: list[dict], max_norm: float) -> torch.Tensor:
    """(w − w̄) / dev_rms_h, capped at max_norm — export_sft's normalization with a fixed cap."""
    stats = load_stats(out_dir)
    v = head_write_deviations(out_dir, rows)
    rms = torch.tensor([float(stats.dev_rms[stats.layer_pos(r["layer"]), r["head"]]) for r in rows])
    v = v / rms[:, None]
    n = v.norm(dim=-1, keepdim=True)
    return v * torch.clamp(max_norm / n.clamp_min(1e-12), max=1.0)


def mean_resid(out_dir: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    """Average residual entering each layer over all its records (a constant per layer)."""
    d = read_meta(out_dir).d_model
    return {layer: from_bf16_bits(fixed_numpy(read_records(out_dir, layer, columns=["resid_pre"])
                                              .column("resid_pre"), np.uint16, d)).float().mean(0)
            for layer in layers}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--per-head", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-norm", type=float, default=4.0)
    p.add_argument("--prompts", default="identity,meaning,assoc")
    p.add_argument("--betas", default="0.5,1,2")
    p.add_argument("--readouts", default="greedy")
    p.add_argument("--sites", default="own", help="comma list of 'own' / 'L<k>'")
    p.add_argument("--carriers", default="X", help="comma list: a word w puts the placeholder ' w'; 'mean' "
                                                    "puts ' X' with its residual replaced by the layer mean")
    p.add_argument("--model", default=None, help="default: the dataset's base model")
    p.add_argument("--embed-model", default=None, help="probe featurizer LM; default the dataset's base model")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-show", type=int, default=12)
    p.add_argument("--legacy", action="store_true", help="also score self_describe's l1_add (the earlier "
                                                         "instruction-prompt self-description) on these examples")
    args = p.parse_args()

    out_dir, work = Path(args.output_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out_dir)

    ex_path = work / "examples.parquet"
    if ex_path.exists():
        rows = pq.read_table(ex_path).to_pylist()
    else:
        heads = pick_heads(meta, args.n_heads, np.random.default_rng(args.seed))
        rows = pick_examples(out_dir, heads, args.per_head, args.seed)
        atomic_write_table(pa.Table.from_pylist(rows), ex_path)
    heads = sorted({(r["layer"], r["head"]) for r in rows})
    print(f"{len(rows)} examples over heads {[f'L{lay}H{h}' for lay, h in heads]}")
    vecs = normalized(out_dir, rows, args.max_norm)

    carriers = [(" X", "mean") if c == "mean" else (f" {c}", "token") for c in args.carriers.split(",")]
    recipes = [r for site in args.sites.split(",") for carrier, state in carriers
               for r in soft_token.recipe_grid(args.prompts.split(","), [float(b) for b in args.betas.split(",")],
                                               args.readouts.split(","), carrier, site, state)]
    assert all(r.site == "own" for r in recipes if r.carrier_state == "mean"), "mean carrier needs --sites own"
    sources: dict[str, list[str]] = {"empty": [""] * len(rows)}
    describer = None
    for rec in recipes:
        path = work / f"{rec.name}.parquet"
        if not path.exists():
            if describer is None:
                means = (mean_resid(out_dir, sorted({lay for lay, _ in heads}))
                         if any(r.carrier_state == "mean" for r in recipes) else None)
                describer = soft_token.SoftTokenDescriber(args.model or meta.base_model, args.device_map,
                                                          args.dtype, args.batch_size, means)
            texts = describer.describe(rec, vecs, [r["layer"] for r in rows])
            atomic_write_table(pa.table({"example_idx": list(range(len(rows))), "description": texts}), path)
            print(f"{rec.name}: {len(set(texts)) / len(texts):.0%} distinct, e.g. {texts[:3]}")
        sources[rec.name] = pq.read_table(path).column("description").to_pylist()
    if describer is not None:   # free it before the next model loads
        del describer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.legacy:
        path = work / "legacy_l1_add.parquet"
        if not path.exists():
            from nla.datagen.heads.self_describe import SelfDescriber, clean_generation
            sd = SelfDescriber(args.model or meta.base_model, args.device_map, args.dtype,
                               batch_size=args.batch_size)
            raw = sd.describe(vecs, [r["layer"] for r in rows], "l1_add")
            atomic_write_table(pa.table({"example_idx": list(range(len(rows))),
                                         "description": [clean_generation(t) for t in raw]}), path)
            del sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sources["legacy_l1_add"] = pq.read_table(path).column("description").to_pylist()

    by_head: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_head[(r["layer"], r["head"])].append(i)
    # within-head permutation: example a gets the description of example perm[a]
    perm = np.arange(len(rows))
    for idxs in by_head.values():
        perm[idxs] = np.random.default_rng([args.seed, len(idxs)]).permutation(idxs)
    base_names = [s for s in sources if s != "empty"]
    for name in base_names:
        sources[f"{name}_shuf"] = [sources[name][b] for b in perm]

    # targets: normalized write deviations in each head's write subspace (orthonormal basis)
    d = meta.head_dim
    y_of = {}
    for (layer, head), idxs in by_head.items():
        u = torch.linalg.svd(load_wo(out_dir, layer)[:, head * d:(head + 1) * d].float(),
                             full_matrices=False).U
        y_of[(layer, head)] = (vecs[idxs] @ u).numpy()

    embed = lm_embedder(args.embed_model or meta.base_model, args.device_map)
    res_path = work / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    emb_of: dict[str, np.ndarray] = {}
    for name, texts in sources.items():
        if name in results:
            continue
        base = name.removesuffix("_shuf")
        if base not in emb_of:
            emb_of[base] = embed(sources[base])
        emb = emb_of[base][perm] if name.endswith("_shuf") else emb_of[base]
        r = {"tfidf": {}, "embed": {}}
        for (layer, head), idxs in by_head.items():
            hd = f"L{layer}H{head}"
            r["tfidf"][hd] = crossfit([texts[i] for i in idxs], y_of[(layer, head)], seed=args.seed)[0]
            r["embed"][hd] = crossfit(emb[idxs], y_of[(layer, head)], seed=args.seed)[0]
        r["unique"] = len(set(texts)) / len(texts)
        results[name] = r
        res_path.write_text(json.dumps(results, indent=1))

    def mean(name: str, key: str) -> float:
        return float(np.mean(list(results[name][key].values())))

    names = ["empty"] + [rec.name for rec in recipes] + (["legacy_l1_add"] if args.legacy else [])
    print(f"\nheld-out FVE of the write from the description, mean over {len(heads)} heads "
          f"(gain = recipe − its within-head shuffle)")
    print(f"{'source':<28}{'tfidf':>7}{'gain':>7}{'embed':>7}{'gain':>7}{'uniq':>6}")
    for name in names:
        sh = f"{name}_shuf" if name != "empty" else name
        print(f"{name:<28}{mean(name, 'tfidf'):>7.3f}{mean(name, 'tfidf') - mean(sh, 'tfidf'):>7.3f}"
              f"{mean(name, 'embed'):>7.3f}{mean(name, 'embed') - mean(sh, 'embed'):>7.3f}"
              f"{results[name]['unique']:>6.2f}")

    # human-readable examples, with the text context for judging (the recipes never saw it)
    tok = load_tokenizer(meta.base_model)
    docs = read_docs(out_dir, columns=["doc_id", "token_ids"])
    doc_row = {dd: i for i, dd in enumerate(docs.column("doc_id").to_pylist())}
    show = np.random.default_rng(args.seed + 1).choice(len(rows), size=min(args.n_show, len(rows)), replace=False)
    lines = ["# Soft-token readouts: examples", "",
             "Context is shown for judging only; every readout saw just the head's write.", ""]
    for i in sorted(show, key=lambda i: (rows[i]["layer"], rows[i]["head"])):
        r = rows[i]
        ids = docs.column("token_ids")[doc_row[r["doc_id"]]].values.to_numpy()
        before = tok.decode(ids[max(0, r["pos"] - 40): r["pos"]].tolist())
        cur = tok.decode(ids[r["pos"]: r["pos"] + 1].tolist())
        lines += [f"## L{r['layer']}H{r['head']} · pos {r['pos']} · ‖v‖ {float(vecs[i].norm()):.2f}", "",
                  f"> …{before}⟦{cur}⟧".replace("\n", "⏎"), ""]
        lines += [f"- **{n}:** {sources[n][i]}" for n in names[1:]] + [""]
    (work / "examples.md").write_text("\n".join(lines))
    print(f"\nexamples → {work / 'examples.md'}")


if __name__ == "__main__":
    main()
