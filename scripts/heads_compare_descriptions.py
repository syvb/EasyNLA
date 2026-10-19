"""Compare warm-start description sources by how well they predict the head's write.

    python scripts/heads_compare_descriptions.py --output-dir <head dataset> --work-dir <dir> [--claude]

Picks --n-heads heads spread over the dataset's layers (skipping heads whose median
first-token attention is ≥ --exclude-sink: their descriptions would all be "little or
nothing"), --per-head examples each (deviation-weighted, as split_select), and scores
every source with the same cross-fitted probe (nla.datagen.heads.probe):

  empty           no description — the floor (FVE ≈ 0)
  template        mechanical: top attended tokens + weights, no LM
  claude          Haiku 5.5 on the warm-start prompt (PAID: only with --claude; cached)
  self_<variant>  the base model describing its own injected head write (self_describe)
  <source>_shuf   descriptions permuted within each head — the "another example's write"
                  control; a source that beats its _shuf is tracking the example

FVE is per head, then averaged. Two featurizers: TF-IDF (literal words) and the
embedding model's mean-pooled hidden states (meaning).
"""

import argparse
import asyncio
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
from nla.datagen.heads import self_describe  # noqa: E402
from nla.datagen.heads.batch_api import request_params  # noqa: E402
from nla.datagen.heads.probe import crossfit, lm_embedder, write_coords  # noqa: E402
from nla.datagen.heads.prompts import PromptConfig, parse_response, render_user_prompt  # noqa: E402
from nla.datagen.heads.split_select import weighted_pick  # noqa: E402
from nla.datagen.heads.store import (  # noqa: E402
    atomic_write_table,
    devnorms_dir,
    fixed_numpy,
    from_bf16_bits,
    load_stats,
    load_wo,
    part_files,
    read_docs,
    read_meta,
    read_records,
)


def pick_heads(out_dir: Path, n_heads: int, exclude_sink: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    meta = read_meta(out_dir)
    per_layer: dict[int, list[int]] = {}
    for layer in meta.layers:
        first = fixed_numpy(read_records(out_dir, layer, columns=["attn_first_w"]).column("attn_first_w"),
                            np.float32, meta.n_heads)
        per_layer[layer] = [h for h in range(meta.n_heads) if np.median(first[:, h]) < exclude_sink]
        rng.shuffle(per_layer[layer])
    heads: list[tuple[int, int]] = []
    while len(heads) < n_heads and any(per_layer.values()):
        for layer in meta.layers:   # round-robin so layers are evenly covered
            if per_layer[layer] and len(heads) < n_heads:
                heads.append((layer, per_layer[layer].pop()))
    return heads


def pick_examples(out_dir: Path, heads: list[tuple[int, int]], per_head: int, seed: int) -> list[dict]:
    stats = load_stats(out_dir)
    meta = read_meta(out_dir)
    rows: list[dict] = []
    for layer, head in heads:
        dn = pa.concat_tables([pq.read_table(f, columns=["doc_id", "pos", "dev_norm"])
                               for f in part_files(devnorms_dir(out_dir, layer))])
        docs = np.array(dn.column("doc_id").to_pylist(), dtype=object)
        dev = fixed_numpy(dn.column("dev_norm"), np.float32, meta.n_heads)[:, head]
        rel = dev / float(stats.dev_rms[stats.layer_pos(layer), head])
        pick = weighted_pick(rel, docs, per_head, 0.75, 4.0, np.random.default_rng([seed, layer, head]))
        pos = dn.column("pos").to_numpy()
        rows += [{"layer": layer, "head": head, "doc_id": docs[i], "pos": int(pos[i]), "dev_rel": float(rel[i])}
                 for i in pick]
    for i, r in enumerate(rows):
        r["example_idx"] = i
    return rows


def attention_rows(out_dir: Path, rows: list[dict]) -> list[dict]:
    """top_idx / top_w / first_w / self_w for each example's head."""
    meta = read_meta(out_dir)
    h, k = meta.n_heads, meta.top_k
    out: list[dict] = [{} for _ in rows]
    by_layer: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_layer[r["layer"]].append(i)
    for layer, idxs in by_layer.items():
        rec = read_records(out_dir, layer, columns=["doc_id", "pos", "attn_top_idx", "attn_top_w",
                                                    "attn_first_w", "attn_self_w"])
        row_of = {(d, p): j for j, (d, p) in enumerate(zip(rec.column("doc_id").to_pylist(),
                                                           rec.column("pos").to_pylist(), strict=True))}
        ti = fixed_numpy(rec.column("attn_top_idx"), np.int32, h * k).reshape(-1, h, k)
        tw = fixed_numpy(rec.column("attn_top_w"), np.float32, h * k).reshape(-1, h, k)
        fw = fixed_numpy(rec.column("attn_first_w"), np.float32, h)
        sw = fixed_numpy(rec.column("attn_self_w"), np.float32, h)
        for i in idxs:
            j, hd = row_of[(rows[i]["doc_id"], rows[i]["pos"])], rows[i]["head"]
            out[i] = {"top_idx": ti[j, hd], "top_w": tw[j, hd], "first_w": float(fw[j, hd]),
                      "self_w": float(sw[j, hd])}
    return out


def template_description(decode, ids, pos: int, att: dict, cfg: PromptConfig) -> str:
    parts = []
    for n, (src, w) in enumerate(zip(att["top_idx"], att["top_w"], strict=True)):
        if n >= cfg.max_sources or (n > 0 and w < cfg.min_source_weight):
            break
        src = int(src)
        tok = decode(ids[src : src + 1]).strip() or "⏎"
        label = ("first token" if src == 0 else f"current token '{tok}'" if src == pos
                 else f"previous token '{tok}'" if src == pos - 1 else f"'{tok}'")
        parts.append(f"{label} {w:.2f}")
    return "; ".join(parts)


async def claude_descriptions(prompts: dict[int, str], cache_path: Path, concurrency: int) -> dict[int, dict]:
    import anthropic
    done: dict[int, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            done[r["example_idx"]] = r
    todo = [i for i in prompts if i not in done]
    if not todo:
        return done
    client = anthropic.AsyncAnthropic(max_retries=8)
    cfg = {"model": "claude-haiku-5-5", "max_tokens": 1024, "effort": "low", "thinking": "adaptive",
           "cache_system": True}
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> dict:
        async with sem:
            try:
                msg = await client.messages.create(**request_params(prompts[i], cfg))
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:   # retries exhausted
                return {"example_idx": i, "stop_reason": "errored", "text": "", "error": str(e)[:300],
                        "usage": [0, 0, 0, 0]}
        u = msg.usage
        return {"example_idx": i, "stop_reason": msg.stop_reason,
                "text": "".join(b.text for b in msg.content if b.type == "text"),
                "usage": [u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                          u.output_tokens]}

    with cache_path.open("a") as f:
        for coro in asyncio.as_completed([one(i) for i in todo]):
            r = await coro
            done[r["example_idx"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--per-head", type=int, default=150)
    p.add_argument("--exclude-sink", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--claude", action="store_true", help="get Haiku 5.5 descriptions (PAID, ~$0.15 per 1k)")
    p.add_argument("--claude-concurrency", type=int, default=16)
    p.add_argument("--variants", default=",".join(self_describe.VARIANTS))
    p.add_argument("--style", default="auto", choices=["auto", "chat", "completion"])
    p.add_argument("--model", default=None, help="self-description model; default the dataset's base model")
    p.add_argument("--embed-model", default=None, help="featurizer LM; default the dataset's base model")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    out_dir, work = Path(args.output_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out_dir)
    model_name = args.model or meta.base_model
    rng = np.random.default_rng(args.seed)

    ex_path = work / "examples.parquet"
    if ex_path.exists():
        rows = pq.read_table(ex_path).to_pylist()
    else:
        heads = pick_heads(out_dir, args.n_heads, args.exclude_sink, rng)
        rows = pick_examples(out_dir, heads, args.per_head, args.seed)
        atomic_write_table(pa.Table.from_pylist(rows), ex_path)
    heads = sorted({(r["layer"], r["head"]) for r in rows})
    print(f"{len(rows)} examples over heads {[f'L{lay}H{h}' for lay, h in heads]}")

    tok = load_tokenizer(meta.base_model)
    def decode(ids):
        return tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    docs = read_docs(out_dir, columns=["doc_id", "token_ids"])
    doc_row = {d: i for i, d in enumerate(docs.column("doc_id").to_pylist())}
    att = attention_rows(out_dir, rows)
    cfg = PromptConfig()
    ids_of = [docs.column("token_ids")[doc_row[r["doc_id"]]].values.to_numpy() for r in rows]

    sources: dict[str, list[str]] = {"empty": [""] * len(rows)}
    sources["template"] = [template_description(decode, ids_of[i], r["pos"], att[i], cfg) for i, r in enumerate(rows)]

    if args.claude:
        prompts = {i: render_user_prompt(decode, ids_of[i], r["pos"], r["layer"], r["head"], meta.n_layers,
                                         meta.n_heads, att[i]["top_idx"], att[i]["top_w"], att[i]["first_w"],
                                         att[i]["self_w"], cfg) for i, r in enumerate(rows)}
        got = asyncio.run(claude_descriptions(prompts, work / "claude.jsonl", args.claude_concurrency))
        sources["claude"] = [parse_response(got[i]["text"], prompts[i]).description or "" for i in range(len(rows))]
        u = np.array([got[i]["usage"] for i in range(len(rows))]).sum(0)
        usd = (u[0] * 0.10 + u[1] * 0.01 + u[2] * 0.125 + u[3] * 0.50) / 1e6
        print(f"claude: {u[0]:,} in + {u[1]:,} cache-read + {u[2]:,} cache-write + {u[3]:,} out ≈ ${usd:.3f}")
    elif (work / "claude.jsonl").exists():
        print("note: claude.jsonl exists but --claude not set; skipping the claude source")

    vecs = None
    for variant in [v for v in args.variants.split(",") if v]:
        path = work / f"self_{variant}.parquet"
        if not path.exists():
            if vecs is None:
                vecs = self_describe.head_write_deviations(out_dir, rows)
                describer = self_describe.SelfDescriber(model_name, args.device_map, style=args.style,
                                                        batch_size=args.batch_size)
                print(f"self-description prompt ({describer.style}):\n{describer.prompt_text}\n")
            raw = describer.describe(vecs, [r["layer"] for r in rows], variant)
            atomic_write_table(pa.table({"example_idx": list(range(len(rows))), "raw_text": raw,
                                         "description": [self_describe.clean_generation(t) for t in raw]}), path)
        sources[f"self_{variant}"] = pq.read_table(path).column("description").to_pylist()
    if vecs is not None:   # free the describer before the embedder loads a second copy of the model
        del describer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # within-head permutation controls
    by_head: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_head[(r["layer"], r["head"])].append(i)
    for name in [s for s in sources if s not in ("empty",)]:
        shuf = list(sources[name])
        for idxs in by_head.values():
            perm = np.random.default_rng([args.seed, len(idxs)]).permutation(idxs)
            for a, b in zip(idxs, perm, strict=True):
                shuf[a] = sources[name][b]
        sources[f"{name}_shuf"] = shuf

    # targets: write deviations in each head's write-space coordinates
    stats = load_stats(out_dir)
    y_of: dict[tuple[int, int], np.ndarray] = {}
    for (layer, head), idxs in by_head.items():
        rec = read_records(out_dir, layer, columns=["doc_id", "pos", "head_out"])
        row_of = {(d, q): j for j, (d, q) in enumerate(zip(rec.column("doc_id").to_pylist(),
                                                           rec.column("pos").to_pylist(), strict=True))}
        take = pa.array([row_of[(rows[i]["doc_id"], rows[i]["pos"])] for i in idxs])
        z = from_bf16_bits(fixed_numpy(rec.column("head_out").take(take), np.uint16, meta.n_heads * meta.head_dim))
        z = z.view(-1, meta.n_heads, meta.head_dim)[:, head]
        z_dev = z - stats.mean_head_out[stats.layer_pos(layer), head]
        d = meta.head_dim
        y_of[(layer, head)] = write_coords(z_dev, load_wo(out_dir, layer)[:, head * d:(head + 1) * d])

    embed = lm_embedder(args.embed_model or meta.base_model, args.device_map)
    results: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for name, texts in sources.items():
        emb = embed(texts)
        for (layer, head), idxs in by_head.items():
            y = y_of[(layer, head)]
            hd = f"L{layer}H{head}"
            results[name]["tfidf"][hd] = crossfit([texts[i] for i in idxs], y, seed=args.seed)[0]
            results[name]["embed"][hd] = crossfit(emb[idxs], y, seed=args.seed)[0]
        results[name]["unique"]["all"] = len(set(texts)) / len(texts)

    (work / "results.json").write_text(json.dumps(results, indent=1))
    head_names = [f"L{lay}H{h}" for lay, h in heads]
    print(f"\nheld-out FVE of the write from the description (mean over {len(heads)} heads; per head below)")
    print(f"{'source':<22}{'tfidf':>8}{'embed':>8}{'unique':>8}   " + " ".join(f"{h:>7}" for h in head_names))
    for name in sources:
        r = results[name]
        print(f"{name:<22}{np.mean(list(r['tfidf'].values())):>8.3f}{np.mean(list(r['embed'].values())):>8.3f}"
              f"{r['unique']['all']:>8.2f}   " + " ".join(f"{r['tfidf'][h]:>7.3f}" for h in head_names))
    print("(per-head columns are tfidf)")

    show = np.random.default_rng(args.seed + 1).choice(len(rows), size=min(6, len(rows)), replace=False)
    for i in show:
        r = rows[i]
        tail = decode(ids_of[i][max(0, r["pos"] - 15) : r["pos"]])
        cur = decode(ids_of[i][r["pos"] : r["pos"] + 1])
        print(f"\n--- L{r['layer']}H{r['head']} dev/rms={r['dev_rel']:.2f}  …{tail!r}⟦{cur!r}⟧")
        for name in sources:
            if not name.endswith("_shuf") and name != "empty":
                print(f"  {name:<18} {sources[name][i]}")


if __name__ == "__main__":
    main()
