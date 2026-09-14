"""Future-lens activation collection — replaces datagen stages 0-3.

One forward per document over the frozen target. For sampled positions t whose
top-1 next-token prediction is correct (Future Lens's filter; keeps "the future"
well defined), record the residual at every requested layer, the next `n_future`
ground-truth tokens, the target's teacher-forced top-5 / log-probs at each of
those offsets, its confidence at t, and the `n_prev` preceding tokens (leakage
analysis only). Eval-split positions also get the target's greedy continuation.

Outputs in --out-dir:
    train.parquet (+ .nla_meta.yaml)   one row per (position, layer)
    eval.parquet  (+ .nla_meta.yaml)   disjoint documents
    docs.parquet                       token ids of every document used
    collect_stats.json                 discard fraction, norm quantiles, counts

Layer convention matches nla/datagen/extractors.py: `layer_index=K` is the output
of decoder block K (HF `hidden_states[K+1]`).

Corpus: an HF dataset id (streamed), a local .parquet with a text column, or a
local .jsonl with a `text` field or a pre-tokenised `ids` field (CPU smoke runs).

    python -m nla.future_lens.collect --base-ckpt Qwen/Qwen3-8B \
        --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
        --n-train-docs 10000 --n-eval-docs 500 --layers 4,8,12,16,20,24 \
        --out-dir /data/fl/qwen3_8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, serialize_sidecar
from nla.future_lens.data import (
    DEFAULT_K_CHOICES, DEFAULT_N_FUTURE, DEFAULT_N_PREV, DEFAULT_TEMPLATE, FL_SIDECAR_KEY,
    FLMeta, build_prompt_messages, docs_schema, fill_template, fl_schema,
)
from nla.schema import sidecar_path_for
from nla.utils.arch_adapters import resolve_text_config

QUANTILES = {"p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}


# ----------------------------------------------------------------------------
# Corpus iteration
# ----------------------------------------------------------------------------

def iter_corpus(args, tokenizer):
    """Yield (doc_id, token_ids list) for documents long enough to use."""
    min_len = args.min_pos + args.n_future + 2
    src = args.corpus

    def _tok(text):
        return tokenizer.encode(text, add_special_tokens=False)[: args.max_len]

    if src.endswith((".jsonl", ".parquet")) and not Path(src).exists():
        raise FileNotFoundError(f"--corpus {src!r} looks like a local file but does not exist")
    if src.endswith(".jsonl"):
        with open(src) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                ids = d.get("ids") if d.get("ids") else (_tok(d["text"]) if d.get("text") else None)
                if ids is None:
                    continue
                ids = list(ids)[: args.max_len]
                if len(ids) >= min_len:
                    yield f"{src}:{d.get('doc_id', i)}", ids
    elif src.endswith(".parquet"):
        pf = pq.ParquetFile(src)
        n = 0
        for batch in pf.iter_batches(batch_size=256, columns=[args.text_column]):
            for text in batch.column(args.text_column).to_pylist():
                ids = _tok(text or "")
                if len(ids) >= min_len:
                    yield f"{src}:{n}", ids
                n += 1
    else:
        from datasets import load_dataset
        ds = load_dataset(src, name=args.corpus_config, split=args.corpus_split, streaming=True)
        if args.corpus_start:
            ds = ds.skip(args.corpus_start)
        for i, ex in enumerate(ds, start=args.corpus_start):
            ids = _tok(ex[args.text_column] or "")
            if len(ids) >= min_len:
                yield f"{src}:{args.corpus_split}:{i}", ids


def sample_positions(candidates: list[int], n: int, doc_id: str, seed: int) -> list[int]:
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    return sorted(rng.sample(candidates, k=min(n, len(candidates))))


# ----------------------------------------------------------------------------
# Model forward
# ----------------------------------------------------------------------------

@torch.no_grad()
def forward_docs(model, docs: list[list[int]], layers: list[int], device, pad_id: int):
    """Right-padded batch forward. Returns (logits [B,L,V] on device, {layer: hs [B,L,d] cpu fp32})."""
    B = len(docs)
    L = max(len(d) for d in docs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, d in enumerate(docs):
        ids[i, : len(d)] = torch.tensor(d, dtype=torch.long)
        attn[i, : len(d)] = 1
    out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                output_hidden_states=True, use_cache=False)
    hs = {l: out.hidden_states[l + 1] for l in layers}
    return out.logits, hs


@torch.no_grad()
def greedy_continuations(model, tokenizer, prefixes: list[list[int]], n_new: int, device,
                         batch_size: int = 16) -> tuple[list[list[int]], list[np.ndarray], list[np.ndarray]]:
    """Greedy `n_new` tokens after each prefix (left-padded batches), plus the target's
    top-5 ids and the log-prob of the chosen token at every step (under its own prefix)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outs: list[list[int]] = []; top5s: list[np.ndarray] = []; logps: list[np.ndarray] = []
    for cs in range(0, len(prefixes), batch_size):
        chunk = prefixes[cs: cs + batch_size]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, L - len(p):] = 1
        gen = model.generate(
            input_ids=ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
            pad_token_id=pad_id, eos_token_id=None, output_scores=True, return_dict_in_generate=True,
        )
        seqs = gen.sequences
        lsm = torch.stack([F.log_softmax(sc.float(), dim=-1) for sc in gen.scores], dim=1)  # [B, n_new, V]
        top5 = lsm.topk(5, dim=-1).indices.cpu()                                          # [B, n_new, 5]
        for i in range(len(chunk)):
            toks = seqs[i, L: L + n_new]
            outs.append(toks.tolist())
            top5s.append(top5[i].reshape(-1).numpy().astype(np.int64))
            logps.append(lsm[i].gather(1, toks.to(lsm.device).unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float32))
    return outs, top5s, logps


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--corpus", required=True, help="HF dataset id | local .parquet | local .jsonl (text or ids)")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument("--text-column", default="text")
    p.add_argument("--n-train-docs", type=int, required=True)
    p.add_argument("--n-eval-docs", type=int, default=0)
    p.add_argument("--layers", default="4,8,12,16,20,24")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--min-pos", type=int, default=50)
    p.add_argument("--positions-per-doc", type=int, default=40)
    p.add_argument("--eval-positions-per-doc", type=int, default=None,
                   help="default: --positions-per-doc")
    p.add_argument("--n-future", type=int, default=DEFAULT_N_FUTURE)
    p.add_argument("--n-prev", type=int, default=DEFAULT_N_PREV)
    p.add_argument("--k-choices", default=",".join(str(k) for k in DEFAULT_K_CHOICES))
    p.add_argument("--template", default=DEFAULT_TEMPLATE)
    p.add_argument("--require-top1", action=argparse.BooleanOptionalAction, default=True,
                   help="keep only positions where the target's top-1 prediction of x_{t+1} is correct")
    p.add_argument("--greedy", choices=["none", "eval", "all"], default="all",
                   help="store the target's greedy n_future continuation (+ top-5, log-probs) for eval "
                        "rows or for all rows (needed for label=greedy training; ~+1 H100-h per 200k positions)")
    p.add_argument("--greedy-batch", type=int, default=32)
    p.add_argument("--alpha-quantile", default="p75",
                   help="which norm quantile becomes injection_scale_by_layer (NLA paper: p75)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--row-group-size", type=int, default=2048)
    args = p.parse_args(argv)

    device = _pick_device(args.device)
    dtype = (torch.bfloat16 if device == "cuda" else torch.float32) if args.dtype == "auto" else \
        {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    layers = [int(x) for x in args.layers.split(",")]
    k_choices = [int(x) for x in args.k_choices.split(",")]
    nf, npv = args.n_future, args.n_prev
    assert max(k_choices) <= nf, f"k_choices {k_choices} exceed n_future={nf}"
    assert args.min_pos >= npv - 1, f"--min-pos {args.min_pos} must be >= n_prev-1={npv - 1}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.base_ckpt, torch_dtype=dtype,
                                                 attn_implementation="sdpa").to(device).eval()
    tcfg = resolve_text_config(model.config)
    d_model = int(tcfg.hidden_size)
    n_layers = int(tcfg.num_hidden_layers)
    # hidden_states[n_layers] is POST final-norm, not the raw output of the last block,
    # so the last block is not a valid extraction layer under the block-output convention.
    assert all(0 <= l < n_layers - 1 for l in layers), (
        f"layers {layers} must be in [0, {n_layers - 2}] for a {n_layers}-block model")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"[collect] {args.base_ckpt}: d_model={d_model} n_layers={n_layers} layers={layers} "
          f"device={device} dtype={dtype}", flush=True)

    # Marker token + neighbours: the sidecar `actor` template is the parametric
    # template with dummy layer/K (neighbours are the <concept> tag bytes, unaffected).
    actor_template_for_sidecar = fill_template(args.template, 0, 1, "{injection_char}")
    token_meta = build_token_meta(tokenizer, actor_template_for_sidecar)
    print(f"[collect] marker {token_meta.injection_char!r} id={token_meta.injection_token_id} "
          f"neighbours=({token_meta.injection_left_neighbor_id},{token_meta.injection_right_neighbor_id})")

    rng = np.random.default_rng(args.seed)
    schema = fl_schema(d_model, nf, npv)
    norms: dict[int, list[float]] = {l: [] for l in layers}
    max_abs: dict[int, float] = {l: 0.0 for l in layers}
    stats = {"n_candidates": 0, "n_top1_correct": 0, "n_docs": {"train": 0, "eval": 0},
             "n_positions": {"train": 0, "eval": 0}, "n_rows": {"train": 0, "eval": 0}}
    docs_rows = {"doc_idx": [], "doc_id": [], "split": [], "ids": []}
    corpus = iter_corpus(args, tokenizer)
    counter = {"doc_idx": 0}
    t_start = time.time()

    def process_split(split: str, n_docs: int, ppd: int):
        if n_docs <= 0:
            return
        writer = pq.ParquetWriter(str(out_dir / f"{split}.parquet"), schema)
        pending: dict[str, list] = {k: [] for k in schema.names}
        n_done = 0

        def flush():
            if pending["doc_id"]:
                writer.write_table(pa.Table.from_pydict(pending, schema=schema))
                for k in pending:
                    pending[k] = []

        batch_docs: list[tuple[str, list[int]]] = []

        def run_batch():
            nonlocal n_done
            ids_list = [d for _, d in batch_docs]
            logits, hs = forward_docs(model, ids_list, layers, device, pad_id)
            argmax = logits.argmax(-1).cpu()          # [B, L]
            for bi, (did, ids) in enumerate(batch_docs):
                L = len(ids)
                ids_t = torch.tensor(ids, dtype=torch.long)
                cands = list(range(args.min_pos, L - nf))   # need ids[t+nf] to exist
                if not cands:
                    continue
                correct = (argmax[bi, cands] == ids_t[[c + 1 for c in cands]]).tolist()
                stats["n_candidates"] += len(cands)
                stats["n_top1_correct"] += int(sum(correct))
                pool = [c for c, ok in zip(cands, correct) if ok] if args.require_top1 else cands
                pos = sample_positions(pool, ppd, did, args.seed)
                if not pos:
                    continue
                doc_idx = counter["doc_idx"]
                docs_rows["doc_idx"].append(doc_idx)
                docs_rows["doc_id"].append(did)
                docs_rows["split"].append(split)
                docs_rows["ids"].append(ids)
                stats["n_docs"][split] += 1
                stats["n_positions"][split] += len(pos)
                # per-position target statistics from the single teacher-forced pass
                greedy = g_top5 = g_logp = None
                if args.greedy == "all" or (split == "eval" and args.greedy == "eval"):
                    greedy, g_top5, g_logp = greedy_continuations(
                        model, tokenizer, [ids[: t + 1] for t in pos], nf, device, batch_size=args.greedy_batch)
                for pi, t in enumerate(pos):
                    lg = logits[bi, t: t + nf].float()            # [nf, V]
                    lsm = F.log_softmax(lg, dim=-1)
                    tgt = ids_t[t + 1: t + 1 + nf]                # [nf]
                    target_logp = lsm.gather(1, tgt.to(lsm.device).unsqueeze(1)).squeeze(1).cpu()
                    top5 = lg.topk(5, dim=-1).indices.cpu()       # [nf, 5]
                    p_top1 = float(lsm[0, tgt[0]].exp())
                    prev = ids_t[t - npv + 1: t + 1]
                    text_full = tokenizer.decode(tgt.tolist())
                    for l in layers:
                        vec = hs[l][bi, t].float().cpu()
                        norms[l].append(float(vec.norm()))
                        max_abs[l] = max(max_abs[l], float(vec.abs().max()))
                        vec16 = vec.to(torch.float16)
                        assert torch.isfinite(vec16).all(), (
                            f"layer {l} activation overflows float16 (max |h| = {vec.abs().max():.0f}) "
                            f"at doc {did} t={t}; store fp32 or exclude this position")
                        k = int(rng.choice(k_choices))
                        pending["prompt"].append(build_prompt_messages(args.template, l, k))
                        pending["response"].append(tokenizer.decode(tgt[:k].tolist()))
                        pending["activation_vector"].append(vec16.numpy())
                        pending["activation_layer"].append(l)
                        pending["doc_id"].append(did)
                        pending["n_raw_tokens"].append(t + 1)
                        pending["target_ids"].append(tgt.numpy())
                        pending["k"].append(k)
                        pending["doc_idx"].append(doc_idx)
                        pending["t"].append(t)
                        pending["p_top1"].append(p_top1)
                        pending["target_top5"].append(top5.reshape(-1).numpy())
                        pending["target_logp"].append(target_logp.numpy())
                        pending["prev_ids"].append(prev.numpy())
                        pending["greedy_ids"].append(
                            np.asarray(greedy[pi] if greedy is not None else [-1] * nf, dtype=np.int64))
                        pending["greedy_top5"].append(
                            g_top5[pi] if g_top5 is not None else np.full(5 * nf, -1, dtype=np.int64))
                        pending["greedy_logp"].append(
                            g_logp[pi] if g_logp is not None else np.full(nf, np.nan, dtype=np.float32))
                        stats["n_rows"][split] += 1
                    del text_full
                counter["doc_idx"] += 1
                n_done += 1
                if len(pending["doc_id"]) >= args.row_group_size:
                    flush()
            batch_docs.clear()
            del logits, hs

        while n_done < n_docs:
            try:
                did, ids = next(corpus)
            except StopIteration:
                print(f"[collect] corpus exhausted after {n_done} {split} docs", flush=True)
                break
            batch_docs.append((did, ids))
            if len(batch_docs) >= args.batch_size:
                run_batch()
                if n_done % (args.batch_size * 25) == 0:
                    el = time.time() - t_start
                    print(f"[collect:{split}] {n_done}/{n_docs} docs, rows={stats['n_rows'][split]}, "
                          f"{el:.0f}s", flush=True)
        if batch_docs:
            run_batch()
        flush()
        writer.close()
        # docs table after every split (a crash in eval leaves train.parquet usable)
        pq.write_table(pa.Table.from_pydict(docs_rows, schema=docs_schema()), str(out_dir / "docs.parquet"))
        print(f"[collect:{split}] done: {stats['n_docs'][split]} docs, "
              f"{stats['n_positions'][split]} positions, {stats['n_rows'][split]} rows", flush=True)

    process_split("train", args.n_train_docs, args.positions_per_doc)
    process_split("eval", args.n_eval_docs, args.eval_positions_per_doc or args.positions_per_doc)

    # ---- norm quantiles -> alpha ----
    norm_q = {l: {q: float(np.quantile(norms[l], f)) if norms[l] else float("nan")
                  for q, f in QUANTILES.items()} for l in layers}
    alpha = {l: norm_q[l][args.alpha_quantile] for l in layers}
    discard = 1.0 - stats["n_top1_correct"] / max(1, stats["n_candidates"])
    stats["discard_fraction_top1"] = discard
    stats["norm_quantiles"] = norm_q
    stats["max_abs_by_layer"] = max_abs
    stats["injection_scale_by_layer"] = alpha
    stats["wall_s"] = time.time() - t_start
    (out_dir / "collect_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[collect] top-1-correct filter discards {discard:.1%} of candidate positions")
    for l in layers:
        print(f"[collect] layer {l:2d}: norm p25/p50/p75/p90 = "
              + "/".join(f"{norm_q[l][q]:.1f}" for q in QUANTILES) + f"  alpha={alpha[l]:.1f}")

    # ---- sidecars ----
    fl_meta = FLMeta(
        layer_indices=layers, n_future=nf, n_prev=npv, k_choices=k_choices,
        template=args.template, norm_quantiles=norm_q, injection_scale_by_layer=alpha,
        docs_parquet="docs.parquet", discard_fraction=discard, d_model=d_model,
        extra={"base_model": args.base_ckpt, "require_top1": bool(args.require_top1), "greedy": args.greedy,
               "max_len": args.max_len, "min_pos": args.min_pos,
               "corpus": args.corpus, "corpus_config": args.corpus_config,
               "corpus_slice": {"start": args.corpus_start, "length": args.n_train_docs + args.n_eval_docs}},
    )
    for split in ("train", "eval"):
        pq_path = out_dir / f"{split}.parquet"
        if not pq_path.exists():
            continue
        meta = NLADatasetMeta(
            dataset_id=f"fl_{args.base_ckpt.split('/')[-1]}_L{'-'.join(map(str, layers))}_{split}",
            stage="av_sft", row_count=stats["n_rows"][split],
            extraction=NLAExtractionMeta(
                base_model=args.base_ckpt, d_model=d_model,
                layer_index=layers[-1],       # scalar slot; future_lens.layer_indices is authoritative
                norm="none", corpus=args.corpus,
                corpus_slice={"start": args.corpus_start, "length": args.n_train_docs + args.n_eval_docs},
                positions_per_doc=args.positions_per_doc,
            ),
            tokens=token_meta,
            prompt_templates={"actor": actor_template_for_sidecar},
            created_by="nla.future_lens.collect",
        )
        d = yaml.safe_load(serialize_sidecar(meta))
        d[FL_SIDECAR_KEY] = fl_meta.to_dict()
        sidecar_path_for(pq_path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    print(f"[collect] wrote {out_dir} in {stats['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    # The `datasets` streaming iterator (aiohttp/pyarrow threads) can abort the interpreter
    # during finalisation *after* every file has been written ("PyGILState_Release ... must be
    # current"), which turns a successful run into exit 134. Everything is flushed by now.
    import os
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
