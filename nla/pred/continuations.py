"""Stage 1: sample the target model's own continuations at each activation position.

For every position we keep, the frozen target model re-reads the document prefix
that produced the activation and samples K continuations of T tokens at
temperature 1.0 - i.e. K draws from the model's actual next-token distribution,
which is exactly the behavior the explanation will later be asked to help predict.

The prefix is reconstructed from `detokenized_text_truncated` and re-tokenized;
the result MUST come back to `n_raw_tokens` tokens or the position is dropped,
because a prefix that does not round-trip is not the context the activation came
from. (Measured on the public warm-start data: 2000/2000 round-trip exactly.)

Nothing here ever sees an explanation. The continuations are fixed once and
reused by the gate, RL and the final eval, so every explanation for a position is
judged against the same sampled futures.

    python -m nla.pred.continuations \
        --source-parquet <rl_shuf.parquet> --sidecar <rl_shuf.parquet> \
        --target-ckpt Qwen/Qwen3-8B --out <positions.parquet> \
        --n-rl 20000 --n-val 1000 --n-eval 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.config import load_nla_config
from nla.pred.data import positions_schema, split_for_doc
from nla.pred.reader import token_char_bounds
from nla.pred.wandb_util import finish_run, init_run


def _iter_source(parquet_path, corpus_filter, max_prefix_tokens, max_per_doc):
    """Stream source rows, filtered by corpus and prefix length, capped per doc."""
    pf = pq.ParquetFile(parquet_path)
    per_doc: dict[str, int] = {}
    cols = ["prompt", "activation_vector", "n_raw_tokens", "doc_id",
            "detokenized_text_truncated"]
    missing = [c for c in cols if c not in pf.schema_arrow.names]
    assert not missing, (
        f"{parquet_path} lacks {missing}. This stage needs the source text "
        f"(datagen keep_debug_metadata: true) to resample continuations."
    )
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        prompts = t.column("prompt").to_pylist()
        docs = t.column("doc_id").to_pylist()
        nraw = t.column("n_raw_tokens").to_pylist()
        texts = t.column("detokenized_text_truncated").to_pylist()
        col = t.column("activation_vector").combine_chunks()
        acts = np.asarray(col.flatten(), dtype=np.float32).reshape(t.num_rows, -1)
        for i in range(t.num_rows):
            d = docs[i]
            if corpus_filter and corpus_filter not in d:
                continue
            if nraw[i] > max_prefix_tokens or not texts[i]:
                continue
            if per_doc.get(d, 0) >= max_per_doc:
                continue
            per_doc[d] = per_doc.get(d, 0) + 1
            yield {"prompt": prompts[i], "activation": acts[i], "doc_id": d,
                   "n_raw_tokens": nraw[i], "prefix_text": texts[i]}


@torch.no_grad()
def sample_branches(model, tokenizer, prefixes, n_branches, n_tokens, device,
                    temperature=1.0, add_special_tokens=False):
    """[len(prefixes)][n_branches] token-id lists sampled from the target model."""
    enc = tokenizer(prefixes, return_tensors="pt", padding=True,
                    add_special_tokens=add_special_tokens).to(device)
    out = model.generate(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=n_tokens, min_new_tokens=n_tokens,
        do_sample=True, temperature=temperature,
        # The target model's TRUE distribution: Qwen3's generation_config ships
        # top_p=0.95/top_k=20, which would make these continuations samples from
        # a truncated model, not the one whose activations we are explaining.
        top_p=1.0, top_k=0, repetition_penalty=1.0,
        num_return_sequences=n_branches,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    new = out.sequences[:, enc.input_ids.shape[1] :]        # [B*K, T]
    new = new.reshape(len(prefixes), n_branches, -1)
    return new.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-parquet", required=True,
                   help="An NLA dataset parquet with detokenized_text_truncated.")
    p.add_argument("--sidecar", required=True,
                   help="Its .nla_meta.yaml (asserted against the live tokenizer).")
    p.add_argument("--out", required=True, help="Output positions parquet.")
    p.add_argument("--target-ckpt", default="Qwen/Qwen3-8B",
                   help="The FROZEN target model whose activations these are.")
    p.add_argument("--corpus-filter", default="finefineweb",
                   help="Keep only doc_ids containing this substring. The default "
                        "keeps the pretraining-like half of a mixed pool and drops "
                        "chat transcripts, which are a different distribution.")
    p.add_argument("--n-branches", type=int, default=4)
    p.add_argument("--n-tokens", type=int, default=24,
                   help="Tokens per branch. Buckets of 8 give bridges of 0/8/16.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-prefix-tokens", type=int, default=1024,
                   help="Skip positions with a longer prefix: the continuation must "
                        "be sampled from the FULL context that produced the "
                        "activation, so long prefixes cost time rather than being "
                        "truncated.")
    p.add_argument("--max-per-doc", type=int, default=2,
                   help="Positions per document, for document diversity.")
    p.add_argument("--n-rl", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--val-permille", type=int, default=60)
    p.add_argument("--eval-permille", type=int, default=120)
    p.add_argument("--batch-prefixes", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-group-size", type=int, default=2000)
    p.add_argument("--wandb-project", default="pred-nla")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="prep")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert not out_path.exists(), f"{out_path} exists - refusing to overwrite"

    tokenizer = AutoTokenizer.from_pretrained(args.target_ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg = load_nla_config(args.sidecar, tokenizer)
    print(f"[cfg] d_model={cfg.d_model} layer={cfg.extraction_layer_index} "
          f"inj_id={cfg.injection_token_id}", flush=True)

    run = None if args.no_wandb else init_run(
        args, project=args.wandb_project, name=args.wandb_name,
        group=args.wandb_group, job_type="continuations",
    )

    print(f"[target] loading {args.target_ckpt}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_ckpt, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
    ).to(args.device).eval()

    quotas = {"rl": args.n_rl, "val": args.n_val, "eval": args.n_eval}
    taken = {k: 0 for k in quotas}
    writer = None
    schema = positions_schema(cfg.d_model, args.n_branches)
    buf: list[dict] = []
    row_id = 0
    n_seen = n_roundtrip_fail = 0
    add_special = None
    t0 = time.time()
    pending: list[dict] = []
    samples_table: list[list] = []

    def flush_batch(pending):
        nonlocal row_id, n_roundtrip_fail, add_special
        if not pending:
            return []
        # Detect the source pipeline's special-token convention once, from a
        # HANDFUL of rows rather than one. Latching the wrong convention off a
        # single unlucky row would fail the round-trip check on every subsequent
        # row, and the job would stream the entire source parquet before dying on
        # "no positions written" - so fail here, loudly, instead.
        if add_special is None:
            probes = pending[: min(8, len(pending))]
            hits = {}
            for cand in (False, True):
                hits[cand] = sum(
                    len(tokenizer(r["prefix_text"], add_special_tokens=cand)["input_ids"])
                    == r["n_raw_tokens"] for r in probes)
            best = max(hits, key=hits.get)
            assert hits[best] > 0, (
                f"No special-token convention reproduces n_raw_tokens on any of "
                f"{len(probes)} probe rows (add_special_tokens=False matched "
                f"{hits[False]}, True matched {hits[True]}). The prefixes in "
                f"{args.source_parquet} do not round-trip under this tokenizer, so "
                f"continuations would be sampled from the wrong context. Check "
                f"that --target-ckpt matches the model that produced the data.")
            add_special = best
            print(f"[prefix] add_special_tokens={add_special} "
                  f"({hits[best]}/{len(probes)} probe rows round-trip exactly)",
                  flush=True)
        keep = []
        for r in pending:
            n = len(tokenizer(r["prefix_text"], add_special_tokens=add_special)["input_ids"])
            if n != r["n_raw_tokens"]:
                n_roundtrip_fail += 1
                # The quota was claimed when the row was queued; give it back, or
                # --n-eval 2000 quietly yields 2000 minus the failures.
                taken[r["split"]] -= 1
                continue
            keep.append(r)
        if not keep:
            return []
        branches = sample_branches(
            model, tokenizer, [r["prefix_text"] for r in keep],
            args.n_branches, args.n_tokens, args.device,
            temperature=args.temperature, add_special_tokens=add_special,
        )
        out_rows = []
        for r, brs in zip(keep, branches):
            texts, idss, boundss = [], [], []
            for ids in brs:
                bounds = token_char_bounds(tokenizer, ids)
                texts.append(tokenizer.decode(ids))
                idss.append([int(x) for x in ids])
                boundss.append([int(x) for x in bounds])
            out_rows.append({
                "row_id": row_id, "doc_id": r["doc_id"], "split": r["split"],
                "n_raw_tokens": int(r["n_raw_tokens"]), "prompt": r["prompt"],
                "activation_vector": r["activation"].tolist(),
                "prefix_text": r["prefix_text"], "cont_text": texts,
                "cont_ids": idss, "cont_bounds": boundss,
            })
            row_id += 1
        return out_rows

    for src in _iter_source(args.source_parquet, args.corpus_filter,
                            args.max_prefix_tokens, args.max_per_doc):
        n_seen += 1
        sp = split_for_doc(src["doc_id"], args.val_permille, args.eval_permille)
        if taken[sp] >= quotas[sp]:
            if all(taken[k] >= quotas[k] for k in quotas):
                break
            continue
        taken[sp] += 1
        src["split"] = sp
        pending.append(src)
        if len(pending) >= args.batch_prefixes:
            buf.extend(flush_batch(pending))
            pending = []
            if len(buf) >= args.row_group_size:
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), schema)
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                done = sum(taken.values())
                rate = done / max(1e-6, time.time() - t0)
                print(f"[prep] {done} positions "
                      f"(rl {taken['rl']} val {taken['val']} eval {taken['eval']}) "
                      f"{rate:.1f} pos/s", flush=True)
                if run is not None:
                    run.log({"prep/positions": done, "prep/pos_per_s": rate,
                             **{f"prep/n_{k}": v for k, v in taken.items()}})
                if len(samples_table) < 20:
                    for b in buf[:2]:
                        samples_table.append([
                            b["row_id"], b["doc_id"], b["split"],
                            b["prefix_text"][-300:], b["cont_text"][0],
                        ])
                buf = []
    buf.extend(flush_batch(pending))
    if buf:
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), schema)
        writer.write_table(pa.Table.from_pylist(buf, schema=schema))
        for b in buf[:2]:
            if len(samples_table) < 20:
                samples_table.append([b["row_id"], b["doc_id"], b["split"],
                                      b["prefix_text"][-300:], b["cont_text"][0]])
    assert writer is not None, "no positions written - check --corpus-filter"
    writer.close()

    # The positions parquet inherits the source's NLA contract (marker token,
    # templates, scales); downstream trainers assert against this copy.
    import shutil

    from nla.schema import sidecar_path_for

    shutil.copy2(sidecar_path_for(args.sidecar), sidecar_path_for(str(out_path)))
    meta = {
        "source_parquet": args.source_parquet, "target_ckpt": args.target_ckpt,
        "n_branches": args.n_branches, "n_tokens": args.n_tokens,
        "temperature": args.temperature, "corpus_filter": args.corpus_filter,
        "max_prefix_tokens": args.max_prefix_tokens, "max_per_doc": args.max_per_doc,
        "val_permille": args.val_permille, "eval_permille": args.eval_permille,
        "add_special_tokens": add_special, "seed": args.seed,
        "counts": taken, "n_source_seen": n_seen,
        "n_roundtrip_fail": n_roundtrip_fail,
    }
    Path(str(out_path) + ".pred_meta.json").write_text(json.dumps(meta, indent=2))
    dt = time.time() - t0
    print(f"[prep] done: {sum(taken.values())} positions in {dt/60:.1f} min "
          f"-> {out_path}\n[prep] {taken} | round-trip failures {n_roundtrip_fail} "
          f"({n_roundtrip_fail / max(1, n_seen):.2%} of scanned)", flush=True)
    if run is not None:
        import wandb

        run.summary.update({f"prep/final_{k}": v for k, v in taken.items()})
        run.summary["prep/roundtrip_fail"] = n_roundtrip_fail
        run.summary["prep/minutes"] = dt / 60
        if samples_table:
            run.log({"prep/samples": wandb.Table(
                columns=["row_id", "doc_id", "split", "prefix_tail", "continuation_0"],
                data=samples_table)})
        finish_run(run)


if __name__ == "__main__":
    main()
