"""Baselines for the future-lens plots, in the same JSONL record format as eval.py.

  ngram   bigram / 4-gram (with backoff) on the training documents' token ids, rolled
          out greedily from x_<=t to predict x_{t+1+N}. The context-free prior every
          decoder gain must be measured against. CPU.
  probe   linear probe h_t^l -> x_{t+1+N} ("Linear Vocab" in Future Lens): one
          Linear(d, vocab) per (layer, N), trained on train.parquet, scored on eval.
  probe --leakage  linear probe h_t^l -> x_{t-j}, j = 1..4: how much of the PAST is
          linearly readable from the state.
  leakage the other half of the spec's leakage check: given a per-row readout dump from
          `eval.py --dump-readouts`, compare each decoder readout with (a) the true
          future and (b) what a preceding-token n-gram model predicts from x_<=t.
          A decoder that agrees with the n-gram model more than with the future is
          reading the past and guessing.

    python -m nla.future_lens.baselines ngram --parquet <data>/eval.parquet --out evals/baselines.jsonl \
        [--hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000 --base-ckpt Qwen/Qwen3-8B]
    python -m nla.future_lens.baselines probe --train-parquet <data>/train.parquet \
        --parquet <data>/eval.parquet --out evals/baselines.jsonl [--leakage]
    python -m nla.future_lens.baselines leakage --parquet <data>/eval.parquet \
        --readouts evals/readouts_rl.jsonl --out evals/leakage.jsonl

Eval documents are never counted by the n-gram models: docs.parquet's eval split is
excluded by doc_idx, and extra sources (--hf-corpus / --extra-docs) are screened by a
hash of each document's first 64 token ids.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from nla.future_lens.data import load_docs, load_fl_meta, load_fl_rows, resolve_docs_path
from nla.future_lens.eval import write_records


# ----------------------------------------------------------------------------
# n-gram
# ----------------------------------------------------------------------------

class NGramModel:
    """Count-based n-gram LM with simple backoff (highest order with any count wins)."""

    def __init__(self, order: int):
        assert order >= 2
        self.order = order
        self.tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(order)]  # index = context len
        self.unigram: Counter = Counter()

    def add(self, ids) -> None:
        ids = [int(x) for x in ids]
        self.unigram.update(ids)
        for i in range(1, len(ids)):
            for c in range(1, self.order):
                if i - c < 0:
                    break
                self.tables[c][tuple(ids[i - c: i])][ids[i]] += 1

    def predict(self, context) -> int:
        ctx = [int(x) for x in context]
        for c in range(min(self.order - 1, len(ctx)), 0, -1):
            tab = self.tables[c].get(tuple(ctx[-c:]))
            if tab:
                return tab.most_common(1)[0][0]
        return self.unigram.most_common(1)[0][0] if self.unigram else 0

    def greedy(self, context, n: int) -> list[int]:
        ctx = [int(x) for x in context]
        out = []
        for _ in range(n):
            nxt = self.predict(ctx)
            out.append(nxt)
            ctx.append(nxt)
        return out


def _doc_key(ids) -> tuple:
    return tuple(int(x) for x in list(ids)[:64])


def ngram_sources(args, fl, docs: dict, eval_docs: set):
    """Yield token-id docs for counting, never an eval document."""
    eval_keys = {_doc_key(docs[d]) for d in eval_docs}
    for di, ids in docs.items():
        if di not in eval_docs:
            yield ids
    if args.extra_docs:
        for ids in _iter_extra_token_docs(args.extra_docs, args.extra_docs_max):
            if _doc_key(ids) not in eval_keys:
                yield ids
    if args.hf_corpus:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_ckpt)
        start = args.hf_start
        if start is None:   # default: right after the slice the collector consumed
            sl = fl.extra.get("corpus_slice") or {}
            start = int(sl.get("start", 0)) + int(sl.get("length", 0))
        ds = load_dataset(args.hf_corpus, name=args.hf_config, split=args.hf_split, streaming=True).skip(start)
        n = 0
        for ex in ds:
            ids = tok.encode(ex[args.text_column] or "", add_special_tokens=False)[: args.hf_max_len]
            if len(ids) < 32:
                continue
            if _doc_key(ids) in eval_keys:
                continue
            yield ids
            n += 1
            if n >= args.hf_docs:
                break


def build_ngram(order: int, sources) -> tuple[NGramModel, int]:
    m = NGramModel(order)
    n_tok = 0
    for ids in sources:
        m.add(ids); n_tok += len(ids)
    return m, n_tok


def run_ngram(args):
    fl = load_fl_meta(args.sidecar or args.parquet)
    docs = load_docs(resolve_docs_path(args.parquet, fl))
    eval_rows = load_fl_rows(args.parquet, keep_activations=False, label=args.label,
                             columns=["doc_idx", "t", "target_ids", "target_top5", "p_top1", "activation_layer"])
    eval_docs = {int(r["doc_idx"]) for r in eval_rows}
    # one row per position (layers duplicate the position)
    positions = {(int(r["doc_idx"]), int(r["t"])): r for r in eval_rows}
    recs = []
    for order in [int(x) for x in args.orders.split(",")]:
        t0 = time.time()
        m, n_tok = build_ngram(order, ngram_sources(args, fl, docs, eval_docs))
        nf = fl.n_future
        hits = defaultdict(list); hits5 = defaultdict(list); hits_tf = defaultdict(list)
        for (di, t), r in positions.items():
            ctx = docs[di][: t + 1]
            ro = m.greedy(ctx, nf)
            tgt = np.asarray(r["target_ids"]); top5 = np.asarray(r["target_top5"]).reshape(-1, 5)
            ctx_l = [int(x) for x in ctx]
            for j in range(nf):
                hits[j].append(int(ro[j] == int(tgt[j])))
                hits5[j].append(int(ro[j] in set(int(x) for x in top5[j])))
                # teacher-forced (Future Lens convention): predict x_{t+1+j} given the LABEL prefix
                hits_tf[j].append(int(m.predict(ctx_l + [int(x) for x in tgt[:j]]) == int(tgt[j])))
        for k in fl.k_choices:
            N = k - 1
            base = {"layer": -1, "N": N, "k": k, "condition": "baseline", "injection": "none",
                    "seed": 0, "checkpoint": f"{order}gram", "n_train_tokens": n_tok, "label": args.label}
            recs.append({**base, "metric": "p1", "value": float(np.mean(hits[N])), "n": len(hits[N])})
            recs.append({**base, "metric": "p5", "value": float(np.mean(hits5[N])), "n": len(hits5[N])})
            recs.append({**base, "metric": "tf_p1", "value": float(np.mean(hits_tf[N])), "n": len(hits_tf[N])})
        print(f"[ngram] order={order} tokens={n_tok} label={args.label} " +
              " ".join(f"p1@{N}={np.mean(hits[N]):.3f}" for N in range(nf)) + " | tf " +
              " ".join(f"{np.mean(hits_tf[N]):.3f}" for N in range(nf)) + f" ({time.time() - t0:.0f}s)")
    write_records(recs, args.out)
    print(f"[ngram] wrote {len(recs)} records -> {args.out}")


def run_leakage(args):
    """Readout agreement with the true future vs with a past-only n-gram prediction.

    For every dumped readout (row = (doc_idx, t, layer, k, condition)) and offset j < k:
        agree_future[j] = 1[y_j == x_{t+1+j}]
        agree_ngram[j]  = 1[y_j == g_j]   where g = n-gram greedy rollout from x_<=t
    plus the n-gram's own accuracy 1[g_j == x_{t+1+j}] and the agreement on positions
    where the n-gram is WRONG (the diagnostic: a lens should not follow a wrong prior).
    Records go out in the standard JSONL format with condition = the dump's condition.
    """
    fl = load_fl_meta(args.sidecar or args.parquet)
    docs = load_docs(resolve_docs_path(args.parquet, fl))
    eval_rows = load_fl_rows(args.parquet, keep_activations=False, columns=["doc_idx", "t", "target_ids"], label=args.label)
    eval_docs = {int(r["doc_idx"]) for r in eval_rows}
    tgt = {(int(r["doc_idx"]), int(r["t"])): np.asarray(r["target_ids"]) for r in eval_rows}
    m, n_tok = build_ngram(args.order, ngram_sources(args, fl, docs, eval_docs))
    print(f"[leakage] {args.order}-gram on {n_tok} tokens")
    import glob as _glob
    paths = [p for pat in args.readouts for p in (sorted(_glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))]
    missing = [pat for pat in args.readouts if not _glob.glob(pat) and not os.path.exists(pat)]
    if missing:
        print(f"[leakage] no files match {missing}; skipped")
    if not paths:
        print(f"[leakage] no readout files; nothing written"); return
    dumps = [json.loads(l) for path in paths for l in open(path) if l.strip()]
    if not dumps:
        print(f"[leakage] no readouts in {paths}; nothing written"); return
    dump_labels = {d.get("label", "text") for d in dumps}
    assert dump_labels == {args.label}, f"readouts were scored under label(s) {dump_labels}, but --label {args.label}"
    cache: dict[tuple, list[int]] = {}
    acc = defaultdict(lambda: defaultdict(list))
    for d in dumps:
        key = (int(d["doc_idx"]), int(d["t"]))
        if key not in cache:
            cache[key] = m.greedy(docs[key[0]][: key[1] + 1], fl.n_future)
        g, x, y = cache[key], tgt[key], d["readout"]
        gk = (d.get("checkpoint", "?"), d.get("group") or d.get("checkpoint", "?"), d["condition"],
              int(d["layer"]), int(d["k"]))
        for j in range(int(d["k"])):
            if j >= len(y):
                break
            af, an, ng = int(y[j] == int(x[j])), int(y[j] == g[j]), int(g[j] == int(x[j]))
            acc[gk][("agree_future", j)].append(af)
            acc[gk][("agree_ngram", j)].append(an)
            acc[gk][("ngram_correct", j)].append(ng)
            if not ng:
                acc[gk][("agree_ngram_when_ngram_wrong", j)].append(an)
                acc[gk][("agree_future_when_ngram_wrong", j)].append(af)
    recs = []
    for (ck, group, cond, layer, k), stats in acc.items():
        for (metric, j), v in stats.items():
            recs.append({"layer": layer, "N": j, "k": k, "condition": cond, "injection": "n/a",
                         "seed": int(dumps[0].get("seed", 0)), "checkpoint": ck, "group": group,
                         "metric": f"leak_{metric}", "value": float(np.mean(v)), "n": len(v),
                         "ngram_order": args.order, "label": args.label})
    write_records(recs, args.out)
    for (ck, group, cond, layer, k), stats in sorted(acc.items()):
        af = np.mean(stats[("agree_future", k - 1)]); an = np.mean(stats[("agree_ngram", k - 1)])
        aw = stats.get(("agree_ngram_when_ngram_wrong", k - 1), [])
        print(f"[leakage] {ck} {cond:9s} L{layer:2d} K={k}: agree(future)={af:.3f} agree(ngram)={an:.3f} "
              f"agree(ngram | ngram wrong)={np.mean(aw) if aw else float('nan'):.3f} n={len(stats[('agree_future', k - 1)])}")
    print(f"[leakage] wrote {len(recs)} records -> {args.out}")


def run_target_window(args):
    """Past-only baseline with the FROZEN TARGET itself: greedy-continue the last m tokens
    before t (prev_ids[-m:]) and score the readout metrics. This bounds what a decoder that
    merely recovers a few recent tokens from h_t and re-simulates the model could achieve
    (the shuffled control does not: it removes token identity along with everything else)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.future_lens.collect import greedy_continuations
    fl = load_fl_meta(args.sidecar or args.parquet)
    rows = load_fl_rows(args.parquet, keep_activations=False, label=args.label,
                        columns=["doc_idx", "t", "target_ids", "target_top5", "prev_ids", "activation_layer"])
    positions = {(int(r["doc_idx"]), int(r["t"])): r for r in rows}
    keys = sorted(positions)
    if args.max_positions and len(keys) > args.max_positions:
        keys = [keys[i] for i in sorted(np.random.default_rng(args.seed).choice(len(keys), size=args.max_positions, replace=False))]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.base_ckpt, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nf = fl.n_future
    recs = []
    for m in [int(x) for x in args.windows.split(",")]:
        t0 = time.time()
        prefixes = [[int(x) for x in positions[k]["prev_ids"][-m:]] for k in keys]
        outs, _, _ = greedy_continuations(model, tok, prefixes, nf, device, batch_size=args.batch)
        hits = defaultdict(list); hits5 = defaultdict(list); hits_tf = defaultdict(list)
        # teacher-forced: argmax at offset j given window + label[:j]
        with torch.no_grad():
            for cs in range(0, len(keys), args.batch):
                chunk = keys[cs: cs + args.batch]
                seqs = [prefixes[cs + i] + [int(x) for x in positions[k]["target_ids"][:nf]] for i, k in enumerate(chunk)]
                L = max(len(q) for q in seqs)
                ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); attn = torch.zeros_like(ids)
                for i, q in enumerate(seqs):
                    ids[i, L - len(q):] = torch.tensor(q); attn[i, L - len(q):] = 1
                pred = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits.argmax(-1).cpu()
                for i, k in enumerate(chunk):
                    tgt = positions[k]["target_ids"]
                    for j in range(nf):
                        hits_tf[j].append(int(pred[i, L - nf - 1 + j] == int(tgt[j])))
        for k, ro in zip(keys, outs):
            r = positions[k]; tgt = np.asarray(r["target_ids"]); top5 = np.asarray(r["target_top5"]).reshape(-1, 5)
            for j in range(nf):
                hits[j].append(int(ro[j] == int(tgt[j])))
                hits5[j].append(int(ro[j] in set(int(x) for x in top5[j])))
        for kk in fl.k_choices:
            N = kk - 1
            base = {"layer": -1, "N": N, "k": kk, "condition": "baseline", "injection": "none", "seed": 0,
                    "checkpoint": f"target_window_{m}", "label": args.label, "window": m}
            recs.append({**base, "metric": "p1", "value": float(np.mean(hits[N])), "n": len(hits[N])})
            recs.append({**base, "metric": "p5", "value": float(np.mean(hits5[N])), "n": len(hits5[N])})
            recs.append({**base, "metric": "tf_p1", "value": float(np.mean(hits_tf[N])), "n": len(hits_tf[N])})
        print(f"[target_window] m={m:2d} n={len(keys)} p1 " + " ".join(f"{np.mean(hits[j]):.3f}" for j in range(nf))
              + " | tf " + " ".join(f"{np.mean(hits_tf[j]):.3f}" for j in range(nf)) + f" ({time.time() - t0:.0f}s)", flush=True)
    write_records(recs, args.out)
    print(f"[target_window] wrote {len(recs)} records -> {args.out}")


def _iter_extra_token_docs(path: str, max_docs: int):
    """Optional extra n-gram counts from a .jsonl with `ids` (pre-tokenised) rows."""
    n = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ids"):
                yield d["ids"]; n += 1
            if max_docs and n >= max_docs:
                break


# ----------------------------------------------------------------------------
# linear probes
# ----------------------------------------------------------------------------

def _stack(rows, key="activation_vector"):
    return torch.tensor(np.stack([np.asarray(r[key], dtype=np.float32) for r in rows]))


def train_probe(X: torch.Tensor, y: torch.Tensor, vocab: int, *, epochs: int, lr: float, batch: int,
                device, weight_decay: float = 0.0, seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    d = X.shape[1]
    lin = torch.nn.Linear(d, vocab).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=weight_decay)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    lin.register_buffer("mu", mu.to(device)); lin.register_buffer("sd", sd.to(device))
    n = X.shape[0]
    steps = max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            xb = ((X[idx].to(device) - lin.mu) / lin.sd)
            loss = torch.nn.functional.cross_entropy(lin(xb), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return lin


@torch.no_grad()
def probe_p1(lin: torch.nn.Linear, X: torch.Tensor, y: torch.Tensor, device, batch: int = 4096) -> float:
    hits = []
    for i in range(0, X.shape[0], batch):
        xb = (X[i:i + batch].to(device) - lin.mu) / lin.sd
        hits.append((lin(xb).argmax(-1).cpu() == y[i:i + batch]).float())
    return float(torch.cat(hits).mean())


def run_probe(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    fl = load_fl_meta(args.sidecar or args.parquet)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else fl.layer_indices
    vocab = args.vocab_size
    if vocab is None:
        from transformers import AutoTokenizer
        vocab = len(AutoTokenizer.from_pretrained(args.base_ckpt))
    train = load_fl_rows(args.train_parquet, n_max=args.max_train_rows, layers=layers, label=args.label,
                         columns=["activation_vector", "activation_layer", "target_ids", "prev_ids", "doc_idx", "t"])
    ev = load_fl_rows(args.parquet, layers=layers, label=args.label,
                      columns=["activation_vector", "activation_layer", "target_ids", "prev_ids", "doc_idx", "t"])
    recs = []
    offsets = [k - 1 for k in fl.k_choices]
    for layer in layers:
        tr = [r for r in train if int(r["activation_layer"]) == layer]
        te = [r for r in ev if int(r["activation_layer"]) == layer]
        if not tr or not te:
            continue
        Xtr, Xte = _stack(tr), _stack(te)
        targets = [("future", N, [int(r["target_ids"][N]) for r in tr], [int(r["target_ids"][N]) for r in te])
                   for N in offsets]
        if args.leakage:
            for j in range(1, 5):   # x_{t-j}: prev_ids[-1] is x_t
                targets.append(("past", -j, [int(r["prev_ids"][-1 - j]) for r in tr],
                                [int(r["prev_ids"][-1 - j]) for r in te]))
        for kind, N, ytr, yte in targets:
            t0 = time.time()
            lin = train_probe(Xtr, torch.tensor(ytr), vocab, epochs=args.epochs, lr=args.lr,
                              batch=args.batch, device=device, weight_decay=args.weight_decay)
            p1 = probe_p1(lin, Xte, torch.tensor(yte), device)
            p1_train = probe_p1(lin, Xtr[: min(len(tr), 5000)], torch.tensor(ytr[: min(len(tr), 5000)]), device)
            rec = {"layer": layer, "N": N, "k": N + 1 if N >= 0 else None, "condition": "baseline", "label": args.label,
                   "injection": "none", "seed": 0, "checkpoint": f"linear_probe_{kind}",
                   "metric": "p1", "value": p1, "n": len(te), "train_p1": p1_train}
            recs.append(rec)
            print(f"[probe] L{layer:2d} {kind} N={N:+d}: p1={p1:.3f} (train {p1_train:.3f}) ({time.time() - t0:.0f}s)", flush=True)
            del lin
    write_records(recs, args.out)
    print(f"[probe] wrote {len(recs)} records -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    p.add_argument("--label", choices=("text", "greedy"), default="text", help="readout label (see data.py)")
    n = sub.add_parser("ngram")
    n.add_argument("--parquet", required=True, help="eval.parquet")
    n.add_argument("--sidecar", default=None)
    n.add_argument("--orders", default="2,4")
    n.add_argument("--out", required=True)
    lk = sub.add_parser("leakage")
    lk.add_argument("--parquet", required=True, help="eval.parquet")
    lk.add_argument("--sidecar", default=None)
    lk.add_argument("--readouts", required=True, nargs="+", help="JSONL file(s) from eval.py --dump-readouts (one n-gram build for all)")
    lk.add_argument("--order", type=int, default=4)
    lk.add_argument("--out", required=True)
    tw = sub.add_parser("target_window", help="frozen target greedy-continues the last m tokens before t")
    tw.add_argument("--parquet", required=True, help="eval.parquet"); tw.add_argument("--sidecar", default=None)
    tw.add_argument("--base-ckpt", default="Qwen/Qwen3-8B-Base"); tw.add_argument("--windows", default="1,2,4,8,16,32")
    tw.add_argument("--max-positions", type=int, default=2000); tw.add_argument("--batch", type=int, default=64)
    tw.add_argument("--device", default="auto"); tw.add_argument("--seed", type=int, default=0)
    tw.add_argument("--out", required=True)
    for sp in (n, lk):
        sp.add_argument("--extra-docs", default=None, help="extra .jsonl with `ids` rows for counts")
        sp.add_argument("--extra-docs-max", type=int, default=0)
        sp.add_argument("--hf-corpus", default=None, help="stream extra counting docs from an HF dataset")
        sp.add_argument("--hf-config", default=None)
        sp.add_argument("--hf-split", default="train")
        sp.add_argument("--hf-start", type=int, default=None, help="default: after the collector's slice")
        sp.add_argument("--hf-docs", type=int, default=20000)
        sp.add_argument("--hf-max-len", type=int, default=1024)
        sp.add_argument("--text-column", default="text")
        sp.add_argument("--base-ckpt", default="Qwen/Qwen3-8B-Base", help="tokenizer for --hf-corpus")
    q = sub.add_parser("probe")
    q.add_argument("--train-parquet", required=True)
    q.add_argument("--parquet", required=True, help="eval.parquet")
    q.add_argument("--sidecar", default=None)
    q.add_argument("--base-ckpt", default="Qwen/Qwen3-8B-Base", help="only used for the vocab size")
    q.add_argument("--vocab-size", type=int, default=None)
    q.add_argument("--layers", default=None)
    q.add_argument("--max-train-rows", type=int, default=None)
    q.add_argument("--epochs", type=int, default=3)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--weight-decay", type=float, default=0.0)
    q.add_argument("--batch", type=int, default=512)
    q.add_argument("--leakage", action="store_true", help="also probe x_{t-1..t-4}")
    q.add_argument("--device", default="auto")
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"ngram": run_ngram, "probe": run_probe, "leakage": run_leakage, "target_window": run_target_window}[args.cmd](args)


if __name__ == "__main__":
    main()
