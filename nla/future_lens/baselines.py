"""Baselines for the future-lens plots, in the same JSONL record format as eval.py.

  ngram   bigram / 4-gram (with backoff) on the training documents' token ids, rolled
          out greedily from x_<=t to predict x_{t+1+N}. The context-free prior every
          decoder gain must be measured against. CPU.
  probe   linear probe h_t^l -> x_{t+1+N} ("Linear Vocab" in Future Lens): one
          Linear(d, vocab) per (layer, N), trained on train.parquet, scored on eval.
  leakage linear probe h_t^l -> x_{t-j}, j = 1..4: how much of the PAST is linearly
          readable from the state. Reported next to the future probes; a decoder
          whose readouts track an n-gram model of the past more than the true
          future is guessing from leaked context.

    python -m nla.future_lens.baselines ngram --parquet <data>/eval.parquet --out evals/baselines.jsonl
    python -m nla.future_lens.baselines probe --train-parquet <data>/train.parquet \
        --parquet <data>/eval.parquet --out evals/baselines.jsonl [--leakage]
"""

from __future__ import annotations

import argparse
import json
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


def run_ngram(args):
    fl = load_fl_meta(args.sidecar or args.parquet)
    docs = load_docs(resolve_docs_path(args.parquet, fl))
    eval_rows = load_fl_rows(args.parquet, keep_activations=False,
                             columns=["doc_idx", "t", "target_ids", "target_top5", "p_top1", "activation_layer"])
    eval_docs = {int(r["doc_idx"]) for r in eval_rows}
    # one row per position (layers duplicate the position)
    positions = {(int(r["doc_idx"]), int(r["t"])): r for r in eval_rows}
    recs = []
    for order in [int(x) for x in args.orders.split(",")]:
        t0 = time.time()
        m = NGramModel(order)
        n_tok = 0
        for di, ids in docs.items():
            if di in eval_docs:
                continue                           # eval docs never counted
            m.add(ids); n_tok += len(ids)
        if args.extra_docs:
            for ids in _iter_extra_token_docs(args.extra_docs, args.extra_docs_max):
                m.add(ids); n_tok += len(ids)
        nf = fl.n_future
        hits = defaultdict(list); hits5 = defaultdict(list)
        for (di, t), r in positions.items():
            ro = m.greedy(docs[di][: t + 1], nf)
            tgt = np.asarray(r["target_ids"]); top5 = np.asarray(r["target_top5"]).reshape(-1, 5)
            for j in range(nf):
                hits[j].append(int(ro[j] == int(tgt[j])))
                hits5[j].append(int(ro[j] in set(int(x) for x in top5[j])))
        for k in fl.k_choices:
            N = k - 1
            base = {"layer": -1, "N": N, "k": k, "condition": "baseline", "injection": "none",
                    "seed": 0, "checkpoint": f"{order}gram", "n_train_tokens": n_tok}
            recs.append({**base, "metric": "p1", "value": float(np.mean(hits[N])), "n": len(hits[N])})
            recs.append({**base, "metric": "p5", "value": float(np.mean(hits5[N])), "n": len(hits5[N])})
        print(f"[ngram] order={order} tokens={n_tok} " +
              " ".join(f"p1@{N}={np.mean(hits[N]):.3f}" for N in range(nf)) + f" ({time.time() - t0:.0f}s)")
    write_records(recs, args.out)
    print(f"[ngram] wrote {len(recs)} records -> {args.out}")


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
    train = load_fl_rows(args.train_parquet, n_max=args.max_train_rows, layers=layers,
                         columns=["activation_vector", "activation_layer", "target_ids", "prev_ids", "doc_idx", "t"])
    ev = load_fl_rows(args.parquet, layers=layers,
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
            rec = {"layer": layer, "N": N, "k": N + 1 if N >= 0 else None, "condition": "baseline",
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
    n = sub.add_parser("ngram")
    n.add_argument("--parquet", required=True, help="eval.parquet")
    n.add_argument("--sidecar", default=None)
    n.add_argument("--orders", default="2,4")
    n.add_argument("--extra-docs", default=None, help="extra .jsonl with `ids` rows for counts")
    n.add_argument("--extra-docs-max", type=int, default=0)
    n.add_argument("--out", required=True)
    q = sub.add_parser("probe")
    q.add_argument("--train-parquet", required=True)
    q.add_argument("--parquet", required=True, help="eval.parquet")
    q.add_argument("--sidecar", default=None)
    q.add_argument("--base-ckpt", default="Qwen/Qwen3-8B", help="only used for the vocab size")
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
    {"ngram": run_ngram, "probe": run_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
