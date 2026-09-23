"""Truncated-context check: does the target's state carry USABLE document memory beyond
the last 20 tokens?

A reader that sees only the last WIN tokens of a document cannot quote the rest, so any
help it gets from a description of the state at t must come from what the state
remembers of the hidden prefix. Before training a verbalizer for that, check the object:

  Part 1  the missing control. The target's state at t from (a) the FULL prefix, (b) the
          WIN-token window ALONE, (c) the window preceded by ANOTHER document's hidden prefix
          (same long-context processing, wrong content). Retrieval: identify each document's
          hidden prefix among all held-out documents, adding each state's ridge prediction to
          the untrained window-word overlap score D (late fusion). PRIMARY: full - shuffled.
          Also checks the recomputed full-prefix state against the stored activation, to pin
          the layer index.
  Part 2  end to end. Keywords decoded from a state (ridge -> TF-IDF vocabulary -> top words)
          are shown to a reader that sees the window, which scores the target's own
          sampled continuations (the ids stored with the positions). Conditions:
            none              window only (no header; descriptive baseline)
            kw_window         extractive keywords of the VISIBLE window: same format, no hidden
                              information and nothing misleading (the no-information baseline)
            kw_extract        extractive keywords of the TRUE hidden prefix, names first;
                              kw_extract - kw_window is the positive control (not vocabulary-limited)
            kw_oracle         top TF-IDF words of the true hidden prefix within the decoders'
                              vocabulary: the ceiling of the decoded channel
            kw_full           decoded from the full-prefix state
            kw_shuf           decoded from the shuffled-prefix state (its own ridge)
            kw_short          decoded from the window-only state (its own ridge)
            *_wrong           the same keywords from another held-out document
            full              the whole prefix (ceiling)
          PRIMARY: kw_full - kw_shuf (header-matched, same processing, only the prefix differs).

The reader must share the target's tokenizer: continuation ids are scored as stored,
after prompt ids tokenized separately, so every condition scores bit-identical tokens.
Selection (one position per document, >= MIN_HIDDEN hidden tokens, seeds) matches the
2026-09-23 CPU check so its numbers are comparable."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WIN, MIN_HIDDEN, K_KW, T_CONT = 20, 128, 14, 24
BUCKETS = ((0, 8), (8, 16), (16, 24))
STOP = set("the a an and or of to in on at for with by from as is was are were be been it its this that these "
           "those he she they we you i his her their our your not no but if then than so which who what when "
           "where while into over about after before under between out up down off more most very can could "
           "would should may might will just also only".split())
WORD = re.compile(r"[A-Za-z0-9]+")


def extract_kw(text, n=K_KW):
    """The 2026-09-20 CPU check's oracle: capitalized words first, then frequent non-stopwords."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text)
    c = collections.Counter(w for w in words if w.lower() not in STOP)
    caps = [w for w in words if w[0].isupper() and w.lower() not in STOP]
    seen, out = set(), []
    for w in [w for w, _ in collections.Counter(caps).most_common(8)] + [w for w, _ in c.most_common(n)]:
        if w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return ", ".join(out[:n])


def log(*a):
    print(f"[trunc {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ selection ----
def select(T, tok, n_docs, max_prefix=None):
    """One position per document with >= MIN_HIDDEN tokens behind the window (as the CPU check)."""
    ids_all = [tok(s, add_special_tokens=False)["input_ids"] for s in T.prefix_text]
    ok = np.array([len(i) == n for i, n in zip(ids_all, T.n_raw_tokens)])
    elig = np.array([o and len(i) - WIN >= MIN_HIDDEN and (max_prefix is None or len(i) <= max_prefix)
                     for o, i in zip(ok, ids_all)])
    log(f"{len(T)} positions; round-trip ok {ok.mean() * 100:.1f}%; eligible {int(elig.sum())}")
    seen, keep = set(), []
    for i in np.random.default_rng(0).permutation(np.flatnonzero(elig)):
        if T.doc_id[i] not in seen:
            seen.add(T.doc_id[i]); keep.append(int(i))
    keep = keep[:n_docs]
    te = np.zeros(len(keep), bool)
    te[np.random.default_rng(1).permutation(len(keep))[:int(0.2 * len(keep))]] = True
    return keep, [ids_all[i] for i in keep], te


# ------------------------------------------------------------------ target states ----
@torch.no_grad()
def last_states(model, seqs, layers, device, budget, pad_id):
    """Residual after `L` blocks (hidden_states[L]) at each sequence's last token."""
    d = model.config.hidden_size
    out = {L: np.zeros((len(seqs), d), np.float32) for L in layers}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    def run(b):
        mx = max(len(seqs[i]) for i in b)
        x = torch.full((len(b), mx), pad_id); am = torch.zeros((len(b), mx), dtype=torch.long)
        for r, i in enumerate(b):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i]); am[r, :len(seqs[i])] = 1
        hs = model.model(input_ids=x.to(device), attention_mask=am.to(device),
                         output_hidden_states=True, use_cache=False).hidden_states
        last = torch.tensor([len(seqs[i]) - 1 for i in b], device=device)
        rows = torch.arange(len(b), device=device)
        for L in layers:
            out[L][b] = hs[L][rows, last].float().cpu().numpy()

    b = []
    for i in order:                                            # right-padded, length-sorted
        if b and (len(b) + 1) * len(seqs[i]) > budget:
            run(b); b = []
        b.append(i)
    run(b)
    return out


# ------------------------------------------------------------------ text space ----
class TextSpace:
    """TF-IDF of hidden prefixes (fit on TRAIN docs only) reduced by SVD to 256 dims."""

    def __init__(self, hid_txt, win_txt, tr):
        low = [[w.lower() for w in WORD.findall(s)] for s in hid_txt]
        df = collections.Counter()
        for j in np.flatnonzero(tr):
            df.update(set(low[j]))
        self.vocab = [w for w, c in df.most_common(20000) if c >= 3]
        vid = {w: i for i, w in enumerate(self.vocab)}
        idf = torch.tensor([math.log(tr.sum() / df[w]) for w in self.vocab], dtype=torch.float32)
        cased = collections.defaultdict(collections.Counter)      # most common surface form per word (train)
        for j in np.flatnonzero(tr):
            for w in WORD.findall(hid_txt[j]):
                cased[w.lower()][w] += 1
        self.surface = [cased[w].most_common(1)[0][0] if cased[w] else w for w in self.vocab]
        self.ok = np.array([len(w) >= 3 and w not in STOP for w in self.vocab])

        def tfidf(tokens):
            M = torch.zeros((len(tokens), len(self.vocab)))
            for r, ws in enumerate(tokens):
                for w, c in collections.Counter(ws).items():
                    if w in vid:
                        M[r, vid[w]] = 1 + math.log(c)
            M = M * idf
            return M / (M.norm(dim=1, keepdim=True) + 1e-9)

        self.H = tfidf(low)
        Hw = tfidf([[w.lower() for w in WORD.findall(s)] for s in win_txt])
        torch.manual_seed(0)
        q = min(256, int(tr.sum()) - 1, len(self.vocab) - 1)
        _, _, self.V = torch.svd_lowrank(self.H[torch.tensor(tr)], q=q, niter=4)
        Y = (self.H @ self.V).numpy()
        self.Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        self.WT = (Hw @ self.V).numpy()                           # window words, same basis

    def top(self, scores, k=K_KW):
        """The k highest-scoring allowed vocabulary words (surface forms)."""
        scores = np.where(self.ok, scores, -np.inf)
        return ", ".join(self.surface[i] for i in np.argsort(-scores)[:k] if np.isfinite(scores[i]))

    def oracle(self, i, k=K_KW):
        """Top TF-IDF words of document i's true hidden prefix: the most a vocabulary decoder can say."""
        h = self.H[i].numpy()
        return self.top(np.where(h > 0, h, -np.inf), k)


def nrm(P):
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)


def zs(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def ranks(S):
    return (S > np.diag(S)[:, None]).sum(1)                       # 0 = correct first


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1], dtype=np.float32), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


LAMS = (1e0, 1e1, 1e2, 1e3, 1e4, 3e4, 1e5)
BETAS = (0, 0.1, 0.25, 0.5, 1, 2, 4, 1e3)


def fit_state(X, ts, tr):
    """Ridge from a state to the hidden-prefix space. Lambda is chosen by RIDGE-ONLY validation MRR
    (never by the fusion, which would let a useless state pick lambda by noise), then beta for the
    late fusion on the same validation split. Returns the predictor refit on all of train."""
    X = X.astype(np.float32); itr = np.flatnonzero(tr); cut = int(0.8 * len(itr)); a, b = itr[:cut], itr[cut:]
    Y, WT = ts.Y, ts.WT

    def val_mrr(S):
        return np.mean(1 / (1 + ranks(S)))
    lam = max(LAMS, key=lambda l: val_mrr(nrm(ridge_fit(X[a], Y[a], l)(X[b])) @ Y[b].T))
    Db = zs(nrm(WT[b]) @ Y[b].T); Sb = zs(nrm(ridge_fit(X[a], Y[a], lam)(X[b])) @ Y[b].T)
    beta = max(BETAS, key=lambda be: val_mrr(Db + be * Sb))
    return ridge_fit(X[tr], Y[tr], lam), lam, beta


def fused_ranks(f, beta, X, ts, te):
    St = zs(nrm(f(X[te].astype(np.float32))) @ ts.Y[te].T)
    return ranks(zs(nrm(ts.WT[te]) @ ts.Y[te].T) + beta * St)


def boot(x, seed=2, B=4000):
    r = np.random.default_rng(seed)
    bs = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------ reader ----
@torch.no_grad()
def score(model, items, device, budget, pad_id, max_rows=128):
    """Teacher-forced log-prob of each T_CONT-token continuation after its prompt ids.
    Right-padded (no fully-masked rows); the unembedding is applied only at the T_CONT
    positions that predict continuation tokens."""
    res = np.zeros((len(items), T_CONT), np.float32)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))

    def run(b):
        seqs = [items[i][0] + items[i][1] for i in b]; mx = max(map(len, seqs))
        x = torch.full((len(b), mx), pad_id); am = torch.zeros((len(b), mx), dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        h = model.model(input_ids=x.to(device), attention_mask=am.to(device), use_cache=False).last_hidden_state
        start = torch.tensor([len(items[i][0]) - 1 for i in b], device=device)
        pos = start[:, None] + torch.arange(T_CONT, device=device)[None]      # position p predicts token p+1
        hsel = h[torch.arange(len(b), device=device)[:, None], pos]
        lp = torch.log_softmax(model.lm_head(hsel).float(), -1)
        tgt = torch.tensor([items[i][1] for i in b], device=device)
        out = lp.gather(-1, tgt[..., None])[..., 0]
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite reader log-probs")
        res[b] = out.cpu().numpy()

    b = []
    for i in order:
        if b and ((len(b) + 1) * (len(items[i][0]) + T_CONT) > budget or len(b) >= max_rows):
            run(b); b = []
        b.append(i)
    run(b)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--reader", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--out", required=True, help="small outputs (summaries, keywords, reader log-probs)")
    ap.add_argument("--states-dir", default=None, help="where states.npz goes (large); default --out")
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--layers", default="12,18,23,24,25,30",
                    help="hidden_states indices saved for every state; the one matching the stored "
                         "activation is used for Part 1/2")
    ap.add_argument("--fallback-layer", type=int, default=24,
                    help="used only when the target's width differs from the stored activation (CPU smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--state-budget", type=int, default=32768, help="padded tokens per target batch")
    ap.add_argument("--score-budget", type=int, default=65536, help="padded tokens per reader batch")
    ap.add_argument("--n-score-docs", type=int, default=None, help="score only the first N held-out docs (smoke)")
    ap.add_argument("--max-prefix-tokens", type=int, default=None, help="restrict selection (smoke only)")
    a = ap.parse_args()
    sd = a.states_dir or a.out
    os.makedirs(a.out, exist_ok=True); os.makedirs(sd, exist_ok=True)
    dt = getattr(torch, a.dtype); layers = [int(x) for x in a.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(a.target)
    rtok = AutoTokenizer.from_pretrained(a.reader)
    probe = ["Hello world, 2026.", " The Hartford Courant reported", "naïve café — ok"]
    if any(tok(s, add_special_tokens=False)["input_ids"] != rtok(s, add_special_tokens=False)["input_ids"]
           for s in probe):
        raise SystemExit("reader and target tokenizers differ; continuation ids cannot be scored as stored")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    T = pq.read_table(a.positions, columns=["doc_id", "n_raw_tokens", "prefix_text", "activation_vector",
                                            "cont_ids"]).to_pandas()
    keep, ids, te = select(T, tok, a.n_docs, a.max_prefix_tokens); tr = ~te; n = len(keep)
    ACT = np.stack([np.asarray(T.activation_vector[i], dtype=np.float32) for i in keep])
    conts = [[[int(t) for t in c] for c in T.cont_ids[i]] for i in keep]
    del T
    hid_txt = [tok.decode(x[:-WIN]) for x in ids]; win_txt = [tok.decode(x[-WIN:]) for x in ids]
    log(f"N={n} docs (train {int(tr.sum())}, test {int(te.sum())}); hidden tokens median "
        f"{int(np.median([len(x) - WIN for x in ids]))}")

    # shuffled-prefix control: this doc's window preceded by ANOTHER TRAIN doc's hidden prefix, so the
    # state has the same long-context processing but the wrong content (never a retrieval candidate)
    trn = np.flatnonzero(tr); r4 = np.random.default_rng(4); src = np.empty(n, int)
    p = r4.permutation(trn); src[p] = np.roll(p, 1)
    src[np.flatnonzero(te)] = r4.choice(trn, int(te.sum()))
    assert not np.any(src == np.arange(n))

    # ---------------- Part 1: states ----------------
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.target, dtype=dt, device_map=a.device).eval()
    FULL = last_states(m, ids, layers, a.device, a.state_budget, pad_id)
    SHORT = last_states(m, [x[-WIN:] for x in ids], layers, a.device, a.state_budget, pad_id)
    SHUF = last_states(m, [ids[src[i]][:-WIN] + ids[i][-WIN:] for i in range(n)], layers, a.device,
                       a.state_budget, pad_id)
    del m
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()
    log(f"target states done in {(time.time() - t0) / 60:.1f} min")
    np.savez_compressed(os.path.join(sd, "states.npz"), keep=np.array(keep), test=te, shuf_src=src,
                        **{f"{k}{L}": S[L].astype(np.float16) for k, S in (("full", FULL), ("short", SHORT),
                                                                           ("shuf", SHUF)) for L in layers})
    if FULL[layers[0]].shape[1] == ACT.shape[1]:
        cos = {L: float(np.mean(np.sum(nrm(FULL[L]) * nrm(ACT), 1))) for L in layers}
        log("cos(recomputed full-prefix state, stored activation): " + "  ".join(f"L{L} {c:.4f}" for L, c in cos.items()))
        LM = max(cos, key=cos.get)
        if cos[LM] < 0.9:
            raise SystemExit(f"no layer reproduces the stored activation (best L{LM} {cos[LM]:.4f}); refusing to "
                             "compare states at an unverified layer (states.npz is saved)")
        if cos[LM] < 0.98:
            log(f"WARNING: best match L{LM} has cos {cos[LM]:.4f} < 0.98; continuing at L{LM}")
    else:
        cos, LM = {}, a.fallback_layer
        log(f"target width {FULL[layers[0]].shape[1]} != stored {ACT.shape[1]} (smoke run): using layer {LM}")
    log(f"matched layer: hidden_states[{LM}]")

    ts = TextSpace(hid_txt, win_txt, tr)
    STATES = {"stored activation": ACT, "full-prefix state": FULL[LM], "shuffled-prefix state": SHUF[LM],
              "window-only state": SHORT[LM]}
    LAM, R, meta = {}, {"D (window words)": ranks(zs(nrm(ts.WT[te]) @ ts.Y[te].T))}, {}
    for k, X in STATES.items():
        f, lam, beta = fit_state(X, ts, tr); LAM[k] = lam; meta[f"D + {k}"] = (lam, beta)
        R[f"D + {k}"] = fused_ranks(f, beta, X, ts, te)
    nte = int(te.sum())
    lines = [f"PART 1: identify the hidden prefix among {nte} held-out docs (chance top-1 {100 / nte:.2f}%), layer {LM}",
             f"{'score':<28} {'top-1':>6} {'top-10':>7} {'MRR':>6}  (lambda, beta)"]
    for k, r in R.items():
        lines.append(f"{k:<28} {np.mean(r == 0) * 100:>5.1f}% {np.mean(r < 10) * 100:>6.1f}% "
                     f"{np.mean(1 / (1 + r)):>6.3f}  {meta.get(k, '')}")
    part1 = {"layer": LM, "cos": cos, "n_test": nte, "meta": meta,
             **{k: {"top1": float(np.mean(r == 0)), "mrr": float(np.mean(1 / (1 + r)))} for k, r in R.items()}}
    for x, y in (("D + full-prefix state", "D + shuffled-prefix state"),          # PRIMARY (Part 1)
                 ("D + full-prefix state", "D (window words)"), ("D + shuffled-prefix state", "D (window words)"),
                 ("D + window-only state", "D (window words)")):
        d, lo, hi = boot(1 / (1 + R[x]) - 1 / (1 + R[y]))
        lines.append(f"  {x} - {y}: MRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"); part1[f"{x} - {y}"] = (d, lo, hi)
    print("\n".join(lines), flush=True)
    json.dump({"args": vars(a), "part1": part1}, open(os.path.join(a.out, "summary_part1.json"), "w"),
              indent=1, default=float)

    # ---------------- Part 2: keywords -> reader ----------------
    ti = np.flatnonzero(te)[:a.n_score_docs]
    order = np.random.default_rng(3).permutation(len(ti)); other = np.empty(len(ti), int)
    other[order] = order[np.r_[1:len(ti), 0]]                   # derangement: another held-out doc
    # decoders: ridge from each state straight to the TF-IDF vocabulary (no SVD bottleneck), with
    # the lambda that state's retrieval probe chose; the vocabulary oracle is then the true ceiling
    Htr = ts.H[torch.tensor(tr)].numpy()

    def decoded(name, X):
        f = ridge_fit(X[tr].astype(np.float32), Htr, LAM[name])
        return [ts.top(v) for v in f(X[ti].astype(np.float32))]
    KW = {"kw_extract": [extract_kw(hid_txt[i]) for i in ti],
          "kw_window": [extract_kw(win_txt[i]) for i in ti],
          "kw_oracle": [ts.oracle(i) for i in ti],
          "kw_full": decoded("full-prefix state", FULL[LM]),
          "kw_shuf": decoded("shuffled-prefix state", SHUF[LM]),
          "kw_short": decoded("window-only state", SHORT[LM])}
    del Htr
    for c in ("kw_extract", "kw_oracle", "kw_full"):
        KW[f"{c}_wrong"] = [KW[c][j] for j in other]
    COND = ["none", "kw_window", "kw_extract", "kw_extract_wrong", "kw_oracle", "kw_oracle_wrong", "kw_full",
            "kw_full_wrong", "kw_shuf", "kw_short", "full"]
    items, where = [], []
    for c in COND:
        for r, i in enumerate(ti):
            if c == "none":
                ctx = ids[i][-WIN:]
            elif c == "full":
                ctx = ids[i]
            else:
                ctx = rtok(f"[Earlier in this document: {KW[c][r]}]\n", add_special_tokens=False)["input_ids"] + ids[i][-WIN:]
            for b, cont in enumerate(conts[i]):
                if len(cont) == T_CONT:
                    items.append((ctx, cont)); where.append((c, r, b))
    t0 = time.time()
    rd = AutoModelForCausalLM.from_pretrained(a.reader, dtype=dt, device_map=a.device).eval()
    LP = score(rd, items, a.device, a.score_budget, pad_id)
    log(f"reader scored {len(items)} continuations in {(time.time() - t0) / 60:.1f} min")
    S = {c: np.full((len(ti), 4, T_CONT), np.nan, np.float32) for c in COND}
    for (c, r, b), lp in zip(where, LP):
        S[c][r, b] = lp
    np.savez_compressed(os.path.join(a.out, "reader_logprobs.npz"), test_rows=ti, **S)
    json.dump({"test_rows": ti.tolist(), **KW}, open(os.path.join(a.out, "keywords.json"), "w"))
    valid = ~np.isnan(S["none"]).all(axis=(1, 2))              # docs with >= 1 full-length continuation
    log(f"docs with a full-length continuation: {int(valid.sum())}/{len(ti)}")

    def per_doc(c, lo, hi):                                     # nats/token, mean over branches
        return np.nanmean(S[c][valid][:, :, lo:hi], axis=(1, 2))
    part2 = {}
    lines = [f"PART 2: reader {a.reader} sees the last {WIN} tokens; nats/token, {int(valid.sum())} held-out docs "
             f"x 4 continuations. X - none (descriptive; 'none' has no header):",
             f"{'condition':<17}" + "".join(f" {f'tok {lo}-{hi}':>16}" for lo, hi in BUCKETS) + f" {'all 24':>16}"]
    for c in COND[1:]:
        row = f"{c:<17}"
        for lo, hi in BUCKETS + ((0, T_CONT),):
            d, l_, h_ = boot(per_doc(c, lo, hi) - per_doc("none", lo, hi))
            row += f" {d:+.3f}[{l_:+.2f},{h_:+.2f}]"; part2[f"{c} - none tok{lo}-{hi}"] = (d, l_, h_)
        lines.append(row)
    lines.append("header-matched contrasts (all 24 tokens | tokens 16-24):")
    for x, y in (("kw_full", "kw_shuf"),                                           # PRIMARY (Part 2)
                 ("kw_extract", "kw_window"),                                      # positive control (S2)
                 ("kw_oracle", "kw_window"), ("kw_extract", "kw_extract_wrong"), ("kw_oracle", "kw_oracle_wrong"),
                 ("kw_full", "kw_full_wrong"), ("kw_full", "kw_short"), ("kw_shuf", "kw_short")):
        s_all = boot(per_doc(x, 0, T_CONT) - per_doc(y, 0, T_CONT)); s_hl = boot(per_doc(x, 16, 24) - per_doc(y, 16, 24))
        lines.append(f"  {x} - {y}: {s_all[0]:+.4f} [{s_all[1]:+.4f}, {s_all[2]:+.4f}] | "
                     f"{s_hl[0]:+.4f} [{s_hl[1]:+.4f}, {s_hl[2]:+.4f}]")
        part2[f"{x} - {y} all"] = s_all; part2[f"{x} - {y} tok16-24"] = s_hl
    lines.append("examples: extractive | vocabulary oracle | decoded full | decoded shuffled-prefix | decoded window-only")
    for r in range(min(5, len(ti))):
        lines.append("  " + "\n    ".join(f"[{KW[c][r]}]" for c in ("kw_extract", "kw_oracle", "kw_full", "kw_shuf", "kw_short")))
    print("\n".join(lines), flush=True)
    json.dump({"args": vars(a), "part1": part1, "part2": part2}, open(os.path.join(a.out, "summary.json"), "w"),
              indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
