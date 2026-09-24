"""Can a TRAINED verbalizer say what the state remembers of the hidden prefix?

The premise checks (trunc_check.py, trunc_expl.py) found the Qwen3-8B L24 state holds some memory of
the text behind the last 20 tokens, that linear/MLP probes recover only ~0.3 hidden-prefix-only words
of 14, and that the existing SFT verbalizer (trained on next-token-feature explanations) says about
as little. This is the supervised ceiling test: fine-tune the verbalizer to name the hidden prefix's
keywords from the state, and compare with an identical verbalizer trained on window-only states.

  build  From the NLA warm-start parquets (activation + prefix text), keep positions with >= 128
         tokens behind a 20-token window, <= 2 per document, none whose text starts like one of
         the 1000 held-out test documents. Target = 14 extractive keywords of the hidden prefix
         (names first; the extractor that gave a 20-token reader +0.13 nats/token), as
         "<explanation>\\nw1, w2, ...\\n</explanation>". Two parquets with IDENTICAL targets:
           F_train  the stored full-context activation
           S_train  the window-only activation (20 tokens alone), computed here at the layer
                    whose full-prefix recompute reproduces the stored activation.
  (train with nla.train_sft, one LoRA per parquet, same base and settings)
  eval   On the 1000 held-out test documents (trunc_check selection), greedy generations from
         F on full-prefix states and from S on window-only states -- each head in its own
         training distribution. PRIMARY: hidden-prefix-only words named, F - S. Reader: the
         20-token Qwen3-4B-Base reader with the keyword header, uniform and hidden-prefix-gap
         weighted (see GPU_RESULTS reward-shaping check)."""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trunc_check import (BUCKETS, STOP, T_CONT, WIN, boot, extract_kw, last_states, log, score,  # noqa: E402
                         select)

MIN_HIDDEN = 128
HEADER = "[Earlier in this document: {}]\n"
CW = re.compile(r"[a-z][a-z0-9']{2,}")


def cw(s):
    return {w for w in CW.findall((s or "").lower()) if w not in STOP}


def start_key(s, n=80):
    return " ".join(s.lower().split())[:n]


def test_prefix_starts(positions, tok):
    T = pq.read_table(positions, columns=["doc_id", "n_raw_tokens", "prefix_text"]).to_pandas()
    keep, _, te = select(T, tok, 5000)
    return {start_key(T.prefix_text[keep[i]]) for i in np.flatnonzero(te)}


# ------------------------------------------------------------------ build ----
def build(a):
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.target)
    avoid = test_prefix_starts(a.positions, tok)
    rng = np.random.default_rng(0)
    rows = []                                             # (source, row index, ids, doc_id)
    stats = {}
    for src in a.sources:
        t = pq.read_table(src, columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated"])
        docs, nraw, texts = (t.column(c).to_pylist() for c in ("doc_id", "n_raw_tokens", "detokenized_text_truncated"))
        n_ok = n_rt = n_test = 0
        for i, (d, n, s) in enumerate(zip(docs, nraw, texts)):
            if n < MIN_HIDDEN + WIN or (a.max_prefix_tokens and n > a.max_prefix_tokens):
                continue
            if start_key(s) in avoid:
                n_test += 1; continue
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) != n:
                n_rt += 1; continue
            rows.append((src, i, ids, d)); n_ok += 1
        stats[os.path.basename(src)] = {"eligible": n_ok, "roundtrip_fail": n_rt, "dropped_test_overlap": n_test}
        log(f"{os.path.basename(src)}: {n_ok} eligible, {n_rt} round-trip failures, {n_test} dropped (test overlap)")
    by_doc = collections.defaultdict(list)
    for r in rows:
        by_doc[r[3]].append(r)
    rows = [r for d in by_doc for r in [by_doc[d][j] for j in rng.permutation(len(by_doc[d]))[:a.max_per_doc]]]
    rows = [rows[j] for j in rng.permutation(len(rows))][:a.n_train]
    log(f"selected {len(rows)} positions from {len({r[3] for r in rows})} documents")

    # stored full-context activations for the selected rows
    need = collections.defaultdict(list)
    for k, r in enumerate(rows):
        need[r[0]].append((r[1], k))
    FULL = np.zeros((len(rows), a.d_model), np.float32)
    for src, lst in need.items():
        col = pq.read_table(src, columns=["activation_vector"]).column("activation_vector").combine_chunks()
        flat = col.values.to_numpy(zero_copy_only=False).reshape(-1, col.type.list_size)
        for i, k in lst:
            FULL[k] = flat[i]
        del col, flat

    # which hidden_states index reproduces the stored activation (checked per source), then window states
    m = AutoModelForCausalLM.from_pretrained(a.target, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cands = [int(x) for x in a.layer_candidates.split(",")]
    cos = {}
    for src, lst in need.items():
        pick = [k for _, k in lst[:a.check_rows]]
        S = last_states(m, [rows[k][2] for k in pick], cands, a.device, a.state_budget, pad)
        if S[cands[0]].shape[1] != FULL.shape[1]:
            cos[os.path.basename(src)] = None; continue
        cos[os.path.basename(src)] = {L: float(np.mean(np.sum(S[L] * FULL[pick], 1) /
                                                     (np.linalg.norm(S[L], axis=1) * np.linalg.norm(FULL[pick], axis=1))))
                                      for L in cands}
        log(f"cos(recomputed, stored) {os.path.basename(src)}: " + "  ".join(f"L{L} {c:.4f}" for L, c in cos[os.path.basename(src)].items()))
    if any(v is None for v in cos.values()):
        LM = a.fallback_layer; log(f"width mismatch (smoke run): window states at layer {LM}")
    else:
        LM = max(cands, key=lambda L: min(c[L] for c in cos.values()))
        worst = min(c[LM] for c in cos.values())
        if worst < 0.98:
            raise SystemExit(f"layer {LM} reproduces a stored activation set only at cos {worst:.4f}; refusing")
        log(f"window-only states at hidden_states[{LM}] (worst per-source cos {worst:.4f})")
    t0 = time.time()
    SHORT = last_states(m, [r[2][-WIN:] for r in rows], [LM], a.device, a.state_budget, pad)[LM]
    log(f"window-only states for {len(rows)} positions in {(time.time() - t0) / 60:.1f} min")
    del m
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # targets and parquets (identical prompts and targets; only the activation differs)
    prompt = pq.read_table(a.prompt_source, columns=["prompt"]).column("prompt")[0].as_py()
    kws = [extract_kw(tok.decode(r[2][:-WIN])) for r in rows]
    ok = [k for k, kw in enumerate(kws) if kw]
    log(f"{len(ok)} of {len(rows)} positions have a non-empty keyword target")
    resp = [f"<explanation>\n{kws[k]}\n</explanation>" for k in ok]
    for name, X in (("F_train", FULL), ("S_train", SHORT)):
        X = X[ok].astype(np.float32)
        tbl = pa.table({"prompt": [prompt] * len(ok), "response": resp,
                        "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1)), X.shape[1]),
                        "doc_id": [rows[k][3] for k in ok]})
        path = os.path.join(a.out, f"{name}.parquet")
        pq.write_table(tbl, path, row_group_size=512)
        shutil.copy(a.sidecar_source + ".nla_meta.yaml", path + ".nla_meta.yaml")
    json.dump({"n": len(ok), "layer": LM, "cos": cos, "sources": stats,
               "example_targets": resp[:5]}, open(os.path.join(a.out, "build.json"), "w"), indent=1)
    log(f"wrote F_train / S_train ({len(ok)} rows each); example target: {resp[0]!r}")


# ------------------------------------------------------------------ eval ----
def evaluate(a):
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.target_tokenizer)
    rtok = AutoTokenizer.from_pretrained(a.reader)
    probe = ["Hello world, 2026.", " The Hartford Courant reported", "naïve café — ok"]
    if any(tok(s, add_special_tokens=False)["input_ids"] != rtok(s, add_special_tokens=False)["input_ids"] for s in probe):
        raise SystemExit("reader and target tokenizers differ")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = pq.read_table(a.positions, columns=["doc_id", "n_raw_tokens", "prefix_text", "activation_vector",
                                            "cont_ids", "prompt"]).to_pandas()
    keep, ids, te = select(T, tok, 5000)
    Z = np.load(a.states)
    if not (np.array_equal(Z["keep"], np.array(keep)) and np.array_equal(Z["test"], te)):
        raise SystemExit("selection does not reproduce states.npz")
    if a.build_json:                                      # S was trained on window states at this layer
        a.layer = json.load(open(a.build_json))["layer"]
        if f"short{a.layer}" not in Z:
            raise SystemExit(f"states.npz has no window-only states at layer {a.layer}")
    ti = np.flatnonzero(te)[:a.n_score_docs]
    ACT = np.stack([np.asarray(T.activation_vector[keep[i]], dtype=np.float32) for i in ti])
    SH = Z[f"short{a.layer}"][ti].astype(np.float32)
    prompt = [dict(m) for m in T.prompt[keep[ti[0]]]]
    conts = [[[int(t) for t in c] for c in T.cont_ids[keep[i]]] for i in ti]
    hid = [tok.decode(ids[i][:-WIN]) for i in ti]; win = [tok.decode(ids[i][-WIN:]) for i in ti]
    win_ids = [ids[i][-WIN:] for i in ti]
    del T

    TX = {}
    if a.fake_gen:
        TX["F_full"] = [extract_kw(h)[:40] for h in hid]; TX["S_short"] = [extract_kw(w) for w in win]
        TX["S_on_full"] = [extract_kw(h)[:20] for h in hid]; TX["F_on_short"] = [extract_kw(w)[:20] for w in win]
    else:
        from nla.config import load_nla_config
        from nla.pred.av import generate_explanations, load_av
        atok = AutoTokenizer.from_pretrained(a.av)
        cfg = load_nla_config(a.positions, atok)
        inj = (cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
        # each head on its own training distribution, plus the cross conditions (head fixed, state swapped)
        for head, adapter, jobs in (("F", a.f_adapter, (("F_full", ACT), ("F_on_short", SH))),
                                    ("S", a.s_adapter, (("S_short", SH), ("S_on_full", ACT)))):
            model, vref = load_av(a.av, adapter, device=a.device, dtype=torch.bfloat16, inj_ids=inj)
            for name, X in jobs:
                t0 = time.time()
                rows = [{"prompt": prompt, "activation": v} for v in X]
                TX[name] = generate_explanations(model, atok, vref, rows, cfg.injection_char, max_new_tokens=a.max_new_tokens,
                                                 temperature=0.0, batch_size=a.gen_batch, device=a.device)
                log(f"{name}: extraction {np.mean([t is not None for t in TX[name]]):.1%} ({(time.time() - t0) / 60:.1f} min)")
            del model, vref
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
    order = np.random.default_rng(3).permutation(len(ti)); other = np.empty(len(ti), int)
    other[order] = order[np.r_[1:len(ti), 0]]
    TX["F_wrong"] = [TX["F_full"][j] for j in other]
    TX["oracle"] = [extract_kw(h) for h in hid]
    json.dump({"test_rows": ti.tolist(), **TX}, open(os.path.join(a.out, "generations.json"), "w"))

    # ---- content (reader-free). A failed extraction (None) names nothing; its rate is reported per condition.
    CAP = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}")
    only = [cw(h) - cw(w) for h, w in zip(hid, win)]
    winw = [cw(w) for w in win]
    names = [set(CAP.findall(h)) - set(CAP.findall(w)) for h, w in zip(hid, win)]
    df = collections.Counter(w for h in hid for w in cw(h))
    idf = {w: np.log(len(hid) / c) for w, c in df.items()}
    M = {"words": lambda t, r: len(cw(t) & only[r]),
         "names": lambda t, r: len(set(CAP.findall(t or "")) & names[r]),
         "idf_words": lambda t, r: sum(idf.get(w, 0.0) for w in cw(t) & only[r]),
         "window_words": lambda t, r: len(cw(t) & winw[r])}
    CONDS = ("F_full", "S_short", "S_on_full", "F_on_short", "F_wrong", "oracle")
    V = {c: {k: np.array([f(t, r) for r, t in enumerate(TX[c])], float) for k, f in M.items()} for c in CONDS}
    res = {"n_docs": len(ti), "extraction": {c: float(np.mean([t is not None for t in TX[c]])) for c in CONDS}}
    lines = [f"CONTENT ({len(ti)} held-out docs), mean per doc; extraction rate in []"]
    for c in CONDS:
        res[c] = {k: float(v.mean()) for k, v in V[c].items()}
        lines.append(f"  {c:<10} [{res['extraction'][c]:.0%}] " + "  ".join(f"{k} {v.mean():.2f}" for k, v in V[c].items()))
    for x, y in (("F_full", "S_short"), ("S_on_full", "S_short"), ("F_full", "F_on_short"), ("F_full", "F_wrong"),
                 ("S_short", "F_wrong")):
        for k in M:
            d = boot(V[x][k] - V[y][k]); res[f"{x} - {y} {k}"] = d
        lines.append(f"  {x} - {y}: " + "  ".join(f"{k} {res[f'{x} - {y} {k}'][0]:+.2f} [{res[f'{x} - {y} {k}'][1]:+.2f},"
                                                    f"{res[f'{x} - {y} {k}'][2]:+.2f}]" for k in ("words", "names", "idf_words", "window_words")))
    hl = np.array([len(ids[i]) - WIN for i in ti]); cut = np.quantile(hl, [1 / 3, 2 / 3])
    for lab, msk in (("short hidden", hl <= cut[0]), ("mid", (hl > cut[0]) & (hl <= cut[1])), ("long hidden", hl > cut[1])):
        d = boot((V["F_full"]["words"] - V["S_short"]["words"])[msk]); res[f"F_full - S_short words {lab}"] = d
        lines.append(f"  F_full - S_short words, {lab} tercile (n={int(msk.sum())}): {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]")
    print("\n".join(lines), flush=True)

    # ---- reader
    COND = ["none", "full", "F_full", "F_wrong", "S_short", "S_on_full", "oracle"]
    items, where = [], []
    for c in COND:
        for r in range(len(ti)):
            if c == "none":
                ctx = win_ids[r]
            elif c == "full":
                ctx = ids[ti[r]]
            elif TX[c][r] is None:
                continue
            else:
                ctx = rtok(HEADER.format(" ".join(TX[c][r].split())), add_special_tokens=False)["input_ids"] + win_ids[r]
            for b, cont in enumerate(conts[r]):
                if len(cont) == T_CONT:
                    items.append((ctx, cont)); where.append((c, r, b))
    dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    rd = AutoModelForCausalLM.from_pretrained(a.reader, dtype=dt, device_map=a.device).eval()
    t0 = time.time(); LP = score(rd, items, a.device, a.score_budget, pad)
    log(f"reader scored {len(items)} continuations in {(time.time() - t0) / 60:.1f} min")
    L = {c: np.full((len(ti), 4, T_CONT), np.nan, np.float32) for c in COND}
    for (c, r, b), lp in zip(where, LP):
        L[c][r, b] = lp
    np.savez_compressed(os.path.join(a.out, "reader_logprobs.npz"), test_rows=ti, **L)
    gap = np.clip(L["full"] - L["none"], 0, None)

    def per_doc(c, w):
        """Weighted mean log-prob per doc; NaN (dropped) where the doc has nothing scored."""
        x = L[c]; ok = ~np.isnan(x) & ~np.isnan(w); x = np.where(ok, x, 0); ww = np.where(ok, w, 0)
        den = ww.sum((1, 2))
        return np.where(den > 0, (ww * x).sum((1, 2)) / np.maximum(den, 1e-9), np.nan)
    lines = ["READER (Qwen3-4B-Base, last 20 tokens + keyword header), nats/token"]
    for wl, w in (("uniform", np.ones_like(gap)), ("gap-weighted", gap)):
        for x, y in (("F_full", "S_short"), ("S_on_full", "S_short"), ("F_full", "F_wrong"), ("oracle", "S_short"),
                     ("F_full", "none"), ("S_short", "none")):
            d = per_doc(x, w) - per_doc(y, w); d = d[~np.isnan(d)]
            s_ = boot(d); res[f"reader {wl} {x} - {y}"] = s_
            lines.append(f"  {wl:<12} {x} - {y}: {s_[0]:+.4f} [{s_[1]:+.4f}, {s_[2]:+.4f}]  (n={len(d)})")
    lines.append("examples (hidden-prefix keywords | F on full state | S on window-only state):")
    for r in range(min(6, len(ti))):
        lines.append(f"  [{TX['oracle'][r]}]\n    [{TX['F_full'][r]}]\n    [{TX['S_short'][r]}]")
    print("\n".join(lines), flush=True)
    json.dump({"args": vars(a), **res}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log("done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", nargs="+", required=True, help="NLA parquets with activation_vector + detokenized_text_truncated")
    b.add_argument("--prompt-source", required=True, help="parquet whose first row's `prompt` is the AV actor prompt")
    b.add_argument("--sidecar-source", required=True, help="parquet whose .nla_meta.yaml is copied to the outputs")
    b.add_argument("--positions", required=True, help="positions parquet (for the held-out test documents)")
    b.add_argument("--target", default="Qwen/Qwen3-8B")
    b.add_argument("--out", required=True)
    b.add_argument("--n-train", type=int, default=None)
    b.add_argument("--max-per-doc", type=int, default=2)
    b.add_argument("--max-prefix-tokens", type=int, default=None)
    b.add_argument("--d-model", type=int, default=4096)
    b.add_argument("--layer-candidates", default="24,25")
    b.add_argument("--fallback-layer", type=int, default=25)
    b.add_argument("--check-rows", type=int, default=64)
    b.add_argument("--device", default="cuda"); b.add_argument("--dtype", default="bfloat16")
    b.add_argument("--state-budget", type=int, default=32768)
    e = sub.add_parser("eval")
    e.add_argument("--positions", required=True); e.add_argument("--states", required=True)
    e.add_argument("--av", default="syvb/nanonla-qwen3-8b-L24-av")
    e.add_argument("--f-adapter"); e.add_argument("--s-adapter")
    e.add_argument("--reader", default="Qwen/Qwen3-4B-Base")
    e.add_argument("--target-tokenizer", default="Qwen/Qwen3-8B")
    e.add_argument("--layer", type=int, default=25)
    e.add_argument("--build-json", default=None, help="take --layer from the build step's verified layer")
    e.add_argument("--out", required=True)
    e.add_argument("--n-score-docs", type=int, default=None)
    e.add_argument("--gen-batch", type=int, default=64)
    e.add_argument("--max-new-tokens", type=int, default=96)
    e.add_argument("--device", default="cuda"); e.add_argument("--score-budget", type=int, default=65536)
    e.add_argument("--fake-gen", action="store_true", help="CPU smoke: text stand-ins, no verbalizer")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()
