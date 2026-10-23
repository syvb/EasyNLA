"""Do quoted terms in head explanations name the tokens the head attends to? Every (token, head) site of the doc
viewer documents, for each explanation source (std RL, matryoshka, SFT, gold).

    python scripts/heads_quote_attention.py --docs <doc_viewer dirs with sites.parquet, attn.npz, mtry.parquet> \\
        [--out <dir>]

A quote is a "…"/“…” span (as in heads_quote_sources.py). It matches a context position j ≤ t (the site's token t
and everything before it) when one of its words (normalized alphanumerics; stopwords dropped unless the quote is only
stopwords) equals token j's normalized text, or a ≥4-character token is a substring of a quote word or vice versa —
"or similar" at the subword level. Each quote is classed as matching the current token, a position among the
head's top-5 attended non-sink positions, another context position, or nothing in context.

Attention is evaluation evidence only. Head-specific control: for a quote whose matches M are in context, the
explaining head's attention mass on M is compared with every other head's mass on the same M at the same token;
if quotes name what THIS head reads, its mass is higher (reported as the mean ratio and the fraction of quotes
where it beats the median other head; the same for "M hits the top-5").
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

QRE = re.compile(r'["“]([^"”]{1,60})["”]')
STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "it", "that", "this", "with", "as", "at",
        "by", "be", "not", "no", "from", "into", "over", "after", "between", "next", "now", "first", "more", "further",
        "than", "up", "out", "but", "so", "if", "then"}


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", "", s.lower())


def quote_words(q: str) -> list[str]:
    w = [norm(x) for x in re.split(r"\s+", q) if norm(x)]
    c = [x for x in w if x not in STOP]
    return c or w


def matches(words: list[str], toks: list[str]) -> np.ndarray:
    """Context positions whose normalized token matches a quote word (exact, or ≥4-char substring either way)."""
    out = []
    for j, t in enumerate(toks):
        if not t:
            continue
        for w in words:
            if t == w or (len(t) >= 4 and (t in w or (len(w) >= 4 and w in t))):
                out.append(j)
                break
    return np.array(out, dtype=int)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--top-k", type=int, default=5)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)

    recs = []
    for dd in sorted(Path(a.docs).iterdir()):
        if not (dd / "sites.parquet").exists():
            continue
        ids = np.load(dd / "token_ids.npy").tolist()
        toks = [norm(tok.decode([i])) for i in ids]
        attn = np.load(dd / "attn.npz")["attn"].astype(np.float32)          # [H, T, T]
        H = attn.shape[0]
        s = pq.read_table(dd / "sites.parquet", columns=["pos", "head", "rl_text", "sft_text", "gold_text"]).to_pandas()
        if (dd / "mtry.parquet").exists():
            s = s.merge(pq.read_table(dd / "mtry.parquet", columns=["pos", "head", "mtry_text"]).to_pandas(),
                        on=["pos", "head"], validate="1:1")
        srcs = [c for c in ("rl_text", "mtry_text", "sft_text", "gold_text") if c in s]
        for r in s.itertuples(index=False):
            t, h = int(r.pos), int(r.head)
            row = attn[h, t, : t + 1]
            ns = row.copy()
            ns[0] = -1                                                         # sink excluded from the top-k
            topk = set(np.argsort(-ns)[: a.top_k].tolist()) if t > 0 else set()
            for src in srcs:
                text = getattr(r, src) or ""
                qs = list(QRE.finditer(text))
                p16 = None
                if src == "mtry_text" and qs:
                    p16 = len(tok.decode(tok.encode(text, add_special_tokens=False)[:16]))
                recs.append({"doc": dd.name, "src": src, "pos": t, "head": h, "n_quotes": len(qs), "quote": None})
                for m in qs:
                    q = m.group(1)
                    M = matches(quote_words(q), toks[: t + 1])
                    rec = {"doc": dd.name, "src": src, "pos": t, "head": h, "n_quotes": None, "quote": q,
                           "early": (m.start() < p16) if p16 is not None else None,
                           "cur": bool(t in set(M.tolist())), "in_ctx": bool(len(M)),
                           "top": bool(set(M.tolist()) & topk), "nonlocal": bool(len(M) and (M < t).any())}
                    if len(M):
                        mass = attn[:, t, M].sum(-1)                           # [H] every head's mass on M at token t
                        others = np.delete(mass, h)
                        tops_o = []
                        for h2 in range(H):
                            if h2 == h:
                                continue
                            ns2 = attn[h2, t, : t + 1].copy()
                            ns2[0] = -1
                            tops_o.append(bool(set(M.tolist()) & set(np.argsort(-ns2)[: a.top_k].tolist())))
                        rec.update(mass=float(mass[h]), mass_oth_mean=float(others.mean()),
                                   beats_median=bool(mass[h] > np.median(others)), top_oth=float(np.mean(tops_o)))
                    recs.append(rec)
        print(f"{dd.name}: {len(s)} sites", flush=True)
    R = pd.DataFrame(recs)
    E = R[R.quote.isna()]
    Q = R[R.quote.notna()].copy()

    names = {"rl_text": "std RL", "mtry_text": "matryoshka", "sft_text": "SFT", "gold_text": "gold (Haiku)"}
    rows = []
    for src, g in Q.groupby("src", sort=False):
        e = E[E.src == src]
        c = g[g.in_ctx]
        rows.append({"source": names[src], "explanations with a quote": (e.n_quotes > 0).mean(),
                     "quotes / explanation": e.n_quotes.mean(), "n quotes": len(g),
                     "= current token": g.cur.mean(), f"in head's top-{a.top_k} attended": g.top.mean(),
                     "in context (anywhere)": g.in_ctx.mean(), "not in context": 1 - g.in_ctx.mean(),
                     f"top-{a.top_k}, other heads (control)": c.top_oth.mean() * len(c) / len(g),
                     "mass on quoted tokens: this head": c["mass"].mean(),
                     "… other heads' mean": c.mass_oth_mean.mean(),
                     "this head > median other head": c.beats_median.mean()})
    T = pd.DataFrame(rows).set_index("source").T
    pd.set_option("display.width", 200)
    print("\nAll quotes (fractions of quotes unless noted; control = the same quote vs the other 31 heads at the token)")
    print(T.to_string(float_format=lambda x: f"{x:.3f}"))

    # matryoshka: quotes in the first 16 tokens vs later
    mq = Q[Q.src == "mtry_text"]
    if len(mq):
        print("\nmatryoshka quotes, first 16 tokens vs later:")
        for k, g in mq.groupby("early"):
            c = g[g.in_ctx]
            print(f"  {'first 16' if k else 'later   '}: n {len(g):6d} · current {g.cur.mean():.3f} · top-{a.top_k} "
                  f"{g.top.mean():.3f} · in ctx {g.in_ctx.mean():.3f} · mass this/other {c['mass'].mean():.3f}/"
                  f"{c.mass_oth_mean.mean():.3f} · beats median {c.beats_median.mean():.3f}")
    # per head: which heads' quotes track their attention
    ph = []
    for (src, h), g in Q[Q.src.isin(["rl_text", "mtry_text"])].groupby(["src", "head"]):
        c = g[g.in_ctx & ~g.cur]
        ph.append({"src": names[src], "head": h, "n": len(g), "current": g.cur.mean(), "top": g.top.mean(),
                   "in_ctx": g.in_ctx.mean(), "nonself_mass_ratio": (c["mass"].mean() / max(c.mass_oth_mean.mean(), 1e-9))
                   if len(c) else np.nan, "beats_median_nonself": c.beats_median.mean() if len(c) else np.nan})
    P = pd.DataFrame(ph).pivot(index="head", columns="src")
    print("\nper head (quotes that are not the current token: this head's attention mass on them ÷ other heads' mean)")
    print(P[["n", "current", "top", "nonself_mass_ratio", "beats_median_nonself"]].round(2).to_string())
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        Q.to_parquet(out / "quotes.parquet")
        ph_json = {f"{k[0]}|{k[1]}": v for k, v in P.round(4).to_dict().items()}
        (out / "summary.json").write_text(json.dumps({"overall": T.to_dict(), "per_head": ph_json}, indent=1, default=float))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
