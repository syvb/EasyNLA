"""Where do quoted strings in AV explanations come from? Matches each quote against the site's evidence.

    python scripts/heads_quote_sources.py --gen <checks dir>/generations.parquet --ctx <eval_ctx.parquet> \
        --texts <texts.parquet> [--column av_true]

Sources per eval row: the current token, the head's top-16 attended tokens, the previous 63 tokens of
context, and the write's avg16 / lens readouts. A quote matches a source if one of its words (normalized
alphanumerics; stopwords skipped for content quotes) equals a source token or a ≥4-character source token
is a substring of it (or vice versa) — rule-sensitive by ~±10 points. Chance = the same test against the
sets of another write of the same head (`shuf_partner_row`). Attention is evaluation evidence only.
"""

import argparse
import ast
import collections
import re

import numpy as np
import pandas as pd

QRE = re.compile(r'["“]([^"”]{1,60})["”]')
STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "it", "that", "this", "with", "as", "at",
        "by", "be", "not", "no", "from", "into", "over", "after", "between", "next", "now", "first", "more", "further",
        "than", "up", "out", "but", "so", "if", "then"}
ORDER = ["current token", "attended top-16", "context (prev 63 tok)", "avg16 readout", "lens readout"]


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", "", s.lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen", required=True)
    p.add_argument("--ctx", required=True)
    p.add_argument("--texts", required=True)
    p.add_argument("--column", default="av_true")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    g = pd.read_parquet(args.gen)
    c = pd.read_parquet(args.ctx).set_index("example_idx")
    tx = pd.read_parquet(args.texts).set_index("example_idx")

    def dec(i):
        return norm(tok.decode([int(i)]))

    def sets(e):
        r, t = c.loc[e], tx.loc[e]
        ap = np.asarray(r.att_pos)
        S = {"current token": {dec(r.tok_cur)},
             "attended top-16": {dec(r.att_tok[k]) for k in range(len(ap)) if ap[k] != r.pos},
             "context (prev 63 tok)": {dec(i) for i in r.ctx_ids[:-1]},
             "avg16 readout": {norm(s) for s in ast.literal_eval("[" + t["avg16"] + "]")},
             "lens readout": {norm(s) for s in ast.literal_eval("[" + t["lens"] + "]")}}
        return {k: {s for s in v if s} for k, v in S.items()}

    def match(q, S, content):
        ws = [norm(x) for x in re.split(r"[\s\-/]+", q)]
        ws = [w for w in ws if w and (not content or w not in STOP)]
        return any(w in S or any(len(s) >= 4 and (s in w or w in s) for s in S) for w in ws)

    ex = list(g.example_idx)
    partner = g.shuf_partner_row.values
    ST = {e: sets(e) for e in ex}
    rows_q = sum(bool(QRE.findall(t)) for t in g[args.column])
    print(f"{rows_q}/{len(ex)} rows contain quotes")
    for content in (False, True):
        res, n, anyt, anyc = collections.defaultdict(lambda: [0, 0]), 0, 0, 0
        for i, e in enumerate(ex):
            for q in QRE.findall(g[args.column].iloc[i]):
                ws = [norm(x) for x in re.split(r"[\s\-/]+", q)]
                if all((not w) or w in STOP for w in ws) == content:
                    continue
                n += 1
                mt = [match(q, ST[e][k], content) for k in ORDER]
                mc = [match(q, ST[ex[partner[i]]][k], content) for k in ORDER]
                for k, a, b in zip(ORDER, mt, mc):
                    res[k][0] += a
                    res[k][1] += b
                anyt += any(mt)
                anyc += any(mc)
        print(f"\n{'CONTENT-word' if content else 'FUNCTION-word-only'} quotes (n={n}): true row vs chance (another write, same head)")
        for k in ORDER:
            print(f"  {k:24s} {100 * res[k][0] / max(n, 1):5.1f}%  vs {100 * res[k][1] / max(n, 1):5.1f}%")
        print(f"  {'ANY source':24s} {100 * anyt / max(n, 1):5.1f}%  vs {100 * anyc / max(n, 1):5.1f}%")


if __name__ == "__main__":
    main()
