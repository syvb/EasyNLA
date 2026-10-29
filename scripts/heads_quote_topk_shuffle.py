"""Do quotes in head explanations name one of the head's top-k attended tokens more often than a shuffled
explanation's quotes do? Every (token, head) site of the doc-viewer texts and chat-viewer chats.

    python scripts/heads_quote_topk_shuffle.py \\
        --doc hf:l24/std_rl/doc_viewer/01_journalism_ocoee_homicide ... --doc hf:l24/chat_viewer/01_ipv4_code ... \\
        --mtry-root hf:l24/mtry/doc_viewer [--out docs/head_nla_quote_topk_shuffle.json]

A quote is a "…"/“…” span (QRE of heads_quote_attention.py). It refers to the context positions M ≤ t it matches:
- strict: the whole quote (whitespace-collapsed, lowercased) equals 1–4 consecutive context tokens;
- loose: one of its words equals a token, or a ≥4-character substring either way (heads_quote_attention.py).
It hits top-k when M meets the head's k most attended positions 1..t at token t (the sink, position 0, excluded).

Controls, for the same quotes:
- shuffled: each site gets the explanation of another site of the same head in the same text (a random derangement,
  --shuffles times), whose quotes are checked against this site's context and attention;
- other heads: the same quote against the other 31 heads' top-k at the same token;
- uniform: k positions drawn uniformly from 1..t.
Sources: std RL (step 400), matryoshka (step 640), its first 16 tokens, SFT and gold. Attention is evaluation
evidence only; neither model sees it.
"""
import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_quote_attention import QRE, matches, quote_words  # noqa: E402

HF_REPO = "syvb/head-nla-qwen3-8b"
DOC_FILES = ("token_ids.npy", "sites.parquet", "attn.npz")
KS = (1, 5, 10, 20)
BIG = 32767


def resolve(spec: str, files: tuple[str, ...], optional: tuple[str, ...] = ()) -> Path:
    """A local folder, or hf:<path in syvb/head-nla-qwen3-8b> → the downloaded folder (anonymous)."""
    if not spec.startswith("hf:"):
        return Path(spec)
    from huggingface_hub import snapshot_download
    sub = spec[3:].strip("/")
    root = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{sub}/{f}" for f in files + optional],
                             token=False)
    d = Path(root) / sub
    missing = [f for f in files if not (d / f).exists()]
    assert not missing, f"{spec}: not in {HF_REPO}: {missing}"
    return d


def snorm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def ranks(attn: np.ndarray) -> np.ndarray:
    """[H, T, T] int16: rank of position j (1..t) in head h's attention at token t, 0 = most attended; BIG elsewhere."""
    H, T, _ = attn.shape
    r = np.full((H, T, T), BIG, dtype=np.int16)
    for t in range(1, T):
        order = np.argsort(-attn[:, t, 1:t + 1], axis=1, kind="stable") + 1
        np.put_along_axis(r[:, t], order, np.arange(t, dtype=np.int16)[None].repeat(H, 0), axis=1)
    return r


class Matcher:
    """Whole-text match positions of each quote, cached; at(q, t) restricts them to context ≤ t."""

    def __init__(self, toks: list[str], rule: str):
        self.rule, self.cache = rule, {}
        if rule == "strict":
            self.spans = {}
            for L in (1, 2, 3, 4):
                for i in range(len(toks) - L + 1):
                    self.spans.setdefault(snorm("".join(toks[i:i + L])), []).append((i, L))
        else:
            from heads_quote_attention import norm
            self.ntoks = [norm(x) for x in toks]

    def whole(self, q: str):
        if q not in self.cache:
            if self.rule == "strict":
                sp = self.spans.get(snorm(q), [])
                pos = np.array([i + d for i, L in sp for d in range(L)], dtype=int)
                end = np.array([i + L - 1 for i, L in sp for d in range(L)], dtype=int)
            else:
                pos = matches(quote_words(q), self.ntoks)
                end = pos
            self.cache[q] = (pos, end)
        return self.cache[q]

    def at(self, q: str, t: int) -> np.ndarray:
        pos, end = self.whole(q)
        return np.unique(pos[end <= t])


@lru_cache(maxsize=None)
def uniform_hit(n: int, m: int) -> np.ndarray:
    """P(k of n positions drawn without replacement include one of m marked ones), for each k in KS."""
    out = []
    for k in KS:
        i = np.arange(min(k, n))
        out.append(1.0 if k >= n else float(1 - np.prod(np.clip((n - m - i) / (n - i), 0, None))))
    return np.array(out)


def score(M: np.ndarray, t: int, h: int, R: np.ndarray):
    """(rank of M in head h, other heads' ranks, non-self rank, |M|) for a quote with context matches M at (t, h); None
    when M has no position the head could rank."""
    M = M[M > 0]                                                       # the sink is never a top-k position
    if not len(M):
        return None
    mr = R[:, t, M].astype(int).min(1)                                 # every head's best rank on M
    Ms = M[M != t]                                                     # non-self: current token dropped from M and
    ns = BIG                                                           # from the ranking
    if len(Ms) and t > 1:
        rs = R[h, t, Ms].astype(int)
        ns = int((rs - (R[h, t, t] < rs)).min())
    return int(mr[h]), np.delete(mr, h), ns, len(M)


# accumulator fields, per (text, head, pairing)
F = (["nq", "ctx"] + [f"hit{k}" for k in KS] + [f"ns{k}" for k in KS] + [f"oth{k}" for k in KS]
     + [f"uni{k}" for k in KS] + ["ne", "neq"] + [f"ehit{k}" for k in KS])
FI = {f: i for i, f in enumerate(F)}
KA = np.array(KS)


def sl(prefix: str) -> slice:
    return slice(FI[f"{prefix}{KS[0]}"], FI[f"{prefix}{KS[-1]}"] + 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", action="append", required=True)
    p.add_argument("--mtry-root", default=None, help="folder of <doc name>/mtry.parquet for docs without one")
    p.add_argument("--shuffles", type=int, default=10)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    rng = np.random.default_rng(a.seed)
    SRCS = ("rl_text", "mtry_text", "mtry16_text", "sft_text", "gold_text")
    H, P, D = 32, a.shuffles + 1, len(a.doc)
    acc = {(rule, src): np.zeros((D, H, P, len(F))) for rule in ("strict", "loose") for src in SRCS}
    seen = set()
    names_docs = []
    for di, spec in enumerate(a.doc):
        d = resolve(spec, DOC_FILES, ("mtry.parquet",))
        name = d.name
        names_docs.append(name)
        mp = d / "mtry.parquet"
        if not mp.exists() and a.mtry_root:
            mp = resolve(f"{a.mtry_root.rstrip('/')}/{name}", ("mtry.parquet",)) / "mtry.parquet"
        ids = np.load(d / "token_ids.npy").tolist()
        toks = [tok.decode([i]) for i in ids]
        attn = np.load(d / "attn.npz")["attn"].astype(np.float32)
        assert attn.shape[0] == H
        R = ranks(attn)
        del attn
        s = pq.read_table(d / "sites.parquet", columns=["pos", "head", "rl_text", "sft_text", "gold_text"]).to_pandas()
        if mp.exists():
            s = s.merge(pq.read_table(mp, columns=["pos", "head", "mtry_text"]).to_pandas(), on=["pos", "head"],
                        validate="1:1")
            s["mtry16_text"] = [tok.decode(tok.encode(x or "", add_special_tokens=False)[:16]) for x in s.mtry_text]
        srcs = [c for c in SRCS if c in s]
        seen.update(srcs)
        s = s.sort_values(["head", "pos"]).reset_index(drop=True)
        quotes = {src: [[m.group(1) for m in QRE.finditer(x or "")] for x in s[src]] for src in srcs}
        # pairings: 0 = true, 1.. = derangements within each head
        part = [np.arange(len(s))]
        for _ in range(a.shuffles):
            perm = np.arange(len(s))
            for g in s.groupby("head").indices.values():
                g = np.asarray(g)
                while True:
                    q = rng.permutation(g)
                    if not (q == g).any():
                        break
                perm[g] = q
            part.append(perm)
        pos, head = s["pos"].to_numpy(), s["head"].to_numpy()
        for rule in ("strict", "loose"):
            mt = Matcher(toks, rule)
            for src in srcs:
                A = acc[rule, src][di]
                for pi, perm in enumerate(part):
                    for i in range(len(s)):
                        t, h = int(pos[i]), int(head[i])
                        qs = quotes[src][perm[i]]
                        row = A[h, pi]
                        row[FI["ne"]] += 1
                        row[FI["neq"]] += bool(qs)
                        hit = np.zeros(len(KS), bool)
                        for q in qs:
                            row[FI["nq"]] += 1
                            r = score(mt.at(q, t), t, h, R)
                            if r is None:
                                continue
                            rk, ro, ns, m = r
                            row[FI["ctx"]] += 1
                            hk = rk < KA
                            row[sl("hit")] += hk
                            row[sl("ns")] += ns < KA
                            row[sl("oth")] += (ro[:, None] < KA).mean(0)
                            row[sl("uni")] += uniform_hit(t, m)
                            hit |= hk
                        row[sl("ehit")] += hit
            print(f"{name}: {rule} done ({len(s)} sites)", flush=True)

    names = {"rl_text": "std RL", "mtry_text": "matryoshka", "mtry16_text": "matryoshka, first 16 tokens",
             "sft_text": "SFT", "gold_text": "gold"}
    rate = lambda x, n: float(x / n) if n else None  # noqa: E731
    out = {"texts": names_docs, "ks": list(KS), "shuffles": a.shuffles, "rules": {}}
    for rule in ("strict", "loose"):
        res = {}
        print(f"\n=== {rule} matching ===")
        for src in SRCS:
            if src not in seen:
                continue
            A = acc[rule, src]
            used = A[..., 0, FI["ne"]].sum((1,)) > 0                     # texts that have this source
            A = A[used]
            T, S = A[:, :, 0], A[:, :, 1:].sum(2)                      # [D, H, F]: true pairing, all shuffles
            t, sh = T.sum((0, 1)), S.sum((0, 1))
            r = {"texts": [n for n, u in zip(names_docs, used) if u], "n_explanations": int(t[FI["ne"]]),
                 "explanations_with_quote": rate(t[FI["neq"]], t[FI["ne"]]), "n_quotes": int(t[FI["nq"]]),
                 "in_context": rate(t[FI["ctx"]], t[FI["nq"]]), "in_context_shuffled": rate(sh[FI["ctx"]], sh[FI["nq"]]),
                 "k": {}}
            if t[FI["nq"]] >= 50:
                Tc, Sc = T.reshape(-1, len(F)), S.reshape(-1, len(F))  # (text, head) clusters
                for j, k in enumerate(KS):
                    hk = FI[f"hit{k}"]
                    bs = []
                    for _ in range(a.boot):
                        b = rng.integers(0, len(Tc), len(Tc))
                        tb, sb = Tc[b].sum(0), Sc[b].sum(0)
                        bs.append(tb[hk] / max(tb[FI["nq"]], 1) - sb[hk] / max(sb[FI["nq"]], 1))
                    per_pair = A[:, :, 1:].sum((0, 1))
                    sh_rates = per_pair[:, hk] / np.maximum(per_pair[:, FI["nq"]], 1)
                    dh = (T.sum(0)[:, hk] / np.maximum(T.sum(0)[:, FI["nq"]], 1)
                          - S.sum(0)[:, hk] / np.maximum(S.sum(0)[:, FI["nq"]], 1))
                    dd = (T.sum(1)[:, hk] / np.maximum(T.sum(1)[:, FI["nq"]], 1)
                          - S.sum(1)[:, hk] / np.maximum(S.sum(1)[:, FI["nq"]], 1))
                    r["k"][k] = {
                        "true": rate(t[hk], t[FI["nq"]]), "shuffled": rate(sh[hk], sh[FI["nq"]]),
                        "shuffled_sd_across_shuffles": float(sh_rates.std()),
                        "diff_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "other_heads": rate(t[FI[f"oth{k}"]], t[FI["nq"]]),
                        "uniform": rate(t[FI[f"uni{k}"]], t[FI["nq"]]),
                        "true_given_in_context": rate(t[hk], t[FI["ctx"]]),
                        "shuffled_given_in_context": rate(sh[hk], sh[FI["ctx"]]),
                        "other_heads_given_in_context": rate(t[FI[f"oth{k}"]], t[FI["ctx"]]),
                        "uniform_given_in_context": rate(t[FI[f"uni{k}"]], t[FI["ctx"]]),
                        "nonself_true": rate(t[FI[f"ns{k}"]], t[FI["nq"]]),
                        "nonself_shuffled": rate(sh[FI[f"ns{k}"]], sh[FI["nq"]]),
                        "explanations_true": rate(t[FI[f"ehit{k}"]], t[FI["ne"]]),
                        "explanations_shuffled": rate(sh[FI[f"ehit{k}"]], sh[FI["ne"]]),
                        "heads_true_gt_shuffled": int((dh > 0).sum()),
                        "texts_true_gt_shuffled": int((dd > 0).sum()),
                        "per_text_diff": {n: float(v) for n, v in zip(r["texts"], dd)},
                        "per_head_diff": [float(v) for v in dh],
                    }
            res[names[src]] = r
            print(f"\n{names[src]}: {r['n_explanations']} explanations, {r['explanations_with_quote']:.1%} quote; "
                  f"{r['n_quotes']} quotes, in context {r['in_context'] or 0:.1%} "
                  f"(shuffled {r['in_context_shuffled'] or 0:.1%})")
            for k, v in r["k"].items():
                print(f"  top-{k:2d}: true {v['true']:.3f} · shuffled {v['shuffled']:.3f} "
                      f"(diff CI {v['diff_ci95'][0]:+.3f}..{v['diff_ci95'][1]:+.3f}) · other heads {v['other_heads']:.3f}"
                      f" · uniform {v['uniform']:.3f} | in ctx {v['true_given_in_context']:.3f} vs "
                      f"{v['shuffled_given_in_context']:.3f} | non-self {v['nonself_true']:.3f} vs "
                      f"{v['nonself_shuffled']:.3f} | expl {v['explanations_true']:.3f} vs "
                      f"{v['explanations_shuffled']:.3f} | heads {v['heads_true_gt_shuffled']}/{H}, texts "
                      f"{v['texts_true_gt_shuffled']}/{len(r['texts'])}")
        out["rules"][rule] = res
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
