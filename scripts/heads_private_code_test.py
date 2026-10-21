"""Is the RL head-NLA's gain carried by a private AV<->AR code, or by readable content?

A private code = information the AR reads from features of the text that do not carry that meaning in
ordinary language (exact phrasing, markup, quotes, word order, odd tokens). Test: rewrite the RL AV's
explanations in ways that keep the meaning but destroy surface form, or keep the form but destroy the
meaning, and see what the RL AR (and the SFT AR) can still recover. Haiku's gold explanations get the
same edits as the reference for a text the RL AR was never co-trained with.

Sites: a stratified sample of the doc-viewer data (4 held-out docs, every head), which stores each site's
normalized write and its RL / gold explanations. Variants of each site's text:
  rl_orig         the RL AV's explanation (T=1), as scored in the viewer
  rl_strip        HTML tags and quotation marks removed, the quoted words kept
  rl_noquoted     HTML tags removed and every quoted span deleted
  rl_first        first sentence only (after rl_strip)
  rl_para         paraphrased by Haiku: same claims, new wording and structure, no markup
  rl_content      content words only (stopwords, punctuation, markup dropped), original order
  rl_skeleton     every content word replaced by "thing"; stopwords, punctuation, markup, layout kept
  rl_bow          the whitespace tokens in random order
  rl_shuf_site    the RL text of another position (|dpos| >= 16) of the same head and doc
  rl_shuf_doc     the RL text of a random position of the same head in another doc
  empty           no explanation (the AR sees only the head name)
  gold_orig / gold_para / gold_content / gold_skeleton / gold_bow   the same edits on the gold text
Readers: the RL AR (co-trained critic of the iter_000400 RL checkpoint) and the SFT AR, on every variant.
Occlusion (RL AR, rl_orig, a subset): delete each whitespace token in turn; the drop in cos is that
token's weight, aggregated by token kind (markup, quoted, stopword, content).

Pre-registered reading (written before the run):
  private code   rl_para keeps much less of rl_orig's gain over `empty` than gold_para keeps of gold_orig's
                 (RL < 0.6 while gold > 0.85), or rl_skeleton keeps > 25% of the gain, or occlusion puts a
                 clearly larger share of the weight on markup / quotes / stopwords than their share of tokens.
  readable       rl_para retention close to gold_para's (both high), rl_skeleton ~ empty, rl_content keeps
                 most of the gain.

    python scripts/heads_private_code_test.py prep --viewer-dir ~/l24_docs/hf/l24/std_rl/doc_viewer --data private_code_data
    python scripts/heads_private_code_test.py score --data private_code_data --rl-critic <iter>/critic --sft-ar <dir> --out results
    python scripts/heads_private_code_test.py report --data private_code_data --results results
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYER = 24
SCALE = 64.0   # sqrt(d_model): the direction-only recipe's target scale

PARA_SYSTEM = """\
You rewrite short descriptions. Rewrite the user's text in your own words: change the wording and the \
sentence structure, but keep every claim, its meaning and its relative emphasis, and add nothing. Keep \
subject-matter terms (topics, fields, named concepts) when there is no plain synonym. Remove any HTML tags \
or other markup. Keep words that appear in quotation marks as plain words, without the quotation marks. Keep \
words whose meaning is unclear as they are. Write plain prose of about the same length. Output only the \
rewritten text."""

TAG = re.compile(r"</?[A-Za-z][^>]*>")
QUOTE_SPAN = re.compile(r"“[^”]*”|\"[^\"]*\"|‘[^’]*’(?![A-Za-z])")
WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")


def stopwords() -> frozenset:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS)


def strip(t: str) -> str:
    t = TAG.sub(" ", t)
    t = re.sub(r"[“”\"]", "", t)
    t = re.sub(r"(?<![A-Za-z])[‘’']|[‘’'](?![A-Za-z])", "", t)   # quote marks, not apostrophes
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", t)).strip()


def noquoted(t: str) -> str:
    return re.sub(r"[ \t]+", " ", QUOTE_SPAN.sub(" ", TAG.sub(" ", t))).strip()


def first_sentence(t: str) -> str:
    s = strip(t)
    m = re.search(r"[.!?](\s|$)", s)
    return s[: m.end()].strip() if m else s


def content_only(t: str, sw) -> str:
    return " ".join(w for w in WORD.findall(TAG.sub(" ", t)) if w.lower() not in sw)


def skeleton(t: str, sw) -> str:
    parts = re.split(r"(</?[A-Za-z][^>]*>)", t)     # tags stay as they are
    sub = lambda m: m.group(0) if m.group(0).lower() in sw else "thing"  # noqa: E731
    return "".join(x if TAG.fullmatch(x) else WORD.sub(sub, x) for x in parts)


def bow(t: str, rng: random.Random) -> str:
    toks = t.split()
    rng.shuffle(toks)
    return " ".join(toks)


def tokens_with_kind(t: str, sw) -> list[tuple[int, int, str]]:
    """Whitespace tokens of t as (start, end, kind); kind is markup / quoted / stopword / content / punct."""
    quoted = [m.span() for m in QUOTE_SPAN.finditer(t)]
    out = []
    for m in re.finditer(r"\S+", t):
        tok = m.group(0)
        if TAG.search(tok):
            kind = "markup"
        elif any(a <= m.start() < b or a < m.end() <= b for a, b in quoted) or re.search(r"[“”\"]", tok):
            kind = "quoted"
        elif not WORD.findall(tok):
            kind = "punct"
        else:
            kind = "stopword" if all(x.lower() in sw for x in WORD.findall(tok)) else "content"
        out.append((m.start(), m.end(), kind))
    return out


# ---------------------------------------------------------------- prep (local, CPU + Haiku)
async def paraphrase_all(texts: dict[str, str], cache: Path, key: str, concurrency: int) -> None:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    usage = {"in": 0, "out": 0, "failed": 0}

    async def one(k: str, t: str) -> None:
        try:
            async with sem:
                msg = await client.messages.create(model="claude-haiku-5-5", max_tokens=1200, system=PARA_SYSTEM,
                                                   messages=[{"role": "user", "content": t}])
        except anthropic.APIError as e:
            print(f"{k}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            usage["failed"] += 1
            return
        usage["in"] += msg.usage.input_tokens
        usage["out"] += msg.usage.output_tokens
        out = "".join(b.text for b in msg.content if b.type == "text").strip()
        if not out:
            usage["failed"] += 1
            return
        with cache.open("a") as f:
            f.write(json.dumps({"id": k, "text": out}) + "\n")

    await asyncio.gather(*(one(k, t) for k, t in texts.items()))
    print(f"paraphrased {len(texts) - usage['failed']}/{len(texts)}; {usage['in']} in + {usage['out']} out tokens "
          f"(~${usage['in'] * 0.1e-6 + usage['out'] * 0.5e-6:.2f})", flush=True)


def prep(a) -> None:
    rng = random.Random(a.seed)
    sw = stopwords()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    docs = sorted(glob.glob(os.path.join(a.viewer_dir, "0*")))
    S = {os.path.basename(d)[:2]: pd.read_parquet(f"{d}/sites.parquet") for d in docs}
    W = {os.path.basename(d)[:2]: np.load(f"{d}/writes.npz")["v"] for d in docs}
    sites, targets = [], []
    for doc, s in S.items():
        for h in range(W[doc].shape[1]):
            sh = s[(s["head"] == h) & (s.pos > 0) & s.gold_ok & (s.gold_text.str.len() > 0)].set_index("pos")
            for pos in rng.sample(sorted(sh.index), a.per_doc_head):
                r = sh.loc[pos]
                far = [q for q in sh.index if abs(q - pos) >= 16]
                odoc = rng.choice([d for d in S if d != doc])
                so = S[odoc][(S[odoc]["head"] == h) & (S[odoc].pos > 0)]
                sites.append(dict(site=len(sites), doc=doc, pos=int(pos), head=h, rl=r.rl_text, gold=r.gold_text,
                                  shuf_site=sh.loc[rng.choice(far), "rl_text"],
                                  shuf_doc=so.iloc[rng.randrange(len(so))].rl_text,
                                  rl_cos=float(r.rl_cos), rl_fve=float(r.rl_fve), gold_rl_cos=float(r.gold_rl_cos),
                                  sft_cos=float(r.sft_cos), gold_sft_cos=float(r.gold_sft_cos)))
                targets.append(W[doc][pos, h])
    sites = pd.DataFrame(sites)
    np.save(data / "targets.npy", np.stack(targets).astype(np.float16))

    cache = data / "paraphrases.jsonl"
    want = {f"{k}:{i}": t for i, (rl, g) in enumerate(zip(sites.rl, sites.gold)) for k, t in (("rl", rl), ("gold", g))}
    have = {json.loads(x)["id"] for x in cache.read_text().splitlines()} if cache.exists() else set()
    todo = {k: v for k, v in want.items() if k not in have}
    if todo:
        key = Path(a.key_file).expanduser().read_text().strip()
        asyncio.run(paraphrase_all(todo, cache, key, a.concurrency))
    para = {json.loads(x)["id"]: json.loads(x)["text"] for x in cache.read_text().splitlines()}

    rows = []
    for r in sites.itertuples():
        v = {"rl_orig": r.rl, "rl_strip": strip(r.rl), "rl_noquoted": noquoted(r.rl), "rl_first": first_sentence(r.rl),
             "rl_para": para.get(f"rl:{r.site}"), "rl_content": content_only(r.rl, sw), "rl_skeleton": skeleton(r.rl, sw),
             "rl_bow": bow(r.rl, rng), "rl_shuf_site": r.shuf_site, "rl_shuf_doc": r.shuf_doc, "empty": "",
             "gold_orig": r.gold, "gold_para": para.get(f"gold:{r.site}"), "gold_content": content_only(r.gold, sw),
             "gold_skeleton": skeleton(r.gold, sw), "gold_bow": bow(r.gold, rng)}
        rows += [dict(site=r.site, head=r.head, variant=k, text=t) for k, t in v.items() if t is not None]
    V = pd.DataFrame(rows)
    occ = sites.groupby(["head", "doc"]).head(a.occlusion_per_doc_head).site.tolist()
    sites.assign(occlusion=sites.site.isin(occ)).to_parquet(data / "sites.parquet")
    V.to_parquet(data / "variants.parquet")
    print(f"{len(sites)} sites, {len(V)} variant texts, {len(occ)} occlusion sites → {data}")
    for k in ["rl_orig", "rl_para", "rl_skeleton", "rl_content", "gold_para"]:
        print(f"--- {k}: {V[(V.variant == k)].text.iloc[3][:300]!r}")


# ---------------------------------------------------------------- score (GPU pod)
def score(a) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from heads_sft_checks import ar_predict
    from show_nla_generations import load_ar_critic
    from transformers import AutoTokenizer

    from nla.datagen.heads.export_sft import critic_explanation

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(data / "sites.parquet")
    V = pd.read_parquet(data / "variants.parquet")
    tgt = np.load(data / "targets.npy").astype(np.float32)
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    sw = stopwords()

    def run(critic, heads, texts) -> np.ndarray:
        """Unit AR predictions in input order (batched by length)."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        ins = [critic_explanation(LAYER, int(heads[i]), texts[i]) for i in order]
        with torch.no_grad():
            p = ar_predict(critic, tok, ins, SCALE, a.device, a.batch_size)
        pred = np.empty_like(p)
        pred[order] = p / np.linalg.norm(p, axis=1, keepdims=True)
        return pred

    for reader, path in (("rl", a.rl_critic), ("sft", a.sft_ar)):
        critic = load_ar_critic(path, a.base, a.device).eval()
        V[f"cos_{reader}"] = (run(critic, V["head"].to_numpy(), V.text.tolist()) * tgt[V.site.to_numpy()]).sum(1)
        chk = V[V.variant == "rl_orig"].merge(sites[["site", "rl_cos", "sft_cos"]], on="site")
        ref = chk.rl_cos if reader == "rl" else chk.sft_cos
        g = V[V.variant == "gold_orig"].merge(sites[["site", "gold_rl_cos", "gold_sft_cos"]], on="site")
        print(f"[{reader}] check vs the viewer's stored scores: rl text {chk[f'cos_{reader}'].mean():.4f} "
              f"(stored {chk.rl_cos.mean() if reader == 'rl' else chk.sft_cos.mean():.4f} for the "
              f"{'RL' if reader == 'rl' else 'SFT'} text), gold text {g[f'cos_{reader}'].mean():.4f} "
              f"(stored {g[f'gold_{reader}_cos'].mean():.4f}); corr with stored "
              f"{np.corrcoef(chk[f'cos_{reader}'], ref)[0, 1] if reader == 'rl' else np.corrcoef(g[f'cos_{reader}'], g[f'gold_{reader}_cos'])[0, 1]:.4f}",
              flush=True)
        if reader == "rl":   # occlusion on the RL pair
            occ_rows = []
            for st in sites[sites.occlusion].itertuples():
                for b, e, kind in tokens_with_kind(st.rl, sw):
                    occ_rows.append(dict(site=st.site, head=st.head, tok=st.rl[b:e], kind=kind,
                                         text=st.rl[:b] + st.rl[e:]))
            O = pd.DataFrame(occ_rows)
            O["cos_del"] = (run(critic, O["head"].to_numpy(), O.text.tolist()) * tgt[O.site.to_numpy()]).sum(1)
            O.drop(columns="text").to_parquet(out / "occlusion.parquet")
            print(f"[rl] occlusion: {len(O)} deletions over {O.site.nunique()} sites", flush=True)
        del critic
        torch.cuda.empty_cache()
    V.drop(columns="text").to_parquet(out / "scores.parquet")
    (out / "DONE").write_text("ok\n")


# ---------------------------------------------------------------- report (local)
def report(a) -> None:
    data, res = Path(a.data), Path(a.results)
    sites = pd.read_parquet(data / "sites.parquet")
    V = pd.read_parquet(res / "scores.parquet").merge(sites[["site", "rl_cos", "rl_fve"]], on="site")
    full = V.groupby("site").variant.nunique()
    V = V[V.site.isin(full[full == full.max()].index)]   # sites with every variant (a few paraphrases were refused)
    base = 2 * (1 - V.rl_cos) / (1 - V.rl_fve)   # per-site FVE denominator (units of SCALE^2), from the viewer
    for r in ("rl", "sft"):
        V[f"fve_{r}"] = 1 - 2 * (1 - V[f"cos_{r}"]) / base
    order = ["rl_orig", "rl_strip", "rl_noquoted", "rl_first", "rl_para", "rl_content", "rl_bow", "rl_skeleton",
             "rl_shuf_site", "rl_shuf_doc", "empty", "gold_orig", "gold_para", "gold_content", "gold_bow", "gold_skeleton"]
    T = V.groupby("variant")[["cos_rl", "fve_rl", "cos_sft", "fve_sft"]].mean().reindex(order)
    rep = {"n_sites": int(V.site.nunique()), "table": T.round(4).to_dict("index")}

    def retention(var, ref, reader):
        lo = T.loc["empty", f"cos_{reader}"]
        return float((T.loc[var, f"cos_{reader}"] - lo) / (T.loc[ref, f"cos_{reader}"] - lo))

    rep["retention_over_empty"] = {reader: {v: round(retention(v, "rl_orig" if v.startswith("rl") else "gold_orig", reader), 3)
                                            for v in order if v not in ("empty", "rl_orig", "gold_orig")}
                                   for reader in ("rl", "sft")}
    P = V.pivot_table(index=["site", "head"], columns="variant", values="cos_rl").reset_index()
    per_head = P.groupby("head").mean()
    ph = ((per_head.rl_para - per_head["empty"]) / (per_head.rl_orig - per_head["empty"])).round(2)
    rep["rl_para_retention_by_head"] = ph.to_dict()
    gh = ((per_head.gold_para - per_head["empty"]) / (per_head.gold_orig - per_head["empty"])).round(2)
    rep["gold_para_retention_by_head"] = gh.to_dict()
    # within-head site-level: does the paraphrase keep which site is which? (corr of per-site cos, demeaned per head)
    dm = P.copy()
    for c in ["rl_orig", "rl_para", "rl_skeleton", "rl_content"]:
        dm[c] = dm[c] - dm.groupby("head")[c].transform("mean")
    rep["site_level_corr_with_rl_orig"] = {c: round(float(np.corrcoef(dm.rl_orig, dm[c])[0, 1]), 3)
                                           for c in ["rl_para", "rl_skeleton", "rl_content"]}

    occ = res / "occlusion.parquet"
    if occ.exists():
        O = pd.read_parquet(occ).merge(V[V.variant == "rl_orig"][["site", "cos_rl"]], on="site")
        O["drop"] = O.cos_rl - O.cos_del
        k = O.groupby("kind").agg(tokens=("drop", "size"), mean_drop=("drop", "mean"), total_pos=("drop", lambda x: x.clip(0).sum()))
        k["share_tokens"] = k.tokens / k.tokens.sum()
        k["share_weight"] = k.total_pos / k.total_pos.sum()
        rep["occlusion_by_kind"] = k.round(4).to_dict("index")
        O["w"] = O.tok.str.lower().str.strip(".,;:!?()“”\"'‘’")
        top = O.groupby("w").agg(n=("drop", "size"), mean_drop=("drop", "mean")).query("n >= 5").sort_values("mean_drop", ascending=False)
        rep["occlusion_top_tokens"] = top.head(30).round(4).to_dict("index")
        rep["occlusion_largest_single_drop"] = O.sort_values("drop", ascending=False).head(15)[["head", "tok", "kind", "drop"]].round(3).to_dict("records")
    (res / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())
    print(json.dumps({k: rep[k] for k in rep if k not in ("table", "occlusion_top_tokens", "rl_para_retention_by_head", "gold_para_retention_by_head")}, indent=1, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--viewer-dir", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--per-doc-head", type=int, default=12)
    q.add_argument("--occlusion-per-doc-head", type=int, default=2)
    q.add_argument("--concurrency", type=int, default=32)
    q.add_argument("--key-file", default="~/.anthropic_key")
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("score")
    q.add_argument("--data", required=True)
    q.add_argument("--rl-critic", required=True)
    q.add_argument("--sft-ar", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--base", default="Qwen/Qwen3-8B")
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--device", default="cuda:0")
    q = sub.add_parser("report")
    q.add_argument("--data", required=True)
    q.add_argument("--results", required=True)
    a = p.parse_args()
    {"prep": prep, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
