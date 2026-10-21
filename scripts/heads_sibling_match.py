"""Can a reader tell GQA-sibling heads apart from their explanations? Blinded matching on doc-viewer data.

Heads in one KV group share value vectors, so at one site their writes are different readouts of the same
attended content (e.g. L24 H16-19: roles / events / institutional register / activity of the doc's domain).
For each sampled site the judge sees the 4 siblings' readouts (soft-token avg16 + logit lens of each
write) labelled A-D, and the 4 siblings' explanations from one source (rl, sft or gold), shuffled and
labelled 1-4, and assigns each explanation to a readout (a permutation). Chance is 25% per explanation.
Source rl_shuf is a control: each head's RL explanation from another random position of the same doc. If it
matches as well as rl, the signal is head-level style, not what this site's write says.

Gold explanations were written by Haiku from these same readouts, so gold is a ceiling, not a peer.

    python scripts/heads_sibling_match.py --viewer-dir ~/l24_docs/hf/l24/std_rl/doc_viewer --heads 16,17,18,19 \
        --out-dir ~/l24_docs/sibling_match --per-doc 50
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

SYSTEM = """\
You match explanations to the vectors they describe. Each vector is one attention head's output at one \
token of a document; the heads all read the same tokens, so they often share a topic and differ in nuance.

Each vector comes with two readouts:
- promoted: words the model makes more likely when the vector is injected in place of a word
- lens: the vocabulary directions the vector points at most directly

Then you get one explanation per vector, shuffled. Assign every explanation to exactly one vector (a \
one-to-one matching). Use the specific content of the readouts; explanations may be noisy or templated.

Answer with JSON only, e.g. {"1": "C", "2": "A", "3": "D", "4": "B"}."""


def item_text(readouts: list[tuple[str, str]], texts: list[str]) -> str:
    labs = "ABCDEFGH"
    parts = ["Vectors:"]
    for lab, (avg, lens) in zip(labs, readouts):
        parts.append(f"{lab}. promoted: {avg}\n   lens: {lens}")
    parts.append("\nExplanations:")
    for i, t in enumerate(texts, 1):
        parts.append(f"{i}. {t.strip()}")
    return "\n".join(parts)


async def run(items, cache: Path, model: str, concurrency: int, key: str) -> None:
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    usage = {"in": 0, "out": 0, "failed": 0}

    async def one(item_id: str, text: str, n: int) -> None:
        for _ in range(2):
            try:
                async with sem:
                    msg = await client.messages.create(
                        model=model, max_tokens=1500, system=SYSTEM,
                        messages=[{"role": "user", "content": text}])
            except anthropic.APIError as e:
                print(f"{item_id}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                break
            usage["in"] += msg.usage.input_tokens
            usage["out"] += msg.usage.output_tokens
            out = "".join(b.text for b in msg.content if b.type == "text")
            m = re.search(r"\{.*\}", out, re.DOTALL)
            try:
                j = json.loads(m.group(0))
                ans = {int(k): v.strip().upper()[:1] for k, v in j.items()}
                assert sorted(ans) == list(range(1, n + 1))
            except (AttributeError, ValueError, AssertionError, json.JSONDecodeError):
                continue
            with cache.open("a") as f:
                f.write(json.dumps({"id": item_id, "ans": ans}) + "\n")
            return
        usage["failed"] += 1

    await asyncio.gather(*(one(*it) for it in items))
    print(f"matched {len(items) - usage['failed']}/{len(items)}; {usage['in']} input + {usage['out']} output tokens",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer-dir", required=True)
    p.add_argument("--heads", default="16,17,18,19")
    p.add_argument("--sources", default="rl,rl_shuf,sft,gold")
    p.add_argument("--per-doc", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--key-file", default="~/.anthropic_key")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    heads = [int(h) for h in a.heads.split(",")]
    labs = "ABCDEFGH"[:len(heads)]
    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    items, truth = [], {}
    for d in sorted(glob.glob(os.path.join(a.viewer_dir, "0*"))):
        doc = os.path.basename(d)[:2]
        S = pd.read_parquet(f"{d}/sites.parquet")
        S = S[S["head"].isin(heads) & (S.pos > 0)].set_index(["pos", "head"]).sort_index()
        positions = sorted(S.index.get_level_values(0).unique())
        for pos in rng.sample(positions, min(a.per_doc, len(positions))):
            vec_order = heads[:]
            rng.shuffle(vec_order)  # which head is A..D
            readouts = [(S.loc[(pos, h), "avg16"], S.loc[(pos, h), "lens"]) for h in vec_order]
            for src in a.sources.split(","):
                txt_order = heads[:]
                rng.shuffle(txt_order)  # which head is explanation 1..4
                if src == "rl_shuf":
                    texts = [S.loc[(rng.choice([q for q in positions if q != pos]), h), "rl_text"] for h in txt_order]
                else:
                    texts = [S.loc[(pos, h), f"{src}_text"] for h in txt_order]
                iid = f"{doc}:{pos}:{src}"
                items.append((iid, item_text(readouts, texts), len(heads)))
                truth[iid] = {str(i + 1): labs[vec_order.index(h)] for i, h in enumerate(txt_order)} | {
                    "heads": {str(i + 1): h for i, h in enumerate(txt_order)}}
    (out / "truth.json").write_text(json.dumps(truth))

    cache = out / "matches.jsonl"
    done = {json.loads(line)["id"] for line in cache.read_text().splitlines()} if cache.exists() else set()
    todo = [it for it in items if it[0] not in done]
    if todo:
        key = Path(a.key_file).expanduser().read_text().strip()
        asyncio.run(run(todo, cache, a.model, a.concurrency, key))

    res = {json.loads(line)["id"]: json.loads(line)["ans"] for line in cache.read_text().splitlines()}
    rows = []
    for iid, ans in res.items():
        doc, pos, src = iid.split(":")
        t = truth[iid]
        for k, lab in ans.items():
            rows.append(dict(doc=doc, src=src, head=t["heads"][k], ok=lab == t[k]))
    r = pd.DataFrame(rows)
    rep = {"chance": 1 / len(heads), "model": a.model, "n_sites_per_source": r.groupby("src").size().div(len(heads)).to_dict(),
           "acc": r.groupby("src").ok.mean().round(3).to_dict(),
           "acc_by_head": r.groupby(["src", "head"]).ok.mean().unstack().round(3).to_dict("index"),
           "acc_by_doc": r.groupby(["src", "doc"]).ok.mean().unstack().round(3).to_dict("index")}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
