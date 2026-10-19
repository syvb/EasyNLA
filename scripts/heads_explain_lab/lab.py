"""Prompt lab for soft-token explanations.

    python lab.py gen v1 v2 ...      generate explanations on the dev set (cached)
    python lab.py disc v1 v2 ...     discrimination: pick the true readout among 5 same-head candidates
    python lab.py judge v1 v2 ...    Opus pairwise/absolute quality judge
    python lab.py report v1 v2 ...   summary table
    python lab.py show v1 [n]        print explanations

Run from anywhere; set EXPLAIN_LAB_DIR (outputs) and EXPLAIN_LAB_TEXTS (heads_soft_token_diag.py's
texts.parquet). Assumes that file's layout: 5 heads × 2400 examples, contiguous per head.
Spends real money (Haiku generation + discrimination, Opus judge); spend.json tracks it.

Variants live in variants.py (dict VARIANTS: name -> dict(system, inputs, model, thinking, effort, max_tokens)).
"""

import asyncio
import collections
import importlib
import json
import os
import random
import re
import sys
from pathlib import Path

import anthropic
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heads_soft_token_explain import FIX, leaks, show

# outputs (gen_*/disc*_*/judge_* jsonl, spend.json) go to $EXPLAIN_LAB_DIR; input is the diag texts.parquet
LAB = Path(os.environ.get("EXPLAIN_LAB_DIR", "explain_lab_out"))
LAB.mkdir(parents=True, exist_ok=True)
TEXTS = os.environ.get("EXPLAIN_LAB_TEXTS", "texts.parquet")
PER_HEAD = 2400
HEADS = ["L6H27", "L12H20", "L18H16", "L24H8", "L30H9"]
T = pq.read_table(TEXTS).to_pydict()
KEY = Path("~/.anthropic_key").expanduser().read_text().strip()
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2, 10), "claude-opus-5-5": (4, 20)}
SPEND = LAB / "spend.json"


def dev_set(n_per_head: int = 30, seed: int = 7, offset: int = 0) -> list[int]:
    r = random.Random(seed)
    out = []
    for h in range(len(HEADS)):
        out += [h * PER_HEAD + i for i in r.sample(range(PER_HEAD), offset + n_per_head)[offset:]]
    return out


DEV = dev_set()
TEST = dev_set(30, 7, 30)
_r = random.Random(21)
BIG = [i for h in range(5) for i in _r.sample([j for j in range(h * PER_HEAD, (h + 1) * PER_HEAD) if j not in set(TEST)], 100)]
SPLITS = {"dev": DEV, "test": TEST, "big": BIG}


# fraction of all 12k avg16 lists each token appears in: a fixed table, the same for every vector
DF = collections.Counter(t for c in T["avg16"] for t in set(eval(f"[{c}]")))
DF = {t: n / len(T["avg16"]) for t, n in DF.items()}


def split_common(idx: int, thr: float) -> tuple[list[str], list[str]]:
    ts = eval(f"[{T['avg16'][idx]}]")
    return [t for t in ts if DF[t] < thr], [t for t in ts if DF[t] >= thr]


def evidence(idx: int, inputs: tuple[str, ...]) -> str:
    names = {"avg16": "Readout A", "lens": "Readout B", "delta24": "Readout C"}
    out = []
    for k in inputs:
        if k.startswith("avg16d"):     # avg16d<pct>: split into distinctive / common filler
            dist, com = split_common(idx, int(k[6:]) / 100)
            j = lambda xs: ", ".join(json.dumps(t, ensure_ascii=False) for t in xs) or "(none)"
            out.append(f"Readout A, distinctive: {j(dist)}\nReadout A, common filler: {j(com)}")
        else:
            out.append(f"{names[k]}: {show(T[k][idx])}")
    return "\n".join(out)


def variants() -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    import variants as v
    importlib.reload(v)
    return v.VARIANTS


def add_spend(model: str, usage: dict) -> None:
    s = json.loads(SPEND.read_text()) if SPEND.exists() else {}
    pi, po = PRICE[model]
    s[model] = s.get(model, 0) + usage["in"] * pi / 1e6 + usage["out"] * po / 1e6
    SPEND.write_text(json.dumps(s))


def text_of(msg) -> str:
    return " ".join("".join(b.text for b in msg.content if b.type == "text").split())


def usage_add(u: dict, msg) -> None:
    u["in"] += msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0) + \
        (msg.usage.cache_creation_input_tokens or 0)
    u["out"] += msg.usage.output_tokens


async def gather_limited(coros, n=32):
    sem = asyncio.Semaphore(n)

    async def w(c):
        async with sem:
            return await c
    return await asyncio.gather(*(w(c) for c in coros))


def kwargs_for(cfg: dict) -> dict:
    kw = {"model": cfg.get("model", "claude-haiku-5-5"), "max_tokens": cfg.get("max_tokens", 200)}
    if cfg.get("thinking", "disabled") in ("disabled", "between_tools"):
        kw["thinking"] = {"type": cfg.get("thinking", "disabled")}
    else:
        kw["thinking"] = {"type": "adaptive"}
        kw["max_tokens"] = cfg.get("max_tokens", 4000)
    if cfg.get("effort"):
        kw["output_config"] = {"effort": cfg["effort"]}
    return kw


async def gen(name: str, cfg: dict, ids: list[int], split: str) -> None:
    path = LAB / f"gen_{split}_{name}.jsonl"
    done = {json.loads(l)["idx"] for l in path.open()} if path.exists() else set()
    todo = [i for i in ids if i not in done]
    if not todo:
        return
    client = anthropic.AsyncAnthropic(api_key=KEY, max_retries=8)
    kw = kwargs_for(cfg)
    u = {"in": 0, "out": 0}
    inputs = tuple(cfg.get("inputs", ("avg16", "lens")))
    check = cfg.get("check", leaks)
    fix = cfg.get("fix", FIX)

    async def one(i):
        ev = evidence(i, inputs)
        msgs = [{"role": "user", "content": ev + cfg.get("suffix", "")}]
        for a in range(1 + cfg.get("retries", 2)):
            m = await client.messages.create(system=[{"type": "text", "text": cfg["system"],
                                                      "cache_control": {"type": "ephemeral"}}],
                                             messages=msgs, **kw)
            usage_add(u, m)
            out = text_of(m)
            if not check(out, ev):
                break
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": fix}]
        with path.open("a") as f:
            f.write(json.dumps({"idx": i, "text": out, "tries": a + 1, "leaky": bool(check(out, ev)),
                                "stop": m.stop_reason}) + "\n")
    await gather_limited([one(i) for i in todo])
    add_spend(kw["model"], u)
    print(f"gen {name}/{split}: {len(todo)} done, ${u['in']*PRICE[kw['model']][0]/1e6 + u['out']*PRICE[kw['model']][1]/1e6:.3f}")


def load_gen(name: str, split: str) -> dict[int, dict]:
    return {r["idx"]: r for r in map(json.loads, (LAB / f"gen_{split}_{name}.jsonl").open())}


LET = "ABCDEFGHIJ"
# distractors come from a fixed pool of 200 per head (not the dev/test examples), same for every variant
POOL = {h: random.Random(11).sample(range(h * PER_HEAD, (h + 1) * PER_HEAD), 200) for h in range(5)}
TOKS = {j: set(eval(f"[{T['avg16'][j]}]")) for js in POOL.values() for j in js}

DISC_SYS = """\
You will read a short explanation of what a vector inside a language model does, then several candidate \
readouts of vectors. Each candidate lists words the model makes more likely when that vector is inserted \
as a stand-in for a word (Readout A), and the output tokens the vector points at directly (Readout B, \
noisy). Exactly one candidate came from the vector the explanation describes. Reply with its letter only."""


def toks(i):
    return set(eval(f"[{T['avg16'][i]}]"))


async def disc(name: str, split: str, draws: int = 2, guesser: str = "claude-haiku-5-5", hard: bool = True) -> None:
    g = load_gen(name, split)
    path = LAB / f"disc{'H' if hard else ''}_{split}_{name}_{guesser.split('-')[1]}.jsonl"
    if path.exists():
        return
    client = anthropic.AsyncAnthropic(api_key=KEY, max_retries=8)
    u = {"in": 0, "out": 0}
    ids = sorted(g)
    by_head = collections.defaultdict(list)
    for i in ids:
        by_head[i // PER_HEAD].append(i)

    async def one(i, d):
        r = random.Random(f"{i}-{d}")    # same candidates and order for every variant
        pool = [j for j in POOL[i // PER_HEAD] if j != i]
        if hard:
            ti = toks(i)
            near = sorted(pool, key=lambda j: -len(ti & TOKS[j]) / len(ti | TOKS[j]))[:5]
            rest = [j for j in pool if j not in near]
            cands = near + r.sample(rest, 4) + [i]
        else:
            cands = r.sample([j for j in by_head[i // PER_HEAD] if j != i], 4) + [i]
        r.shuffle(cands)
        body = "\n\n".join(f"{LET[k]}.\n{evidence(j, ('avg16', 'lens'))}" for k, j in enumerate(cands))
        m = await client.messages.create(
            model=guesser, max_tokens=5, thinking={"type": "disabled"},
            system=[{"type": "text", "text": DISC_SYS, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Explanation: {g[i]['text']}\n\n{body}"}])
        usage_add(u, m)
        ans = text_of(m).strip()[:1].upper()
        return {"idx": i, "draw": d, "correct": ans == LET[cands.index(i)], "ans": ans}
    res = await gather_limited([one(i, d) for i in ids for d in range(draws)])
    path.write_text("".join(json.dumps(r) + "\n" for r in res))
    add_spend(guesser, u)


def opening(t: str, n: int = 2) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", t.lower())[:n])


def report(names: list[str], split: str) -> None:
    print(f"{'variant':<14}{'disc':>7}{'±':>5}{'L6':>6}{'L12':>6}{'L18':>6}{'L24':>6}{'L30':>6}"
          f"{'leak':>6}{'retry':>6}{'top2w':>7}{'words':>6}  most common opening")
    for n in names:
        g = load_gen(n, split)
        dp = LAB / f"discH_{split}_{n}_haiku.jsonl"
        d = [json.loads(l) for l in dp.open()] if dp.exists() else []
        acc = sum(r["correct"] for r in d) / max(1, len(d))
        se = (acc * (1 - acc) / max(1, len(d))) ** 0.5
        per = []
        for h in range(5):
            dh = [r for r in d if r["idx"] // PER_HEAD == h]
            per.append(sum(r["correct"] for r in dh) / max(1, len(dh)))
        ops = collections.Counter(opening(r["text"]) for r in g.values())
        top, c = ops.most_common(1)[0]
        words = sum(len(r["text"].split()) for r in g.values()) / len(g)
        print(f"{n:<14}{acc:>7.3f}{se:>5.2f}" + "".join(f"{p:>6.2f}" for p in per) +
              f"{sum(r['leaky'] for r in g.values())/len(g):>6.2f}{sum(r['tries']>1 for r in g.values())/len(g):>6.2f}"
              f"{c/len(g):>7.2f}{words:>6.0f}  {top!r}")
    s = json.loads(SPEND.read_text()) if SPEND.exists() else {}
    print("spend so far:", {k: round(v, 3) for k, v in s.items()}, f"total ${sum(s.values()):.2f}")


JUDGE_SYS = """\
You are judging explanations of vectors from inside a language model, written for researchers. For one \
vector you get its readouts (Readout A: words the model makes more likely when the vector is inserted as a \
stand-in for a word, ranked; Readout B: output tokens the vector points at directly, noisy) and several \
candidate explanations, labelled. Score each explanation from 1 to 10 on overall quality as a natural-language \
explanation of what the vector does: faithful to the readouts without inventing things, as specific and \
informative as the readouts allow, clear and natural to read, and NOT a list or mention of individual \
words/tokens. Penalize formulaic or boilerplate phrasing. Reply with JSON only: {"<label>": score, ...}"""


JUDGE_SYS2 = """\
You are judging explanations of vectors from inside a language model, written for researchers. For one \
vector you get its readouts and several candidate explanations, labelled. Readout A: words the model makes \
more likely when the vector is inserted as a stand-in for a word, split into a distinctive part (words rare \
across vectors, so most telling) and a common filler part (words that show up for a large share of all \
vectors, so they say little). Readout B: output tokens the vector points at directly, noisy.

Score each explanation from 1 to 10 on overall quality as a natural-language explanation of what the vector \
does: faithful to the readouts without inventing things, as specific and informative about THIS vector as the \
readouts allow, clear and natural to read, and NOT a list or mention of individual words/tokens. Penalize \
formulaic or boilerplate phrasing. Reply with JSON only: {"<label>": score, ...}"""
REASONS = """ — except: reply with JSON {"scores": {"<label>": score, ...}, "why": "<two sentences on what separated the best from the worst>"}"""


async def judge(names: list[str], split: str, n: int = 100, ev_inputs=("avg16d8", "lens"), reasons: bool = False,
                tag: str = "") -> None:
    gens = {nm: load_gen(nm, split) for nm in names}
    ids = sorted(set.intersection(*(set(g) for g in gens.values())))
    ids = random.Random(3).sample(ids, min(n, len(ids)))
    client = anthropic.AsyncAnthropic(api_key=KEY, max_retries=8)
    u = {"in": 0, "out": 0}

    async def one(i):
        order = names[:]
        random.Random(i).shuffle(order)
        lab = {nm: chr(80 + k) for k, nm in enumerate(order)}   # P, Q, R...
        body = "\n".join(f"{lab[nm]}: {gens[nm][i]['text']}" for nm in order)
        m = await client.messages.create(
            model="claude-opus-5-5", max_tokens=3000, output_config={"effort": "low"},
            system=JUDGE_SYS2 + (REASONS if reasons else ""),
            messages=[{"role": "user", "content": f"{evidence(i, ev_inputs)}\n\nExplanations:\n{body}"}])
        usage_add(u, m)
        t = text_of(m)
        try:
            sc = json.loads(t[t.index("{"):t.rindex("}") + 1])
            if reasons:
                with (LAB / f"judge_reasons{tag}.jsonl").open("a") as f:
                    f.write(json.dumps({"idx": i, "labels": lab, "reply": sc}) + "\n")
                sc = sc["scores"]
            return {nm: float(sc[lab[nm]]) for nm in names}
        except (ValueError, KeyError, TypeError):
            return None
    res = [r for r in await gather_limited([one(i) for i in ids], 16) if r]
    add_spend("claude-opus-5-5", u)
    print(f"judge over {len(res)} vectors (Opus, ${u['in']*4/1e6 + u['out']*20/1e6:.2f}):")
    for nm in names:
        print(f"  {nm:<14}{sum(r[nm] for r in res)/len(res):.2f}")


def main() -> None:
    cmd, *rest = sys.argv[1:]
    split = "dev"
    if rest and rest[0].startswith("--"):
        split, rest = rest[0][2:], rest[1:]
    ids = SPLITS[split]
    V = variants()
    if cmd == "gen":
        async def go():
            await asyncio.gather(*(gen(n, V[n], ids, split) for n in rest))
        asyncio.run(go())
    elif cmd == "disc":
        async def go():
            await asyncio.gather(*(disc(n, split) for n in rest))
        asyncio.run(go())
    elif cmd == "run":
        async def go():
            await asyncio.gather(*(gen(n, V[n], ids, split) for n in rest))
            await asyncio.gather(*(disc(n, split) for n in rest))
        asyncio.run(go())
        report(rest, split)
    elif cmd == "report":
        report(rest, split)
    elif cmd == "judge":
        asyncio.run(judge(rest, split))
    elif cmd == "show":
        g = load_gen(rest[0], split)
        k = int(rest[1]) if len(rest) > 1 else 15
        for i in sorted(g)[::max(1, len(g) // k)][:k]:
            print(f"[{HEADS[i // PER_HEAD]} {i}] {g[i]['text']}")


if __name__ == "__main__":
    main()
