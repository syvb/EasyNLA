"""Rewrite soft-token readouts into short explanations with Claude.

    python scripts/heads_soft_token_explain.py --texts <work>/texts.parquet --out <work>/explain.parquet

Input is heads_soft_token_diag.py's texts.parquet. Claude sees only readouts that are functions of
the write itself: avg16 (the tokens the write raises most when injected as a soft token into fixed
prompts) and lens (the write's direct vocabulary projection). Never the document, the attention
pattern, the residual or the head's name, so the explanation is still a function of the write.

avg16 is shown split into distinctive and common-filler tokens. A token is filler when it appears
in at least --common-threshold of all avg16 lists in the input table: one fixed table for the
whole dataset, like the dev_rms normalization, so each explanation still depends on its own write
only. On a prompt lab over the 8B diagnostics (500 examples, 10-way same-head discrimination by
a second model plus an Opus quality judge) the split, thinking at medium effort and the rule
against stock openings each helped.

An answer that quotes words, lists the evidence's words, mentions readouts or tokens, or opens
with a stock phrase is sent back for a rewrite (up to --retries times).

Results are cached per example_idx in <out>.jsonl, so reruns only send what is missing. An example
whose call fails (API error after the SDK's retries, refusal, truncation) is skipped and left for a
rerun; <out> holds only the explained examples, with ok=False on any that still break the rules
after the rewrites (export_sft and the probe drop those).
"""

import argparse
import ast
import asyncio
import collections
import json
import os
import re
from pathlib import Path

import anthropic
import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM = """\
You explain what a vector from inside a language model does, using only readouts of it.

Readout A lists words the model makes MORE likely when the vector is placed into it as a stand-in for a word, \
in the prompts "Words that come to mind: <vector>:" and "Here is some text: <vector>". It is the main evidence. \
It comes in two parts. The distinctive part holds the words that are rare across vectors in general, so they \
say the most about this one. The common filler part holds words that show up for a large share of all vectors \
(vague nouns for things and people, reporting verbs, line breaks); on their own they mean little, so lean on \
them only when the distinctive part is thin or agrees with them.

Readout B lists the output tokens the vector points at directly. It is noisy (subword fragments, other \
languages) and is often uninformative; use it only when it agrees with A or is itself clearly coherent \
(e.g. many casings of one word).

Write one or two plain sentences (at most 45 words) explaining what the vector does: the main meaning, \
topic, entity, grammatical role, sentiment or kind of continuation it carries or pushes the model toward, \
and a secondary tendency if one is clearly supported. Describe its function in your own words, as \
concepts, never as a list of the words it promotes.

Lead with whatever is most distinctive about this vector. Fall back on a vague or generic reading only \
when nothing more specific is supported.
Ignore weak or noisy evidence silently: never comment on what is noisy, unclear, weak or uninformative.

Hard rules:
- Never quote any word, and never use quotation marks.
- Never give examples of words (no "such as X, Y and Z" with words from the evidence).
- Never mention readouts, evidence, tokens, words, casings, spellings or languages.
- Be as specific as the evidence allows without speculating beyond it. When it is diffuse, describe the \
broad tendency rather than saying there is no theme.

Write like a researcher's notes, varying the phrasing from vector to vector. Do not open with a stock \
phrase (for example "Pushes the model toward", "This vector", "It carries", "It pushes", "Promotes", \
"A slot", "A placeholder"); lead with the substance and pick whatever sentence structure fits it. Avoid \
opening with a bare verb (such as Leans, Sets up, Tilts, Points, Primes, Marks, Pulls, Favors); most \
explanations should open with the concept itself or with the situation it applies to.

These examples show the range of styles; do not reuse their wording:
- Doors, windows and other openings in buildings, along with where or when something happens in them.
- Obligation and possibility, setting up a statement about what someone must, may or can do.
- Something has gone wrong or is out of place, as when a problem or a fault is being described.
- Reported speech and past events lead the continuation: what someone said, or what happened next.
- Mostly grammatical, marking the point right after a subject where a verb of being or feeling belongs.
- Counting and measurement of time, especially yearly cycles and anniversaries.

Reply with the explanation only."""

FIX = ("Rewrite it. Explain only what the vector does, in your own words, as concepts: no quoted or example words, "
       "no mention of readouts, tokens, words or how reliable anything is, and do not open with a stock phrase "
       "such as This vector, It pushes, Promotes or It carries; lead with the substance. "
       "Reply with the explanation only.")


def tokens(cell: str) -> list[str]:
    """The readout cells are repr()s joined by ", "; parse them back to strings."""
    return list(ast.literal_eval(f"[{cell}]"))


def show(cell: str) -> str:
    return ", ".join(json.dumps(t, ensure_ascii=False) for t in tokens(cell))


_META = re.compile(r"[\"“”]|(^|\s)'\w|readout|token|casing|spelling", re.IGNORECASE)
_STOCK = re.compile(r"^(this|the|a) (vector|representation)\b|^(it )?(pushes|promotes|steers|nudges)\b|"
                    r"^it carries\b", re.IGNORECASE)


_SCAFFOLD = re.compile(r"Readout [AB](, distinctive|, common filler)?:|\(none\)")


def _stem(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def leaks(text: str, evidence: str) -> bool:
    """Quotes, meta talk, or a "such as"/"like" run of two or more of the evidence's own words
    (the token lists only, not the prompt's labels; plurals match singulars)."""
    if _META.search(text):
        return True
    words = {_stem(w) for w in re.findall(r"[A-Za-z]{3,}", _SCAFFOLD.sub(" ", evidence))}
    return any(sum(_stem(w) in words for w in re.findall(r"[A-Za-z]{3,}", m)) >= 2
               for m in re.findall(r"(?:such as|like|as in|e\.g\.|including|namely)\s([^.;:]{0,60})", text,
                                   re.IGNORECASE))


def breaks_rules(text: str, evidence: str) -> bool:
    return leaks(text, evidence) or bool(_STOCK.search(text))


def doc_freq(avg16: list[str]) -> dict[str, float]:
    """Fraction of avg16 lists each token appears in."""
    n = collections.Counter(t for cell in avg16 for t in set(tokens(cell)))
    return {t: c / len(avg16) for t, c in n.items()}


def user_text(avg16: str, lens: str, df: dict[str, float], threshold: float) -> str:
    ts = tokens(avg16)

    def j(xs: list[str]) -> str:
        return ", ".join(json.dumps(t, ensure_ascii=False) for t in xs) or "(none)"
    return (f"Readout A, distinctive: {j([t for t in ts if df[t] < threshold])}\n"
            f"Readout A, common filler: {j([t for t in ts if df[t] >= threshold])}\n"
            f"Readout B: {show(lens)}")


async def run(rows: list[tuple[int, str]], cache: Path, args: argparse.Namespace, key: str) -> None:
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    usage = {"input": 0, "output": 0, "broken": 0, "failed": 0, "done": 0}

    async def one(idx: int, text: str) -> None:
        messages = [{"role": "user", "content": text}]
        out = ""
        try:
            for attempt in range(1 + args.retries):
                async with sem:
                    msg = await client.messages.create(
                        model=args.model, max_tokens=4000, thinking={"type": "adaptive"},
                        output_config={"effort": args.effort},
                        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        messages=messages)
                new = " ".join("".join(b.text for b in msg.content if b.type == "text").split())
                usage["input"] += msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0)
                usage["output"] += msg.usage.output_tokens
                if msg.stop_reason != "end_turn" or not new:   # refusal, max_tokens: no usable answer
                    if not out:
                        usage["failed"] += 1
                        return
                    break    # keep the earlier (rule-breaking) answer; it is saved with ok=False
                out = new
                if not breaks_rules(out, text):
                    break
                messages += [{"role": "assistant", "content": out}, {"role": "user", "content": FIX}]
        except anthropic.APIError as e:   # after the SDK's own retries; a rerun picks it up
            usage["failed"] += 1
            if usage["failed"] <= 5:
                print(f"example {idx}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        bad = breaks_rules(out, text)
        usage["broken"] += bad
        usage["done"] += 1
        with cache.open("a") as f:   # no await inside: the event loop keeps writes whole
            f.write(json.dumps({"example_idx": idx, "explain": out, "stop": msg.stop_reason,
                                "attempts": attempt + 1, "breaks_rules": bad}) + "\n")
        if usage["done"] % 2000 == 0:
            print(f"  {usage['done']}/{len(rows)} explained ({usage['failed']} failed so far)", flush=True)

    await asyncio.gather(*(one(i, t) for i, t in rows))
    print(f"sent {len(rows)}: {usage['done']} explained, {usage['failed']} failed (rerun to retry them); "
          f"{usage['input']} input + {usage['output']} output tokens; "
          f"{usage['broken']} still break the rules after {args.retries} rewrites", flush=True)


def load_cache(cache: Path) -> dict[int, dict]:
    """example_idx → cached record; a line cut off by a killed pass is skipped (that example reruns)."""
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["example_idx"]] = r
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default=None, help="comma list of example_idx (default: all)")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--common-threshold", type=float, default=0.08,
                   help="avg16 tokens in at least this fraction of all lists are shown as filler")
    p.add_argument("--retries", type=int, default=4, help="rewrites asked for when an answer breaks the rules")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--key-file", default="~/.anthropic_key", help="used when ANTHROPIC_API_KEY is unset")
    args = p.parse_args()

    t = pq.read_table(args.texts).to_pydict()
    df = doc_freq(t["avg16"])
    want = [int(x) for x in args.rows.split(",")] if args.rows else t["example_idx"]
    pos = {e: i for i, e in enumerate(t["example_idx"])}
    out, cache = Path(args.out), Path(args.out).with_suffix(".jsonl")
    done = load_cache(cache)
    todo = [(e, user_text(t["avg16"][pos[e]], t["lens"][pos[e]], df, args.common_threshold))
            for e in want if e not in done]
    if todo:
        key = os.environ.get("ANTHROPIC_API_KEY") or Path(args.key_file).expanduser().read_text().strip()
        if cache.exists() and not cache.read_bytes().endswith(b"\n"):
            with cache.open("a") as f:   # a pass killed mid-write: don't glue the next record onto it
                f.write("\n")
        asyncio.run(run(todo, cache, args, key))
        done = load_cache(cache)
    got = [e for e in want if e in done]
    pq.write_table(pa.table({"example_idx": got, "explain": [done[e]["explain"] for e in got],
                             "ok": [not done[e]["breaks_rules"] for e in got],
                             "attempts": [done[e]["attempts"] for e in got]}), out)
    print(f"wrote {len(got)} explanations to {out} ({len(want) - len(got)} missing, "
          f"{sum(done[e]['breaks_rules'] for e in got)} with ok=False)")


if __name__ == "__main__":
    main()
