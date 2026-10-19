"""Blinded faithfulness judge for head-NLA explanations, and its relation to reconstruction.

    python scripts/heads_faithfulness_judge.py --checks-dir <dir> --texts <texts.parquet> --out-dir <dir>

Inputs: heads_sft_checks.py's generations.parquet + row_errors.npz (row-aligned) and the readouts
(heads_soft_token_readouts.py's texts.parquet). For a stratified sample of eval rows (--per-head per
head, plus --include-rows), each of three texts is judged against THAT row's readouts, one per call,
with no hint of where the text came from:

  gold      the Haiku explanation the models were trained on
  av_true   the AV's explanation of this row's write
  av_shuf   the AV's explanation of ANOTHER row's write (same head): the control, which should score low

The judge (default Claude Sonnet 5.5) sees the readouts as the explanation prompt shows them
(distinctive / common-filler split, the vocabulary projection) and returns a 1–5 faithfulness score,
plus flags for unsupported specifics and generic template text. The analysis relates the av_true score to
the row's reconstruction gain (baseline error − AV error, ÷ the head's mean baseline error), raw and
controlling for how reconstructible the write is at all (the gold text's gain).

Judgments are cached in <out-dir>/judgments.jsonl; reruns only send what is missing.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import anthropic
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from heads_soft_token_explain import doc_freq, user_text  # noqa: E402

SYSTEM = """\
You judge how faithfully a short explanation describes a vector from inside a language model, given readouts of \
that vector. The readouts are the only ground truth.

Readout A lists words the model makes MORE likely when the vector is placed into it as a stand-in for a word. It \
comes in two parts: the distinctive part holds words that are rare across vectors, so they say the most; the common \
filler part holds words that show up for a large share of all vectors and mean little on their own. Readout B lists \
output tokens the vector points at directly; it is noisy but, when coherent (many forms of one word or concept, \
possibly in several languages), it is strong evidence.

Score the explanation's FAITHFULNESS to these readouts, 1-5:
5  captures the main signal of the readouts accurately (e.g. the coherent theme of B, or the distinctive words of A), \
with nothing important contradicted or invented
4  captures the main signal, with minor omissions or minor unsupported detail
3  partly related: gets a secondary or vague aspect, misses or blurs the main signal
2  mostly unrelated, with at most a superficial overlap
1  unrelated to or contradicted by the readouts

Also flag:
- unsupported_specifics: the explanation names specific things (an event, entity, topic or detail) that nothing in \
the readouts supports
- generic_template: the explanation is generic boilerplate that would fit many vectors (e.g. only "hedged, tentative \
phrasing" or "a slot where a noun is expected") rather than anything specific to these readouts

Judge only against the readouts. Reply with JSON only: \
{"faithful": <1-5>, "unsupported_specifics": <true|false>, "generic_template": <true|false>}"""

CONDS = ("gold", "av_true", "av_shuf")


def item_text(evidence: str, explanation: str) -> str:
    return f"{evidence}\n\nExplanation: {explanation}"


async def judge_all(items: list[tuple[str, str]], cache: Path, model: str, effort: str, concurrency: int,
                    key: str) -> None:
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    usage = {"in": 0, "out": 0, "failed": 0}

    async def one(item_id: str, text: str) -> None:
        for _ in range(2):
            try:
                async with sem:
                    msg = await client.messages.create(
                        model=model, max_tokens=2000, thinking={"type": "between_tools"},
                        output_config={"effort": effort},
                        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        messages=[{"role": "user", "content": text}])
            except anthropic.APIError as e:
                print(f"{item_id}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                break
            usage["in"] += msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0)
            usage["out"] += msg.usage.output_tokens
            out = "".join(b.text for b in msg.content if b.type == "text")
            m = re.search(r"\{.*\}", out, re.DOTALL)
            try:
                j = json.loads(m.group(0)) if m else None
                score = int(j["faithful"])
                assert 1 <= score <= 5
            except (ValueError, KeyError, TypeError, AssertionError, json.JSONDecodeError):
                continue
            with cache.open("a") as f:
                f.write(json.dumps({"id": item_id, "faithful": score,
                                    "unsupported_specifics": bool(j.get("unsupported_specifics")),
                                    "generic_template": bool(j.get("generic_template"))}) + "\n")
            return
        usage["failed"] += 1

    await asyncio.gather(*(one(i, t) for i, t in items))
    print(f"judged {len(items) - usage['failed']}/{len(items)}; {usage['in']} input + {usage['out']} output tokens",
          flush=True)


def load_cache(cache: Path) -> dict[str, dict]:
    out = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["id"]] = r
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y controlling for z (Pearson on ranks, z regressed out)."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(a) for a in (x, y, z))
    res = [r - np.polyval(np.polyfit(rz, r, 1), rz) for r in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checks-dir", required=True)
    p.add_argument("--texts", required=True, help="readouts (texts.parquet) for the same examples")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-head", type=int, default=25)
    p.add_argument("--include-rows", default="", help="comma list of eval rows to always include")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--effort", default="low")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--common-threshold", type=float, default=0.08, help="as heads_soft_token_explain.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key-file", default="~/.anthropic_key")
    args = p.parse_args()

    ck, out = Path(args.checks_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = pq.read_table(ck / "generations.parquet").to_pydict()
    z = np.load(ck / "row_errors.npz")
    t = pq.read_table(args.texts).to_pydict()
    df = doc_freq(t["avg16"])
    at = {e: i for i, e in enumerate(t["example_idx"])}
    heads = np.array(g["head"])

    rng = random.Random(args.seed)
    rows = sorted({r for h in sorted(set(g["head"])) for r in
                   rng.sample([i for i in range(len(heads)) if heads[i] == h], args.per_head)}
                  | {int(x) for x in args.include_rows.split(",") if x})
    items = []
    for r in rows:
        i = at[g["example_idx"][r]]
        ev = user_text(t["avg16"][i], t["lens"][i], df, args.common_threshold)
        items += [(f"{r}:{c}", item_text(ev, g[c][r])) for c in CONDS]
    order = list(range(len(items)))
    rng.shuffle(order)   # interleave conditions so no batch or time window is one condition
    items = [items[k] for k in order]
    cache = out / "judgments.jsonl"
    done = load_cache(cache)
    todo = [it for it in items if it[0] not in done]
    print(f"{len(rows)} rows × {len(CONDS)} conditions = {len(items)} judgments; {len(todo)} to send", flush=True)
    if todo:
        key = os.environ.get("ANTHROPIC_API_KEY") or Path(args.key_file).expanduser().read_text().strip()
        asyncio.run(judge_all(todo, cache, args.model, args.effort, args.concurrency, key))
        done = load_cache(cache)

    # --- analysis ---
    base, err = z["base"], {c: z[k] for c, k in (("gold", "ar__gold"), ("av_true", "av__true"), ("av_shuf", "av__shuf"))}
    hm = {h: base[heads == h].mean() for h in np.unique(heads)}
    norm = np.array([hm[h] for h in heads])
    gain = {c: (base - e) / norm for c, e in err.items()}
    rows = [r for r in rows if all(f"{r}:{c}" in done for c in CONDS)]
    J = {c: np.array([done[f"{r}:{c}"]["faithful"] for r in rows]) for c in CONDS}
    F = {c: {f: np.mean([done[f"{r}:{c}"][f] for r in rows]) for f in ("unsupported_specifics", "generic_template")}
         for c in CONDS}
    G = {c: gain[c][rows] for c in CONDS}
    rep = {"n_rows": len(rows), "judge_model": args.model}
    print(f"\n{len(rows)} rows judged in all three conditions")
    print(f"{'condition':<9}{'faithful':>9}{'≥4':>7}{'specifics':>11}{'template':>10}{'mean gain':>11}")
    for c in CONDS:
        rep[c] = {"faithful_mean": float(J[c].mean()), "frac_ge4": float((J[c] >= 4).mean()),
                  **{k: float(v) for k, v in F[c].items()}, "gain_mean": float(G[c].mean())}
        print(f"{c:<9}{J[c].mean():>9.2f}{(J[c] >= 4).mean():>7.0%}{F[c]['unsupported_specifics']:>11.0%}"
              f"{F[c]['generic_template']:>10.0%}{G[c].mean():>+11.3f}")

    a, gg, gold_gain = J["av_true"], G["av_true"], G["gold"]
    rho = spearmanr(a, gg)[0]
    rho_p = partial_spearman(a, gg, gold_gain)
    within = [spearmanr(a[ix], gg[ix])[0] for h in np.unique(heads[rows])
              if (ix := np.nonzero(heads[rows] == h)[0]).size > 5 and np.ptp(a[ix]) > 0]
    print(f"\nAV faithfulness vs AV reconstruction gain: Spearman {rho:+.3f} (all rows); "
          f"{rho_p:+.3f} controlling for the gold text's gain; within-head mean {np.mean(within):+.3f} "
          f"({len(within)} heads)")
    print(f"gold faithfulness vs gold gain: Spearman {spearmanr(J['gold'], gold_gain)[0]:+.3f}")
    print("\nAV gain by AV faithfulness score:")
    by = {}
    for s in range(1, 6):
        ix = a == s
        if ix.any():
            by[s] = {"n": int(ix.sum()), "av_gain_mean": float(gg[ix].mean()), "av_gain_median": float(np.median(gg[ix])),
                     "gold_gain_mean": float(gold_gain[ix].mean())}
            print(f"  {s}: n={ix.sum():>4}  AV gain mean {gg[ix].mean():+.3f} median {np.median(gg[ix]):+.3f}   "
                  f"(gold gain on the same rows {gold_gain[ix].mean():+.3f})")
    # does a faithful AV text help beyond what the write type allows? residual of AV gain after gold gain
    resid = gg - np.polyval(np.polyfit(gold_gain, gg, 1), gold_gain)
    print("AV gain residual (after regressing on gold gain) by AV faithfulness:",
          {s: round(float(resid[a == s].mean()), 3) for s in range(1, 6) if (a == s).any()})
    for flag in ("unsupported_specifics", "generic_template"):
        f = np.array([done[f"{r}:av_true"][flag] for r in rows])
        if f.any() and (~f).any():
            print(f"AV gain with {flag}: {gg[f].mean():+.3f} (n={f.sum()}) vs without: {gg[~f].mean():+.3f}")
    rep.update({"spearman_av": float(rho), "partial_spearman_av_given_gold_gain": float(rho_p),
                "within_head_spearman_mean": float(np.mean(within)), "by_score": by})
    (out / "faithfulness.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {out / 'faithfulness.json'}")


if __name__ == "__main__":
    main()
