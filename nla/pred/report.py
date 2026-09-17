"""Stage 5: turn the eval outputs into the report the experiment is supposed to end with.

Reads an eval directory (summary.json + scores.jsonl + explanations.jsonl) and
writes report.md: the primary test, the headline tables (context-free and
context-conditioned), the bridge-length breakdown, paired checkpoint
comparisons, the transfer check, the verbatim-overlap and length diagnostics,
blinded representative explanations, and the caveats.

TWO DIAGNOSTICS ARE NOT DECORATION.

Verbatim overlap: with no document in the reader's prompt, an explanation that
just quotes what the model was reading scores well, transfers across readers
and beats a shuffled control. The share of an explanation's 4-grams that appear
verbatim in the document prefix says how much of a checkpoint's gain is context
recovery; the tail_quote reference row says how much gain that route yields.

Length: the reward rises with explanation length essentially for free, so a
checkpoint can top the table by writing longer. The report prints length beside
every gain and fits gain ~ length + checkpoint, so the length-adjusted
checkpoint effect is visible next to the raw one.

    python -m nla.pred.report --eval-dir evals/final --blind --n-examples 100
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nla.pred.stats import paired_bootstrap_diff


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(mean, lo, hi, digits=4):
    if mean is None or lo is None or hi is None or not np.isfinite(mean):
        return "n/a"
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def _p(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.3f}"


def _ngrams(text, n=4):
    w = (text or "").split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlap_fraction(explanation, source, n=4):
    """Share of the explanation's word n-grams that occur verbatim in `source`."""
    e = _ngrams(explanation, n)
    if not e:
        return float("nan")
    s = _ngrams(source, n)
    return len(e & s) / len(e)


def length_adjusted_effects(gain_by_ck, tok_by_ck, checkpoints, base):
    """OLS of gain on explanation length + checkpoint dummies, pooled over
    positions. Returns {ck: (raw_diff_vs_base, adjusted_diff_vs_base, slope)}."""
    ys, lens, dums = [], [], []
    others = [c for c in checkpoints if c != base]
    for ck in checkpoints:
        for rid, g in gain_by_ck[ck].items():
            t = tok_by_ck[ck].get(rid, 0)
            if not np.isfinite(g) or t <= 0:
                continue
            ys.append(g); lens.append(t)
            dums.append([1.0 if ck == o else 0.0 for o in others])
    if len(ys) < 10 or not others:
        return {}
    X = np.column_stack([np.ones(len(ys)), np.asarray(lens, float), np.asarray(dums)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(ys), rcond=None)
    slope = float(beta[1])
    out = {}
    for j, ck in enumerate(others):
        raw = (np.nanmean(list(gain_by_ck[ck].values()))
               - np.nanmean(list(gain_by_ck[base].values())))
        out[ck] = (float(raw), float(beta[2 + j]), slope)
    return out


def build_report(eval_dir: Path, title: str, wandb_url: str | None = None,
                 n_examples: int = 12, seed: int = 0, blind: bool = False) -> str:
    summary = json.loads((eval_dir / "summary.json").read_text())
    scores = _read_jsonl(eval_dir / "scores.jsonl")
    expls = _read_jsonl(eval_dir / "explanations.jsonl")
    buckets = [tuple(b) for b in summary["buckets"]]
    hb = buckets.index(tuple(summary["headline_bucket"]))
    ctxs = summary.get("context_words_list", [0])
    refs = set(summary.get("reference_sets", []))
    checkpoints = sorted({r["checkpoint"] for r in scores} - refs)
    readers = sorted({r["reader"] for r in scores})
    train_reader = summary.get("training_reader", readers[0])
    held_out = [r for r in readers if r != train_reader]
    primary = summary.get("primary_test")

    L = [f"# {title}", ""]
    L.append(
        f"Frozen-reader predictive gain on {summary['n_positions']} held-out "
        f"positions, {len(buckets)} bridge lengths, {len(checkpoints)} checkpoints x "
        f"{len(readers)} readers, context settings {ctxs}. Gain is nats per target "
        f"token relative to the same reader shown no explanation; intervals are 95% "
        f"bootstrap over documents, and every comparison is paired over positions.")
    if wandb_url:
        L += ["", f"Training curve: {wandb_url}"]

    # ---- 0. primary test ----
    L += ["", "## 0. Primary test", ""]
    if primary:
        L.append(f"**{primary['comparison']}**, matched explanations, held-out reader "
                 f"`{primary['reader']}`, bucket {primary['bucket'][0]}-{primary['bucket'][1]}, "
                 f"reader shown the last {primary['context_words']} words of the document: "
                 f"**{_fmt(primary['diff'], primary['lo'], primary['hi'])}** nats/token, "
                 f"p = {_p(primary['p'])}, n = {primary['n']}.")
        pf = primary.get("plan_spec_context_free")
        if pf:
            L.append("")
            L.append(f"The plan's original context-free version of the same comparison: "
                     f"{_fmt(pf['diff'], pf['lo'], pf['hi'])}, p = {_p(pf['p'])}. It is "
                     f"reported, not decisive: without document context a verbatim "
                     f"quote of the prefix out-scores a real explanation, so a "
                     f"context-free win can be context recovery.")
        L.append("")
        L.append(f"This is the one pre-registered number; every other comparison below is "
                 f"secondary and uncorrected. {primary['note']}.")
    else:
        L.append("Not available (needs a `behavioral_rl` checkpoint, a baseline "
                 "checkpoint and a held-out reader).")

    # ---- 1. headline tables ----
    L += ["", "## 1. Headline: predictive gain and activation-dependence", ""]
    hbl = summary["headline_bucket"]
    L.append(f"Bucket {hbl[0]}-{hbl[1]} of the continuation, i.e. the reader has "
             f"already been given {hbl[0]} real tokens as a bridge. `shuffled` pairs "
             f"the explanation with another document's continuation; `same_doc` with "
             f"the other position of the SAME document, which holds topic and register "
             f"constant and is the informative control.")
    med_tok = summary.get("explanation_tokens_median", {})
    for ctx in ctxs:
        L += ["", (f"### Reader sees no document text" if ctx == 0 else
                   f"### Reader also sees the last {ctx} words of the document"), "",
              "| checkpoint | reader | gain [95% CI] | shuffled | matched - shuffled | p "
              "| same_doc | matched - same_doc | p | split-half r | median tokens |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cell in summary["cells"]:
            if cell["context_words"] != ctx:
                continue
            h = cell["headline"]
            name = cell["checkpoint"] + (" (reference)" if cell["checkpoint"] in refs else "")
            rel = h.get("reliability_gain")
            L.append(
                f"| {name} | {cell['reader'].split('/')[-1]} "
                f"| {_fmt(h['gain'], h['gain_lo'], h['gain_hi'])} "
                f"| {h.get('shuffled_gain', float('nan')):+.4f} "
                f"| {_fmt(h.get('matched_minus_shuffled'), h.get('mms_lo'), h.get('mms_hi'))} "
                f"| {_p(h.get('mms_p'))} "
                f"| {h.get('same_doc_gain', float('nan')):+.4f} "
                f"| {_fmt(h.get('matched_minus_same_doc'), h.get('msd_lo'), h.get('msd_hi'))} "
                f"| {_p(h.get('msd_p'))} "
                f"| {'n/a' if rel is None else f'{rel:.2f}'} "
                f"| {med_tok.get(cell['checkpoint'], float('nan')):.0f} |")
    if refs:
        L += ["", "The `tail_quote` reference is not a checkpoint: it is the last words "
              "of the document, presented as if they were an explanation. In the "
              "context-free setting it shows how much gain pure context recovery is "
              "worth; a checkpoint that does not clearly beat it is reconstructing the "
              "text, not explaining the activation. In the context-conditioned setting "
              "it should be near zero by construction."]

    # ---- 2. by bridge length ----
    L += ["", "## 2. Gain by bridge length (no document context)", "",
          "Each bucket is scored in the same forward pass; a later bucket means the "
          "reader has already seen more of the real continuation, so the explanation "
          "has to add something the text itself has not already revealed.", "",
          "| checkpoint | reader | " + " | ".join(f"{lo}-{hi}" for lo, hi in buckets)
          + " | split-half r per bucket |",
          "|---|---|" + "---|" * len(buckets) + "---|"]
    for cell in summary["cells"]:
        if cell["context_words"] != 0:
            continue
        vals = " | ".join(f"{e['gain']:+.4f}" for e in cell["per_bucket"])
        rels = ", ".join("n/a" if e.get("reliability_gain") is None
                         else f"{e['reliability_gain']:.2f}" for e in cell["per_bucket"])
        L.append(f"| {cell['checkpoint']} | {cell['reader'].split('/')[-1]} | {vals} | {rels} |")

    # ---- per-(ck, reader, ctx) matched gains from raw records ----
    by = defaultdict(dict)
    docs = {}
    for r in scores:
        if r["condition"] != "matched":
            continue
        by[(r["checkpoint"], r["reader"], r["context_words"])][r["row_id"]] = r["gain"][hb]
        docs[r["row_id"]] = r["doc_id"]

    # ---- 3. paired comparisons ----
    L += ["", "## 3. Checkpoint comparisons (paired)", "",
          "| comparison | reader | context | difference [95% CI] | p | n |",
          "|---|---|---|---|---|---|"]
    for ctx in ctxs:
        for a in checkpoints:
            for b in checkpoints:
                if a >= b:
                    continue
                for rd in readers:
                    ga, gb = by[(a, rd, ctx)], by[(b, rd, ctx)]
                    keys = sorted(set(ga) & set(gb))
                    if not keys:
                        continue
                    d, lo, hi, p, n = paired_bootstrap_diff(
                        [ga[k] for k in keys], [gb[k] for k in keys],
                        clusters=[docs[k] for k in keys], seed=seed)
                    L.append(f"| {a} - {b} | {rd.split('/')[-1]} | {ctx} "
                             f"| {_fmt(d, lo, hi)} | {_p(p)} | {n} |")

    # ---- 4. transfer ----
    if held_out:
        L += ["", "## 4. Does the gain transfer to the held-out reader?", "",
              f"Training reader: `{train_reader}`. Held out: "
              + ", ".join(f"`{h}`" for h in held_out) + ".", "",
              "| checkpoint | context | training-reader gain | held-out-reader gain | ratio |",
              "|---|---|---|---|---|"]
        for ctx in ctxs:
            for ck in checkpoints:
                gt = np.nanmean(list(by[(ck, train_reader, ctx)].values())) if by[(ck, train_reader, ctx)] else np.nan
                for hr in held_out:
                    gh = np.nanmean(list(by[(ck, hr, ctx)].values())) if by[(ck, hr, ctx)] else np.nan
                    ratio = (f"{gh / gt:.2f}" if np.isfinite(gt) and np.isfinite(gh)
                             and gt > 0.001 else "n/a")
                    L.append(f"| {ck} | {ctx} | {gt:+.4f} | {gh:+.4f} ({hr.split('/')[-1]}) "
                             f"| {ratio} |")
        L.append("")
        L.append("The ratio is the held-out gain as a fraction of the training-reader "
                 "gain, shown only where the training-reader gain is positive - a "
                 "ratio of two negative numbers says nothing about transfer. A ratio "
                 "near zero alongside a solid training-reader gain is the "
                 "reader-specific-phrasing failure mode.")
        hr = held_out[0]
        best_ck = next((c for c in checkpoints if "behavioral" in c), checkpoints[-1])
        gt_map, gh_map = by[(best_ck, train_reader, 0)], by[(best_ck, hr, 0)]
        both = [(k, gt_map[k], gh_map[k]) for k in sorted(set(gt_map) & set(gh_map))
                if np.isfinite(gt_map[k]) and np.isfinite(gh_map[k])]
        both.sort(key=lambda t: t[1] - t[2], reverse=True)
        L += ["", "### Positions where the readers disagree most", "",
              f"Checkpoint `{best_ck}`, no context: the cases to read first when the "
              f"training reader improves and the held-out one does not.", "",
              "| position | training-reader gain | held-out gain | difference |",
              "|---|---|---|---|"]
        for k, a_, b_ in both[:5]:
            L.append(f"| {k} | {a_:+.4f} | {b_:+.4f} | {a_ - b_:+.4f} |")

    # ---- 5. verbatim overlap + length ----
    tok_by = defaultdict(dict)
    ov_prefix = defaultdict(list)
    ov_cont = defaultdict(list)
    for e in expls:
        tok_by[e["checkpoint"]][e["row_id"]] = e["n_tokens"]
        if e.get("explanation"):
            ov_prefix[e["checkpoint"]].append(
                overlap_fraction(e["explanation"], e.get("prefix_tail", "")))
            ov_cont[e["checkpoint"]].append(
                overlap_fraction(e["explanation"], " ".join(e.get("continuations") or
                                                            [e.get("continuation_0", "")])))
    L += ["", "## 5. Is it explanation or context recovery? Overlap and length", "",
          "Share of each explanation's word 4-grams found verbatim in the document "
          "prefix (what the model was reading) and in the sampled continuations (what "
          "it wrote next). A rising prefix overlap under RL means the verbalizer is "
          "learning to quote; a rising continuation overlap means it is leaking the "
          "future, which the reader will reward without understanding anything.", "",
          "| checkpoint | extraction | median tokens | 4-gram overlap with prefix "
          "| 4-gram overlap with continuations |",
          "|---|---|---|---|---|"]
    for ck in checkpoints + sorted(refs):
        toks = [t for t in tok_by[ck].values() if t > 0]
        L.append(f"| {ck} | {summary['extraction_rate'].get(ck, float('nan')):.1%} "
                 f"| {np.median(toks) if toks else float('nan'):.0f} "
                 f"| {np.nanmean(ov_prefix[ck]) if ov_prefix[ck] else float('nan'):.3f} "
                 f"| {np.nanmean(ov_cont[ck]) if ov_cont[ck] else float('nan'):.3f} |")
    base_ck = "sft" if "sft" in checkpoints else checkpoints[0]
    gain_ck0 = {ck: by[(ck, train_reader, 0)] for ck in checkpoints}
    adj = length_adjusted_effects(gain_ck0, tok_by, checkpoints, base_ck)
    if adj:
        L += ["", f"Length-adjusted effects on the training reader (OLS of gain on "
              f"explanation tokens with checkpoint dummies, pooled; baseline `{base_ck}`):",
              "", "| checkpoint | raw difference vs baseline | length-adjusted difference "
              "| gain per extra token |", "|---|---|---|---|"]
        for ck, (raw, a, slope) in adj.items():
            L.append(f"| {ck} | {raw:+.4f} | {a:+.4f} | {slope:+.5f} |")
        L.append("")
        L.append("If the adjusted difference is much smaller than the raw one, the "
                 "checkpoint's advantage is mostly that it writes more.")

    # ---- 6. examples ----
    L += ["", "## 6. Representative explanations", ""]
    rng = np.random.default_rng(seed)
    label = dict(zip(checkpoints, checkpoints))
    if blind:
        shuffled_names = list(checkpoints)
        rng.shuffle(shuffled_names)
        label = {ck: f"model {chr(65 + i)}" for i, ck in enumerate(shuffled_names)}
        L += ["Checkpoint identities are blinded; the key is at the end of this "
              "section. Read for grammar, repetition, unexplained shorthand, quoted "
              "continuations and invented context before unblinding.", ""]
    per_ck = defaultdict(list)
    idx = {}
    for e in expls:
        idx[(e["checkpoint"], e["row_id"])] = e
        if e["checkpoint"] in checkpoints and e["explanation"]:
            per_ck[e["checkpoint"]].append(e)
    common = None
    for ck in checkpoints:
        ids = {e["row_id"] for e in per_ck[ck]}
        common = ids if common is None else (common & ids)
    common = sorted(common or [])
    pick = (list(rng.choice(common, size=min(n_examples, len(common)), replace=False))
            if common else [])
    for row_id in pick:
        any_e = idx[(checkpoints[0], row_id)]
        L += [f"### position {row_id}", "",
              "Document text before the activation (tail):", "",
              "> " + (any_e.get("prefix_tail") or "").replace("\n", " ")[-350:], "",
              "What the target model actually wrote next:", "",
              "> " + (any_e.get("continuation_0") or "").replace("\n", " "), ""]
        for ck in checkpoints:
            e = idx.get((ck, row_id))
            if not e:
                continue
            gv = by[(ck, train_reader, 0)].get(row_id, float("nan"))
            L += [f"**{label[ck]}** (gain {gv:+.3f}, {e['n_tokens']} tokens):", "",
                  (e["explanation"] or "<extraction failed>").strip(), ""]
    if blind:
        L += ["", "Key: " + ", ".join(f"{v} = `{k}`" for k, v in sorted(
            label.items(), key=lambda kv: kv[1])), ""]

    # ---- 7. caveats ----
    L += ["", "## 7. Notes and caveats", "",
          "- Gain is a difference between two prompts over an identical set of "
          "scored tokens. It is not an information-theoretic quantity and does "
          "not show that the explanation is sufficient for the activation.",
          "- In the context-free setting the reader's baseline knows nothing about "
          "the document, so any information about the document is rewarded - "
          "including a quotation of it. The context-conditioned tables and the "
          "tail_quote reference exist to separate explanation from context recovery.",
          "- The continuation is tokenized on its own and appended to the reader "
          "prompt's tokens, so the scored token set is identical across "
          "conditions. That costs a little naturalness at the seam in exchange "
          "for exact pairing.",
          "- `shuffled` reuses another document's explanation under a derangement; "
          "`same_doc` reuses the other position of the same document. Explanation "
          "texts therefore appear in two rows each, a mild cross-row coupling the "
          "document-clustered bootstrap does not model (variance slightly "
          "understated, estimates unaffected).",
          "- Held-out reader results never influenced training or checkpoint "
          "selection; the trainer does not load that model at all.",
          "- One RL seed. Intervals are over positions, not over training runs.",
          ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", default=None, help="default <eval-dir>/report.md")
    p.add_argument("--title", default="Frozen-readout NLA pilot")
    p.add_argument("--wandb-url", default=None)
    p.add_argument("--n-examples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blind", action="store_true",
                   help="Hide checkpoint identities in the examples section (the "
                        "key is printed after them) so the writing-quality read "
                        "happens before you know which objective wrote it.")
    args = p.parse_args()
    d = Path(args.eval_dir)
    md = build_report(d, args.title, args.wandb_url, args.n_examples, args.seed,
                      blind=args.blind)
    out = Path(args.out) if args.out else d / "report.md"
    out.write_text(md)
    print(md)
    print(f"\n[report] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
