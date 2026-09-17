"""Stage 5: turn the eval outputs into the report the experiment is supposed to end with.

Reads an eval directory (summary.json + scores.jsonl + explanations.jsonl) and
writes report.md: the headline table, the bridge-length breakdown, the paired
checkpoint comparisons, the activation-dependence result, a length check, and
representative explanations.

THE LENGTH CHECK is not decoration. The frozen-reader reward has an obvious
degenerate solution - write more words, help the reader more - so a behavioral
checkpoint that simply writes longer explanations could top the table without
communicating anything better about the activation. The report therefore always
prints explanation length beside every gain, and breaks gain down by length
tercile, so a length-driven win is visible rather than buried.

    python -m nla.pred.report --eval-dir evals/final --out evals/final/report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nla.pred.stats import bootstrap_mean, paired_bootstrap_diff


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(mean, lo, hi, digits=4):
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def build_report(eval_dir: Path, title: str, wandb_url: str | None = None,
                 n_examples: int = 12, seed: int = 0) -> str:
    summary = json.loads((eval_dir / "summary.json").read_text())
    scores = _read_jsonl(eval_dir / "scores.jsonl")
    expls = _read_jsonl(eval_dir / "explanations.jsonl")
    buckets = [tuple(b) for b in summary["buckets"]]
    hb = buckets.index(tuple(summary["headline_bucket"]))
    checkpoints = sorted({r["checkpoint"] for r in scores})
    readers = sorted({r["reader"] for r in scores})
    train_reader = summary.get("training_reader", readers[0])

    L = [f"# {title}", ""]
    L.append(
        f"Frozen-reader predictive gain on {summary['n_positions']} held-out "
        f"positions, {len(buckets)} bridge lengths, "
        f"{len(checkpoints)} checkpoints x {len(readers)} readers. Gain is nats "
        f"per target token relative to the same reader shown no explanation; "
        f"intervals are 95% bootstrap over documents, and every comparison is "
        f"paired over positions."
    )
    if wandb_url:
        L += ["", f"Training curve: {wandb_url}"]
    L += ["", "## 1. Headline: predictive gain and activation-dependence", ""]
    hbl = summary["headline_bucket"]
    L.append(f"Bucket {hbl[0]}-{hbl[1]} of the continuation, i.e. the reader has "
             f"already been given {hbl[0]} real tokens as a bridge.")
    L += ["",
          "| checkpoint | reader | gain [95% CI] | shuffled-activation gain | "
          "matched - shuffled [95% CI] | p | median expl. tokens |",
          "|---|---|---|---|---|---|---|"]
    med_tok = summary.get("explanation_tokens_median", {})
    for cell in summary["cells"]:
        h = cell["headline"]
        mms = h.get("matched_minus_shuffled")
        L.append(
            f"| {cell['checkpoint']} | {cell['reader'].split('/')[-1]} "
            f"| {_fmt(h['gain'], h['gain_lo'], h['gain_hi'])} "
            f"| {h.get('shuffled_gain', float('nan')):+.4f} "
            f"| {_fmt(mms, h.get('mms_lo', float('nan')), h.get('mms_hi', float('nan')))} "
            f"| {h.get('mms_p', float('nan')):.3f} "
            f"| {med_tok.get(cell['checkpoint'], float('nan')):.0f} |")

    L += ["", "## 2. Gain by bridge length", "",
          "Each bucket is scored in the same forward pass; a later bucket means the "
          "reader has already seen more of the real continuation, so the "
          "explanation has to add something the text itself has not already "
          "revealed.", "",
          "| checkpoint | reader | " + " | ".join(f"{lo}-{hi}" for lo, hi in buckets) + " |",
          "|---|---|" + "---|" * len(buckets)]
    for cell in summary["cells"]:
        vals = " | ".join(f"{e['gain']:+.4f}" for e in cell["per_bucket"])
        L.append(f"| {cell['checkpoint']} | {cell['reader'].split('/')[-1]} | {vals} |")

    # ---- paired comparisons ----
    L += ["", "## 3. Checkpoint comparisons (paired)", ""]
    by = defaultdict(dict)
    docs = {}
    for r in scores:
        if r["condition"] != "matched":
            continue
        by[(r["checkpoint"], r["reader"])][r["row_id"]] = r["gain"][hb]
        docs[r["row_id"]] = r["doc_id"]
    pairs = []
    for a in checkpoints:
        for b in checkpoints:
            if a >= b:
                continue
            pairs.append((a, b))
    L += ["| comparison | reader | difference [95% CI] | p | n |", "|---|---|---|---|---|"]
    for a, b in pairs:
        for rd in readers:
            ga, gb = by[(a, rd)], by[(b, rd)]
            keys = sorted(set(ga) & set(gb))
            if not keys:
                continue
            d, lo, hi, p, n = paired_bootstrap_diff(
                [ga[k] for k in keys], [gb[k] for k in keys],
                clusters=[docs[k] for k in keys], seed=seed)
            L.append(f"| {a} - {b} | {rd.split('/')[-1]} | {_fmt(d, lo, hi)} "
                     f"| {p:.3f} | {n} |")

    # ---- transfer check ----
    held_out = [r for r in readers if r != train_reader]
    if held_out:
        L += ["", "## 4. Does the gain transfer to the held-out reader?", "",
              f"Training reader: `{train_reader}`. Held out: "
              + ", ".join(f"`{h}`" for h in held_out) + ".", "",
              "| checkpoint | training-reader gain | held-out-reader gain | ratio |",
              "|---|---|---|---|"]
        for ck in checkpoints:
            gt = np.nanmean(list(by[(ck, train_reader)].values())) if by[(ck, train_reader)] else np.nan
            for hr in held_out:
                gh = np.nanmean(list(by[(ck, hr)].values())) if by[(ck, hr)] else np.nan
                ratio = gh / gt if np.isfinite(gt) and abs(gt) > 1e-9 else float("nan")
                L.append(f"| {ck} | {gt:+.4f} | {gh:+.4f} ({hr.split('/')[-1]}) "
                         f"| {ratio:.2f} |")
        L.append("")
        L.append("A ratio near zero with a positive training-reader gain is the "
                 "reader-specific-phrasing failure mode: the verbalizer found "
                 "words that suit one reader rather than words that communicate.")

    # ---- length check ----
    L += ["", "## 5. Length check", "",
          "The reward rises with explanation length for free, so a longer "
          "checkpoint can win without explaining better. Gains below are split by "
          "tercile of the checkpoint's own explanation length.", "",
          "| checkpoint | extraction | median tokens | short third | middle third | long third |",
          "|---|---|---|---|---|---|"]
    tok_by = defaultdict(dict)
    for e in expls:
        tok_by[e["checkpoint"]][e["row_id"]] = e["n_tokens"]
    for ck in checkpoints:
        toks = tok_by[ck]
        g = by[(ck, train_reader)]
        keys = [k for k in sorted(set(toks) & set(g)) if toks[k] > 0]
        if not keys:
            continue
        tv = np.array([toks[k] for k in keys], dtype=float)
        gv = np.array([g[k] for k in keys], dtype=float)
        q1, q2 = np.percentile(tv, [33.3, 66.7])
        cells = []
        for lo_m, hi_m in ((-np.inf, q1), (q1, q2), (q2, np.inf)):
            m = (tv > lo_m) & (tv <= hi_m)
            cells.append(f"{np.nanmean(gv[m]):+.4f}" if m.sum() else "n/a")
        L.append(f"| {ck} | {summary['extraction_rate'].get(ck, float('nan')):.1%} "
                 f"| {np.median(tv):.0f} | " + " | ".join(cells) + " |")

    # ---- examples ----
    L += ["", "## 6. Representative explanations", ""]
    rng = np.random.default_rng(seed)
    per_ck = defaultdict(list)
    for e in expls:
        if e["explanation"]:
            per_ck[e["checkpoint"]].append(e)
    common = None
    for ck in checkpoints:
        ids = {e["row_id"] for e in per_ck[ck]}
        common = ids if common is None else (common & ids)
    common = sorted(common or [])
    pick = list(rng.choice(common, size=min(n_examples, len(common)), replace=False)) if common else []
    idx = {(e["checkpoint"], e["row_id"]): e for e in expls}
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
            gv = by[(ck, train_reader)].get(row_id, float("nan"))
            L += [f"**{ck}** (gain {gv:+.3f}, {e['n_tokens']} tokens):", "",
                  (e["explanation"] or "<extraction failed>").strip(), ""]

    # ---- failure notes ----
    L += ["", "## 7. Notes and caveats", "",
          "- Gain is a difference between two prompts over an identical set of "
          "scored tokens. It is not an information-theoretic quantity and does "
          "not show that the explanation is sufficient for the activation.",
          "- The continuation is tokenized on its own and appended to the reader "
          "prompt's tokens, so the scored token set is identical across "
          "conditions. That costs a little naturalness at the seam in exchange "
          "for exact pairing.",
          "- The shuffled condition reuses another position's explanation text "
          "under a derangement that never pairs two positions from the same "
          "document.",
          "- Held-out reader results never influenced training or checkpoint "
          "selection; the trainer does not load that model at all.",
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
    args = p.parse_args()
    d = Path(args.eval_dir)
    md = build_report(d, args.title, args.wandb_url, args.n_examples, args.seed)
    out = Path(args.out) if args.out else d / "report.md"
    out.write_text(md)
    print(md)
    print(f"\n[report] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
