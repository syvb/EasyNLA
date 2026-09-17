"""Stage 4: the final evaluation. All checkpoints x both readers x matched/shuffled.

Same machinery as the gate (nla/pred/evaluate.py), pointed at the held-out `eval`
split, with the checkpoint-vs-checkpoint comparisons the experiment is actually
about added on top:

  * Evaluation 1 - training-reader gain: did behavioral RL optimize its objective?
  * Evaluation 2 - held-out-reader gain: did that survive changing the reader?
  * Evaluation 3 - matched minus shuffled: is the gain about THIS activation?

Every comparison is paired over positions and bootstrapped over documents.

    python -m nla.pred.eval --positions <positions.parquet> \
        --checkpoint sft= \
        --checkpoint recon_rl=syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0 \
        --checkpoint behavioral_rl=<ckpts>/rl/iter_000300 \
        --out-dir evals/final --baseline-checkpoint sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nla.pred.evaluate import compare_checkpoints
from nla.pred.gate import add_common_args, run
from nla.pred.reader import DEFAULT_BUCKETS
from nla.pred.stats import fmt_ci
from nla.pred.wandb_util import finish_run, log_table


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n-positions", type=int, default=2000)
    p.add_argument("--split", default="eval")
    p.add_argument("--baseline-checkpoint", default="sft",
                   help="Checkpoint every other one is compared against.")
    args = p.parse_args()

    summary, records, _rows, _expl, run_ = run(
        args, split=args.split, n_positions=args.n_positions,
        job_type="eval", group=args.wandb_group or "eval",
    )

    names = sorted({r["checkpoint"] for r in records})
    base = args.baseline_checkpoint
    comparisons = {}
    for ctx in summary["context_words_list"]:
        if base in names:
            for name in names:
                if name == base:
                    continue
                comparisons[f"{name}_vs_{base}/ctx{ctx}"] = compare_checkpoints(
                    records, name, base, buckets=DEFAULT_BUCKETS, context_words=ctx,
                    n_boot=args.n_boot, seed=args.seed)
        # Behavioral vs reconstruction, the H4 comparison, when both are present.
        if "behavioral_rl" in names and "recon_rl" in names:
            comparisons[f"behavioral_rl_vs_recon_rl/ctx{ctx}"] = compare_checkpoints(
                records, "behavioral_rl", "recon_rl", buckets=DEFAULT_BUCKETS,
                context_words=ctx, n_boot=args.n_boot, seed=args.seed)

    # The PRIMARY, pre-registered test. Everything else is secondary and there
    # is no multiplicity correction, so this is the one number that decides H2.
    # It is decided in the CONTEXT-CONDITIONED setting (the largest context
    # scored): without document context a verbatim quote of the prefix beats a
    # real explanation, so a context-free win could be pure context recovery.
    # The context-free comparison is reported beside it as the plan's original
    # specification.
    held_out = [r for r in args.readers if r != args.readers[0]]
    primary = None
    if "behavioral_rl" in names and base in names and held_out:
        ctx_primary = max(summary["context_words_list"])
        primary = {"comparison": f"behavioral_rl - {base}", "reader": held_out[0],
                   "condition": "matched", "context_words": ctx_primary,
                   "bucket": summary["headline_bucket"],
                   **comparisons[f"behavioral_rl_vs_{base}/ctx{ctx_primary}"][held_out[0]],
                   "plan_spec_context_free":
                       comparisons[f"behavioral_rl_vs_{base}/ctx0"][held_out[0]],
                   "note": "single RL seed; the CI is over positions, not over runs"}

    print("\nPAIRED COMPARISONS (headline bucket, nats per target token)")
    print("-" * 78)
    rows_tbl = []
    for label, per_reader in comparisons.items():
        for reader, d in per_reader.items():
            print(f"  {label:<40} {reader.split('/')[-1]:<18} "
                  f"{fmt_ci(d['diff'], d['lo'], d['hi'])}  p={d['p']:.3f}  n={d['n']}")
            rows_tbl.append([label, reader, d["diff"], d["lo"], d["hi"], d["p"], d["n"]])
    print("-" * 78)
    if primary:
        pf = primary["plan_spec_context_free"]
        print(f"PRIMARY TEST  {primary['comparison']} on {primary['reader'].split('/')[-1]}, "
              f"reader shown {primary['context_words']} words of context: "
              f"{fmt_ci(primary['diff'], primary['lo'], primary['hi'])} "
              f"p={primary['p']:.3f} n={primary['n']}  ({primary['note']})")
        print(f"  plan's context-free version: {fmt_ci(pf['diff'], pf['lo'], pf['hi'])} "
              f"p={pf['p']:.3f}")

    out = Path(args.out_dir)
    summary["comparisons"] = comparisons
    summary["primary_test"] = primary
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if run_ is not None:
        flat = {f"cmp/{label}/{rd.split('/')[-1]}": d["diff"]
                for label, per in comparisons.items() for rd, d in per.items()}
        flat.update({f"cmp_p/{label}/{rd.split('/')[-1]}": d["p"]
                     for label, per in comparisons.items() for rd, d in per.items()})
        run_.log(flat)
        run_.summary.update(flat)
        log_table(run_, "table/comparisons",
                  ["comparison", "reader", "diff", "lo", "hi", "p", "n"], rows_tbl)
        finish_run(run_)
    print(f"\nwrote {out}/summary.json, scores.jsonl, explanations.jsonl")


if __name__ == "__main__":
    main()
