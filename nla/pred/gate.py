"""Stage 2: the pre-RL gate. Does this reward see the ACTIVATION at all?

Before spending a run optimizing the frozen-reader score, check that the score
already responds to which activation an explanation came from. On held-out
validation positions, score:

  * the SFT explanation for the position's own activation  (matched)
  * an SFT explanation from a different position           (shuffled)
  * the reconstruction-RL explanation, matched and shuffled (if a checkpoint is given)
  * no explanation                                          (the baseline, implicit)

against the SAME stored continuations. The comparison that matters is
matched > shuffled. A gain over "no explanation" alone proves nothing: a generic
note about how documents continue can help a reader without saying anything about
this activation.

VERDICT: pass if matched - shuffled is positive with a 95% interval excluding
zero on at least one reader. Failing that, the run exits non-zero and RL should
not start - the thing to fix is the scoring task, not the optimizer.

    python -m nla.pred.gate --positions <positions.parquet> \
        --base-ckpt syvb/nanonla-qwen3-8b-L24-av \
        --checkpoint sft= --checkpoint recon_rl=<repo>#p0.0 \
        --out-dir evals/gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nla.pred.data import load_positions
from nla.pred.evaluate import (
    CheckpointSpec,
    format_table,
    generate_all,
    score_all,
    summarize,
    write_outputs,
)
from nla.pred.reader import DEFAULT_BUCKETS, ReaderTemplates
from nla.pred.wandb_util import finish_run, init_run, log_table


def add_common_args(p):
    p.add_argument("--positions", required=True, help="positions parquet from nla.pred.continuations")
    p.add_argument("--sidecar", default=None, help="defaults to the positions parquet's own sidecar")
    p.add_argument("--base-ckpt", default="syvb/nanonla-qwen3-8b-L24-av",
                   help="The merged AV every condition sits on (SFT = this, no adapter).")
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=ADAPTER[#SUB]",
                   help="Repeatable. Empty adapter = the bare base (the SFT condition).")
    p.add_argument("--readers", nargs="+",
                   default=["Qwen/Qwen3-4B-Base", "google/gemma-3-4b-pt"],
                   help="Frozen readers. The FIRST is the training reader; the rest "
                        "are held out and must never influence training.")
    p.add_argument("--branches", type=int, default=4, help="continuation branches to score")
    p.add_argument("--max-new-tokens", type=int, default=192,
                   help="AV generation cap. Must exceed the SFT length distribution "
                        "(median ~119, p99 ~149 tokens) or a fifth of explanations "
                        "are truncated and the comparison measures brevity.")
    p.add_argument("--gen-temperature", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--reader-batch-rows", type=int, default=32)
    p.add_argument("--reader-batch-tokens", type=int, default=32768)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reader-dtype", default="bfloat16")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--wandb-project", default="pred-nla")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", default=None)
    p.add_argument("--no-wandb", action="store_true")


def run(args, *, split, n_positions, job_type, group):
    import torch

    sidecar = args.sidecar or args.positions
    specs = [CheckpointSpec.parse(s) for s in args.checkpoint]
    assert specs, "give at least one --checkpoint (e.g. --checkpoint sft=)"
    rows = load_positions(args.positions, split=split, limit=n_positions,
                          with_prefix=True)
    assert rows, f"no rows in split {split!r} of {args.positions}"
    n_br = min(args.branches, len(rows[0]["cont_text"]))
    branches = tuple(range(n_br))
    print(f"[data] {len(rows)} {split} positions, {n_br} branches each", flush=True)

    run_ = None if args.no_wandb else init_run(
        args, project=args.wandb_project, name=args.wandb_name,
        group=group or args.wandb_group, job_type=job_type,
        extra_config={"split": split, "n_positions": len(rows)},
    )

    expl_sets, cfg = generate_all(
        rows, specs, base_ckpt=args.base_ckpt, sidecar=sidecar, quant=args.quant,
        device=args.device, max_new_tokens=args.max_new_tokens,
        temperature=args.gen_temperature, batch_size=args.gen_batch, seed=args.seed,
    )
    del cfg
    records, partner, reader_diag = score_all(
        rows, expl_sets, args.readers, buckets=DEFAULT_BUCKETS, branches=branches,
        device=args.device, dtype=args.reader_dtype, seed=args.seed,
        max_batch_rows=args.reader_batch_rows,
        max_batch_tokens=args.reader_batch_tokens,
    )
    del partner
    summary = summarize(records, expl_sets, buckets=DEFAULT_BUCKETS,
                        n_boot=args.n_boot, seed=args.seed)
    summary["reader_diagnostics"] = reader_diag
    summary["readers"] = list(args.readers)
    summary["training_reader"] = args.readers[0]
    summary["split"] = split
    summary["reader_templates"] = ReaderTemplates().as_dict()
    summary["gen_temperature"] = args.gen_temperature
    write_outputs(args.out_dir, rows, expl_sets, records, summary)
    table = format_table(summary)
    print("\n" + table + "\n", flush=True)
    if run_ is not None:
        flat = {}
        for cell in summary["cells"]:
            tag = f"{cell['checkpoint']}/{cell['reader'].split('/')[-1]}"
            h = cell["headline"]
            flat[f"gain/{tag}"] = h["gain"]
            if "matched_minus_shuffled" in h:
                flat[f"mms/{tag}"] = h["matched_minus_shuffled"]
                flat[f"mms_p/{tag}"] = h["mms_p"]
            for bi, e in enumerate(cell["per_bucket"]):
                flat[f"gain_bucket{bi}/{tag}"] = e["gain"]
        for n, v in summary["extraction_rate"].items():
            flat[f"extraction_rate/{n}"] = v
        run_.log(flat)
        run_.summary.update(flat)
        log_table(
            run_, "table/headline",
            ["checkpoint", "reader", "bucket", "gain", "gain_lo", "gain_hi",
             "shuffled", "matched_minus_shuffled", "p"],
            [[c["checkpoint"], c["reader"], f"{e['bucket'][0]}-{e['bucket'][1]}",
              e["gain"], e["gain_lo"], e["gain_hi"], e.get("shuffled_gain"),
              e.get("matched_minus_shuffled"), e.get("mms_p")]
             for c in summary["cells"] for e in c["per_bucket"]],
        )
        log_table(
            run_, "table/explanations",
            ["row_id", "checkpoint", "n_tokens", "explanation", "continuation_0"],
            [[rows[i]["row_id"], n, es.n_tokens[i], (es.texts[i] or "<failed>")[:800],
              rows[i]["cont_text"][0]]
             for n, es in expl_sets.items() for i in range(min(40, len(rows)))],
        )
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary, records, rows, expl_sets, run_


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n-positions", type=int, default=500)
    p.add_argument("--split", default="val")
    p.add_argument("--min-gap", type=float, default=0.0,
                   help="Required matched-minus-shuffled gain (nats/target token) "
                        "for a pass, in addition to the CI excluding zero.")
    args = p.parse_args()

    summary, _records, _rows, _expl, run_ = run(
        args, split=args.split, n_positions=args.n_positions,
        job_type="gate", group=args.wandb_group or "gate",
    )

    # ---- verdict ----
    verdict_rows = []
    passed_any = False
    for cell in summary["cells"]:
        h = cell["headline"]
        d, lo = h.get("matched_minus_shuffled"), h.get("mms_lo")
        if d is None:
            continue
        ok = bool(np.isfinite(d) and d > args.min_gap and lo > 0)
        passed_any = passed_any or ok
        verdict_rows.append((cell["checkpoint"], cell["reader"], d, lo, h["mms_hi"], ok))
    print("GATE VERDICT")
    print("-" * 72)
    for ck, rd, d, lo, hi, ok in verdict_rows:
        print(f"  {'PASS' if ok else 'fail'}  {ck:<12} {rd:<28} "
              f"matched-shuffled {d:+.4f} [{lo:+.4f}, {hi:+.4f}] nats/token")
    print("-" * 72)
    print("Reward is activation-specific -> RL may start."
          if passed_any else
          "NO activation-specific signal on any reader -> investigate the scoring\n"
          "task before running RL (continuation length, bridge, reader choice,\n"
          "explanation truncation).")
    Path(args.out_dir, "verdict.json").write_text(json.dumps({
        "passed": passed_any, "min_gap": args.min_gap,
        "rows": [{"checkpoint": c, "reader": r, "matched_minus_shuffled": d,
                  "lo": lo, "hi": hi, "pass": ok}
                 for c, r, d, lo, hi, ok in verdict_rows],
    }, indent=2))
    if run_ is not None:
        run_.summary["gate/passed"] = passed_any
        finish_run(run_)
    sys.exit(0 if passed_any else 2)


if __name__ == "__main__":
    main()
