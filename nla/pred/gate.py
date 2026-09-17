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

VERDICT: decided on ONE cell - the RL-init checkpoint (the first --checkpoint,
normally sft) under the TRAINING reader (the first --readers entry), with no
document context - because that is the reward RL will see. Pass requires the
matched-minus-shuffled AND matched-minus-same_doc differences to be positive
with 95% intervals excluding zero (and above --min-gap). Every other cell,
the context-conditioned tables and the tail-quote reference are printed as
information. Failing that, the run exits non-zero and RL should not start - the
thing to fix is the scoring task, not the optimizer.

The gate also reports SPLIT-HALF RELIABILITY: the per-position gain scored on
branches {0,1} vs {2,3}. A reward whose per-position values do not agree with
themselves across sampled futures is noise no optimizer can see through.

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

from nla.pred.data import load_positions, take_by_doc
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
    p.add_argument("--context-words", type=int, nargs="*", default=[0, 64],
                   help="Context settings to score: 0 = the reader sees no "
                        "document text; N = both prompts also show the last N "
                        "words of the document, so the explanation must add "
                        "something the text does not already say.")
    p.add_argument("--tail-quote-words", type=int, default=40,
                   help="Add a synthetic 'explanation' = the last N words of the "
                        "prefix, scored like a checkpoint. 0 disables.")
    p.add_argument("--reliability", action=argparse.BooleanOptionalAction, default=None,
                   help="Split-half reliability of the per-position gain (scores "
                        "each condition twice more on half the branches). Default: "
                        "on for the gate, off for the final eval.")
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
    # Whole documents, so every position's same-document mate is present.
    rows = take_by_doc(load_positions(args.positions, split=split, with_prefix=True),
                       n_positions)
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
        tail_quote_words=args.tail_quote_words,
    )
    del cfg
    reliability = (job_type == "gate") if args.reliability is None else args.reliability
    records, partner, reader_diag = score_all(
        rows, expl_sets, args.readers, buckets=DEFAULT_BUCKETS, branches=branches,
        device=args.device, dtype=args.reader_dtype, seed=args.seed,
        context_words_list=tuple(args.context_words), reliability=reliability,
        max_batch_rows=args.reader_batch_rows,
        max_batch_tokens=args.reader_batch_tokens,
    )
    del partner
    summary = summarize(records, expl_sets, buckets=DEFAULT_BUCKETS,
                        n_boot=args.n_boot, seed=args.seed)
    summary["reader_diagnostics"] = reader_diag
    summary["readers"] = list(args.readers)
    summary["training_reader"] = args.readers[0]
    summary["init_checkpoint"] = specs[0].name
    summary["split"] = split
    summary["reader_templates"] = ReaderTemplates().as_dict()
    summary["gen_temperature"] = args.gen_temperature
    write_outputs(args.out_dir, rows, expl_sets, records, summary)
    table = format_table(summary)
    print("\n" + table + "\n", flush=True)
    if run_ is not None:
        flat = {}
        for cell in summary["cells"]:
            tag = f"{cell['checkpoint']}/{cell['reader'].split('/')[-1]}/ctx{cell['context_words']}"
            h = cell["headline"]
            flat[f"gain/{tag}"] = h["gain"]
            for k_src, k_dst in (("matched_minus_shuffled", "mms"), ("mms_p", "mms_p"),
                                 ("matched_minus_same_doc", "msd"), ("msd_p", "msd_p"),
                                 ("reliability_gain", "reliability")):
                if k_src in h:
                    flat[f"{k_dst}/{tag}"] = h[k_src]
            for bi, e in enumerate(cell["per_bucket"]):
                flat[f"gain_bucket{bi}/{tag}"] = e["gain"]
        for n, v in summary["extraction_rate"].items():
            flat[f"extraction_rate/{n}"] = v
        run_.log(flat)
        run_.summary.update(flat)
        log_table(
            run_, "table/headline",
            ["checkpoint", "reader", "context_words", "bucket", "gain", "gain_lo",
             "gain_hi", "shuffled", "matched_minus_shuffled", "p_mms", "same_doc",
             "matched_minus_same_doc", "p_msd", "reliability"],
            [[c["checkpoint"], c["reader"], c["context_words"],
              f"{e['bucket'][0]}-{e['bucket'][1]}", e["gain"], e["gain_lo"],
              e["gain_hi"], e.get("shuffled_gain"), e.get("matched_minus_shuffled"),
              e.get("mms_p"), e.get("same_doc_gain"), e.get("matched_minus_same_doc"),
              e.get("msd_p"), e.get("reliability_gain")]
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
                   help="Required matched-minus-control gain (nats/target token) "
                        "for a pass, in addition to the CI excluding zero.")
    p.add_argument("--gate-checkpoint", default=None,
                   help="Checkpoint the verdict is decided on. Default: the first "
                        "--checkpoint (the RL init).")
    p.add_argument("--gate-reader", default=None,
                   help="Reader the verdict is decided on. Default: the first "
                        "--readers entry (the training reader).")
    args = p.parse_args()

    summary, _records, _rows, _expl, run_ = run(
        args, split=args.split, n_positions=args.n_positions,
        job_type="gate", group=args.wandb_group or "gate",
    )

    # ---- verdict: ONE pre-registered cell, not "any cell passes" ----
    # With 2 checkpoints x 2 readers, "any cell" passes ~9% of the time under the
    # null, and a Gemma-only pass says nothing about the reward RL will train on.
    gate_ck = args.gate_checkpoint or summary["init_checkpoint"]
    gate_rd = args.gate_reader or summary["training_reader"]
    decisive = None
    rows_out = []
    for cell in summary["cells"]:
        h = cell["headline"]
        d, lo = h.get("matched_minus_shuffled"), h.get("mms_lo")
        sd_d, sd_lo = h.get("matched_minus_same_doc"), h.get("msd_lo")
        ok_sh = bool(d is not None and np.isfinite(d) and d > args.min_gap and lo > 0)
        ok_sd = bool(sd_d is not None and np.isfinite(sd_d) and sd_d > args.min_gap
                     and sd_lo > 0)
        is_gate = (cell["checkpoint"] == gate_ck and cell["reader"] == gate_rd
                   and cell["context_words"] == 0)
        row = {"checkpoint": cell["checkpoint"], "reader": cell["reader"],
               "context_words": cell["context_words"],
               "matched_minus_shuffled": d, "mms_lo": lo, "mms_hi": h.get("mms_hi"),
               "matched_minus_same_doc": sd_d, "msd_lo": sd_lo, "msd_hi": h.get("msd_hi"),
               "reliability_gain": h.get("reliability_gain"),
               "pass_shuffled": ok_sh, "pass_same_doc": ok_sd, "decisive": is_gate}
        rows_out.append(row)
        if is_gate:
            decisive = row
    passed = bool(decisive and decisive["pass_shuffled"] and decisive["pass_same_doc"])
    print("GATE VERDICT  (decided on "
          f"{gate_ck} / {gate_rd.split('/')[-1]} / no document context)")
    print("-" * 100)
    for r in rows_out:
        def _f(v):
            return "   n/a " if v is None or not np.isfinite(v) else f"{v:+.4f}"
        mark = "==>" if r["decisive"] else "   "
        print(f"{mark} {r['checkpoint']:<12} {r['reader'].split('/')[-1]:<16} "
              f"ctx{r['context_words']:<3} "
              f"m-shuf {_f(r['matched_minus_shuffled'])} "
              f"[{_f(r['mms_lo'])},{_f(r['mms_hi'])}] {'ok ' if r['pass_shuffled'] else 'no '} "
              f"m-samedoc {_f(r['matched_minus_same_doc'])} "
              f"[{_f(r['msd_lo'])},{_f(r['msd_hi'])}] {'ok ' if r['pass_same_doc'] else 'no '} "
              f"rel {_f(r['reliability_gain'])}")
    print("-" * 100)
    if decisive is None:
        print(f"no cell for {gate_ck} / {gate_rd} at ctx 0 - cannot decide")
    print("PASS: the reward separates this position's explanation from a mismatched "
          "one, including a same-document mismatch -> RL may start."
          if passed else
          "FAIL: no reliable activation-specific signal in the reward RL would train "
          "on. Read the same_doc and context-conditioned rows and the tail_quote "
          "reference before touching the optimizer.")
    Path(args.out_dir, "verdict.json").write_text(json.dumps({
        "passed": passed, "min_gap": args.min_gap, "decided_on":
        {"checkpoint": gate_ck, "reader": gate_rd, "context_words": 0},
        "rows": rows_out}, indent=2))
    if run_ is not None:
        run_.summary["gate/passed"] = passed
        finish_run(run_)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
