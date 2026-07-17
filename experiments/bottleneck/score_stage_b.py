"""Score Stage B outputs and build the retention table (CPU-ok, offline).

    python -m experiments.bottleneck.score_stage_b --results <out_dir>/stage_b \
        [--conditions clean,identity,nla]

Reads every {task}/{condition}_seed{seed}.parquet, scores per problem, and
writes retention.md / scores.parquet next to the results. Retention =
score(cond) / score(clean), paired bootstrap CI over problems. IFEval falls
back to writing official-format JSONL when the reference scorer isn't installed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.bottleneck.scoring import (
    bootstrap_mean_ci,
    extract_boxed,
    extract_letter,
    paired_retention_ci,
    repetition_rate,
    score_ifeval,
    score_row,
)

MC_CHANCE = 0.1  # MMLU-Pro: 10 options


def format_ok(task: str, output: str) -> float:
    """Did the output comply with the instructed answer format? Reported per
    condition so 'capability lost' vs 'format compliance lost' is separable —
    a degraded model may know the answer but stop emitting ####/boxed/letters."""
    if task in ("gsm8k", "mgsm"):
        return float("####" in output)
    if task == "math500":
        return float(extract_boxed(output) is not None)
    if task in ("humaneval", "mbpp"):
        return float("```" in output)
    if task == "mmlu_pro":
        return float(extract_letter(output) is not None)
    if task in ("triviaqa", "popqa", "gsm8k_short"):
        return float(len(output.strip().split("\n")[0].split()) <= 12)
    return float("nan")


def aux_stats(task: str, rows: list[dict]) -> str:
    n = max(len(rows), 1)
    out_toks = np.array([r["n_output_tokens"] for r in rows], dtype=float)
    cap = np.mean([r["hit_cap"] for r in rows])
    rep = np.mean([repetition_rate(r["output"]) for r in rows])
    fmt = np.array([format_ok(task, r["output"]) for r in rows], dtype=float)
    fmt_s = f" fmt_ok={np.nanmean(fmt):.2f}" if not np.all(np.isnan(fmt)) else ""
    return (f"len={out_toks.mean():.0f} hit_cap={cap:.2f} rep3={rep:.2f}{fmt_s}"
            f" n={n}")


def load_runs(results_dir: Path) -> dict:
    """{(task, condition, seed): {pid: row}}"""
    runs = {}
    for f in sorted(results_dir.glob("*/*.parquet")):
        if f.stem.endswith("_steplogs"):
            continue
        m = re.fullmatch(r"(\w+)_seed(\d+)", f.stem)
        if not m:
            continue
        task = f.parent.name
        rows = pq.read_table(f).to_pylist()
        runs[(task, m.group(1), int(m.group(2)))] = {r["pid"]: r for r in rows}
    return runs


def score_run(task: str, rows_by_pid: dict, out_dir: Path, tag: str) -> dict[str, float]:
    pids = sorted(rows_by_pid)
    rows = [rows_by_pid[p] for p in pids]
    if task == "ifeval":
        s = score_ifeval(rows)
        if s is None:
            jl = out_dir / f"ifeval_responses_{tag}.jsonl"
            with jl.open("w") as f:
                for r in rows:
                    f.write(json.dumps({"prompt": json.loads(r["gold"])["prompt"],
                                        "response": r["output"]}) + "\n")
            print(f"[ifeval] scorer not installed (pip install instruction-following-eval); "
                  f"wrote {jl} for the official script")
            return {}
        return dict(zip(pids, s))
    scores = {}
    for pid, r in zip(pids, rows):
        res = score_row(task, json.loads(r["gold"]), r["output"])
        if task == "fluency":
            # fluency has no accuracy; report clean-M NLL + repetition instead
            scores[pid] = r.get("clean_nll", float("nan"))
        else:
            scores[pid] = res["score"]
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="<out_dir>/stage_b")
    args = p.parse_args()
    results_dir = Path(args.results)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no run parquets under {results_dir}")

    scored: dict[tuple, dict[str, float]] = {}
    for (task, cond, seed), rows in sorted(runs.items()):
        scored[(task, cond, seed)] = score_run(task, rows, results_dir, f"{cond}_s{seed}")
        vals = np.array([v for v in scored[(task, cond, seed)].values()
                         if not np.isnan(v)])
        aux = aux_stats(task, list(rows.values()))
        if len(vals):
            mean, lo, hi = bootstrap_mean_ci(vals)
            print(f"{task:12s} {cond:10s} seed{seed}: "
                  f"{'nll' if task == 'fluency' else 'score'} "
                  f"{mean:.4f} [{lo:.4f}, {hi:.4f}]  | {aux}")
        else:
            print(f"{task:12s} {cond:10s} seed{seed}: (no inline score) | {aux}")

    # Retention vs clean (seed-matched to clean seed 0 — clean is deterministic).
    lines = ["| task | condition | seed | score | clean | retention [95% CI] | n |",
             "|---|---|---|---|---|---|---|"]
    out_rows = []
    by_task_cond = defaultdict(list)
    for (task, cond, seed), s in sorted(scored.items()):
        clean = scored.get((task, "clean", 0))
        if not s or not clean or cond == "clean":
            continue
        pids = sorted(set(s) & set(clean))
        cv = np.array([s[p] for p in pids])
        kv = np.array([clean[p] for p in pids])
        ok = ~(np.isnan(cv) | np.isnan(kv))
        cv, kv = cv[ok], kv[ok]
        if not len(cv):
            continue
        if task == "fluency":  # lower NLL is better — report delta, not ratio
            lines.append(f"| {task} | {cond} | {seed} | {cv.mean():.3f} | {kv.mean():.3f} "
                         f"| ΔNLL {cv.mean()-kv.mean():+.3f} | {len(cv)} |")
            out_rows.append({"task": task, "condition": cond, "seed": seed,
                             "score": float(cv.mean()), "clean": float(kv.mean()),
                             "retention": float("nan"), "lo": float("nan"),
                             "hi": float("nan"), "n": len(cv)})
            continue
        ret, lo, hi = paired_retention_ci(cv, kv)
        by_task_cond[(task, cond)].append(ret)
        extra = ""
        if task == "mmlu_pro":
            # raw retention has a hidden floor of chance/clean (a floored model
            # still scores ~10% by luck); report chance-adjusted alongside
            adj = ((cv.mean() - MC_CHANCE) / (kv.mean() - MC_CHANCE)
                   if kv.mean() > MC_CHANCE else float("nan"))
            extra = f" (chance-adj {adj:.3f})"
        lines.append(f"| {task} | {cond} | {seed} | {cv.mean():.3f} | {kv.mean():.3f} "
                     f"| {ret:.3f} [{lo:.3f}, {hi:.3f}]{extra} | {len(cv)} |")
        out_rows.append({"task": task, "condition": cond, "seed": seed,
                         "score": float(cv.mean()), "clean": float(kv.mean()),
                         "retention": ret, "lo": lo, "hi": hi, "n": len(cv)})

    md = results_dir / "retention.md"
    md.write_text("# Stage B retention\n\n" + "\n".join(lines) + "\n")
    if out_rows:
        pq.write_table(pa.Table.from_pylist(out_rows), results_dir / "scores.parquet")
    print(f"\nwrote {md}")

    # Per-slice breakdowns that the aggregate hides.
    for (task, cond, seed), s in sorted(scored.items()):
        rows = runs[(task, cond, seed)]
        if task == "mgsm":
            by_lang = defaultdict(list)
            for pid, v in s.items():
                by_lang[json.loads(rows[pid]["gold"])["lang"]].append(v)
            print(f"mgsm {cond} seed{seed}: " + "  ".join(
                f"{la}={np.mean(v):.3f}(n={len(v)})" for la, v in sorted(by_lang.items())))
        if task == "mmlu_pro":
            by_cat = defaultdict(list)
            for pid, v in s.items():
                by_cat[json.loads(rows[pid]["gold"])["category"]].append(v)
            print(f"mmlu_pro {cond} seed{seed} (per-category, NOISY at n~40): " + "  ".join(
                f"{c}={np.mean(v):.2f}" for c, v in sorted(by_cat.items())))


if __name__ == "__main__":
    main()
