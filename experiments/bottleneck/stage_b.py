"""Stage B: generative benchmarks under the continuous NLA bottleneck.

Per (task, condition, seed): batched greedy generation with the layer-24 tap
(clean / identity / nla), outputs + per-token codec logs written as parquet.
Scoring is OFFLINE (score_stage_b.py, CPU-ok) so scorer debugging never burns
GPU hours.

    python -m experiments.bottleneck.stage_b --config experiments/bottleneck/config.yaml \
        --condition nla --tasks gsm8k,triviaqa --seed 0

Outputs land in {out_dir}/stage_b/{task}/{condition}_seed{seed}.parquet
(+ _steplogs.parquet for nla). Existing outputs are skipped (crash-resumable);
--overwrite to redo.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from experiments.bottleneck.patched import BottleneckModel, StepLog
from experiments.bottleneck.tasks import ALL_TASKS, DEFAULT_SIZES, load_task


def write_parquet(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"[out] wrote {len(rows)} rows -> {path}", flush=True)


def run_task(m: BottleneckModel, codec, task_name: str, condition: str, seed: int,
             cfg: dict, out_dir: Path, overwrite: bool):
    out_path = out_dir / "stage_b" / task_name / f"{condition}_seed{seed}.parquet"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return
    n = cfg.get("task_sizes", {}).get(task_name, DEFAULT_SIZES[task_name])
    problems = load_task(task_name, n, seed=cfg.get("data_seed", 0))
    print(f"[task {task_name}] {len(problems)} problems, condition={condition}, "
          f"seed={seed}", flush=True)

    batch_size = cfg.get("gen_batch_size", 64)
    rows, all_logs = [], []
    t_start = time.time()
    for cs in range(0, len(problems), batch_size):
        cohort = problems[cs:cs + batch_size]
        prompt_ids = [m.build_chat_ids(p.messages) for p in cohort]
        max_new = max(p.max_new_tokens for p in cohort)
        logs: list[StepLog] = []
        t0 = time.time()
        gen = m.generate(prompt_ids, max_new, codec=codec, condition=condition,
                         step_logs=logs if condition == "nla" else None)
        dt = time.time() - t0
        n_tok = sum(len(g) for g in gen)
        for j, p in enumerate(cohort):
            out_ids = gen[j]
            text = m.tokenizer.decode(out_ids, skip_special_tokens=True)
            row = {
                "task": p.task, "pid": p.pid, "condition": condition, "seed": seed,
                "prompt": p.messages[-1]["content"],
                "output": text,
                "n_prompt_tokens": len(prompt_ids[j]),
                "n_output_tokens": len(out_ids),
                "hit_cap": len(out_ids) >= p.max_new_tokens
                           and (not out_ids or out_ids[-1] not in m.eos_ids),
                "gold": json.dumps(p.gold),
            }
            if task_name == "fluency" and out_ids:
                # PPL of the output under CLEAN M — the fluency metric.
                row["clean_nll"] = m.score_continuation_nll(prompt_ids[j], out_ids)
            rows.append(row)
        for lg in logs:
            d = dataclasses.asdict(lg)
            d["pid"] = cohort[lg.row].pid
            del d["row"]
            all_logs.append(d)
        done = cs + len(cohort)
        print(f"[task {task_name}] {done}/{len(problems)} "
              f"({dt:.0f}s cohort, {n_tok} tokens, {n_tok/max(dt,1e-9):.1f} tok/s, "
              f"elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)

    write_parquet(out_path, rows)
    if all_logs:
        write_parquet(out_path.with_name(out_path.stem + "_steplogs.parquet"), all_logs)
    if codec is not None:
        print(f"[codec stats after {task_name}] {codec.report_stats()}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True, choices=["clean", "identity", "nla"])
    p.add_argument("--tasks", default="all", help="comma list or 'all'")
    p.add_argument("--seed", type=int, default=0,
                   help="codec sampling seed (z_i are temp-1 stochastic); "
                        "clean/identity are deterministic so seed only labels files")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"])
    tasks = list(ALL_TASKS) if args.tasks == "all" else args.tasks.split(",")
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    m = BottleneckModel(cfg.get("m_ckpt", "Qwen/Qwen3-8B"),
                        layer_index=cfg.get("layer_index", 24))
    codec = None
    if args.condition == "nla":
        from experiments.bottleneck.codec import NLACodec
        codec = NLACodec(
            av_merged_dir=cfg["av_merged_dir"], ar_dir=cfg["ar_dir"],
            vllm_gpu_mem=cfg.get("vllm_gpu_mem", 0.35),
            vllm_max_len=cfg.get("vllm_max_len", 1024),
            av_max_tokens=cfg.get("av_max_tokens", 150),
            av_temperature=cfg.get("av_temperature", 1.0),
            seed=args.seed,
        )
        assert codec.cfg.extraction_layer_index == cfg.get("layer_index", 24), (
            "config layer_index disagrees with the checkpoint sidecar")

    for t in tasks:
        run_task(m, codec, t, args.condition, args.seed, cfg, out_dir, args.overwrite)
    if codec is not None:
        print(f"[codec stats FINAL] {codec.report_stats()}", flush=True)
    print("STAGE_B_DONE", flush=True)


if __name__ == "__main__":
    main()
