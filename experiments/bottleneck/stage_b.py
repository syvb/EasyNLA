"""Stage B: generative benchmarks under the continuous NLA bottleneck.

Per (task, condition, seed): batched greedy generation with the layer-24 tap
(clean / identity / nla / nla_prompt), outputs + per-token codec logs written
as parquet. Scoring is OFFLINE (score_stage_b.py, CPU-ok) so scorer debugging
never burns GPU hours.

    python -m experiments.bottleneck.stage_b --config experiments/bottleneck/config.yaml \
        --condition nla --tasks gsm8k,triviaqa --seed 0

Outputs land in {out_dir}/stage_b/{task}/{condition}_seed{seed}.parquet
(+ _steplogs.parquet for codec conditions). Resume is per-cohort: each cohort
writes a shard under .../{condition}_seed{seed}_shards/ and completed shards
are skipped on relaunch; the final parquet is concatenated from shards.
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

from experiments.bottleneck.patched import CONDITIONS, BottleneckModel, StepLog
from experiments.bottleneck.tasks import ALL_TASKS, DEFAULT_SIZES, load_task

CODEC_CONDITIONS = ("nla", "nla_prompt")


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
    shard_dir = out_path.with_name(f"{out_path.stem}_shards")
    if overwrite:
        for old in shard_dir.glob("*.parquet"):
            old.unlink()
    n = cfg.get("task_sizes", {}).get(task_name, DEFAULT_SIZES[task_name])
    kw = {}
    if task_name == "mgsm" and cfg.get("mgsm_langs"):
        kw["mgsm_langs"] = cfg["mgsm_langs"]
    problems = load_task(task_name, n, seed=cfg.get("data_seed", 0), **kw)
    # optional per-task generation-budget cap (a cohort runs until its slowest
    # row finishes, so max_new — not n below cohort size — is the cost knob)
    cap = cfg.get("max_new_caps", {}).get(task_name)
    if cap:
        for p_ in problems:
            p_.max_new_tokens = min(p_.max_new_tokens, cap)
    print(f"[task {task_name}] {len(problems)} problems, condition={condition}, "
          f"seed={seed}" + (f", max_new capped {cap}" if cap else ""), flush=True)

    batch_size = cfg.get("gen_batch_size", 192)
    t_start = time.time()
    for cs in range(0, len(problems), batch_size):
        shard = shard_dir / f"cohort_{cs:06d}.parquet"
        log_shard = shard_dir / f"cohort_{cs:06d}_steplogs.parquet"
        if shard.exists():
            print(f"[task {task_name}] cohort {cs} shard exists — skipping")
            continue
        cohort = problems[cs:cs + batch_size]
        prompt_ids = [m.build_chat_ids(p.messages) for p in cohort]
        max_new = max(p.max_new_tokens for p in cohort)
        logs: list[StepLog] = []
        t0 = time.time()
        gen = m.generate(prompt_ids, max_new, codec=codec, condition=condition,
                         step_logs=logs if condition in CODEC_CONDITIONS else None)
        dt = time.time() - t0
        n_tok = sum(len(g) for g in gen)
        rows = []
        for j, p in enumerate(cohort):
            out_ids = gen[j]
            row = {
                "task": p.task, "pid": p.pid, "condition": condition, "seed": seed,
                "prompt": p.messages[-1]["content"],
                "output": m.tokenizer.decode(out_ids, skip_special_tokens=True),
                # exact ids: text->token round-trips are not guaranteed exact,
                # and deferred controls (shuffled-z, noise) replay from ids
                "output_ids": json.dumps(out_ids),
                "n_prompt_tokens": len(prompt_ids[j]),
                "n_output_tokens": len(out_ids),
                "hit_cap": len(out_ids) >= p.max_new_tokens
                           and (not out_ids or out_ids[-1] not in m.eos_ids),
                "gold": json.dumps(p.gold),
            }
            if task_name == "fluency" and out_ids:
                # NLL of the output under CLEAN M. Report jointly with rep3 +
                # length in score_stage_b — PPL alone rewards degenerate loops.
                row["clean_nll"] = m.score_continuation_nll(prompt_ids[j], out_ids)
            rows.append(row)
        # steplogs first: the data shard is the resume sentinel, so its
        # existence must imply the logs are already on disk
        if logs:
            log_rows = []
            for lg in logs:
                d = dataclasses.asdict(lg)
                d["pid"] = cohort[lg.row].pid
                del d["row"]
                log_rows.append(d)
            write_parquet(log_shard, log_rows)
        write_parquet(shard, rows)
        done = min(cs + batch_size, len(problems))
        print(f"[task {task_name}] {done}/{len(problems)} "
              f"({dt:.0f}s cohort, {n_tok} tokens, {n_tok/max(dt,1e-9):.1f} tok/s, "
              f"elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)

    # concatenate shards into the final per-task parquets
    shards = sorted(shard_dir.glob("cohort_*.parquet"))
    out_rows = [r for s in shards if not s.stem.endswith("_steplogs")
                for r in pq.read_table(s).to_pylist()]
    write_parquet(out_path, out_rows)
    log_shards = sorted(shard_dir.glob("cohort_*_steplogs.parquet"))
    if log_shards:
        all_logs = [r for s in log_shards for r in pq.read_table(s).to_pylist()]
        write_parquet(out_path.with_name(out_path.stem + "_steplogs.parquet"), all_logs)
    if codec is not None:
        print(f"[codec stats after {task_name}] {codec.report_stats()}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True, choices=list(CONDITIONS))
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
    if args.condition in CODEC_CONDITIONS:
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
