"""Head-contribution datagen orchestrator — run stages from one YAML config.

Each stage shells out to its CLI (printed, so any stage can be re-run by hand).
Every stage resumes or is idempotent, so re-running a stage list is safe.

    python -m nla.datagen.heads.run_pipeline --config configs/datagen/heads_qwen3_8b_finefineweb_100k.yaml
        # default: corpus,extract,stats,select,prepare — all local, nothing paid

    python -m nla.datagen.heads.run_pipeline --config ... --stages submit                  # dry run
    python -m nla.datagen.heads.run_pipeline --config ... --stages submit --confirm-spend  # PAID
    python -m nla.datagen.heads.run_pipeline --config ... --stages wait    # poll + collect until done
    python -m nla.datagen.heads.run_pipeline --config ... --stages status
    python -m nla.datagen.heads.run_pipeline --config ... --stages retry,submit --confirm-spend
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from nla.datagen.model_presets import resolve as resolve_model_preset

_DEFAULT_STAGES = ("corpus", "extract", "stats", "select", "prepare")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _flags(d: dict[str, Any]) -> list[str]:
    """{'max_tokens': 1024, 'cache_system': False} → ['--max-tokens', '1024', '--no-cache-system']."""
    out: list[str] = []
    for k, v in d.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out.append(flag if v else "--no-" + k.replace("_", "-"))
        elif isinstance(v, (list, tuple)):
            out += [flag, ",".join(str(x) for x in v)]
        elif v is not None:
            out += [flag, str(v)]
    return out


def _corpus(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    c = cfg["corpus"]
    if "build" not in c:
        print(f"corpus.build not set — using {c['path']} as-is")
        return
    if Path(c["path"]).exists():
        print(f"{c['path']} exists — skipping corpus build (delete it to rebuild)")
        return
    _run([sys.executable, "-m", "nla.datagen.heads.build_corpus", *_flags(c["build"]), "--output", c["path"]])


def _extract(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    c, e = cfg["corpus"], dict(cfg["extract"])
    num_gpus = int(e.pop("num_gpus", 1))
    kwargs = e.pop("extractor_kwargs", {})
    cmd = [
        sys.executable, "-m", "nla.datagen.heads.extract",
        "--base-model", cfg["base_model"],
        "--corpus", c["path"],
        "--corpus-start", str(c.get("start", 0)),
        "--corpus-length", str(c["length"]),
        "--text-column", c.get("text_column", "text"),
        *(["--id-column", c["id_column"]] if c.get("id_column") else []),
        *(["--meta-columns", ",".join(c["meta_columns"])] if c.get("meta_columns") else []),
        *_flags(e),
        "--extractor-kwargs", json.dumps(kwargs),
        "--output-dir", cfg["output_dir"],
    ]
    if num_gpus == 1:
        _run(cmd)
        return
    # One process per GPU over interleaved parts of the same output_dir.
    procs = []
    for i in range(num_gpus):
        shard_cmd = [*cmd, "--num-shards", str(num_gpus), "--shard-index", str(i)]
        print(f"\n$ CUDA_VISIBLE_DEVICES={i} {' '.join(shard_cmd)}\n", flush=True)
        procs.append(subprocess.Popen(shard_cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}))
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"extract shards exited with {codes} — re-run the stage to resume"


def _stats(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    _run([sys.executable, "-m", "nla.datagen.heads.stats", "--output-dir", cfg["output_dir"],
          *_flags(cfg.get("stats", {}))])


def _select(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    s, sel = cfg["split"], cfg["select"]
    _run([sys.executable, "-m", "nla.datagen.heads.split_select", "--output-dir", cfg["output_dir"],
          "--av-sft-frac", str(s["av_sft"]), "--ar-sft-frac", str(s["ar_sft"]),
          "--rl-frac", str(s["rl"]), "--eval-frac", str(s["eval"]), "--split-seed", str(s["seed"]),
          *_flags(sel)])


def _batch(cmd: str, flags: list[str]):
    def stage(cfg: dict[str, Any], args: argparse.Namespace) -> None:
        _run([sys.executable, "-m", "nla.datagen.heads.batch_api", cmd, "--output-dir", cfg["output_dir"],
              *flags(cfg, args)])
    return stage


def _prices(cfg: dict[str, Any]) -> list[str]:
    p = cfg["warmstart"].get("price_per_mtok", {})
    return _flags({"price_input": p.get("input"), "price_output": p.get("output")})


def _prepare_flags(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    w = {k: v for k, v in cfg["warmstart"].items() if k not in ("price_per_mtok", "max_words", "poll_seconds")}
    return [*_flags(w), *_prices(cfg)]


def _submit_flags(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [*(["--confirm-spend"] if args.confirm_spend else []),
            *(["--max-shards", str(args.max_shards)] if args.max_shards is not None else [])]


def _collect_flags(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [*_flags({"max_words": cfg["warmstart"].get("max_words")}), *_prices(cfg)]


def _wait_flags(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [*_collect_flags(cfg, args), *_flags({"poll_seconds": cfg["warmstart"].get("poll_seconds")})]


_STAGES = {
    "corpus": _corpus,
    "extract": _extract,
    "stats": _stats,
    "select": _select,
    "prepare": _batch("prepare", _prepare_flags),
    "submit": _batch("submit", _submit_flags),
    "status": _batch("status", lambda cfg, args: []),
    "collect": _batch("collect", _collect_flags),
    "wait": _batch("wait", _wait_flags),
    "retry": _batch("retry", lambda cfg, args: []),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default=",".join(_DEFAULT_STAGES),
                    help=f"comma-separated, from {sorted(_STAGES)}; default {','.join(_DEFAULT_STAGES)}")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted-key overrides, e.g. 'corpus.length=512 output_dir=/tmp/h1'")
    ap.add_argument("--confirm-spend", action="store_true", help="let `submit` actually submit (costs money)")
    ap.add_argument("--max-shards", type=int, default=None, help="`submit`: at most this many shards")
    args = ap.parse_args()

    cfg = resolve_model_preset(yaml.safe_load(Path(args.config).read_text()))
    cfg.pop("stage0", None)   # preset adds residual-pipeline extractor kwargs; this pipeline sets its own
    for ov in args.override:
        key, _, val = ov.partition("=")
        d = cfg
        *path, leaf = key.split(".")
        for k in path:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    stages = args.stages.split(",")
    for s in stages:
        assert s in _STAGES, f"unknown stage {s!r}, valid: {sorted(_STAGES)}"

    print(f"=== head datagen: {args.config} → {cfg['output_dir']} ===")
    print(f"    base_model={cfg['base_model']} docs={cfg['corpus']['length']} "
          f"positions/doc={cfg['extract']['positions_per_doc']} "
          f"layers/position={cfg['extract']['layers_per_position']} layers={cfg['extract']['layers']}")
    print(f"    stages={stages}")
    for s in stages:
        print(f"\n{'=' * 20} STAGE {s} {'=' * 20}")
        _STAGES[s](cfg, args)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
