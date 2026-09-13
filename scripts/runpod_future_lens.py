"""Launch future-lens stages on RunPod (one pod per stage, shared network volume).

NOT run automatically — every stage costs money. Adapted from the metamodelling
repo's launcher: the pod boots the official PyTorch image, clones this repo's
branch, `pip install -e .`, runs ONE stage command with logs to wandb and to the
network volume, then terminates itself (unless --keep).

  python scripts/runpod_future_lens.py volume --size 150 --dc EU-RO-1      # once
  python scripts/runpod_future_lens.py launch collect  --volume <id>
  python scripts/runpod_future_lens.py launch alpha_sweep --volume <id>
  python scripts/runpod_future_lens.py launch sft      --volume <id> --alpha-mult 1 --injection replace_embed
  python scripts/runpod_future_lens.py launch rl       --volume <id> --av-ckpt sft_a1/iter_0018750 --seed 0
  python scripts/runpod_future_lens.py launch eval     --volume <id> --adapters sft_a1/iter_0018750,rl_s0/iter_004000
  python scripts/runpod_future_lens.py launch baselines --volume <id>
  python scripts/runpod_future_lens.py status | terminate <pod_id>

Quoting: the whole bootstrap runs inside `bash -lc '...'`, so stage commands must
never contain single quotes; JSON tags use escaped double quotes (see _tag_arg).

Credentials: ~/.runpod_key, ~/.wandb_key, ~/.hf_token. Layout on the volume
(mounted at /workspace): fl/data (parquets), fl/ckpts/<run>, fl/evals/*.jsonl, fl/logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx

REPO = "https://github.com/syvb/EasyNLA.git"
BRANCH = os.environ.get("FL_BRANCH", "sv/future-rl")
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU_PREF = [("NVIDIA H100 80GB HBM3", "SECURE"), ("NVIDIA H100 PCIe", "SECURE"),
            ("NVIDIA A100 80GB PCIe", "SECURE"), ("NVIDIA A100-SXM4-80GB", "SECURE"),
            ("NVIDIA H100 PCIe", "COMMUNITY"), ("NVIDIA A100 80GB PCIe", "COMMUNITY")]
BASE = "Qwen/Qwen3-8B"
WORK = "/workspace/fl"
WANDB_PROJECT = "rl-future-lens"
LAYERS = "4,8,12,16,20,24"


def _read(path):
    return open(os.path.expanduser(path)).read().strip()


def _runpod():
    import runpod
    runpod.api_key = _read("~/.runpod_key")
    return runpod


# ----------------------------------------------------------------------------
# stage commands (run inside the pod, cwd = repo)
# ----------------------------------------------------------------------------

def stage_cmd(stage: str, a) -> str:
    D, C, E = f"{WORK}/data", f"{WORK}/ckpts", f"{WORK}/evals"
    if stage == "collect":
        return (f"python -m nla.future_lens.collect --base-ckpt {BASE} --corpus HuggingFaceFW/fineweb "
                f"--corpus-config sample-10BT --n-train-docs {a.n_train_docs} --n-eval-docs {a.n_eval_docs} "
                f"--layers {LAYERS} --positions-per-doc 40 --eval-positions-per-doc 20 --max-len 1024 "
                f"--batch-size 8 --out-dir {D} && "
                f"(huggingface-cli upload {a.hf_repo} {D} data --repo-type dataset --private || true)")
    if stage == "alpha_sweep":
        runs = []
        for inj, mults in (("replace_embed", (0.5, 1.0, 2.0, 4.0)), ("karvonen", (1.0,))):
            for m in mults:
                for shuf in ("", "--shuffle-activations"):
                    tag = f"{inj}_a{m}{'_shuf' if shuf else ''}"
                    runs.append(
                        f"python -m nla.train_sft --config configs/future_lens/sft_alpha_sweep.yaml "
                        f"--base-ckpt {BASE} --parquet {D}/train.parquet --heldout-parquet {D}/eval.parquet "
                        f"--save-dir {C}/sweep_{tag} --injection {inj} --alpha-mult {m} {shuf} "
                        f"--wandb-name sweep_{tag} --seed 0")
        return " && ".join(runs)
    if stage == "sft":
        return (f"python -m nla.train_sft --config configs/future_lens/sft.yaml --base-ckpt {BASE} "
                f"--parquet {D}/train.parquet --heldout-parquet {D}/eval.parquet "
                f"--save-dir {C}/{a.run_name} --injection {a.injection} --alpha-mult {a.alpha_mult} "
                f"{'--affine' if a.affine else ''} --wandb-name {a.run_name} --seed {a.seed} {a.extra}")
    if stage == "rl":
        return (f"python -m nla.future_lens.train_rl --config configs/future_lens/rl.yaml --base-ckpt {BASE} "
                f"--av-ckpt {C}/{a.av_ckpt} --parquet {D}/train.parquet --eval-parquet {D}/eval.parquet "
                f"--save-dir {C}/{a.run_name} --reward {a.reward} --seed {a.seed} --wandb-name {a.run_name} {a.extra}")
    if stage == "eval":
        cmds = []
        for ad in a.adapters.split(","):
            name = ad.replace("/", "_")
            # `group` = run family (seed stripped) so plots.py pools seeds; `seed` from the run name.
            run = ad.split("/")[0]
            m = re.search(r"_s(\d+)$", run)
            seed = int(m.group(1)) if m else a.seed
            group = run[: m.start()] if m else run
            cmds.append(f"python -m nla.future_lens.eval --base-ckpt {BASE} --adapter {C}/{ad} "
                        f"--parquet {D}/eval.parquet --out {E}/{name}.jsonl "
                        f"--conditions real,shuffled,none,wrong_layer --wrong-layer 4 --batch-size 64 "
                        f"--layers 8,12,16,20,24 --dump-readouts {E}/readouts_{name}.jsonl "
                        f"--seed {seed} --tag {_tag_arg({'checkpoint': name, 'group': group, 'seed': seed})} {a.extra}")
        return " && ".join(cmds)
    if stage == "baselines":
        return (f"python -m nla.future_lens.baselines ngram --parquet {D}/eval.parquet --out {E}/baselines.jsonl "
                f"--base-ckpt {BASE} --hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000 && "
                f"python -m nla.future_lens.baselines probe --train-parquet {D}/train.parquet --parquet {D}/eval.parquet "
                f"--base-ckpt {BASE} --leakage --epochs 3 --batch 1024 --out {E}/baselines.jsonl && "
                f"for f in {E}/readouts_*.jsonl; do python -m nla.future_lens.baselines leakage --parquet {D}/eval.parquet "
                f"--readouts $f --out {E}/leakage.jsonl --base-ckpt {BASE} --hf-corpus HuggingFaceFW/fineweb "
                f"--hf-config sample-10BT --hf-docs 20000; done")
    raise SystemExit(f"unknown stage {stage}")


def _tag_arg(d: dict) -> str:
    """JSON for --tag that survives inside the single-quoted `bash -lc '...'` wrapper."""
    return '"' + json.dumps(d, separators=(",", ":")).replace('"', '\\"') + '"'


def bootstrap(stage_command: str, stage: str, keep: bool) -> str:
    assert "'" not in stage_command, f"stage command contains a single quote (breaks bash -lc quoting): {stage_command}"
    log = f"{WORK}/logs/{stage}_$(date +%Y%m%d_%H%M%S).log"
    finish = "" if keep else "python scripts/pod_terminate.py"
    return (
        "/start.sh >/dev/null 2>&1 & "
        f"mkdir -p {WORK}/logs {WORK}/data {WORK}/ckpts {WORK}/evals && exec > >(tee -a {log}) 2>&1; set -x; "
        f"cd /workspace && rm -rf EasyNLA && git clone -q -b {BRANCH} {REPO} && cd EasyNLA && "
        "pip install -q -e . bitsandbytes runpod 2>&1 | tail -2 && nvidia-smi --query-gpu=name,memory.total --format=csv && "
        f"export HF_HOME=/workspace/hf && ({stage_command}) ; echo STAGE_EXIT=$? ; "
        f"wandb artifact put --type evals --name {WANDB_PROJECT}/fl-evals-{stage} {WORK}/evals >/dev/null 2>&1 || true; "
        f"{finish}; echo FINISHED; sleep infinity"
    )


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_volume(a):
    r = httpx.post("https://rest.runpod.io/v1/networkvolumes",
                   headers={"Authorization": f"Bearer {_read('~/.runpod_key')}"},
                   json={"name": a.name, "size": a.size, "dataCenterId": a.dc}, timeout=60)
    print(r.status_code, r.text)


def cmd_launch(a):
    cmd = bootstrap(stage_cmd(a.stage, a), a.stage, a.keep)
    if a.dry_run:
        print(cmd); return
    runpod = _runpod()
    env = {"WANDB_API_KEY": _read("~/.wandb_key"), "WANDB_PROJECT": WANDB_PROJECT, "HF_TOKEN": _read("~/.hf_token"),
           "RUNPOD_API_KEY": runpod.api_key, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    attempts = [(a.gpu, a.cloud)] if a.gpu else GPU_PREF
    pod = None
    for gpu, cloud in attempts:
        try:
            pod = runpod.create_pod(
                name=f"fl-{a.stage}-{a.run_name or ''}".rstrip("-"), image_name=IMAGE, gpu_type_id=gpu,
                cloud_type=cloud, gpu_count=1, container_disk_in_gb=80, volume_in_gb=0,
                network_volume_id=a.volume, volume_mount_path="/workspace",
                min_memory_in_gb=48, min_vcpu_count=8, ports="22/tcp",
                docker_args=f"bash -lc '{cmd}'", env=env,
            )
            print("launched on", gpu, cloud); break
        except Exception as e:   # runpod.error.QueryError for availability
            if "available" not in str(e).lower() and "instances" not in str(e).lower():
                sys.exit(f"launch error: {str(e)[:400]}")
            print("unavailable:", gpu, cloud)
    if pod is None:
        sys.exit("no GPU available")
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "machineId")}, indent=1))
    print("POD_ID", pod["id"])


def cmd_status(a):
    for p in _runpod().get_pods():
        print(p["id"], p["name"], p["desiredStatus"], p.get("machine", {}).get("gpuDisplayName"),
              f"${p.get('costPerHr')}/h", (p.get("runtime") or {}).get("uptimeInSeconds"))


def cmd_terminate(a):
    _runpod().terminate_pod(a.pod_id); print("terminated", a.pod_id)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volume"); v.add_argument("--name", default="fl-qwen3-8b"); v.add_argument("--size", type=int, default=150)
    v.add_argument("--dc", default="EU-RO-1")
    l = sub.add_parser("launch"); l.add_argument("stage", choices=["collect", "alpha_sweep", "sft", "rl", "eval", "baselines"])
    l.add_argument("--volume", required=True, help="network volume id")
    l.add_argument("--gpu", default=None); l.add_argument("--cloud", default="SECURE")
    l.add_argument("--keep", action="store_true"); l.add_argument("--dry-run", action="store_true")
    l.add_argument("--run-name", default=None); l.add_argument("--seed", type=int, default=0)
    l.add_argument("--extra", default="", help="extra CLI flags appended to the stage command")
    l.add_argument("--n-train-docs", type=int, default=5500); l.add_argument("--n-eval-docs", type=int, default=300)
    l.add_argument("--hf-repo", default="syvb/rl-future-lens-qwen3-8b")
    l.add_argument("--injection", default="replace_embed"); l.add_argument("--alpha-mult", type=float, default=1.0)
    l.add_argument("--affine", action="store_true")
    l.add_argument("--av-ckpt", default=None, help="rl: SFT iter dir relative to ckpts/")
    l.add_argument("--reward", default="exact_match")
    l.add_argument("--adapters", default="", help="eval: comma list of ckpt dirs relative to ckpts/")
    s = sub.add_parser("status")
    t = sub.add_parser("terminate"); t.add_argument("pod_id")
    a = p.parse_args(argv)
    if a.cmd == "launch" and a.run_name is None:
        a.run_name = {"sft": f"sft_{a.injection}_a{a.alpha_mult}", "rl": f"rl_{a.reward}_s{a.seed}"}.get(a.stage, a.stage)
    {"volume": cmd_volume, "launch": cmd_launch, "status": cmd_status, "terminate": cmd_terminate}[a.cmd](a)


if __name__ == "__main__":
    main()
