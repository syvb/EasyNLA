"""Launch the KL-NLA experiment (docs/kl_nla.md) on RunPod.

    python scripts/runpod_kl_nla.py plan   --stages audit0            # price it, no spend
    python scripts/runpod_kl_nla.py launch --stages audit0 --dry-run
    python scripts/runpod_kl_nla.py launch --stages audit0 --no-keep
    python scripts/runpod_kl_nla.py status
    python scripts/runpod_kl_nla.py terminate <pod_id>

Same design as sv/pred-nla's launcher: the pod runs the COMMITTED
scripts/pod_kl_nla.sh, parameterised by environment variables, from a fixed
bootstrap with no single quotes (RunPod's `bash -lc '<cmd>'` truncates at one);
no network volume, HuggingFace is the store; --no-keep self-terminates after the
last push, and the pod script's watchdog removes the pod after --max-hours.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REPO = "https://github.com/syvb/EasyNLA.git"
FALLBACK_GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200",
                 "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]
STAGES = ("audit0", "ar_kl", "ar_mse", "audit1")
# Wall-clock guesses on an H100 (782 AR steps at eff. batch 64); replace with
# measured numbers after the first run.
STAGE_HOURS = {"audit0": 0.3, "ar_kl": 0.9, "ar_mse": 0.5, "audit1": 0.4}
STARTUP_HOURS = 0.3      # image pull + pip + ~30 GB of model/data downloads

BOOTSTRAP = (
    "/start.sh >/dev/null 2>&1 & mkdir -p /workspace && cd /workspace && "
    "rm -rf EasyNLA && git clone -q -b $KL_BRANCH $KL_REPO && cd EasyNLA && "
    "pip install -q -e . 2>&1 | tail -2 && "
    "exec bash scripts/pod_kl_nla.sh 2>&1 | tee -a /workspace/boot.log"
)
assert "'" not in BOOTSTRAP


def _key(name):
    return open(os.path.expanduser(f"~/{name}")).read().strip()


def _stages(s):
    st = s.split(",")
    bad = [x for x in st if x not in STAGES]
    if bad:
        sys.exit(f"--stages: unknown {bad}; choose from {list(STAGES)}")
    return st


def pod_env(a) -> dict:
    _stages(a.stages)
    return {
        "KL_BRANCH": a.branch, "KL_REPO": REPO, "STAGES": a.stages, "HF_REPO": a.hf_repo,
        "AR_CKPT": a.ar_ckpt, "TARGET_CKPT": a.target_ckpt,
        "AR_LR": str(a.ar_lr), "LORA_R": str(a.lora_r), "AR_BATCH": str(a.ar_batch),
        "AR_ACCUM": str(a.ar_accum), "AR_STEPS": str(a.ar_steps),
        "KL_MICRO_BATCH": str(a.kl_micro_batch), "SEED": str(a.seed),
        "MAX_HOURS": str(a.max_hours), "KEEP_POD": "1" if a.keep else "0",
        "WANDB_PROJECT": a.wandb_project, "WANDB_GROUP": a.wandb_group,
        "HF_HOME": "/workspace/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
    }


def hours(stages):
    return sum(STAGE_HOURS[s] for s in stages) + STARTUP_HOURS


def launch(a):
    env = pod_env(a)
    st = _stages(a.stages)
    if a.dry_run:
        print("docker_args:\n  bash -lc '" + BOOTSTRAP + "'\n\nenv:")
        for k, v in env.items():
            print(f"  {k}={v}")
        print(f"\n[dry-run] stages={st} gpu={a.gpu} est {hours(st):.1f} h incl. startup")
        return
    import runpod
    runpod.api_key = _key(".runpod_key")
    env.update({"WANDB_API_KEY": _key(".wandb_key"), "HF_TOKEN": _key(".hf_token"),
                "RUNPOD_API_KEY": runpod.api_key})
    attempts = [(a.gpu, a.cloud)] + [(g, c) for g in FALLBACK_GPUS for c in ("SECURE", "COMMUNITY")
                                     if (g, c) != (a.gpu, a.cloud)]
    pod = None
    for gpu, cloud in attempts:
        try:
            pod = runpod.create_pod(
                name=f"kl-nla-{a.stages.replace(',', '-')}", image_name=IMAGE,
                gpu_type_id=gpu, cloud_type=cloud, gpu_count=1,
                container_disk_in_gb=a.disk_gb, volume_in_gb=0,
                min_memory_in_gb=64, min_vcpu_count=8, ports="22/tcp",
                docker_args=f"bash -lc '{BOOTSTRAP}'", env=env,
            )
            print("launched on", gpu, cloud)
            break
        except runpod.error.QueryError as e:
            msg = str(e)
            if "no longer any instances" not in msg and "not available" not in msg.lower():
                sys.exit(f"launch error: {msg[:300]}")
            print("unavailable:", gpu, cloud)
    if pod is None:
        sys.exit("no GPU available in any fallback")
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "machineId")},
                     indent=1))
    print("POD_ID", pod["id"])


def plan(a):
    import runpod
    runpod.api_key = _key(".runpod_key")
    st = _stages(a.stages)
    h = hours(st)
    for s in st:
        print(f"  {s:<8} {STAGE_HOURS[s]:>4.1f} h")
    print(f"  startup  {STARTUP_HOURS:>4.1f} h\nGPU-hours: {h:.1f}")
    for g in [a.gpu] + [x for x in FALLBACK_GPUS if x != a.gpu]:
        try:
            r = (runpod.get_gpu(g).get("lowestPrice") or {}).get("uninterruptablePrice")
        except Exception as e:                                  # noqa: BLE001
            r = None
            print(f"  (no price for {g}: {type(e).__name__})")
        if r:
            print(f"{g:<28} ${r:.2f}/h listed  -> ${h * r:.2f}")


_UPTIME_Q = ("query { myself { pods { id name desiredStatus costPerHr "
             "runtime { uptimeInSeconds } machine { gpuDisplayName } } } }")


def status(a):
    import requests
    r = requests.post(f"https://api.runpod.io/graphql?api_key={_key('.runpod_key')}",
                      json={"query": _UPTIME_Q}, timeout=30)
    r.raise_for_status()
    pods = r.json()["data"]["myself"]["pods"]
    if not pods:
        print("no pods")
    for p in pods:
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        cost = p.get("costPerHr") or 0
        print(f"{p['id']} {p['name']} {p['desiredStatus']} {(p.get('machine') or {}).get('gpuDisplayName')} "
              f"${cost}/h up={up / 3600:.2f}h spent=${cost * up / 3600:.2f}")


def terminate(a):
    import runpod
    runpod.api_key = _key(".runpod_key")
    runpod.terminate_pod(a.pod_id)
    print("terminated", a.pod_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--stages", default="audit0", help=f"comma list of {','.join(STAGES)}")
        p.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
        p.add_argument("--cloud", default="SECURE")
        p.add_argument("--branch", default="sv/kl-nla")
        p.add_argument("--hf-repo", default="syvb/kl-nla-qwen3-8b")
        p.add_argument("--ar-ckpt", default="syvb/nanonla-qwen3-8b-L24-ar")
        p.add_argument("--target-ckpt", default="Qwen/Qwen3-8B")
        p.add_argument("--ar-lr", type=float, default=5e-5)
        p.add_argument("--lora-r", type=int, default=128)
        p.add_argument("--ar-batch", type=int, default=16)
        p.add_argument("--ar-accum", type=int, default=4)
        p.add_argument("--ar-steps", type=int, default=782, help="one epoch of ar_sft_full at eff. batch 64")
        p.add_argument("--kl-micro-batch", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-hours", type=int, default=6, help="pod watchdog and per-stage timeout")
        p.add_argument("--wandb-project", default="kl-nla")
        p.add_argument("--wandb-group", default="kl-nla-phase1")
        p.add_argument("--disk-gb", type=int, default=150)
        p.add_argument("--keep", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    t = sub.add_parser("terminate")
    t.add_argument("pod_id")
    a = ap.parse_args()
    {"launch": launch, "plan": plan, "status": status, "terminate": terminate}[a.cmd](a)


if __name__ == "__main__":
    main()
