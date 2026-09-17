"""Launch the pred-NLA experiment on RunPod, one stage (or the whole chain) per pod.

    python scripts/runpod_pred_nla.py plan                  # cost/time estimate, no spend
    python scripts/runpod_pred_nla.py launch --stages prep,gate --dry-run
    python scripts/runpod_pred_nla.py launch --stages prep,gate,rl,eval
    python scripts/runpod_pred_nla.py status
    python scripts/runpod_pred_nla.py terminate <pod_id>

Design notes
    * The pod runs the COMMITTED script scripts/pod_pred_nla.sh, parameterised
      entirely by environment variables. RunPod's docker_args is passed as
      `bash -lc '<cmd>'`, and any single quote in a generated script silently
      truncates it - the previous launcher generated ~12 of them and would have
      run nothing. The bootstrap here is a fixed one-liner with no quotes at all,
      asserted as such.
    * No network volume. HuggingFace is the store: every stage pulls what it
      needs from the dataset repo and pushes what it made, so a pod can run in
      whatever datacenter has stock, and a later stage can run on another pod.
    * The gate is a real gate. In the default chain a non-zero exit from
      nla.pred.gate stops the pod before RL spends anything; --force-rl
      overrides that deliberately rather than by accident.
    * The pod stays alive after FINISHED unless --no-keep, because the cheapest
      way to lose a run is to terminate before the upload finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The only runpod/pytorch 2.8.0 tag that exists (the "-cuda12.8.1-devel" spelling
# without "-cudnn" is a 404 and would stall the pod on image pull while billing).
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
REPO = "https://github.com/syvb/EasyNLA.git"
FALLBACK_GPUS = [
    "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL",
    "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
]
STAGES = ("prep", "gate", "rl", "rl_recon", "eval")
# Wall-clock per stage at the default pilot sizes. The RL figures come from a
# per-step cost model (rollout decode + reward forwards + the GRPO update's
# forward/backward/reference passes): ~55 s/step for the reader reward and
# ~45 s/step for reconstruction on an H100-class card.
STAGE_HOURS = {"prep": 0.7, "gate": 0.8, "rl": 4.6, "rl_recon": 3.7, "eval": 1.5}
STARTUP_HOURS = 0.35     # image pull + pip + four model downloads, per pod

# The bootstrap must contain NO single quotes (see the module docstring).
BOOTSTRAP = (
    "/start.sh >/dev/null 2>&1 & mkdir -p /workspace && cd /workspace && "
    "rm -rf EasyNLA && git clone -q -b $PRED_BRANCH $PRED_REPO && cd EasyNLA && "
    "pip install -q -e . 2>&1 | tail -2 && "
    "exec bash scripts/pod_pred_nla.sh 2>&1 | tee -a /workspace/boot.log"
)
assert "'" not in BOOTSTRAP


def _key(name):
    return open(os.path.expanduser(f"~/{name}")).read().strip()


def _check_stages(stages):
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"--stages: unknown {bad}; choose from {list(STAGES)}")


def pod_env(a) -> dict:
    """Everything scripts/pod_pred_nla.sh reads, from the CLI flags."""
    _check_stages(a.stages.split(","))
    return {
        "PRED_BRANCH": a.branch, "PRED_REPO": REPO,
        "STAGES": a.stages, "HF_REPO": a.hf_repo,
        "AV_CKPT": a.av_ckpt, "AR_CKPT": a.ar_ckpt,
        "RECON_ADAPTER": a.recon_adapter or "",
        "BEHAVIORAL_ADAPTER": a.behavioral_adapter or "",
        "TARGET_CKPT": a.target_ckpt,
        "TRAIN_READER": a.train_reader, "HELDOUT_READER": a.heldout_reader,
        "SOURCE_REPO": a.source_repo, "SOURCE_FILE": a.source_file,
        "CORPUS_FILTER": a.corpus_filter,
        "N_RL": str(a.n_rl), "N_VAL": str(a.n_val), "N_EVAL": str(a.n_eval),
        "N_BRANCHES": str(a.n_branches), "N_TOKENS": str(a.n_tokens),
        "MAX_PER_DOC": str(a.max_per_doc), "BATCH_PREFIXES": str(a.batch_prefixes),
        "RL_STEPS": str(a.rl_steps), "GATE_POSITIONS": str(a.gate_positions),
        "EVAL_POSITIONS": str(a.eval_positions), "MAX_NEW_TOKENS": str(a.max_new_tokens),
        "SEED": str(a.seed), "MAX_HOURS": str(a.max_hours),
        "WANDB_PROJECT": a.wandb_project,
        "WANDB_GROUP": a.wandb_group or f"pred-nla-s{a.seed}",
        "FORCE_RL": "1" if a.force_rl else "0",
        "KEEP_POD": "1" if a.keep else "0",
        "N_EXAMPLES": str(a.n_examples),
        "HF_HOME": "/workspace/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
    }


def estimate_hours(stages) -> float:
    return sum(STAGE_HOURS[s] for s in stages) + STARTUP_HOURS


def launch(a):
    env = pod_env(a)
    stages = a.stages.split(",")
    if a.dry_run:
        print("docker_args:\n  bash -lc '" + BOOTSTRAP + "'\n\nenv:")
        for k, v in env.items():
            print(f"  {k}={v}")
        print(f"\n[dry-run] stages={stages} gpu={a.gpu} "
              f"est {estimate_hours(stages):.1f} h incl. startup")
        return
    import runpod

    runpod.api_key = _key(".runpod_key")
    env.update({"WANDB_API_KEY": _key(".wandb_key"), "HF_TOKEN": _key(".hf_token"),
                "RUNPOD_API_KEY": runpod.api_key})
    attempts = [(a.gpu, a.cloud)] + [(g, c) for g in FALLBACK_GPUS
                                     for c in ("SECURE", "COMMUNITY")
                                     if (g, c) != (a.gpu, a.cloud)]
    pod = None
    for gpu, cloud in attempts:
        try:
            pod = runpod.create_pod(
                name=f"pred-nla-{a.stages.replace(',', '-')}", image_name=IMAGE,
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
    print(json.dumps({k: pod.get(k) for k in
                      ("id", "name", "desiredStatus", "costPerHr", "machineId")}, indent=1))
    print("POD_ID", pod["id"])


def plan(a):
    """Price the run from live RunPod rates. Spends nothing."""
    import runpod

    runpod.api_key = _key(".runpod_key")
    stages = a.stages.split(",")
    _check_stages(stages)
    rates = {}
    for g in runpod.get_gpus():
        try:
            d = runpod.get_gpu(g["id"])
            rates[g["id"]] = (d.get("lowestPrice") or {}).get("uninterruptablePrice")
        except Exception as e:                                 # noqa: BLE001
            print(f"  (no price for {g['id']}: {type(e).__name__})")
    hours = estimate_hours(stages)
    for st in stages:
        print(f"  {st:<10} {STAGE_HOURS[st]:>4.1f} h")
    print(f"  startup    {STARTUP_HOURS:>4.2f} h\nGPU-hours total: {hours:.1f}")
    print(f"{'gpu':<28} {'$/h':>7} {'run $':>8}")
    for g in [a.gpu] + [x for x in FALLBACK_GPUS if x != a.gpu]:
        r = rates.get(g)
        if r:
            print(f"{g:<28} {r:>7.2f} {hours * r:>8.2f}")
    try:
        import requests
        r = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {runpod.api_key}"},
            json={"query": "{ myself { clientBalance } }"}, timeout=20)
        bal = r.json()["data"]["myself"]["clientBalance"]
        print(f"\nRunPod balance: ${bal:.2f}")
        sel = rates.get(a.gpu)
        if sel and bal < hours * sel:
            print(f"WARNING: ${bal:.2f} will not cover these stages on {a.gpu} "
                  f"(${hours * sel:.2f}). Top up, pick a cheaper GPU, or split "
                  f"the chain across pods.")
    except Exception as e:                                     # noqa: BLE001
        print(f"\n(balance unavailable: {type(e).__name__})")


def status(a):
    import runpod

    runpod.api_key = _key(".runpod_key")
    pods = runpod.get_pods()
    if not pods:
        print("no pods")
    for p in pods:
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        print(f"{p['id']} {p['name']} {p['desiredStatus']} "
              f"{(p.get('machine') or {}).get('gpuDisplayName')} "
              f"${p.get('costPerHr')}/h up={up / 3600:.2f}h "
              f"spent=${(p.get('costPerHr') or 0) * up / 3600:.2f}")


def terminate(a):
    import runpod

    runpod.api_key = _key(".runpod_key")
    runpod.terminate_pod(a.pod_id)
    print("terminated", a.pod_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--stages", default="prep,gate,rl,eval",
                       help=f"comma list of {','.join(STAGES)}")
        p.add_argument("--gpu", default="NVIDIA H200")
        p.add_argument("--cloud", default="SECURE")
        p.add_argument("--branch", default="sv/pred-nla")
        p.add_argument("--hf-repo", default="syvb/pred-nla-qwen3-8b")
        p.add_argument("--wandb-project", default="pred-nla")
        p.add_argument("--wandb-group", default=None,
                       help="one group for every stage; default pred-nla-s<seed>")
        p.add_argument("--av-ckpt", default="syvb/nanonla-qwen3-8b-L24-av")
        p.add_argument("--ar-ckpt", default="syvb/nanonla-qwen3-8b-L24-ar")
        p.add_argument("--recon-adapter",
                       default="syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0",
                       help="published reconstruction-RL comparator (adapter#subfolder)")
        p.add_argument("--behavioral-adapter", default=None,
                       help="eval-only: a LOCAL path or HF MODEL repo of an already "
                            "trained behavioral adapter. Omit to pull the one this "
                            "experiment pushed to --hf-repo.")
        p.add_argument("--target-ckpt", default="Qwen/Qwen3-8B")
        p.add_argument("--train-reader", default="Qwen/Qwen3-4B-Base")
        p.add_argument("--heldout-reader", default="google/gemma-3-4b-pt")
        p.add_argument("--source-repo", default="asher577/nla-rl-data-free8")
        p.add_argument("--source-file", default="rl_shuf.parquet")
        p.add_argument("--corpus-filter", default="finefineweb",
                       help="the source pool is half chat transcripts; keep the "
                            "pretraining-like half (its doc_ids contain this)")
        p.add_argument("--n-rl", type=int, default=12000)
        p.add_argument("--n-val", type=int, default=1000)
        p.add_argument("--n-eval", type=int, default=2000)
        p.add_argument("--n-branches", type=int, default=4)
        p.add_argument("--n-tokens", type=int, default=24)
        p.add_argument("--max-per-doc", type=int, default=2)
        p.add_argument("--batch-prefixes", type=int, default=24)
        p.add_argument("--rl-steps", type=int, default=300)
        p.add_argument("--gate-positions", type=int, default=500)
        p.add_argument("--eval-positions", type=int, default=2000)
        p.add_argument("--max-new-tokens", type=int, default=192)
        p.add_argument("--n-examples", type=int, default=100,
                       help="blinded explanations per checkpoint in report.md")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-hours", type=int, default=8,
                       help="per-stage timeout; RL at 300 steps is a ~4.6 h stage")
        p.add_argument("--disk-gb", type=int, default=250)
        p.add_argument("--keep", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--force-rl", action="store_true",
                       help="run RL even if the gate fails (deliberate override)")
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    t = sub.add_parser("terminate")
    t.add_argument("pod_id")
    a = ap.parse_args()
    {"launch": launch, "plan": plan, "status": status, "terminate": terminate}[a.cmd](a)


if __name__ == "__main__":
    main()
