"""Launch the pred-NLA experiment on RunPod, one stage (or the whole chain) per pod.

    python scripts/runpod_pred_nla.py plan                  # cost/time estimate, no spend
    python scripts/runpod_pred_nla.py launch --stages prep,gate
    python scripts/runpod_pred_nla.py launch --stages rl,eval --dry-run
    python scripts/runpod_pred_nla.py status
    python scripts/runpod_pred_nla.py terminate <pod_id>

Design notes
    * No network volume. HuggingFace is the store: every stage pulls what it
      needs from the dataset repo and pushes what it made, so a pod can run in
      whatever datacenter has stock.
    * The gate is a real gate. In the default chain a non-zero exit from
      nla.pred.gate stops the pod before RL spends anything; --force-rl overrides
      that deliberately rather than by accident.
    * The pod stays alive after FINISHED unless --no-keep, because the cheapest
      way to lose a run is to terminate before the upload finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04"
REPO = "https://github.com/syvb/EasyNLA.git"
FALLBACK_GPUS = [
    "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL",
    "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
]
# Wall-clock per stage at the default pilot sizes, used by `plan`. The RL figures
# come from a per-step cost model (rollout decode + reward forwards + the GRPO
# update's forward/backward/reference passes) at ~55 s/step for the reader reward
# and ~45 s/step for reconstruction on an H100-class card, not from a guess.
STAGE_HOURS = {"prep": 0.6, "gate": 0.5, "rl": 4.6, "rl_recon": 3.7, "eval": 1.0}


def _key(name):
    return open(os.path.expanduser(f"~/{name}")).read().strip()


def build_script(a) -> str:
    """The pod's whole job as one bash string."""
    data = "/workspace/data"
    ck = "/workspace/ckpts"
    ev = "/workspace/evals"
    pos = f"{data}/positions.parquet"
    hf = a.hf_repo
    sync = "python scripts/pred_hf_sync.py"
    readers = f"{a.train_reader} {a.heldout_reader}"
    setup = [
        "/start.sh >/dev/null 2>&1 & mkdir -p /workspace && "
        "exec > >(tee -a /workspace/boot.log) 2>&1; set -o pipefail; set -x",
        f"cd /workspace && rm -rf EasyNLA && git clone -q -b {a.branch} {REPO} && cd EasyNLA",
        "pip install -q -e . 2>&1 | tail -2",
        "pip install -q bitsandbytes 2>&1 | tail -1",
        "python -c \"import torch,transformers;print(torch.__version__,transformers.__version__)\"",
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        f"mkdir -p {data} {ck} {ev}",
    ]
    parts = []
    stages = a.stages.split(",")

    if "prep" in stages:
        parts.append(
            f"huggingface-cli download {a.source_repo} --repo-type dataset "
            f"--include '{a.source_file}*' --local-dir {data}/source")
        parts.append(
            f"timeout {a.max_hours}h python -m nla.pred.continuations "
            f"--source-parquet {data}/source/{a.source_file} "
            f"--sidecar {data}/source/{a.source_file} "
            f"--target-ckpt {a.target_ckpt} --out {pos} "
            f"--n-rl {a.n_rl} --n-val {a.n_val} --n-eval {a.n_eval} "
            f"--n-branches {a.n_branches} --n-tokens {a.n_tokens} "
            f"--corpus-filter {a.corpus_filter} --max-per-doc {a.max_per_doc} "
            f"--batch-prefixes {a.batch_prefixes} "
            f"--wandb-project {a.wandb_project} --wandb-name prep")
        parts.append(f"{sync} push {hf} {data} data")
    else:
        parts.append(f"{sync} pull {hf} data {data}_dl && "
                     f"cp {data}_dl/data/positions.parquet* {data}/")

    if "gate" in stages:
        gate = (
            f"python -m nla.pred.gate --positions {pos} "
            f"--base-ckpt {a.av_ckpt} --checkpoint sft= "
            + (f"--checkpoint recon_rl={a.recon_adapter} " if a.recon_adapter else "")
            + f"--readers {readers} --n-positions {a.gate_positions} "
            f"--branches {a.n_branches} --max-new-tokens {a.max_new_tokens} "
            f"--out-dir {ev}/gate --wandb-project {a.wandb_project} --wandb-name gate")
        # A failing gate must stop the chain BEFORE the expensive stage.
        parts.append(f"{gate}; GATE=$?; {sync} push {hf} {ev}/gate evals/gate; "
                     + ("echo gate exit $GATE (forced on)"
                        if a.force_rl else
                        "if [ $GATE -ne 0 ]; then echo 'GATE FAILED - stopping before RL'; "
                        "echo FINISHED; sleep infinity; fi"))

    if "rl" in stages:
        parts.append(
            f"timeout {a.max_hours}h python -m nla.pred.train_rl "
            f"--config configs/pred/rl_behavioral.yaml --positions {pos} "
            f"--base-ckpt {a.av_ckpt} --reader {a.train_reader} "
            f"--save-dir {ck}/behavioral_rl --num-steps {a.rl_steps} "
            f"--wandb-project {a.wandb_project} --wandb-name behavioral_rl "
            f"--seed {a.seed}")
        parts.append(f"{sync} push {hf} {ck}/behavioral_rl ckpts/behavioral_rl")

    if "rl_recon" in stages:
        parts.append(
            f"timeout {a.max_hours}h python -m nla.pred.train_rl "
            f"--config configs/pred/rl_recon_matched.yaml --positions {pos} "
            f"--base-ckpt {a.av_ckpt} --ar-ckpt {a.ar_ckpt} "
            f"--save-dir {ck}/recon_matched --num-steps {a.rl_steps} "
            f"--wandb-project {a.wandb_project} --wandb-name recon_matched "
            f"--seed {a.seed}")
        parts.append(f"{sync} push {hf} {ck}/recon_matched ckpts/recon_matched")

    if "eval" in stages:
        ckpts = ["--checkpoint sft="]
        if a.recon_adapter:
            ckpts.append(f"--checkpoint recon_rl={a.recon_adapter}")
        if "rl" in stages or a.behavioral_adapter:
            # Prefer the checkpoint the training reader liked best over the last
            # one written. train_rl records that in best.json as a directory name
            # that exists on disk (evals and saves happen on different cadences,
            # so the best STEP usually has no checkpoint of its own).
            parts.append(
                "BEST=$(python -c \"import json,pathlib;"
                f"p=pathlib.Path('{ck}/behavioral_rl/best.json');"
                "d=json.loads(p.read_text()) if p.exists() else {};"
                f"print(d.get('best_ckpt') or 'iter_{a.rl_steps:06d}')\")"
                f" && echo \"[eval] behavioral checkpoint: $BEST\"")
            ckpts.append("--checkpoint behavioral_rl="
                         + (a.behavioral_adapter
                            or f"{ck}/behavioral_rl/$BEST"))
        if "rl_recon" in stages:
            ckpts.append("--checkpoint recon_matched="
                         f"{ck}/recon_matched/iter_{a.rl_steps:06d}")
        parts.append(
            f"python -m nla.pred.eval --positions {pos} --base-ckpt {a.av_ckpt} "
            + " ".join(ckpts)
            + f" --readers {readers} --n-positions {a.eval_positions} "
            f"--branches {a.n_branches} --max-new-tokens {a.max_new_tokens} "
            f"--out-dir {ev}/final --wandb-project {a.wandb_project} "
            f"--wandb-name eval")
        parts.append(f"python -m nla.pred.report --eval-dir {ev}/final "
                     f"--title 'Frozen-readout NLA pilot'")
        parts.append(f"{sync} push {hf} {ev}/final evals/final")

    parts.append("echo FINISHED")
    parts.append("sleep infinity" if a.keep else "runpodctl remove pod $RUNPOD_POD_ID || true")
    # Setup aborts the pod on failure; stages continue so partial results upload.
    return " && ".join(setup) + " ; " + " ; ".join(parts)


def launch(a):
    # Build and print the script BEFORE touching the SDK or the keys, so a
    # dry-run works anywhere (and reviewing the command costs nothing).
    script = build_script(a)
    if a.dry_run:
        print(script.replace("; ", ";\n"))
        print(f"\n[dry-run] stages={a.stages} gpu={a.gpu} "
              f"est {sum(STAGE_HOURS.get(s, 1) for s in a.stages.split(','))*1.15:.1f} h")
        return
    import runpod

    runpod.api_key = _key(".runpod_key")
    env = {
        "WANDB_API_KEY": _key(".wandb_key"),
        "HF_TOKEN": _key(".hf_token"),
        "RUNPOD_API_KEY": runpod.api_key,
        "HF_HOME": "/workspace/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
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
                docker_args=f"bash -lc '{script}'", env=env,
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
    rates = {}
    for g in runpod.get_gpus():
        try:
            d = runpod.get_gpu(g["id"])
            rates[g["id"]] = (d.get("lowestPrice") or {}).get("uninterruptablePrice")
        except Exception as e:                                 # noqa: BLE001
            # One unpriceable GPU should not stop the estimate for the others.
            print(f"  (no price for {g['id']}: {type(e).__name__})")
    stages = a.stages.split(",")
    hours = sum(STAGE_HOURS.get(s, 1.0) for s in stages)
    overhead = 0.35        # image pull + pip + model downloads, per pod
    for st in stages:
        print(f"  {st:<10} {STAGE_HOURS.get(st, 1.0):>4.1f} h")
    print(f"GPU-hours (compute): {hours:.1f}  + {overhead:.2f} h startup")
    print(f"{'gpu':<28} {'$/h':>7} {'run $':>8}")
    for g in [a.gpu] + [x for x in FALLBACK_GPUS if x != a.gpu]:
        r = rates.get(g)
        if r:
            print(f"{g:<28} {r:>7.2f} {(hours + overhead) * r:>8.2f}")
    # runpod.get_user() does not carry the balance; ask GraphQL directly.
    try:
        import requests
        r = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {runpod.api_key}"},
            json={"query": "{ myself { clientBalance } }"}, timeout=20)
        bal = r.json()["data"]["myself"]["clientBalance"]
        print(f"\nRunPod balance: ${bal:.2f}")
        # Warn against the GPU actually selected, not the cheapest one listed -
        # the default is what will be launched.
        sel = rates.get(a.gpu)
        if sel and bal < (hours + overhead) * sel:
            print(f"WARNING: ${bal:.2f} will not cover these stages on {a.gpu} "
                  f"(${(hours + overhead) * sel:.2f}). Top up, pick a cheaper GPU, "
                  f"or split the chain across pods.")
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
                       help="comma list of prep,gate,rl,rl_recon,eval")
        p.add_argument("--gpu", default="NVIDIA H200")
        p.add_argument("--cloud", default="SECURE")
        p.add_argument("--branch", default="sv/pred-nla")
        p.add_argument("--hf-repo", default="syvb/pred-nla-qwen3-8b")
        p.add_argument("--wandb-project", default="pred-nla")
        # models
        p.add_argument("--av-ckpt", default="syvb/nanonla-qwen3-8b-L24-av")
        p.add_argument("--ar-ckpt", default="syvb/nanonla-qwen3-8b-L24-ar")
        p.add_argument("--recon-adapter",
                       default="syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0",
                       help="published reconstruction-RL comparator (adapter#subfolder)")
        p.add_argument("--behavioral-adapter", default=None,
                       help="eval-only: an already-trained behavioral adapter")
        p.add_argument("--target-ckpt", default="Qwen/Qwen3-8B")
        p.add_argument("--train-reader", default="Qwen/Qwen3-4B-Base")
        p.add_argument("--heldout-reader", default="google/gemma-3-4b-pt")
        # data
        p.add_argument("--source-repo", default="asher577/nla-rl-data-free8")
        p.add_argument("--source-file", default="rl_shuf.parquet")
        p.add_argument("--corpus-filter", default="finefineweb",
                       help="the source pool is half chat transcripts; keep the "
                            "pretraining-like half")
        p.add_argument("--n-rl", type=int, default=12000)
        p.add_argument("--n-val", type=int, default=1000)
        p.add_argument("--n-eval", type=int, default=2000)
        p.add_argument("--n-branches", type=int, default=4)
        p.add_argument("--n-tokens", type=int, default=24)
        p.add_argument("--max-per-doc", type=int, default=2)
        p.add_argument("--batch-prefixes", type=int, default=24)
        # run sizes
        p.add_argument("--rl-steps", type=int, default=300)
        p.add_argument("--gate-positions", type=int, default=500)
        p.add_argument("--eval-positions", type=int, default=2000)
        p.add_argument("--max-new-tokens", type=int, default=192)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-hours", type=int, default=8,
                       help="per-stage timeout; RL at the default 300 "
                            "steps is a ~4.6 h stage")
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
