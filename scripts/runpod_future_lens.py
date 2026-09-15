"""Launch future-lens stages on RunPod (one pod per stage, shared network volume).

NOT run automatically — every stage costs money. Adapted from the metamodelling
repo's launcher: the pod boots the official PyTorch image, clones this repo's
branch, `pip install -e .`, runs ONE stage command with logs to wandb and to the
network volume, then terminates itself (unless --keep).

  python scripts/runpod_future_lens.py volume --size 150 --dc US-CA-2      # once (a DC with H100 stock that supports volumes)
  python scripts/runpod_future_lens.py launch collect  --volume <id>
  python scripts/runpod_future_lens.py launch alpha_sweep --volume <id>
  python scripts/runpod_future_lens.py launch sft      --volume <id> --alpha-mult 2 --extra "--num-steps 8000"
  python scripts/runpod_future_lens.py launch rl       --volume <id> --av-ckpt sft_replace_embed_a2.0_greedy_distill/iter_0008000 --seed 0
  python scripts/runpod_future_lens.py launch eval     --volume <id> --adapters sft_replace_embed_a2.0_greedy_distill/iter_0008000,rl_exact_match_s0/iter_004000
  python scripts/runpod_future_lens.py launch baselines --volume <id>
  python scripts/runpod_future_lens.py status | terminate <pod_id>

Quoting: the whole bootstrap runs inside `bash -lc '...'`, so stage commands must
never contain single quotes; JSON tags use escaped double quotes (see _tag_arg).

Credentials: ~/.runpod_key, ~/.wandb_key, ~/.hf_token. Layout on the volume (data dir = --data-dir, default data_base_v2)
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
GPU_PREF = [("NVIDIA H100 80GB HBM3", "SECURE"), ("NVIDIA H100 NVL", "SECURE"), ("NVIDIA H100 PCIe", "SECURE"),
            ("NVIDIA A100 80GB PCIe", "SECURE"), ("NVIDIA A100-SXM4-80GB", "SECURE"),
            ("NVIDIA H100 PCIe", "COMMUNITY"), ("NVIDIA A100 80GB PCIe", "COMMUNITY")]
# --model: pretrained base + plain prompt (spec: "base / non-thinking"; Future Lens used base GPT-J).
# Layers are depth-matched to the 8B choice (36 blocks: collect 4..32, train 8..32; the first collected
# layer is the wrong-layer control only). 28-block models use the same relative depths (x28/36, rounded).
MODELS = {  # size -> (base ckpt, collected layers, trained layers)
    "8b":   ("Qwen/Qwen3-8B-Base",   "4,8,12,16,20,24,28,32", "8,12,16,20,24,28,32"),
    "4b":   ("Qwen/Qwen3-4B-Base",   "4,8,12,16,20,24,28,32", "8,12,16,20,24,28,32"),
    "1.7b": ("Qwen/Qwen3-1.7B-Base", "3,6,9,12,16,19,22,25",  "6,9,12,16,19,22,25"),
    "0.6b": ("Qwen/Qwen3-0.6B-Base", "3,6,9,12,16,19,22,25",  "6,9,12,16,19,22,25"),
}
BASE, LAYERS, TRAIN_LAYERS = MODELS["8b"]   # overridden from --model in main()
PROMPT_FORMAT = "plain"
TOPK = 64            # stored target top-K per greedy step: distillation targets for SFT --distill
WORK = "/workspace/fl"
WANDB_PROJECT = "rl-future-lens"
MAX_POD_HOURS = 16   # watchdog: the longest stage (RL, ~4-5 h) plus a wide margin


def _read(path):
    return open(os.path.expanduser(path)).read().strip()


def _runpod():
    import runpod
    runpod.api_key = _read("~/.runpod_key")
    return runpod


# ----------------------------------------------------------------------------
# stage commands (run inside the pod, cwd = repo)
# ----------------------------------------------------------------------------

def evals_name(a) -> str:
    """evals dir on the volume / in the HF repo: `evals` for the 8B, `evals_{size}` otherwise."""
    return "evals" if a.model == "8b" else f"evals_{a.model}"


def stage_cmd(stage: str, a) -> str:
    D, C, E = f"{WORK}/{a.data_dir}", f"{WORK}/ckpts", f"{WORK}/{evals_name(a)}"
    if stage == "pipeline":
        # one pod per model: collect -> SFT -> eval -> baselines, each step skipped if its output exists
        # (GPU stock is scarce: one queue wait instead of four). SFT flags via --extra go to the SFT only.
        run = a.run_name
        sub = argparse.Namespace(**vars(a)); sub.extra = ""; sub.adapters = f"{run}/iter_{a.sft_steps:07d}"
        sft = argparse.Namespace(**vars(a)); sft.extra = f"--num-steps {a.sft_steps} {a.extra}".strip()
        return " && ".join([
            f"[ -f {D}/eval.parquet ] || ({stage_cmd('collect', sub)})",
            f"[ -d {C}/{run}/iter_{a.sft_steps:07d} ] || ({stage_cmd('sft', sft)})",
            stage_cmd("eval", sub), stage_cmd("baselines", sub)])
    if stage == "collect":
        return (f"python -m nla.future_lens.collect --base-ckpt {BASE} --corpus HuggingFaceFW/fineweb "
                f"--corpus-config sample-10BT --n-train-docs {a.n_train_docs} --n-eval-docs {a.n_eval_docs} "
                f"--layers {LAYERS} --positions-per-doc 40 --eval-positions-per-doc 20 --max-len 1024 "
                f"--batch-size 8 --greedy all --topk {TOPK} --prompt-format {PROMPT_FORMAT} --out-dir {D} && "
                f"(python scripts/hf_upload.py {D} {a.data_dir} --repo {a.hf_repo} || true)")
    if stage == "alpha_sweep":
        # mirror data/ to HF in the background (idempotent) while the sweep runs
        runs = [f"python scripts/hf_upload.py {D} {a.data_dir} --repo {a.hf_repo} > {WORK}/logs/hf_upload_data.log 2>&1 & UP=$!"]
        # --sweep "replace_embed:0.5,1,2,4;karvonen:1" ; shuffled-control runs for --sweep-shuffle-mults
        for spec in a.sweep.split(";"):
            inj, ms = spec.split(":")
            for m in [float(x) for x in ms.split(",")]:
                for shuf in ("", "--shuffle-activations") if m in a.sweep_shuffle_mults else ("",):
                    # run names carry the data split: the finished-run guard below must not match a
                    # sweep done on another split (it skipped every run of the base-model check once)
                    tag = f"{a.data_dir}_{inj}_a{m}{'_shuf' if shuf else ''}"
                    runs.append(
                        f"[ -d {C}/sweep_{tag}/iter_0000400 ] || python -m nla.train_sft --config configs/future_lens/sft_alpha_sweep.yaml "
                        f"--base-ckpt {BASE} --parquet {D}/train.parquet --heldout-parquet {D}/eval.parquet "
                        f"--save-dir {C}/sweep_{tag} --injection {inj} --alpha-mult {m} {shuf} "
                        f"--layers {TRAIN_LAYERS} --label {a.label} "
                        f"--wandb-name sweep_{tag} --seed 0 && "
                        f"(python scripts/hf_upload.py {C}/sweep_{tag} ckpts/sweep_{tag} --repo {a.hf_repo} || true)")
        runs.append("wait $UP")
        return " && ".join(runs)
    if stage == "sft":
        return (f"python -m nla.train_sft --config configs/future_lens/sft.yaml --base-ckpt {BASE} "
                f"--parquet {D}/train.parquet --heldout-parquet {D}/eval.parquet "
                f"--save-dir {C}/{a.run_name} --injection {a.injection} --alpha-mult {a.alpha_mult} "
                f"--layers {TRAIN_LAYERS} --label {a.label} {'--distill' if a.distill else '--no-distill'} "
                f"{'--affine' if a.affine else ''} --wandb-name {a.run_name} --seed {a.seed} {a.extra} && "
                f"(python scripts/hf_upload.py {C}/{a.run_name} ckpts/{a.run_name} --repo {a.hf_repo} || true)")
    if stage == "rl":
        return (f"python -m nla.future_lens.train_rl --config configs/future_lens/rl.yaml --base-ckpt {BASE} "
                f"--av-ckpt {C}/{a.av_ckpt} --parquet {D}/train.parquet --eval-parquet {D}/eval.parquet "
                f"--save-dir {C}/{a.run_name} --reward {a.reward} --seed {a.seed} --layers {TRAIN_LAYERS} "
                f"--wandb-name {a.run_name} {a.extra} && "
                f"(python scripts/hf_upload.py {C}/{a.run_name} ckpts/{a.run_name} --repo {a.hf_repo} || true)")
    if stage == "eval":
        cmds = []
        for ad in a.adapters.split(","):
            name = ad.replace("/", "_")
            # `group` = run family (seed stripped) so plots.py pools seeds; `seed` from the run name.
            run = ad.split("/")[0]
            m = re.search(r"_s(\d+)$", run)
            seed = int(m.group(1)) if m else a.seed
            group = run[: m.start()] if m else run
            cmds.append(f"rm -f {E}/{name}.jsonl {E}/readouts_{name}.jsonl && "
                        f"python -m nla.future_lens.eval --base-ckpt {BASE} --adapter {C}/{ad} "
                        f"--parquet {D}/eval.parquet --out {E}/{name}.jsonl "
                        f"--conditions real,shuffled,none,wrong_layer,cross_layer --wrong-layer {LAYERS.split(',')[0]} "
                        f"--batch-size 128 --surprisal --surprisal-rows 256 "
                        f"--layers {TRAIN_LAYERS} --dump-readouts {E}/readouts_{name}.jsonl "
                        f"--seed {seed} --tag-kv {_tag_arg({'checkpoint': name, 'group': group, 'seed': seed, 'model': a.model})} {a.extra}")
        return " && ".join(cmds)
    if stage == "filter_ablation":
        # How much does the top-1-correct position filter matter? Unfiltered split from fresh
        # docs (corpus offset past the filtered slice), two identical 2000-step SFT runs
        # (filtered vs unfiltered data), each evaluated on both eval sets. Readouts on the
        # unfiltered eval set are dumped so the top-1-correct subset can be split off locally.
        # --part unf: collect + train/eval the unfiltered arm; --part filt: train/eval the
        # filtered arm (waits for the unf collection's DONE marker before its cross-eval);
        # --part all: everything sequentially in one pod.
        U = f"{WORK}/data_unf"
        sets = {"filt": D, "unf": U}
        parts = ["filt", "unf"] if a.part == "all" else [a.part]
        cmds = []
        if "unf" in parts:
            cmds += [f"[ -f {U}/DONE ] || (python -m nla.future_lens.collect --base-ckpt {BASE} --corpus HuggingFaceFW/fineweb "
                     f"--corpus-config sample-10BT --corpus-start 6000 --n-train-docs 2700 --n-eval-docs 300 "
                     f"--layers {LAYERS} --positions-per-doc 40 --eval-positions-per-doc 20 --max-len 1024 "
                     f"--batch-size 8 --no-require-top1 --greedy all --prompt-format {PROMPT_FORMAT} --out-dir {U} && touch {U}/DONE)",
                     f"[ -f {U}/DONE ]",
                     f"python scripts/hf_upload.py {U} data_unf --repo {a.hf_repo} > {WORK}/logs/hf_upload_data_unf.log 2>&1 & UP=$!"]
        for name in parts:
            data, run = sets[name], f"sft_{name}_2k"
            cmds.append(f"[ -d {C}/{run}/iter_0002000 ] || python -m nla.train_sft --config configs/future_lens/sft.yaml --base-ckpt {BASE} "
                        f"--parquet {data}/train.parquet --heldout-parquet {data}/eval.parquet "
                        f"--save-dir {C}/{run} --injection {a.injection} --alpha-mult {a.alpha_mult} --no-distill "
                        f"--num-steps 2000 --save-every 2000 --wandb-name {run} --seed {a.seed} {a.extra}")
            cmds.append(f"(python scripts/hf_upload.py {C}/{run} ckpts/{run} --repo {a.hf_repo} || true)")
        if a.part == "filt":
            cmds.append(f"until [ -f {U}/DONE ]; do sleep 30; done")
        for name in parts:
            data, run = sets[name], f"sft_{name}_2k"
            for ename, edata in sets.items():
                tag = _tag_arg({"checkpoint": run, "group": run, "seed": a.seed, "evalset": ename})
                cmds.append(f"python -m nla.future_lens.eval --base-ckpt {BASE} --adapter {C}/{run}/iter_0002000 "
                            f"--parquet {edata}/eval.parquet --sidecar {data}/train.parquet "
                            f"--out {E}/ablation_{run}_on_{ename}.jsonl --conditions real,shuffled,none "
                            f"--layers {TRAIN_LAYERS} --max-rows 800 --batch-size 128 --seed {a.seed} --tag-kv {tag} "
                            f"--dump-readouts {E}/readouts_ablation_{run}_on_{ename}.jsonl")
        if "unf" in parts:
            cmds.append("wait $UP")
        return " && ".join(cmds)
    if stage == "futurelens":
        # Future Lens learned-prompt baseline (Pal et al. 2023): soft prompt + same-layer transplant,
        # KL at N=1 from the stored top-K; eval on the same 2000-position subsample as `eval`
        return (f"python -m nla.future_lens.futurelens_prompt --base-ckpt {BASE} --train-parquet {D}/train.parquet "
                f"--parquet {D}/eval.parquet --layers {TRAIN_LAYERS} --n-train 10000 --steps 600 --max-rows 2000 "
                f"--seed {a.seed} --save-dir {C}/futurelens --out {E}/futurelens.jsonl {a.extra} && "
                f"(python scripts/hf_upload.py {C}/futurelens ckpts/futurelens --repo {a.hf_repo} || true)")
    if stage == "leakage":
        # rerun only the readout-leakage diagnostic (e.g. after new adapters were evaluated)
        return (f"python -m nla.future_lens.baselines --label {a.label} leakage --parquet {D}/eval.parquet "
                f"--readouts {E}/readouts_sft_*.jsonl {E}/readouts_rl_*.jsonl --out {E}/leakage.jsonl --base-ckpt {BASE} "
                f"--hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000 {a.extra}")
    if stage == "baselines":
        return (f"rm -f {E}/baselines.jsonl {E}/leakage.jsonl && "
                f"python -m nla.future_lens.baselines --label {a.label} ngram --parquet {D}/eval.parquet --out {E}/baselines.jsonl "
                f"--base-ckpt {BASE} --hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000 && "
                f"python -m nla.future_lens.baselines --label {a.label} probe --train-parquet {D}/train.parquet --parquet {D}/eval.parquet "
                f"--base-ckpt {BASE} --leakage --epochs 3 --batch 1024 --layers {TRAIN_LAYERS} --out {E}/baselines.jsonl && "
                f"python -m nla.future_lens.baselines --label {a.label} target_window --parquet {D}/eval.parquet "
                f"--base-ckpt {BASE} --out {E}/baselines.jsonl && "
                f"python -m nla.future_lens.baselines --label {a.label} leakage --parquet {D}/eval.parquet "
                f"--readouts {E}/readouts_sft_*.jsonl {E}/readouts_rl_*.jsonl --out {E}/leakage.jsonl --base-ckpt {BASE} "
                f"--hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000")
    raise SystemExit(f"unknown stage {stage}")


def _tag_arg(d: dict) -> str:
    """eval.py --tag-kv value: k=v,k=v. No quotes or braces — the runpod SDK pastes dockerArgs
    into a GraphQL string unescaped, and the bash -lc wrapper is single-quoted."""
    for k, v in d.items():
        assert "," not in f"{k}{v}" and "=" not in f"{k}{v}", (k, v)
    return ",".join(f"{k}={v}" for k, v in d.items())


def bootstrap(stage_command: str, stage: str, keep: bool, hf_repo: str, evals: str = "evals") -> str:
    assert "'" not in stage_command, f"stage command contains a single quote (breaks bash -lc quoting): {stage_command}"
    log = f"{WORK}/logs/{stage}_$(date +%Y%m%d_%H%M%S).log"
    finish = "" if keep else "python /root/EasyNLA/scripts/pod_terminate.py || runpodctl remove pod $RUNPOD_POD_ID"
    return (
        "/start.sh >/dev/null 2>&1 & "
        # watchdog: nothing here should take a day; a failed clone/pip/terminate must not bill forever
        f"(sleep {MAX_POD_HOURS}h; runpodctl remove pod $RUNPOD_POD_ID) >/dev/null 2>&1 & "
        f"mkdir -p {WORK}/logs {WORK}/ckpts {WORK}/{evals} && exec > >(tee -a {log}) 2>&1; set -x; set -o pipefail; "
        # clone onto the pod's own container disk: two pods sharing the volume must not rm -rf each other's repo
        f"cd /root && rm -rf EasyNLA && git clone -q -b {BRANCH} {REPO} && cd EasyNLA && "
        "pip install -q -e . bitsandbytes runpod 2>&1 | tail -2 && nvidia-smi --query-gpu=name,memory.total --format=csv && "
        f"export HF_HOME=/workspace/hf && ({stage_command}) ; echo STAGE_EXIT=$? ; "
        f"python scripts/hf_upload.py {WORK}/{evals} {evals} --repo {hf_repo} || true; "
        f"python scripts/hf_upload.py {WORK}/logs logs --repo {hf_repo} || true; "
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
    cmd = bootstrap(stage_cmd(a.stage, a), a.stage, a.keep, a.hf_repo, evals_name(a))
    assert '"' not in cmd and "{" not in cmd and "}" not in cmd, "dockerArgs is pasted into GraphQL unescaped"
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
    l = sub.add_parser("launch"); l.add_argument("stage", choices=["collect", "alpha_sweep", "sft", "rl", "eval", "baselines", "filter_ablation", "futurelens", "leakage", "pipeline"])
    l.add_argument("--model", default="8b", choices=list(MODELS), help="target = decoder size (sets base ckpt, layers, "
                   "default --data-dir data_{size}, evals_{size}/, and a _{size} suffix on sft/rl run names)")
    l.add_argument("--sft-steps", type=int, default=8000, help="pipeline: SFT --num-steps (8000 = the 8B run)")
    l.add_argument("--volume", required=True, help="network volume id")
    l.add_argument("--gpu", default=None); l.add_argument("--cloud", default="SECURE")
    l.add_argument("--keep", action="store_true"); l.add_argument("--dry-run", action="store_true")
    l.add_argument("--run-name", default=None); l.add_argument("--seed", type=int, default=0)
    l.add_argument("--extra", default="", help="extra CLI flags appended to the stage command")
    l.add_argument("--n-train-docs", type=int, default=5500); l.add_argument("--n-eval-docs", type=int, default=300)
    l.add_argument("--hf-repo", default="syvb/rl-future-lens-qwen3-8b")
    l.add_argument("--data-dir", default=None, help="split dir under the volume and path in the HF repo (default: "
                   "data_base_v2 for --model 8b, data_{size} otherwise) "
                   "(data = chat-model split, text labels; data_base = base model, greedy labels, no top-K; "
                   "data_base_v2 = base model, greedy labels, top-64 distillation targets)")
    l.add_argument("--injection", default="replace_embed"); l.add_argument("--alpha-mult", type=float, default=1.0)
    l.add_argument("--sweep", default="replace_embed:0.5,1,2,4;karvonen:1", help="alpha_sweep: inj:mults;inj:mults")
    l.add_argument("--sweep-shuffle-mults", default="0.5,1,2,4", help="alpha_sweep: mults that also get a shuffled-control run")
    l.add_argument("--distill", action=argparse.BooleanOptionalAction, default=True,
                   help="sft: soft targets from the stored top-K distributions (Future Lens KL objective)")
    l.add_argument("--label", default="greedy", choices=["text", "greedy"],
                   help="readout label for sft/alpha_sweep/baselines (rl and eval read it from the checkpoint)")
    l.add_argument("--affine", action="store_true")
    l.add_argument("--av-ckpt", default=None, help="rl: SFT iter dir relative to ckpts/")
    l.add_argument("--reward", default="exact_match")
    l.add_argument("--part", default="all", choices=["all", "filt", "unf"], help="filter_ablation: which arm this pod runs")
    l.add_argument("--adapters", default="", help="eval: comma list of ckpt dirs relative to ckpts/")
    s = sub.add_parser("status")
    t = sub.add_parser("terminate"); t.add_argument("pod_id")
    a = p.parse_args(argv)
    if a.cmd == "launch":
        a.sweep_shuffle_mults = {float(x) for x in a.sweep_shuffle_mults.split(",") if x}
    if a.cmd == "launch":
        global BASE, LAYERS, TRAIN_LAYERS
        BASE, LAYERS, TRAIN_LAYERS = MODELS[a.model]
        if a.data_dir is None:
            a.data_dir = "data_base_v2" if a.model == "8b" else f"data_{a.model}"
        if a.run_name is None:
            sfx = "" if a.model == "8b" else f"_{a.model}"
            sft_name = f"sft_{a.injection}_a{a.alpha_mult}_{a.label}{'_distill' if a.distill else ''}{sfx}"
            a.run_name = {"sft": sft_name, "pipeline": sft_name, "rl": f"rl_{a.reward}_s{a.seed}{sfx}",
                          "filter_ablation": f"ablation_{a.part}"}.get(a.stage, a.stage)
    {"volume": cmd_volume, "launch": cmd_launch, "status": cmd_status, "terminate": cmd_terminate}[a.cmd](a)


if __name__ == "__main__":
    main()
