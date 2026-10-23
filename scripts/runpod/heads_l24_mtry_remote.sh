#!/usr/bin/env bash
# Continue the L24 head-NLA std RL pair (step 400) with the matryoshka objective — executes ON the pod
# (docs/head_nla_l24_matryoshka_plan.md): each GRPO group keeps its first L ~ U[1,120] explanation
# tokens, the AR scores and co-trains on that prefix, KL tripled (0.03) and tapered past content token
# 16 to 0.1× at 120, after a 40-step AR-only warm-up (AV frozen) so the AR first learns to read prefixes.
# Launch with:
#   GPUS="NVIDIA H100 80GB HBM3|3.5|4.5" scripts/runpod/launch_with_retry.sh <out dir> heads-l24-mtry \
#       --remote-script scripts/runpod/heads_l24_mtry_remote.sh --secrets hf,wandb \
#       --fetch results --no-checkpoints --max-hours 16 --avoid-machine bdh8atjj57i7
# Markers: /workspace/results/DONE | FAILED (first failing stage); log /workspace/results/run.log.
set -eEuo pipefail

REPO=/workspace/EasyNLA
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
BASE=Qwen/Qwen3-8B
SRC_PREFIX=syvb/head-nla-qwen3-8b-l24-std
FROM_ITER=${FROM_ITER:-iter_000400}
EXTRA_STEPS=${EXTRA_STEPS:-200}     # matryoshka steps after the AR-only warm-up
FREEZE=${FREEZE:-40}
OBJ=(--trunc-min-tokens 1 --trunc-max-tokens 120 --kl-beta 0.03
     --kl-taper-start 16 --kl-taper-end 120 --kl-taper-floor 0.1 --av-freeze-steps "$FREEZE")
START=${FROM_ITER#iter_}; START=$((10#$START))
NUM_STEPS=$((START + FREEZE + EXTRA_STEPS))
HF_RUN=syvb/head-nla-qwen3-8b-l24-mtry
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
keep_ckpt() {   # keep_ckpt <iter dir> <dest>: policy adapter + co-trained AR, minus the frozen reference
  mkdir -p "$2"
  cp -r "$1" "$2/" && rm -rf "$2/$(basename "$1")/reference"
}
save_partial() {   # the newest checkpoint into results/ (the launcher fetches results/, then deletes the pod)
  local latest
  latest=$(ls -d "$CK"/rl/iter_* 2>/dev/null | sort | tail -1 || true)
  if [ -n "$latest" ] && [ ! -d "$RES/rl_ckpt/$(basename "$latest")" ]; then
    keep_ckpt "$latest" "$RES/rl_ckpt_partial"
  fi
}
# copy the newest checkpoint out BEFORE writing FAILED: the launcher fetches and deletes on the marker
trap 'save_partial || true; echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, WANDB_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false FROM_ITER SRC_PREFIX
unset PYTORCH_CUDA_ALLOC_CONF
echo "resume=$SRC_PREFIX-rl-ckpts/$FROM_ITER steps $START -> $NUM_STEPS (freeze $FREEZE) ${OBJ[*]}"
wandb_flags() {   # wandb_flags <run name>
  if [ -n "${WANDB_API_KEY:-}" ]; then
    echo --wandb-project head-nla-qwen3-8b --wandb-group l24-mtry --wandb-name "$1"
  else
    echo --no-wandb
  fi
}

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate bitsandbytes
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', \
transformers.__version__, 'peft', peft.__version__)"

stage download
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok, it, pre = os.environ["HF_TOKEN"], os.environ["FROM_ITER"], os.environ["SRC_PREFIX"]
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset",
                      allow_patterns=["l24/sft_data/*", "l24/rl_data/*"], local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for sub in ("sft_data", "rl_data"):
    for f in Path(d, "l24", sub).iterdir():
        shutil.copy(f, Path("/workspace/sft", f.name))
snapshot_download(f"{pre}-av", local_dir="/workspace/ckpt/sft_av", token=tok)
snapshot_download(f"{pre}-rl-ckpts", allow_patterns=[f"{it}/*"], local_dir="/workspace/ckpt/prev", token=tok)
# the trainer finds the optimizer state next to the resumed iter dir (nla/utils/resume.py)
shutil.copy(f"/workspace/ckpt/prev/{it}/optim_latest.pt", "/workspace/ckpt/prev/optim_latest.pt")
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
print(sorted(os.listdir("/workspace/sft")), sorted(os.listdir(f"/workspace/ckpt/prev/{it}")))
PY

rl_train() {   # rl_train <save dir> <num steps> <rollout batch> <logp mb> <AR mb> <eval gen batch> <extra...>
  local dir=$1 steps=$2 rb=$3 lmb=$4 cmb=$5 egb=$6; shift 6
  python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml \
    --base-ckpt "$BASE" --quant none --device cuda \
    --av-ckpt "$CK/sft_av" --ar-ckpt "$CK/prev/$FROM_ITER/critic" \
    --resume-from-lora "$CK/prev/$FROM_ITER" --start-step "$START" \
    --rl-parquet "$SFT/rl.parquet" --sidecar "$SFT/rl.parquet" --save-dir "$dir" \
    --critic-prefix-column critic_prefix \
    --rollout-batch "$rb" --logp-micro-batch "$lmb" --critic-micro-batch "$cmb" --eval-gen-batch "$egb" \
    --num-steps "$steps" --save-every 50 --val-rows 2000 --evals base_fve \
    --eval-trunc-lengths 4 16 64 "${OBJ[@]}" --seed 0 "$@"
}

# Three steps on a throwaway save dir, no wandb: one AV-frozen step (with the held-out eval and its
# prefix FVE), then two matryoshka steps — shapes, memory and the step times at the real batch.
stage smoke
rl_train "$CK/smoke" $((START + 3)) 256 8 16 64 --no-wandb --eval-every "$START" --av-freeze-steps 1 \
  2>&1 | tee "$RES/smoke.log"
grep -E "^step " "$RES/smoke.log"
rm -rf "$CK/smoke"

stage rl
setsid nohup nice -n 19 python scripts/runpod/hf_ckpt_uploader.py --log "$RES/run.log" --ckpt-dir "$CK/rl" \
  --repo "$HF_RUN-rl-ckpts" > "$RES/hf_upload.log" 2>&1 < /dev/null &
# OOM fallback as in the std run: half batches if it dies before any checkpoint past the warm-up (the
# first AV update, the memory-heavy step, comes right after the warm-up's checkpoint). The retry
# starts over, warm-up included (~15 min).
past_warmup() {
  local latest
  latest=$(ls -d "$CK"/rl/iter_* 2>/dev/null | sort | tail -1 || true)
  [ -n "$latest" ] && [ "$((10#${latest##*iter_}))" -gt $((START + FREEZE)) ]
}
if ! rl_train "$CK/rl" "$NUM_STEPS" 256 8 16 64 $(wandb_flags l24-mtry-rl) 2>&1 | tee "$RES/rl_train.log"; then
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$RES/rl_train.log" && ! past_warmup; then
    echo "OOM before any checkpoint past the warm-up: retrying at half batch"
    rm -rf "$CK/rl"
    rl_train "$CK/rl" "$NUM_STEPS" 128 4 8 32 $(wandb_flags l24-mtry-rl) 2>&1 | tee "$RES/rl_train.log"
  else
    false
  fi
fi

stage keep_ckpts
LAST=$(ls -d "$CK"/rl/iter_* | sort | tail -1)
keep_ckpt "$LAST" "$RES/rl_ckpt"
if [ "$FREEZE" -gt 0 ]; then keep_ckpt "$CK/rl/iter_$(printf %06d $((START + FREEZE)))" "$RES/rl_ckpt"; fi
cp "$CK/rl/run_config.yaml" "$CK/rl/nla_meta.yaml" "$RES/rl_ckpt/"

# The standard held-out checks (all 6,360 eval rows, T=1, whole explanations) on the final pair.
stage checks
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --no-inject-preserve-magnitude \
    --batch-size 64 --max-new-tokens 256 --temperature 1 --no-fallback-cut \
    --av-ckpt "$LAST" --ar-ckpt "$LAST/critic" --out-dir "$RES/checks_t1"

# The sweep pod downloads the warm-up's and the last checkpoint from HF: wait (≤ 60 min) until both are
# uploaded, unless the uploader is gone (then say so; results/rl_ckpt still has both).
stage wait_uploads
WARM=iter_$(printf %06d $((START + FREEZE)))
for _ in $(seq 1 120); do
  up=/workspace/hf_upload_stage/uploaded.txt
  if grep -qx "$(basename "$LAST")" "$up" 2>/dev/null && grep -qx "$WARM" "$up" 2>/dev/null; then break; fi
  if ! pgrep -f hf_ckpt_uploader.py >/dev/null; then
    echo "WARNING: the HF uploader is not running; last lines of its log:"; tail -n 20 "$RES/hf_upload.log"; break
  fi
  sleep 30
done
cat /workspace/hf_upload_stage/uploaded.txt || true

date -u > "$RES/DONE"
echo "=== all stages done ==="
