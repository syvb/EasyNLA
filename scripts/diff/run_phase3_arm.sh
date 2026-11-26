#!/usr/bin/env bash
# End-to-end Phase 3 launcher for one experimental arm:
#   AV SFT -> AR SFT -> Merge AV/AR to bf16 HF -> 4-GPU DP vLLM GRPO RL
#
# Automatically:
#   - Unsets PYTORCH_CUDA_ALLOC_CONF (mandatory for vLLM IPC weight sync!)
#   - Reads tau_web_median from phase0_config.json for Arm 2 & Arm 2b
#   - Passes --av-adapter into RL (continues tuning SFT LoRA, avoiding the
#     documented ~12pp cold-start penalty from re-initializing LoRA B=0!)
#   - Saves lightweight LoRA + critic_latest + optim_latest every 25 steps (~7 min)
#     and archival 11 GB iter_NNNNNN/critic snapshots at steps 0, 50, 100, 200, 300, 400
#   - With `--resume`: automatically skips already-completed AV SFT, AR SFT, and
#     merge stages, and resumes RL from the latest atomically-committed checkpoint!
#
# Usage:
#   bash scripts/diff/run_phase3_arm.sh \
#       --arm arm2 \
#       --phase0-dir /data/diff_nla/phase0 \
#       --corpus-dir /data/diff_nla/corpus \
#       --ckpt-root  /ckpts/diff_nla \
#       --vllm-python /opt/vllm-lens-venv/bin/python \
#       --nproc 4 \
#       [--hardware h200|h100] [--seed 0] \
#       [--resume] [--skip-sft] [--skip-merge] [--dry-run]

set -euo pipefail

ARM=""
PHASE0_DIR=""
CORPUS_DIR=""
CKPT_ROOT=""
CORE_PYTHON="${CORE_PYTHON:-python}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"
NPROC=4
HARDWARE="h200"
SEED=0
AUTO_RESUME=0
SKIP_SFT=0
SKIP_MERGE=0
DRY_RUN=0
EXTRA_RL_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --phase0-dir) PHASE0_DIR="$2"; shift 2 ;;
    --corpus-dir) CORPUS_DIR="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --core-python) CORE_PYTHON="$2"; shift 2 ;;
    --vllm-python) VLLM_PYTHON="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --hardware) HARDWARE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume) AUTO_RESUME=1; shift ;;
    --skip-sft) SKIP_SFT=1; shift ;;
    --skip-merge) SKIP_MERGE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --extra-rl-args) EXTRA_RL_ARGS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ARM" || -z "$PHASE0_DIR" || -z "$CORPUS_DIR" || -z "$CKPT_ROOT" ]]; then
  echo "Required: --arm {arm0|arm0c|arm1|arm2|arm2b} --phase0-dir --corpus-dir --ckpt-root" >&2
  exit 1
fi

P0_CFG="${PHASE0_DIR}/phase0_config.json"
if [[ ! -f "$P0_CFG" ]]; then
  echo "FATAL: missing ${P0_CFG} — run Phase 0 first!" >&2
  exit 1
fi

TARGET_MODEL="$("$CORE_PYTHON" -c "import json; print(json.load(open('$P0_CFG'))['target_model'])")"
LAYER_IDX="$("$CORE_PYTHON" -c "import json; print(json.load(open('$P0_CFG'))['layer_index'])")"
AR_NUM_LAYERS=$((LAYER_IDX + 1))
TAU_MED="$("$CORE_PYTHON" -c "import json; print(json.load(open('$P0_CFG'))['tau_web_median'])")"

if [[ "$ARM" == "arm2" || "$ARM" == "arm2b" ]]; then
  DATA_DIR="${CORPUS_DIR}/arm2"
  TAU="$TAU_MED"
else
  DATA_DIR="${CORPUS_DIR}/${ARM}"
  TAU="0.0"
fi

if [[ "$HARDWARE" == "h100" ]]; then
  RL_CFG="configs/diff/rl_vllm_h100_80gb.yaml"
else
  case "$ARM" in
    arm0)  RL_CFG="configs/diff/rl_vllm_arm0_h200.yaml" ;;
    arm0c) RL_CFG="configs/diff/rl_vllm_arm0c_h200.yaml" ;;
    arm1)  RL_CFG="configs/diff/rl_vllm_arm1_h200.yaml" ;;
    arm2)  RL_CFG="configs/diff/rl_vllm_arm2_h200.yaml" ;;
    arm2b) RL_CFG="configs/diff/rl_vllm_arm2b_coldstart_h200.yaml" ;;
    *) echo "Unknown arm: $ARM" >&2; exit 1 ;;
  esac
fi

SEED_TAG=""
if [[ "$SEED" -ne 0 ]]; then
  SEED_TAG="_s${SEED}"
fi

ARM_CKPT="${CKPT_ROOT}/${ARM}${SEED_TAG}"
AV_SFT_DIR="${ARM_CKPT}/av_sft"
AR_SFT_DIR="${ARM_CKPT}/ar_sft"
MERGED_AV="${ARM_CKPT}/av_merged_hf"
MERGED_AR="${ARM_CKPT}/ar_merged_hf"
RL_SAVE_DIR="${ARM_CKPT}/rl_vllm"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

# Detect latest valid RL checkpoint under RL_SAVE_DIR when --resume is enabled.
# Prefers optim_latest.pt["step"] (written atomically LAST after LoRA + critic_latest)
# so actor, critic, and optimizer are guaranteed synchronized; falls back to the
# highest iter_NNNNNN with adapter_config.json and step > 0.
detect_rl_resume_args() {
  "$CORE_PYTHON" - "$RL_SAVE_DIR" "$RL_CFG" <<'PY'
import re, sys
from pathlib import Path
import yaml

save_dir = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
target_steps = 400
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    target_steps = int(cfg.get("num_steps", 400))

if not save_dir.exists():
    sys.exit(0)

step = 0
opt_p = save_dir / "optim_latest.pt"
if opt_p.exists():
    try:
        import torch
        st = torch.load(str(opt_p), map_location="cpu", weights_only=True)
        cand = int(st.get("step", 0))
        if cand > 0 and (save_dir / f"iter_{cand:06d}" / "adapter_config.json").exists():
            step = cand
    except Exception:
        step = 0

if step == 0:
    best = 0
    for d in save_dir.glob("iter_*"):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and (d / "adapter_config.json").exists():
            s = int(m.group(1))
            if s > best:
                best = s
    step = best

if step >= target_steps and target_steps > 0:
    print(f"COMPLETE {step} {target_steps}")
elif step > 0:
    print(f"RESUME {step} {save_dir / f'iter_{step:06d}'}")
PY
}

RL_RESUME_ARGS=()
if [[ "$AUTO_RESUME" -eq 1 && "$EXTRA_RL_ARGS" != *"--resume-from-lora"* ]]; then
  RESUME_INFO="$(detect_rl_resume_args || true)"
  if [[ "$RESUME_INFO" == COMPLETE* ]]; then
    echo "[phase3/resume] ${ARM}${SEED_TAG} RL already complete (${RESUME_INFO}) — nothing to do."
    exit 0
  elif [[ "$RESUME_INFO" == RESUME* ]]; then
    read -r _tag RESUME_STEP RESUME_LORA <<< "$RESUME_INFO"
    echo "[phase3/resume] Auto-resuming RL from step ${RESUME_STEP} (${RESUME_LORA})"
    RL_RESUME_ARGS=(--resume-from-lora "$RESUME_LORA" --start-step "$RESUME_STEP")
  fi
fi

echo "========================================================================"
echo "Phase 3 Launcher — ARM=${ARM} | SEED=${SEED} | MODEL=${TARGET_MODEL} | L${LAYER_IDX}"
echo "DATA_DIR=${DATA_DIR} | TAU=${TAU} | HARDWARE=${HARDWARE} | NPROC=${NPROC} | RESUME=${AUTO_RESUME}"
echo "========================================================================"

if [[ "$ARM" == "arm2b" ]]; then
  # Cold-start: NO SFT, initialize identity 25-layer AR checkpoint directly (skip if already built under --resume)
  if [[ "$AUTO_RESUME" -eq 1 && -f "${MERGED_AR}/value_head.safetensors" ]]; then
    echo "[phase3/resume] Reusing existing cold-start AR checkpoint: ${MERGED_AR}"
  else
    run_cmd "$CORE_PYTHON" scripts/init_coldstart_ar.py \
      --base-ckpt "$TARGET_MODEL" \
      --ar-num-layers "$AR_NUM_LAYERS" \
      --output-dir "$MERGED_AR" \
      --sidecar "${DATA_DIR}/rl_shuf.parquet"
  fi

  echo "[phase3] Launching Cold-Start RL (Arm 2b) with PYTORCH_CUDA_ALLOC_CONF unset..."
  unset PYTORCH_CUDA_ALLOC_CONF || true
  VLLM_TORCHRUN="$(dirname "$VLLM_PYTHON")/torchrun"
  run_cmd "$VLLM_TORCHRUN" --standalone --nproc_per_node="$NPROC" -m nla.train_rl_vllm \
    --config "$RL_CFG" \
    --base-ckpt "$TARGET_MODEL" \
    --av-ckpt "$TARGET_MODEL" \
    --ar-ckpt "$MERGED_AR" \
    --rl-parquet "${DATA_DIR}/rl_shuf.parquet" \
    --eval-parquet "${DATA_DIR}/val.parquet" \
    --sidecar "${DATA_DIR}/rl_shuf.parquet" \
    --save-dir "$RL_SAVE_DIR" \
    --soft-norm-tau "$TAU" \
    --seed "$SEED" \
    --disable-av-thinking \
    --wandb-name "diff_${ARM}${SEED_TAG}_L${LAYER_IDX}" \
    "${RL_RESUME_ARGS[@]}" \
    $EXTRA_RL_ARGS
  exit 0
fi

# 1. Warm-Start SFT (AV + AR, 1 epoch each; eff. batch = 64 on both H200 and H100)
if [[ "$HARDWARE" == "h100" ]]; then
  SFT_BS=16
  SFT_ACCUM=4
  AR_EXTRA_ARGS=(--use-lora --lora-r 128 --lora-alpha 16 --gradient-checkpointing)
else
  SFT_BS=64
  SFT_ACCUM=1
  AR_EXTRA_ARGS=()
fi

has_av_sft() {
  compgen -G "${AV_SFT_DIR}/iter_*/adapter_config.json" > /dev/null 2>&1
}

has_ar_sft() {
  compgen -G "${AR_SFT_DIR}/iter_*/value_head.safetensors" > /dev/null 2>&1 \
    || compgen -G "${AR_SFT_DIR}/iter_*/ar_lora_value_head.safetensors" > /dev/null 2>&1
}

if [[ "$SKIP_SFT" -eq 0 ]]; then
  if [[ "$AUTO_RESUME" -eq 1 ]] && has_av_sft; then
    echo "[phase3/resume] AV SFT already completed under ${AV_SFT_DIR} — skipping."
  else
    run_cmd "$CORE_PYTHON" -m nla.train_sft \
      --mode av \
      --base-ckpt "$TARGET_MODEL" \
      --parquet "${DATA_DIR}/av_sft_train.parquet" \
      --heldout-parquet "${DATA_DIR}/av_sft_val.parquet" \
      --sidecar "${DATA_DIR}/av_sft_train.parquet" \
      --save-dir "$AV_SFT_DIR" \
      --use-lora --lora-r 128 --lora-alpha 16 \
      --batch-size "$SFT_BS" --gradient-accumulation-steps "$SFT_ACCUM" \
      --lr 1e-4 --min-lr 1e-5 --lr-warmup-steps 25 \
      --soft-norm-tau "$TAU" \
      --seed "$SEED" \
      --heldout-every 50 --save-every 500 \
      --wandb-project easynla-diff-qwen3-8b \
      --wandb-name "sft_av_${ARM}${SEED_TAG}"
  fi

  if [[ "$AUTO_RESUME" -eq 1 ]] && has_ar_sft; then
    echo "[phase3/resume] AR SFT already completed under ${AR_SFT_DIR} — skipping."
  else
    run_cmd "$CORE_PYTHON" -m nla.train_sft \
      --mode ar \
      --base-ckpt "$TARGET_MODEL" \
      --parquet "${DATA_DIR}/ar_sft_train.parquet" \
      --heldout-parquet "${DATA_DIR}/av_sft_val.parquet" \
      --sidecar "${DATA_DIR}/ar_sft_train.parquet" \
      --save-dir "$AR_SFT_DIR" \
      --ar-num-layers "$AR_NUM_LAYERS" \
      --batch-size "$SFT_BS" --gradient-accumulation-steps "$SFT_ACCUM" \
      --lr 2e-5 --min-lr 2e-6 --lr-warmup-steps 25 \
      --soft-norm-tau "$TAU" \
      --seed "$SEED" \
      --heldout-every 50 --save-every 500 \
      --wandb-project easynla-diff-qwen3-8b \
      --wandb-name "sft_ar_${ARM}${SEED_TAG}" \
      "${AR_EXTRA_ARGS[@]}"
  fi
fi

# Locate latest SFT checkpoints
if [[ "$DRY_RUN" -eq 1 && ! -d "$AV_SFT_DIR" ]]; then
  AV_LATEST="${AV_SFT_DIR}/iter_FINAL"
  AR_LATEST="${AR_SFT_DIR}/iter_FINAL"
else
  AV_LATEST="$(ls -d "${AV_SFT_DIR}"/iter_* 2>/dev/null | sort | tail -n 1 || echo "${AV_SFT_DIR}/iter_FINAL")"
  AR_LATEST="$(ls -d "${AR_SFT_DIR}"/iter_* 2>/dev/null | sort | tail -n 1 || echo "${AR_SFT_DIR}/iter_FINAL")"
fi

# 2. Merge AV LoRA to bf16 HF (for vLLM rollout serving); AR full-FT is already HF!
AR_NEEDS_MERGE=0
if [[ -f "${AR_LATEST}/ar_meta.json" || ( "$DRY_RUN" -eq 1 && "$HARDWARE" == "h100" ) ]]; then
  AR_NEEDS_MERGE=1
fi

MERGE_ALREADY_DONE=0
if [[ "$AUTO_RESUME" -eq 1 && -f "${MERGED_AV}/config.json" ]]; then
  if [[ "$AR_NEEDS_MERGE" -eq 0 || -f "${MERGED_AR}/value_head.safetensors" ]]; then
    MERGE_ALREADY_DONE=1
  fi
fi

if [[ "$SKIP_MERGE" -eq 0 && "$MERGE_ALREADY_DONE" -eq 0 ]]; then
  if [[ "$AR_NEEDS_MERGE" -eq 1 ]]; then
    # AR was LoRA-trained -> merge both AV and AR
    run_cmd "$CORE_PYTHON" scripts/merge_lora_to_hf.py \
      --base-ckpt "$TARGET_MODEL" \
      --av-dir "$AV_LATEST" --ar-dir "$AR_LATEST" \
      --av-out "$MERGED_AV" --ar-out "$MERGED_AR" \
      --mode both
  else
    # AR was full-FT bf16/fp32 -> only merge AV LoRA; use AR dir directly
    run_cmd "$CORE_PYTHON" scripts/merge_lora_to_hf.py \
      --base-ckpt "$TARGET_MODEL" \
      --av-dir "$AV_LATEST" --ar-dir "$AR_LATEST" \
      --av-out "$MERGED_AV" --ar-out "$MERGED_AR" \
      --mode av
    MERGED_AR="$AR_LATEST"
  fi
else
  if [[ "$MERGE_ALREADY_DONE" -eq 1 ]]; then
    echo "[phase3/resume] Merged HF checkpoints already exist — skipping merge."
  fi
  if [[ ! -d "$MERGED_AR" && -d "$AR_LATEST" ]]; then
    MERGED_AR="$AR_LATEST"
  fi
fi

# 3. On-Policy DP GRPO RL (MUST unset PYTORCH_CUDA_ALLOC_CONF for IPC sync!)
echo "[phase3] Launching GRPO RL (${ARM}) with PYTORCH_CUDA_ALLOC_CONF unset..."
unset PYTORCH_CUDA_ALLOC_CONF || true
VLLM_TORCHRUN="$(dirname "$VLLM_PYTHON")/torchrun"
run_cmd "$VLLM_TORCHRUN" --standalone --nproc_per_node="$NPROC" -m nla.train_rl_vllm \
  --config "$RL_CFG" \
  --base-ckpt "$TARGET_MODEL" \
  --av-ckpt "$MERGED_AV" \
  --av-adapter "$AV_LATEST" \
  --ar-ckpt "$MERGED_AR" \
  --rl-parquet "${DATA_DIR}/rl_shuf.parquet" \
  --eval-parquet "${DATA_DIR}/val.parquet" \
  --sidecar "${DATA_DIR}/rl_shuf.parquet" \
  --save-dir "$RL_SAVE_DIR" \
  --soft-norm-tau "$TAU" \
  --seed "$SEED" \
  --wandb-name "diff_${ARM}${SEED_TAG}_L${LAYER_IDX}" \
  "${RL_RESUME_ARGS[@]}" \
  $EXTRA_RL_ARGS
