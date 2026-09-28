#!/bin/bash
# The KL-NLA job as it runs ON THE POD (docs/kl_nla.md). Driven entirely by
# environment variables, which scripts/runpod_kl_nla.py sets from its flags.
# A committed file rather than a generated one-liner: RunPod passes docker_args
# as `bash -lc '<script>'`, so a single quote in a generated script silently
# truncates it.
#
# Stages (comma list in $STAGES):
#   audit0  Phase 0: KL audit of the SFT AR; exits the pod if a gate fails
#   ar_kl   Phase 1: continue the SFT AR on KL      (LoRA, one epoch of ar_sft_full)
#   ar_mse  Phase 1: continue the SFT AR on MSE     (matched control)
#   audit1  Phase 1: KL audit of sft / ar_kl / ar_mse
#   hedge   hedging control: no-info baselines + cross-fit shrinkage (kl_hedge_control.py)
#   rl_kl   RL pilot arm: reward -KL, AR co-trained on KL   (docs/kl_nla_phase2.md)
#   rl_mse  RL pilot arm: reward -MSE, AR co-trained on MSE
#           each RL stage then scores every saved checkpoint with frozen judges (kl_curve_eval.py)
# Everything is pushed to the HF dataset repo $HF_REPO as it is produced;
# stages that need earlier outputs pull them from there.

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
log() { echo "[pod $(date -u +%H:%M:%S)] $*"; }

: "${STAGES:?}" "${HF_REPO:?}"
: "${AR_CKPT:=syvb/nanonla-qwen3-8b-L24-ar}"
: "${TARGET_CKPT:=Qwen/Qwen3-8B}"
: "${EXPL_REPO:=syvb/pred-nla-qwen3-8b}"
: "${AR_LR:=5e-5}" "${LORA_R:=128}" "${AR_BATCH:=16}" "${AR_ACCUM:=4}" "${AR_STEPS:=782}"
: "${KL_MICRO_BATCH:=8}" "${SEED:=0}" "${MAX_HOURS:=6}" "${KEEP_POD:=1}"
: "${WANDB_PROJECT:=kl-nla}" "${WANDB_GROUP:=kl-nla-phase1}"
: "${AV_CKPT:=syvb/nanonla-qwen3-8b-L24-av}" "${RL_WANDB_GROUP:=kl-nla-rl-pilot}"
: "${RL_STEPS:=100}" "${RL_BATCH:=32}" "${RL_GROUP:=8}" "${RL_EVAL_EVERY:=10}" "${RL_SAVE_EVERY:=20}"
: "${VLLM_GPU_MEM:=0.30}" "${CURVE_N:=500}"

finish() {
  log "FINISHED ($1)"
  if [ "$KEEP_POD" = "1" ]; then sleep infinity; fi
  runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  exit 0
}
has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
for s in ${STAGES//,/ }; do
  case "$s" in audit0|ar_kl|ar_mse|audit1|hedge|rl_kl|rl_mse) ;; *) finish "unknown stage '$s'";; esac
done

# Whole-pod watchdog: whatever hangs, the pod does not outlive MAX_HOURS.
( sleep $((MAX_HOURS * 3600)); log "WATCHDOG: ${MAX_HOURS} h reached"; \
  runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null ) &

CK=/workspace/ckpts; EV=/workspace/evals; SRC=/workspace/source
mkdir -p "$CK" "$EV" "$SRC"
SYNC="python scripts/hf_sync.py"
push_retry() {
  for i in 1 2 3; do
    timeout -k 1m 30m $SYNC push "$HF_REPO" "$1" "$2" && return 0
    log "push of $2 failed (attempt $i)"; sleep 60
  done
  log "push of $2 FAILED; keeping the pod up 2 h for manual rescue"; sleep 7200; return 1
}

# ---------------------------------------------------------------- inputs ----
# The AR must be a LOCAL dir: train_sft only treats --base-ckpt as a prepared
# critic when <dir>/value_head.safetensors exists (a repo id would silently be
# re-truncated from scratch with an identity head).
huggingface-cli download "$AR_CKPT" --local-dir "$SRC/ar" >/dev/null || finish "AR download failed"
[ -f "$SRC/ar/value_head.safetensors" ] || finish "AR has no value_head.safetensors"
huggingface-cli download asher577/easynla-warmstart-data --repo-type dataset \
    --include "av_sft_val.parquet*" --local-dir "$SRC/ws" >/dev/null || finish "val download failed"
huggingface-cli download syvb/nanonla-qwen3-8b-L24-data-full --repo-type dataset \
    --include "av_sft_full.parquet*" "ar_sft_full.parquet*" --local-dir "$SRC/nla8b" >/dev/null \
    || finish "training parquet download failed"
huggingface-cli download "$EXPL_REPO" evals/fvecmp/explanations.json --repo-type dataset \
    --local-dir "$SRC/expl" >/dev/null || finish "explanations download failed"
huggingface-cli download "$TARGET_CKPT" --exclude "*.pth" >/dev/null || finish "target download failed"
AUDIT_ARGS=(--val "$SRC/ws/av_sft_val.parquet"
            --exclude "$SRC/nla8b/av_sft_full.parquet" "$SRC/nla8b/ar_sft_full.parquet"
            --explanations "$SRC/expl/evals/fvecmp/explanations.json"
            --target "$TARGET_CKPT" --micro-batch "$KL_MICRO_BATCH")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- audit0 ----
if has audit0; then
  mkdir -p "$EV/audit0"
  timeout -k 2m "${MAX_HOURS}h" python scripts/kl_audit.py "${AUDIT_ARGS[@]}" \
      --ar "sft=$SRC/ar" --out "$EV/audit0" 2>&1 | tee "$EV/audit0/audit0.log"
  RC=${PIPESTATUS[0]}
  log "audit0 exit $RC"
  push_retry "$EV/audit0" evals/audit0
  [ "$RC" = "0" ] || finish "audit0 failed or a gate failed (exit $RC) - stopping before training"
fi

# ---------------------------------------------------------------- phase 1 ----
train_ar() {  # $1 = name, $2 = recon loss
  mkdir -p "$CK/$1"
  timeout -k 2m "${MAX_HOURS}h" python -m nla.train_sft --mode ar \
      --base-ckpt "$SRC/ar" --parquet "$SRC/nla8b/ar_sft_full.parquet" \
      --heldout-parquet "$SRC/ws/av_sft_val.parquet" --heldout-rows 500 --heldout-every 200 \
      --save-dir "$CK/$1" --save-every 100000 --num-steps "$AR_STEPS" \
      --use-lora --lora-r "$LORA_R" --lr "$AR_LR" --seed "$SEED" \
      --batch-size "$AR_BATCH" --gradient-accumulation-steps "$AR_ACCUM" \
      --recon-loss "$2" --kl-micro-batch "$KL_MICRO_BATCH" \
      --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name "$1" \
      2>&1 | tee "$CK/$1/train.log"
  RC=${PIPESTATUS[0]}
  log "$1 exit $RC"
  push_retry "$CK/$1" "ckpts/$1"
  [ "$RC" = "0" ] || finish "$1 training failed (exit $RC)"
}
has ar_kl && train_ar ar_kl kl
has ar_mse && train_ar ar_mse mse

# ---------------------------------------------------------------- audit1 ----
pull_ars() {   # Phase 1 ARs: trained on this pod, else pulled from HF; sets KLD / MSED
  for n in ar_kl ar_mse; do
    if [ ! -d "$CK/$n" ]; then
      $SYNC pull "$HF_REPO" "ckpts/$n" "${CK}_dl" && mkdir -p "$CK/$n" \
          && cp -r "${CK}_dl/ckpts/$n/." "$CK/$n/" || finish "could not pull ckpts/$n"
    fi
  done
  KLD=$(ls -d "$CK"/ar_kl/iter_* | tail -1); MSED=$(ls -d "$CK"/ar_mse/iter_* | tail -1)
}
if has audit1; then
  pull_ars
  log "audit1: ar_kl=$KLD ar_mse=$MSED"
  mkdir -p "$EV/audit1"
  timeout -k 2m "${MAX_HOURS}h" python scripts/kl_audit.py "${AUDIT_ARGS[@]}" \
      --ar "sft=$SRC/ar" --ar "kl=$SRC/ar:$KLD" --ar "mse=$SRC/ar:$MSED" \
      --out "$EV/audit1" 2>&1 | tee "$EV/audit1/audit1.log"
  log "audit1 exit ${PIPESTATUS[0]}"
  push_retry "$EV/audit1" evals/audit1
fi

# ----------------------------------------------------------------- hedge ----
if has hedge; then
  pull_ars
  log "hedge: ar_kl=$KLD ar_mse=$MSED"
  mkdir -p "$EV/hedge"
  timeout -k 2m "${MAX_HOURS}h" python scripts/kl_hedge_control.py \
      --val "$SRC/ws/av_sft_val.parquet" \
      --exclude "$SRC/nla8b/av_sft_full.parquet" "$SRC/nla8b/ar_sft_full.parquet" \
      --explanations "$SRC/expl/evals/fvecmp/explanations.json" \
      --target "$TARGET_CKPT" --micro-batch "$KL_MICRO_BATCH" \
      --ar-kl "$SRC/ar:$KLD" --ar-mse "$SRC/ar:$MSED" \
      --out "$EV/hedge" 2>&1 | tee "$EV/hedge/hedge.log"
  log "hedge exit ${PIPESTATUS[0]}"
  push_retry "$EV/hedge" evals/hedge
fi

# -------------------------------------------------------------------- rl ----
# One arm per pod. The AR is ALWAYS co-trained with the AV (--train-critic --ar-lora),
# never frozen. Everything goes to HF as it lands and to W&B group $RL_WANDB_GROUP.
run_rl() {  # $1 = kl | mse
  local ARM=$1 NAME=rl_$1 VENV=/workspace/vllm-venv P1
  mkdir -p "$CK/$NAME"
  # setup failures push their logs before the pod goes away
  fail_rl() { log "$1"; push_retry "$CK/$NAME" "ckpts/$NAME"; finish "$1"; }
  pull_ars
  if [ "$ARM" = kl ]; then P1=$KLD; else P1=$MSED; fi
  # rollout env: pinned vllm 0.19.0 + vllm-lens 1.1.0 + the injection patch (docs/vllm-lens-setup.md)
  pip install -q uv || fail_rl "uv install failed"
  bash scripts/install_vllm_lens.sh "$VENV" > "$CK/$NAME/vllm_install.log" 2>&1
  log "vllm-lens install exit $? (log: ckpts/$NAME/vllm_install.log)"
  [ -x "$VENV/bin/python" ] || fail_rl "vllm-lens venv build failed"
  uv pip install -q --python "$VENV/bin/python" pyarrow pyyaml orjson httpx tqdm safetensors \
      huggingface_hub >> "$CK/$NAME/vllm_install.log" 2>&1 || fail_rl "venv deps failed"
  uv pip install -q --python "$VENV/bin/python" --no-deps -e . >> "$CK/$NAME/vllm_install.log" 2>&1 \
      || fail_rl "venv repo install failed"
  # the patch is idempotent; re-run it and CHECK the hunks the trainer requires
  "$VENV/bin/python" utils/patch_vllm_lens.py 2>&1 | tee -a "$CK/$NAME/vllm_install.log"
  local VW
  VW=$("$VENV/bin/python" -c "import importlib.util as u; print(u.find_spec('vllm_lens._worker_ext').origin)")
  for m in _meta5 get_and_reset_steer_log log_key=per_req_log_key get_and_reset_steer_count; do
    grep -q "$m" "$VW" || fail_rl "vllm-lens patch incomplete: '$m' missing in $VW"
  done
  log "vllm-lens patch verified ($VW)"
  huggingface-cli download "$AV_CKPT" --local-dir "$SRC/av" >/dev/null || fail_rl "AV download failed"
  huggingface-cli download syvb/nanonla-qwen3-8b-L24-data-full --repo-type dataset \
      --include "rl_full.parquet*" --local-dir "$SRC/nla8b" >/dev/null || fail_rl "rl_full download failed"
  mkdir -p /workspace/data
  python scripts/kl_rl_prep.py --src "$SRC/nla8b/rl_full.parquet" \
      --explanations "$SRC/expl/evals/fvecmp/explanations.json" \
      --out /workspace/data/rl_pilot.parquet || fail_rl "rl data prep failed"
  python scripts/merge_prepared_ar.py --base "$SRC/ar" --lora "$P1" --out "$CK/ar_${ARM}_merged" \
      || fail_rl "AR merge failed"
  push_retry "$CK/ar_${ARM}_merged" "ckpts/ar_${ARM}_merged"
  # push adapters + logs every 20 min while training (the merged co-trained AR at the end)
  ( while sleep 1200; do
      timeout -k 1m 20m $SYNC push "$HF_REPO" "$CK/$NAME" "ckpts/$NAME" \
          --ignore "optim_latest.pt,*.tmp,critic_latest*" >/dev/null 2>&1 && log "periodic push of ckpts/$NAME"
    done ) &
  local PUSHER=$!
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 60 > "$CK/$NAME/gpu_mem.log" 2>&1 &
  local SMI=$!
  env -u PYTORCH_CUDA_ALLOC_CONF timeout -k 2m "${MAX_HOURS}h" "$VENV/bin/python" -m nla.train_rl_vllm \
      --config configs/rl_vllm.yaml --av-ckpt "$SRC/av" --ar-ckpt "$CK/ar_${ARM}_merged" \
      --rl-parquet /workspace/data/rl_pilot.parquet --sidecar /workspace/data/rl_pilot.parquet \
      --save-dir "$CK/$NAME" --num-steps "$RL_STEPS" --batch-prompts "$RL_BATCH" --group-size "$RL_GROUP" \
      --train-critic --ar-lora --evals base_fve --eval-every "$RL_EVAL_EVERY" --eval-n-prompts 128 \
      --save-every "$RL_SAVE_EVERY" --val-rows 2000 --vllm-gpu-mem "$VLLM_GPU_MEM" \
      --recon-loss "$ARM" --kl-micro-batch "$KL_MICRO_BATCH" --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-group "$RL_WANDB_GROUP" --wandb-name "$NAME" \
      2>&1 | tee "$CK/$NAME/train.log"
  local RC=${PIPESTATUS[0]}
  kill "$PUSHER" "$SMI" 2>/dev/null
  log "$NAME exit $RC"
  push_retry "$CK/$NAME" "ckpts/$NAME"
  # checkpoint curves with the FROZEN Phase 1 judges (also when training died part-way)
  if ls -d "$CK/$NAME"/iter_* >/dev/null 2>&1; then
    mkdir -p "$EV/curves_$NAME"
    timeout -k 2m 2h python scripts/kl_curve_eval.py \
        --val "$SRC/ws/av_sft_val.parquet" \
        --exclude "$SRC/nla8b/av_sft_full.parquet" "$SRC/nla8b/ar_sft_full.parquet" \
        --explanations "$SRC/expl/evals/fvecmp/explanations.json" \
        --av "$SRC/av" --run-dir "$CK/$NAME" --ar-kl "$SRC/ar:$KLD" --ar-mse "$SRC/ar:$MSED" \
        --target "$TARGET_CKPT" --n "$CURVE_N" --micro-batch "$KL_MICRO_BATCH" \
        --wandb-project "$WANDB_PROJECT" --wandb-group "$RL_WANDB_GROUP" --wandb-name "curves_$NAME" \
        --out "$EV/curves_$NAME" 2>&1 | tee "$EV/curves_$NAME/curves.log"
    log "curves_$NAME exit ${PIPESTATUS[0]}"
    push_retry "$EV/curves_$NAME" "evals/curves_$NAME"
  fi
  [ "$RC" = "0" ] || finish "$NAME training failed (exit $RC)"
}
has rl_kl && run_rl kl
has rl_mse && run_rl mse

finish "all stages done"
