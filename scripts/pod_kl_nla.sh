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

finish() {
  log "FINISHED ($1)"
  if [ "$KEEP_POD" = "1" ]; then sleep infinity; fi
  runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  exit 0
}
has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
for s in ${STAGES//,/ }; do
  case "$s" in audit0|ar_kl|ar_mse|audit1) ;; *) finish "unknown stage '$s'";; esac
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
if has audit1; then
  for n in ar_kl ar_mse; do
    if [ ! -d "$CK/$n" ]; then
      $SYNC pull "$HF_REPO" "ckpts/$n" "${CK}_dl" && mkdir -p "$CK/$n" \
          && cp -r "${CK}_dl/ckpts/$n/." "$CK/$n/" || finish "could not pull ckpts/$n"
    fi
  done
  KLD=$(ls -d "$CK"/ar_kl/iter_* | tail -1); MSED=$(ls -d "$CK"/ar_mse/iter_* | tail -1)
  log "audit1: ar_kl=$KLD ar_mse=$MSED"
  mkdir -p "$EV/audit1"
  timeout -k 2m "${MAX_HOURS}h" python scripts/kl_audit.py "${AUDIT_ARGS[@]}" \
      --ar "sft=$SRC/ar" --ar "kl=$SRC/ar:$KLD" --ar "mse=$SRC/ar:$MSED" \
      --out "$EV/audit1" 2>&1 | tee "$EV/audit1/audit1.log"
  log "audit1 exit ${PIPESTATUS[0]}"
  push_retry "$EV/audit1" evals/audit1
fi

finish "all stages done"
