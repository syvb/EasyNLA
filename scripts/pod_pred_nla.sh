#!/bin/bash
# The pred-NLA job as it runs ON THE POD. Driven entirely by environment
# variables, which scripts/runpod_pred_nla.py sets from its CLI flags.
#
# Why a committed script rather than a generated one-liner: RunPod's
# docker_args is passed as `bash -lc '<script>'`, so any single quote inside a
# generated script silently truncates it and nothing runs. A file on disk has
# no such constraint, can be read and tested, and is what actually executed.
#
# Stages (comma list in $STAGES): prep, gate, rl, rl_recon, eval.
# Anything written is pushed to the HF dataset repo $HF_REPO; stages that need
# earlier outputs pull them from there, so the chain can span several pods.

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
log() { echo "[pod $(date -u +%H:%M:%S)] $*"; }

: "${STAGES:?}" "${HF_REPO:?}"
: "${AV_CKPT:=syvb/nanonla-qwen3-8b-L24-av}"
: "${AR_CKPT:=syvb/nanonla-qwen3-8b-L24-ar}"
: "${RECON_ADAPTER:=syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0}"
: "${BEHAVIORAL_ADAPTER:=}"
: "${TARGET_CKPT:=Qwen/Qwen3-8B}"
: "${TRAIN_READER:=Qwen/Qwen3-4B-Base}"
: "${HELDOUT_READER:=google/gemma-3-4b-pt}"
: "${SOURCE_REPO:=asher577/nla-rl-data-free8}"
: "${SOURCE_FILE:=rl_shuf.parquet}"
: "${CORPUS_FILTER:=finefineweb}"
: "${N_RL:=12000}" "${N_VAL:=1000}" "${N_EVAL:=2000}"
: "${N_BRANCHES:=4}" "${N_TOKENS:=24}" "${MAX_PER_DOC:=2}" "${BATCH_PREFIXES:=24}"
: "${RL_STEPS:=300}" "${GATE_POSITIONS:=500}" "${EVAL_POSITIONS:=2000}"
: "${MAX_NEW_TOKENS:=192}" "${SEED:=0}" "${MAX_HOURS:=8}"
: "${WANDB_PROJECT:=pred-nla}" "${WANDB_GROUP:=pred-nla-s${SEED}}"
: "${FORCE_RL:=0}" "${KEEP_POD:=1}" "${N_EXAMPLES:=100}" "${CHECK_OVERLAP:=1}"

finish() {
  # The cheapest way to lose a run is to terminate before the upload finishes,
  # so the default is to stay up. --no-keep opts into self-termination.
  log "FINISHED ($1)"
  if [ "$KEEP_POD" = "1" ]; then sleep infinity; fi
  runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  exit 0
}

has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
for s in ${STAGES//,/ }; do
  case "$s" in prep|gate|rl|rl_recon|eval|trunc|trunc_expl|memsft|bon|fvecmp) ;; *) finish "unknown stage '$s'";; esac
done

DATA=/workspace/data; CK=/workspace/ckpts; EV=/workspace/evals; SRC=/workspace/source
POS=$DATA/positions.parquet
mkdir -p "$DATA" "$CK" "$EV" "$SRC"
SYNC="python scripts/pred_hf_sync.py"
READERS="$TRAIN_READER $HELDOUT_READER"


# ---------------------------------------------------------------- data ----
if has prep; then
  # The 11 GB source parquet lives OUTSIDE $DATA on purpose: $DATA is what gets
  # pushed, and every later pod would otherwise download the source again.
  huggingface-cli download "$SOURCE_REPO" --repo-type dataset \
      --include "$SOURCE_FILE*" --local-dir "$SRC" || finish "source download failed"
  timeout "${MAX_HOURS}h" python -m nla.pred.continuations \
      --source-parquet "$SRC/$SOURCE_FILE" --sidecar "$SRC/$SOURCE_FILE" \
      --target-ckpt "$TARGET_CKPT" --out "$POS" \
      --n-rl "$N_RL" --n-val "$N_VAL" --n-eval "$N_EVAL" \
      --n-branches "$N_BRANCHES" --n-tokens "$N_TOKENS" \
      --corpus-filter "$CORPUS_FILTER" --max-per-doc "$MAX_PER_DOC" \
      --batch-prefixes "$BATCH_PREFIXES" --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name prep
  if [ "${CHECK_OVERLAP:-1}" = "1" ]; then
    # Document overlap between the pilot's val/eval positions and the data the
    # SFT and recon_rl checkpoints trained on. Streams the reference parquets;
    # a few minutes, and the only way to know rather than assume.
    python scripts/pred_check_overlap.py --positions "$POS" \
        --reference asher577/easynla-warmstart-data:av_sft_train.parquet \
        --reference asher577/easynla-warmstart-data:ar_sft_train.parquet \
        --reference syvb/nanonla-qwen3-8b-L24-data-full:rl_full.parquet \
        --out "$DATA/positions.overlap.json" || log "overlap check failed (non-fatal)"
  fi
  $SYNC push "$HF_REPO" "$DATA" data
  [ -f "$POS" ] || finish "prep produced no positions"
else
  $SYNC pull "$HF_REPO" data "${DATA}_dl" && cp "${DATA}_dl"/data/positions.parquet* "$DATA/" \
      || finish "could not pull positions from $HF_REPO"
fi

# Retry a push under a timeout; the premise-check stages call this and, if it still fails, keep
# the pod up for 2 h so the results can be copied off by hand rather than die with it.
push_retry() {
  for i in 1 2 3; do
    timeout -k 1m 20m $SYNC push "$HF_REPO" "$1" "$2" && return 0
    log "push of $2 failed (attempt $i)"; sleep 60
  done
  return 1
}

# --------------------------------------------------------------- trunc ----
# Premise check for a truncated-context reader (scripts/trunc_check.py): does the
# target's state hold usable memory of the document beyond its last 20 tokens?
if has trunc; then
  timeout -k 2m "${MAX_HOURS}h" python scripts/trunc_check.py --positions "$POS" \
      --target "$TARGET_CKPT" --reader "$TRAIN_READER" --out "$EV/trunc" \
      --states-dir "$EV/trunc_states" 2>&1 | tee "$EV/trunc.log"
  log "trunc_check exit ${PIPESTATUS[0]}"
  mkdir -p "$EV/trunc" && cp "$EV/trunc.log" "$EV/trunc/"
  # small results first, then the large states file (push_retry: see above)
  RESCUE=0
  push_retry "$EV/trunc" evals/trunc || RESCUE=1
  if [ -d "$EV/trunc_states" ]; then push_retry "$EV/trunc_states" evals/trunc_states || RESCUE=1; fi
  if [ "$RESCUE" = "1" ]; then log "a push failed; keeping the pod up 2 h for manual rescue"; sleep 7200; fi
fi

# ---------------------------------------------------------- trunc_expl ----
# Does the existing SFT verbalizer express the memory trunc found? (scripts/trunc_expl.py)
if has trunc_expl; then
  $SYNC pull "$HF_REPO" evals/trunc_states "${EV}_dl" || finish "could not pull evals/trunc_states"
  timeout -k 2m "${MAX_HOURS}h" python scripts/trunc_expl.py --positions "$POS" \
      --states "${EV}_dl/evals/trunc_states/states.npz" --av "$AV_CKPT" --reader "$TRAIN_READER" \
      --out "$EV/trunc_expl" 2>&1 | tee "$EV/trunc_expl.log"
  log "trunc_expl exit ${PIPESTATUS[0]}"
  mkdir -p "$EV/trunc_expl" && cp "$EV/trunc_expl.log" "$EV/trunc_expl/"
  push_retry "$EV/trunc_expl" evals/trunc_expl \
      || { log "push failed; keeping the pod up 2 h for manual rescue"; sleep 7200; }
fi

# -------------------------------------------------------------- memsft ----
# Supervised ceiling: can a verbalizer TRAINED to name the hidden prefix's keywords do so from the
# full-context state (F) better than from the window-only state (S)? (scripts/memsft.py)
if has memsft; then
  MS=/workspace/memsft
  mkdir -p "$EV/memsft"
  msrun() { timeout -k 2m "${MAX_HOURS}h" "$@" 2>&1 | tee -a "$EV/memsft/memsft.log"; return ${PIPESTATUS[0]}; }
  huggingface-cli download syvb/nanonla-qwen3-8b-L24-data-full --repo-type dataset \
      --include "*_full.parquet*" --local-dir "$SRC/nla8b" || finish "could not download the NLA parquets"
  $SYNC pull "$HF_REPO" evals/trunc_states "${EV}_dl" || finish "could not pull evals/trunc_states"
  if msrun python scripts/memsft.py build \
        --sources "$SRC/nla8b/rl_full.parquet" "$SRC/nla8b/ar_sft_full.parquet" \
        --prompt-source "$SRC/nla8b/av_sft_full.parquet" --sidecar-source "$SRC/nla8b/av_sft_full.parquet" \
        --positions "$POS" --target "$TARGET_CKPT" --out "$MS/data"; then
    for H in F S; do
      msrun python -m nla.train_sft --mode av --base-ckpt "$AV_CKPT" --parquet "$MS/data/${H}_train.parquet" \
          --save-dir "$MS/ckpt_$H" --use-lora --lora-r 128 --lora-alpha 16 --save-every 100000 \
          --seed "$SEED" --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name "memsft_$H" \
        || { log "memsft: training $H failed"; break; }
    done
  else
    log "memsft: build failed"
  fi
  F_AD=$(ls -d "$MS"/ckpt_F/iter_* 2>/dev/null | sort | tail -1)
  S_AD=$(ls -d "$MS"/ckpt_S/iter_* 2>/dev/null | sort | tail -1)
  if [ -n "$F_AD" ] && [ -n "$S_AD" ]; then
    msrun python scripts/memsft.py eval --positions "$POS" --states "${EV}_dl/evals/trunc_states/states.npz" \
        --av "$AV_CKPT" --f-adapter "$F_AD" --s-adapter "$S_AD" --reader "$TRAIN_READER" \
        --build-json "$MS/data/build.json" --out "$EV/memsft" || log "memsft: eval failed"
  else
    log "memsft: missing adapter(s) (F='$F_AD' S='$S_AD'); skipping eval"
  fi
  cp "$MS/data/build.json" "$EV/memsft/" 2>/dev/null
  RESCUE=0
  push_retry "$EV/memsft" evals/memsft || RESCUE=1
  if [ -n "$F_AD" ]; then push_retry "$F_AD" ckpts/memsft_F || RESCUE=1; fi
  if [ -n "$S_AD" ]; then push_retry "$S_AD" ckpts/memsft_S || RESCUE=1; fi
  if [ "$RESCUE" = "1" ]; then log "a push failed; keeping the pod up 2 h for manual rescue"; sleep 7200; fi
fi

# ----------------------------------------------------------------- bon ----
# Best-of-N from the memsft heads: headroom for RL near the SFT policy? (scripts/bon.py)
if has bon; then
  $SYNC pull "$HF_REPO" evals/trunc_states "${EV}_dl" || finish "could not pull evals/trunc_states"
  $SYNC pull "$HF_REPO" ckpts/memsft_F "${CK}_dl" || finish "could not pull ckpts/memsft_F"
  $SYNC pull "$HF_REPO" ckpts/memsft_S "${CK}_dl" || finish "could not pull ckpts/memsft_S"
  mkdir -p "$EV/bon"
  timeout -k 2m "${MAX_HOURS}h" python scripts/bon.py --positions "$POS" \
      --states "${EV}_dl/evals/trunc_states/states.npz" --av "$AV_CKPT" \
      --f-adapter "${CK}_dl/ckpts/memsft_F" --s-adapter "${CK}_dl/ckpts/memsft_S" \
      --reader "$TRAIN_READER" --out "$EV/bon" 2>&1 | tee "$EV/bon/bon.log"
  log "bon exit ${PIPESTATUS[0]}"
  push_retry "$EV/bon" evals/bon || { log "push failed; keeping the pod up 2 h for manual rescue"; sleep 7200; }
fi

# -------------------------------------------------------------- fvecmp ----
# Do normal NLA explanations reconstruct better than a text-only description? (scripts/fve_cmp.py)
if has fvecmp; then
  huggingface-cli download asher577/easynla-warmstart-data --repo-type dataset \
      --include "av_sft_val.parquet*" --local-dir "$SRC/ws" || finish "could not download av_sft_val"
  huggingface-cli download syvb/nanonla-qwen3-8b-L24-data-full --repo-type dataset \
      --include "av_sft_full.parquet" "ar_sft_full.parquet" --local-dir "$SRC/nla8b" || finish "could not download training parquets"
  mkdir -p "$EV/fvecmp"
  timeout -k 2m "${MAX_HOURS}h" python scripts/fve_cmp.py --val "$SRC/ws/av_sft_val.parquet" \
      --exclude "$SRC/nla8b/av_sft_full.parquet" "$SRC/nla8b/ar_sft_full.parquet" \
      --av "$AV_CKPT" --ar "$AR_CKPT" --rl-adapter "$RECON_ADAPTER" --out "$EV/fvecmp" \
      --rl-adapters "p0.0=syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0" "p0.001=syvb/nanonla-qwen3-8b-L24-rl-lora#p0.001" \
      2>&1 | tee "$EV/fvecmp/fvecmp.log"
  log "fvecmp exit ${PIPESTATUS[0]}"
  push_retry "$EV/fvecmp" evals/fvecmp || { log "push failed; keeping the pod up 2 h for manual rescue"; sleep 7200; }
fi

# ---------------------------------------------------------------- gate ----
if has gate; then
  python -m nla.pred.gate --positions "$POS" --base-ckpt "$AV_CKPT" \
      --checkpoint "sft=" ${RECON_ADAPTER:+--checkpoint "recon_rl=$RECON_ADAPTER"} \
      --readers $READERS --n-positions "$GATE_POSITIONS" \
      --branches "$N_BRANCHES" --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED" \
      --out-dir "$EV/gate" --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" \
      --wandb-name gate
  GATE=$?
  $SYNC push "$HF_REPO" "$EV/gate" evals/gate
  if [ "$GATE" -ne 0 ]; then
    if [ "$FORCE_RL" = "1" ]; then
      log "gate exit $GATE - continuing because FORCE_RL=1"
    else
      finish "GATE FAILED (exit $GATE) - stopping before RL"
    fi
  fi
fi

# ---------------------------------------------------------------- rl ----
if has rl; then
  timeout "${MAX_HOURS}h" python -m nla.pred.train_rl \
      --config configs/pred/rl_behavioral.yaml --positions "$POS" \
      --base-ckpt "$AV_CKPT" --reader "$TRAIN_READER" \
      --save-dir "$CK/behavioral_rl" --num-steps "$RL_STEPS" --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name behavioral_rl
  $SYNC push "$HF_REPO" "$CK/behavioral_rl" ckpts/behavioral_rl
fi

if has rl_recon; then
  timeout "${MAX_HOURS}h" python -m nla.pred.train_rl \
      --config configs/pred/rl_recon_matched.yaml --positions "$POS" \
      --base-ckpt "$AV_CKPT" --ar-ckpt "$AR_CKPT" \
      --save-dir "$CK/recon_matched" --num-steps "$RL_STEPS" --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name recon_matched
  $SYNC push "$HF_REPO" "$CK/recon_matched" ckpts/recon_matched
fi

# ---------------------------------------------------------------- eval ----
if has eval; then
  CKPTS=(--checkpoint "sft=")
  [ -n "$RECON_ADAPTER" ] && CKPTS+=(--checkpoint "recon_rl=$RECON_ADAPTER")

  # The behavioral checkpoint: trained on this pod, named explicitly, or pulled
  # from the HF mirror (a separate eval pod). Prefer the checkpoint the training
  # reader scored best (train_rl records one that exists on disk).
  if has rl || [ -n "$BEHAVIORAL_ADAPTER" ]; then
    if [ -n "$BEHAVIORAL_ADAPTER" ]; then
      B="$BEHAVIORAL_ADAPTER"
    else
      B="$CK/behavioral_rl/$(python scripts/pred_best_ckpt.py "$CK/behavioral_rl")"
    fi
  else
    $SYNC pull "$HF_REPO" ckpts/behavioral_rl "${CK}_dl" \
      && B="${CK}_dl/ckpts/behavioral_rl/$(python scripts/pred_best_ckpt.py "${CK}_dl/ckpts/behavioral_rl")" \
      || B=""
  fi
  [ -n "$B" ] && CKPTS+=(--checkpoint "behavioral_rl=$B")
  if has rl_recon; then
    CKPTS+=(--checkpoint "recon_matched=$CK/recon_matched/$(python scripts/pred_best_ckpt.py "$CK/recon_matched")")
  fi
  log "eval checkpoints: ${CKPTS[*]}"

  python -m nla.pred.eval --positions "$POS" --base-ckpt "$AV_CKPT" "${CKPTS[@]}" \
      --readers $READERS --n-positions "$EVAL_POSITIONS" \
      --branches "$N_BRANCHES" --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED" \
      --out-dir "$EV/final" --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" \
      --wandb-name eval
  python -m nla.pred.report --eval-dir "$EV/final" --blind --n-examples "$N_EXAMPLES"
  $SYNC push "$HF_REPO" "$EV/final" evals/final
fi

finish "all stages done"
