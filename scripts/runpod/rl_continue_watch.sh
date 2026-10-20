#!/usr/bin/env bash
# Continue an RL run past its --max-hours stop — executes ON the pod, beside a PAUSED pipeline script.
#   rl_continue_watch.sh <trainer pid> <trainer cmdline file (NUL- or newline-separated argv)> <wandb run id>
# Waits for the trainer to exit (it stays a zombie while its parent pipeline is SIGSTOPped). If it
# stopped on its time cap ("[time] --max-hours ... stopping"), relaunches the same argv with
# --resume-from-lora <newest iter> --max-hours 0 --stop-file $STOP_FILE, logging into the same
# wandb run and appending to the pipeline's logs, so the HF uploader and the best-checkpoint pick
# see the continuation. Any other exit: does nothing (the operator decides). To stop the
# continuation cleanly: touch $STOP_FILE (it saves at the end of the current step, then exits);
# then SIGCONT the pipeline for the checks.
set -uo pipefail
PID=$1; CMDFILE=$2; RUN_ID=$3
RES=${RES:-/workspace/results}
CK=${CK:-/workspace/ckpt/rl}
STOP_FILE=${STOP_FILE:-/workspace/STOP_RL}
log() { echo "[continue $(date -u +%H:%M:%S)] $*" | tee -a "$RES/run.log"; }

while [ -e "/proc/$PID" ] && ! grep -q "^State:.*Z" "/proc/$PID/status" 2>/dev/null; do sleep 15; done
log "trainer $PID exited"
if ! tail -n 50 "$RES/rl_train.log" | grep -q "\[time\] --max-hours .* stopping"; then
  log "no time-cap stop in the log: not continuing (operator decides)"
  exit 0
fi
LATEST=$(ls -d "$CK"/iter_* | sort | tail -1)
log "time-cap stop; resuming from $LATEST (wandb run $RUN_ID, stop file $STOP_FILE)"
mapfile -t ARGV < <(tr '\0' '\n' < "$CMDFILE" | sed '/^$/d')
cd "${REPO_DIR:-/workspace/EasyNLA}"
set -a; source "${SECRETS:-/workspace/.secrets}"; set +a
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_RUN_ID="$RUN_ID" WANDB_RESUME=allow
unset PYTORCH_CUDA_ALLOC_CONF
"${ARGV[@]}" --resume-from-lora "$LATEST" --max-hours 0 --stop-file "$STOP_FILE" 2>&1 \
  | tee -a "$RES/rl_train.log" >> "$RES/run.log"
log "continuation exited (rc ${PIPESTATUS[0]})"
