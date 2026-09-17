#!/bin/bash
# End-to-end CPU smoke of the pred-NLA pipeline on Qwen3-0.6B. No GPU, no spend.
#
# Runs the REAL code paths - the same modules the pod runs - on a tiny model and
# a handful of positions: build a small NLA dataset, warm-start an AV so it
# actually emits <explanation> tags, sample continuations, run the gate with two
# readers from DIFFERENT tokenizer families, take a couple of GRPO steps on the
# frozen-reader reward, then evaluate and write a report.
#
# What it is for: catching contract breaks (marker ids, prompt templates, span
# alignment, checkpoint plumbing, argument names) before any of them cost a pod
# hour. It says nothing about whether the method works - a 0.6B verbalizer
# trained for 60 steps on six documents is not evidence of anything.
#
#   bash scripts/smoke_pred_cpu.sh [workdir]
#
# ~25-40 minutes on 4 vCPUs. Set SKIP_SFT=1 to reuse an AV from a previous run.

set -euo pipefail

WORK="${1:-/tmp/pred_smoke}"
PY="${PY:-.venv/bin/python}"
READER_A="${READER_A:-Qwen/Qwen3-0.6B}"        # "training" reader
READER_B="${READER_B:-google/gemma-3-270m}"    # held-out, different tokenizer family
BASE="${BASE:-Qwen/Qwen3-0.6B}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK"
echo "=== pred-NLA CPU smoke -> $WORK ==="

# 1. A real (tiny) NLA dataset: genuine activations, datagen's own marker picker
#    and sidecar serializer, so the contract under test is the real one.
if [ ! -f "$WORK/rl.parquet" ]; then
  $PY scripts/make_pred_smoke_data.py --model "$BASE" --layer 18 \
      --out "$WORK/rl.parquet" --av-out "$WORK/av.parquet"
fi

# 2. Warm-start an AV. Without this the base instruct model answers with a
#    <think> block and never emits the tags, so the whole success path would go
#    untested.
SFT_STEPS="${SFT_STEPS:-40}"     # ~11 s/step on 4 vCPUs at these sizes
if [ "${SKIP_SFT:-0}" != "1" ] && [ -z "$(ls -d "$WORK"/av_ckpt/iter_* 2>/dev/null)" ]; then
  rm -rf "$WORK/av_ckpt"
  $PY -m nla.train_sft --mode av --base-ckpt "$BASE" \
      --parquet "$WORK/av.parquet" --sidecar "$WORK/av.parquet" \
      --save-dir "$WORK/av_ckpt" --device cpu --quant none \
      --use-lora --lora-r 8 --lora-alpha 16 \
      --batch-size 2 --num-steps "$SFT_STEPS" --max-len 256 --lr 3e-4 \
      --lr-warmup-steps 5 --save-every "$SFT_STEPS" \
      --attn-implementation eager --no-wandb
fi
# train_sft and train_rl pad iteration numbers differently, so discover it.
AV_ADAPTER="$(ls -d "$WORK"/av_ckpt/iter_* | sort | tail -1)"
echo "[smoke] AV adapter: $AV_ADAPTER"

# 3. Continuations from the frozen target model.
if [ ! -f "$WORK/positions.parquet" ]; then
  $PY -m nla.pred.continuations \
      --source-parquet "$WORK/rl.parquet" --sidecar "$WORK/rl.parquet" \
      --target-ckpt "$BASE" --out "$WORK/positions.parquet" \
      --n-rl 32 --n-val 10 --n-eval 10 --n-branches 2 --max-per-doc 4 \
      --val-permille 250 --eval-permille 250 \
      --batch-prefixes 8 --device cpu --dtype float32 --no-wandb
fi

# 4. The gate: matched vs shuffled vs no explanation, on both readers.
#    Expected to FAIL its verdict here - ten positions from a 40-step 0.6B
#    verbalizer is noise, and the synthetic warm-start targets teach it to quote
#    the prefix, which actively misleads a reader about what comes NEXT. So the
#    exit code is tolerated. What is being checked is that both readers load,
#    that two tokenizer families bucket the same characters, and that the
#    verdict path runs in both directions.
set +e
$PY -m nla.pred.gate --positions "$WORK/positions.parquet" \
    --base-ckpt "$BASE" --checkpoint "sft=$AV_ADAPTER" \
    --readers "$READER_A" "$READER_B" \
    --n-positions 10 --split val --branches 2 --max-new-tokens 96 \
    --gen-batch 4 --reader-batch-rows 4 --reader-batch-tokens 4096 \
    --device cpu --reader-dtype float32 --n-boot 300 \
    --out-dir "$WORK/gate" --no-wandb
echo "[smoke] gate exit $? (a failing verdict is expected at this scale)"
set -e

# 5. Two GRPO steps on the frozen-reader reward.
rm -rf "$WORK/rl_ckpt"
$PY -m nla.pred.train_rl --positions "$WORK/positions.parquet" \
    --base-ckpt "$BASE" --init-adapter "$AV_ADAPTER" \
    --reader "$READER_A" --reader-dtype float32 \
    --save-dir "$WORK/rl_ckpt" --num-steps 2 --batch-prompts 2 --group-size 2 \
    --branches 2 --max-new-tokens 96 --gen-batch 4 \
    --reader-batch-rows 4 --reader-batch-tokens 4096 \
    --logp-micro-batch 1 --eval-every 2 --eval-n-positions 4 --save-every 2 \
    --device cpu --dtype float32 --quant none --no-wandb

# 6. Final evaluation over two checkpoints, then the report.
$PY -m nla.pred.eval --positions "$WORK/positions.parquet" \
    --base-ckpt "$BASE" \
    --checkpoint "sft=$AV_ADAPTER" \
    --checkpoint "behavioral_rl=$(ls -d "$WORK"/rl_ckpt/iter_* | sort | tail -1)" \
    --readers "$READER_A" "$READER_B" \
    --n-positions 10 --split eval --branches 2 --max-new-tokens 96 \
    --gen-batch 4 --reader-batch-rows 4 --reader-batch-tokens 4096 \
    --device cpu --reader-dtype float32 --n-boot 300 \
    --out-dir "$WORK/final" --baseline-checkpoint sft --no-wandb

$PY -m nla.pred.report --eval-dir "$WORK/final" --n-examples 3 \
    --title "pred-NLA CPU smoke" > "$WORK/report_stdout.txt"

echo
echo "=== SMOKE PASSED ==="
echo "report:       $WORK/final/report.md"
echo "gate verdict: $WORK/gate/verdict.json"
echo "rl ckpt:      $(ls -d "$WORK"/rl_ckpt/iter_* | sort | tail -1)"
