#!/usr/bin/env bash
# Quick head-contribution NLA viability run — executes ON the RunPod pod.
# Started by scripts/runpod/launch_heads_quick_sft.py, which uploads the repo to
# /workspace/EasyNLA and secrets to /workspace/.secrets, then polls the markers:
#   /workspace/results/DONE    every stage finished
#   /workspace/results/FAILED  first failing stage (contents) — the run stops there
#
# Stages: install → corpus/extract/stats/select/prepare → Haiku descriptions (standard
# API, PAID) alongside the 8B self-description comparison → export → AR SFT → AV SFT →
# pre-RL checks. Logs go to /workspace/results/run.log.
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (pip_install)

REPO=/workspace/EasyNLA
OUT=/workspace/heads_q
SFT=$OUT/sft
CK=/workspace/ckpt
RES=/workspace/results
CFG=configs/datagen/heads_qwen3_8b_quick.yaml
BASE=Qwen/Qwen3-8B
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, ANTHROPIC_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__, 'peft', peft.__version__)"

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats,select,prepare

# The 8B self-description comparison (GPU + a few $0.1 of Claude) overlaps the Haiku
# descriptions (API only); both finish before SFT needs the GPU.
stage compare_start
timeout 60m python scripts/heads_compare_descriptions.py --output-dir "$OUT" --work-dir "$RES/compare" \
    --n-heads 8 --per-head 300 --claude --device-map cuda:0 --batch-size 64 > "$RES/compare.log" 2>&1 &
COMPARE_PID=$!

# Org limits (checked): 10k requests/min, 10M input / 2M output tokens/min — 96 in flight is ~2k RPM.
# The deadline ends the stage with whatever came back; `timeout` is only the hang backstop.
stage descriptions
timeout 75m python -m nla.datagen.heads.batch_api run-sync --output-dir "$OUT" --confirm-spend \
    --concurrency 96 --deadline-minutes 60
python -m nla.datagen.heads.batch_api retry --output-dir "$OUT"
timeout 25m python -m nla.datagen.heads.batch_api run-sync --output-dir "$OUT" --confirm-spend \
    --concurrency 32 --deadline-minutes 15

stage compare_wait
if wait "$COMPARE_PID"; then echo "comparison done"; else echo "comparison FAILED (see compare.log) — continuing"; fi
tail -40 "$RES/compare.log" || true

stage export
python -m nla.datagen.heads.export_sft --output-dir "$OUT" --sft-dir "$SFT"
steps() {   # epochs × ceil(rows / batch)
  python -c "import math, pyarrow.parquet as pq; print($2 * math.ceil(pq.ParquetFile('$1').metadata.num_rows / $3))"
}

stage ar_sft
python -m nla.train_sft --mode ar --base-ckpt "$BASE" --parquet "$SFT/ar_sft.parquet" --save-dir "$CK/ar" \
    --use-lora --ar-raw-loss --value-head-lr-mult 10 --ar-num-layers 25 \
    --batch-size 64 --num-steps "$(steps "$SFT/ar_sft.parquet" 2 64)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 --no-wandb --seed 0

stage av_sft
python -m nla.train_sft --mode av --base-ckpt "$BASE" --parquet "$SFT/av_sft.parquet" --save-dir "$CK/av" \
    --use-lora --inject-preserve-magnitude \
    --batch-size 32 --num-steps "$(steps "$SFT/av_sft.parquet" 2 32)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 --no-wandb --seed 0 \
    --heldout-parquet "$SFT/eval.parquet" --heldout-rows 4000 --heldout-every 200 \
    --sample-every 200 --n-samples 4 --sample-max-new-tokens 64

stage checks
AV_CK=$(ls -d "$CK"/av/iter_* | sort | tail -1)
AR_CK=$(ls -d "$CK"/ar/iter_* | sort | tail -1)
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$AV_CK" --ar-ckpt "$AR_CK" \
    --out-dir "$RES/checks" --batch-size 64 --max-new-tokens 96

stage collect
cp "$OUT/warmstart/descriptions.parquet" "$OUT/warmstart/selection.parquet" "$OUT/warmstart/requests/prepare.json" \
    "$OUT/head_meta.yaml" "$OUT/head_stats.safetensors" "$RES/"
du -sh "$OUT" "$SFT" "$CK" || true
date -u > "$RES/DONE"
echo "=== all stages done ==="
