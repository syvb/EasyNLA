#!/usr/bin/env bash
# Soft-token readout sweep on Qwen3-8B — executes ON the RunPod pod (launched with
# scripts/runpod/launch_heads_quick_sft.py --remote-script <this> --hold --secrets hf).
# Extracts head writes, then runs the first sweep; later sweeps run over SSH while held.
#   /workspace/results/DONE | FAILED   markers;  /workspace/results/run.log   log
set -euo pipefail

REPO=/workspace/EasyNLA
OUT=/workspace/heads_st
RES=/workspace/results
CFG=configs/datagen/heads_qwen3_8b_softtoken.yaml
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats

stage sweep1
python scripts/heads_soft_token_sweep.py --output-dir "$OUT" --work-dir "$RES/sweep1" \
    --n-heads 20 --per-head 200 --prompts identity,meaning,assoc,continue --betas 2,4,8,16 \
    --readouts greedy,delta --batch-size 64 --device-map cuda:0 --legacy

date -u > "$RES/DONE"
echo "=== all stages done ==="
