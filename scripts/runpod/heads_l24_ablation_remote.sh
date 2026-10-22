#!/usr/bin/env bash
# Per-token head ablation effects (scripts/heads_ablation_effects.py) — executes ON the pod. Launch with:
#   scripts/runpod/launch_with_retry.sh <out dir> heads-l24-ablation \
#       --remote-script scripts/runpod/heads_l24_ablation_remote.sh --secrets hf \
#       --fetch results --no-checkpoints --max-hours 2 --avoid-machine bdh8atjj57i7
# Needs ablate_data/<doc>/{token_ids.npy,check_writes.npz} inside the uploaded repo.
set -eEuo pipefail

REPO=/workspace/EasyNLA
RES=/workspace/results
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . accelerate
ls ablate_data

stage download
python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', allow_patterns=['*.json', '*.safetensors', '*.txt'], token=os.environ['HF_TOKEN'])"

stage smoke
python scripts/heads_ablation_effects.py --data ablate_data --out "$RES/smoke" --max-positions 12

stage effects
python scripts/heads_ablation_effects.py --data ablate_data --out "$RES/effects"

date -u > "$RES/DONE"
echo "=== all stages done ==="
