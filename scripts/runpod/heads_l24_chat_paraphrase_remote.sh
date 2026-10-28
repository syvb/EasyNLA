#!/usr/bin/env bash
# Paraphrase test on the chat viewer's sites (scripts/heads_chat_paraphrase_test.py score) — executes ON the pod.
# Needs chat_paraphrase_data/ (from the local `prep` step) in the synced repo. Launch with:
#   GPUS="NVIDIA H100 80GB HBM3|3.5|4.5" scripts/runpod/launch_with_retry.sh <out dir> chat-paraphrase \
#       --remote-script scripts/runpod/heads_l24_chat_paraphrase_remote.sh --secrets hf --fetch results \
#       --no-checkpoints --max-hours 1.5 --avoid-machine bdh8atjj57i7
set -eEuo pipefail

REPO=/workspace/EasyNLA
CK=/workspace/ckpt
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
pip_install -e . scikit-learn accelerate
test -f chat_paraphrase_data/variants.parquet

stage download
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("syvb/head-nla-qwen3-8b-l24-std-rl-ckpts", allow_patterns=["iter_000400/critic/*"], local_dir="/workspace/ckpt/std", token=tok)
snapshot_download("syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts", allow_patterns=["iter_000640/critic/*"], local_dir="/workspace/ckpt/mtry", token=tok)
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
PY
ls "$CK"/std/iter_000400/critic "$CK"/mtry/iter_000640/critic

stage score
python scripts/heads_chat_paraphrase_test.py score --data chat_paraphrase_data --rl-critic "$CK/std/iter_000400/critic" \
    --mtry-critic "$CK/mtry/iter_000640/critic" --out "$RES" --batch-size 64

date -u > "$RES/DONE"
echo "=== all stages done ==="
