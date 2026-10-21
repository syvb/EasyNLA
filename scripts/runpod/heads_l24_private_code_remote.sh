#!/usr/bin/env bash
# Private-code test (scripts/heads_private_code_test.py score) — executes ON the pod. Launch with:
#   GPUS="NVIDIA A100-SXM4-80GB|1.5|2.2;NVIDIA A100 80GB PCIe|1.5|2.2;NVIDIA H100 80GB HBM3|3.5|4.5" \
#   scripts/runpod/launch_with_retry.sh <out dir> heads-l24-private-code \
#       --remote-script scripts/runpod/heads_l24_private_code_remote.sh --secrets hf \
#       --fetch results --no-checkpoints --max-hours 1.5 --avoid-machine bdh8atjj57i7
# Needs private_code_data/ (from `heads_private_code_test.py prep`) inside the uploaded repo. Pulls the RL
# checkpoint's co-trained AR critic and the SFT AR from HF, scores every variant with both, plus occlusion.
set -eEuo pipefail

REPO=/workspace/EasyNLA
CK=/workspace/ckpt
RES=/workspace/results
BASE=Qwen/Qwen3-8B
HF_PREFIX=${HF_PREFIX:-syvb/head-nla-qwen3-8b-l24-std}
CKPT_ITER=${CKPT_ITER:-iter_000400}
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_PREFIX CKPT_ITER
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate bitsandbytes
test -f private_code_data/variants.parquet

stage download
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok, it, pre = os.environ["HF_TOKEN"], os.environ["CKPT_ITER"], os.environ["HF_PREFIX"]
snapshot_download(f"{pre}-rl-ckpts", allow_patterns=[f"{it}/critic/*"], local_dir="/workspace/ckpt/rl", token=tok)
snapshot_download(f"{pre}-ar", local_dir="/workspace/ckpt/sft_ar", token=tok)
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
PY
ls -la "$CK/rl/$CKPT_ITER/critic" "$CK/sft_ar"

stage score
python scripts/heads_private_code_test.py score --data private_code_data --rl-critic "$CK/rl/$CKPT_ITER/critic" \
    --sft-ar "$CK/sft_ar" --out "$RES" --base "$BASE" --batch-size 64

date -u > "$RES/DONE"
echo "=== all stages done ==="
