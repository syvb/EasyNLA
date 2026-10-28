#!/usr/bin/env bash
# Standard (residual-stream) NLA explanations for every token of the chat-viewer chats and the 4 web doc-viewer documents
# (scripts/resid_nla_viewer_job.py) — executes ON the pod. Launch with:
#   GPUS="NVIDIA H100 80GB HBM3|3.5|4.5" scripts/runpod/launch_with_retry.sh <out dir> resid-nla-viewer \
#       --remote-script scripts/runpod/resid_nla_viewer_remote.sh --secrets hf --fetch results --no-checkpoints \
#       --max-hours 2 --avoid-machine bdh8atjj57i7
# The NLA: nanoNLA Qwen3-8B L24 (syvb/nanonla-qwen3-8b-L24-av + RL LoRA p0.0 = no length penalty, frozen AR
# syvb/nanonla-qwen3-8b-L24-ar).
# Outputs go to the HF dataset next to each document: l24/chat_viewer/<chat>/resid.parquet and
# l24/resid_nla/doc_viewer/<doc>/resid.parquet.
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

stage download
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset", local_dir="/workspace/hfdata", token=tok,
                  allow_patterns=["l24/chat_viewer/*/token_ids.npy", "l24/std_rl/doc_viewer/*/token_ids.npy"])
snapshot_download("syvb/nanonla-qwen3-8b-L24-data-full", repo_type="dataset", local_dir="/workspace/nanonla_data", token=tok,
                  allow_patterns=["rl_full.parquet", "rl_full.parquet.nla_meta.yaml"])
snapshot_download("syvb/nanonla-qwen3-8b-L24-av", local_dir="/workspace/ckpt/av", token=tok)
snapshot_download("syvb/nanonla-qwen3-8b-L24-ar", local_dir="/workspace/ckpt/ar", token=tok)
snapshot_download("syvb/nanonla-qwen3-8b-L24-rl-lora", allow_patterns=["p0.0/*"], local_dir="/workspace/ckpt/rl", token=tok)
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
PY
ls /workspace/hfdata/l24/chat_viewer /workspace/hfdata/l24/std_rl/doc_viewer /workspace/ckpt/*

stage resid
python scripts/resid_nla_viewer_job.py \
    --docs /workspace/hfdata/l24/chat_viewer=l24/chat_viewer \
    --docs /workspace/hfdata/l24/std_rl/doc_viewer=l24/resid_nla/doc_viewer \
    --out "$RES/resid" --av-base /workspace/ckpt/av --av-lora /workspace/ckpt/rl/p0.0 --ar /workspace/ckpt/ar \
    --rl-data /workspace/nanonla_data/rl_full.parquet --max-new-tokens 1024 --hf-repo syvb/head-nla-qwen3-8b

date -u > "$RES/DONE"
echo "=== all stages done ==="
