#!/usr/bin/env bash
# Every token × every head of L24 for a queue of documents (scripts/heads_doc_viewer_job.py) — executes ON the
# pod. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_docviewer_remote.sh \
#       --secrets hf,anthropic --fetch results --no-checkpoints --max-hours 6 --out <local dir>
# Initial queue: $REPO/docviewer_queue/*.json (copied to /workspace/queue). More specs can be scp'd into
# /workspace/queue while it runs; touch /workspace/queue/CLOSE to finish once the queue drains (otherwise it
# exits after IDLE_MIN idle minutes). Each finished document is uploaded to the HF dataset right away.
set -eEuo pipefail

REPO=/workspace/EasyNLA
RES=/workspace/results
CK=/workspace/ckpt
HF_PREFIX=${HF_PREFIX:-syvb/head-nla-qwen3-8b-l24-std}
IDLE_MIN=${IDLE_MIN:-20}
mkdir -p "$RES" /workspace/queue
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, ANTHROPIC_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_PREFIX
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate anthropic

stage download
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok, pre = os.environ["HF_TOKEN"], os.environ["HF_PREFIX"]
snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset",
                  allow_patterns=["l24/sft_data/*", "l24/meta/head_meta.yaml", "l24/meta/head_stats.safetensors",
                                  "l24/meta/wo.safetensors", "l24/probe/texts.parquet"], local_dir="/workspace/hfdata", token=tok)
for name in ("av", "ar"):
    snapshot_download(f"{pre}-{name}", local_dir=f"/workspace/ckpt/sft_{name}", token=tok)
snapshot_download(f"{pre}-rl-ckpts", allow_patterns=["iter_000400/*"], ignore_patterns=["*optim_latest.pt"],
                  local_dir="/workspace/ckpt/rl", token=tok)
PY
cp "$REPO"/docviewer_queue/*.json /workspace/queue/
ls /workspace/queue

stage docs
python scripts/heads_doc_viewer_job.py --queue /workspace/queue --out "$RES/docs" \
    --meta /workspace/hfdata/l24/meta --sft-dir /workspace/hfdata/l24/sft_data \
    --texts-ref /workspace/hfdata/l24/probe/texts.parquet \
    --sft-av "$CK/sft_av" --sft-ar "$CK/sft_ar" --rl-ckpt "$CK/rl/iter_000400" \
    --idle-minutes "$IDLE_MIN" --hf-repo syvb/head-nla-qwen3-8b --hf-path l24/std_rl/doc_viewer

date -u > "$RES/DONE"
echo "=== all stages done ==="
