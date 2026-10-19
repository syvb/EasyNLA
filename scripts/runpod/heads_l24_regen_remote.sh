#!/usr/bin/env bash
# Re-run the L24 pre-RL checks from the saved checkpoints, keeping every AV generation
# (results/checks/generations.parquet) for the faithfulness judge — executes ON the pod. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_regen_remote.sh \
#       --secrets hf --fetch results --no-checkpoints --max-hours 2 --out <local dir>
# Pulls the AV / AR checkpoints and the SFT parquets from the private HF repos (HF_TOKEN).
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (pip_install)

REPO=/workspace/EasyNLA
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
BASE=Qwen/Qwen3-8B
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

stage download
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for name in ("av", "ar"):
    snapshot_download(f"syvb/head-nla-qwen3-8b-l24-{name}", local_dir=f"/workspace/ckpt/{name}", token=tok)
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset", allow_patterns=["l24/sft_data/*"],
                      local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for f in Path(d, "l24/sft_data").iterdir():
    shutil.copy(f, Path("/workspace/sft", f.name))
print(sorted(os.listdir("/workspace/sft")))
PY

stage checks
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$CK/av" --ar-ckpt "$CK/ar" \
    --out-dir "$RES/checks" --batch-size 64 --max-new-tokens 128

date -u > "$RES/DONE"
echo "=== all stages done ==="
