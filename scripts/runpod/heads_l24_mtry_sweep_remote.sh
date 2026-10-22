#!/usr/bin/env bash
# Prefix-length sweep (scripts/heads_matryoshka_sweep.py) over the matryoshka run and the step-400 std
# pair it resumed from — executes ON the pod, after the run uploaded its checkpoints:
#   AVs: rl400 (std step 400), mtry (matryoshka final), + the gold descriptions
#   ARs: rl400, mtry_w (end of the AR-only warm-up: the rl400 AV's prefixes, AV untouched), mtry
# Launch with:
#   scripts/runpod/launch_with_retry.sh <out dir> heads-l24-mtry-sweep \
#       --remote-script scripts/runpod/heads_l24_mtry_sweep_remote.sh --secrets hf \
#       --fetch results --no-checkpoints --max-hours 3 --avoid-machine bdh8atjj57i7
set -eEuo pipefail

REPO=/workspace/EasyNLA
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
BASE=Qwen/Qwen3-8B
export STD_ITER=${STD_ITER:-iter_000400} WARM_ITER=${WARM_ITER:-iter_000440}
export MTRY_ITER=${MTRY_ITER:-iter_000840}
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
pip_install -e . scikit-learn accelerate bitsandbytes

stage download
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok, e = os.environ["HF_TOKEN"], os.environ
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset", allow_patterns=["l24/sft_data/*"],
                      local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for f in Path(d, "l24/sft_data").iterdir():
    shutil.copy(f, Path("/workspace/sft", f.name))
skip = ["*optim_latest.pt"]
for name, repo, it in (("std", "syvb/head-nla-qwen3-8b-l24-std-rl-ckpts", e["STD_ITER"]),
                       ("mtry", "syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts", e["WARM_ITER"]),
                       ("mtry", "syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts", e["MTRY_ITER"])):
    snapshot_download(repo, allow_patterns=[f"{it}/*"], ignore_patterns=skip, local_dir=f"/workspace/ckpt/{name}",
                      token=tok)
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
PY
ls "$CK"/*

stage sweep
python scripts/heads_matryoshka_sweep.py --sft-dir "$SFT" --base-ckpt "$BASE" --out-dir "$RES/sweep" \
    --av rl400="$CK/std/$STD_ITER" --av mtry="$CK/mtry/$MTRY_ITER" \
    --ar rl400="$CK/std/$STD_ITER/critic" --ar mtry_w="$CK/mtry/$WARM_ITER/critic" \
    --ar mtry="$CK/mtry/$MTRY_ITER/critic" \
    --max-rows 2000 --batch-size 64 --max-new-tokens 256

date -u > "$RES/DONE"
echo "=== all stages done ==="
