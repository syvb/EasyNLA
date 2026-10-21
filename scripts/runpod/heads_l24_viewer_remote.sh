#!/usr/bin/env bash
# Data for the head-NLA viewer (scripts/heads_viewer_html.py) — executes ON the pod. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_viewer_remote.sh \
#       --secrets hf,anthropic --fetch results --no-checkpoints --max-hours 2 --out <local dir>
# All 32 heads of L24 at N_POS positions of N_DOCS eval-split (held-out) documents: the write v exactly as
# the models read it, its readouts, the Haiku gold explanation (readouts only), the SFT and RL (step 400)
# AV explanations at T=1 with their ARs' reconstruction scores, and attention evidence (evaluation only).
set -eEuo pipefail

REPO=/workspace/EasyNLA
OUT=/workspace/heads_l24
VDS=/workspace/viewer_ds
VSFT=/workspace/viewer_sft
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
CFG=configs/datagen/heads_qwen3_8b_l24.yaml
BASE=Qwen/Qwen3-8B
N_DOCS=${N_DOCS:-8}
N_POS=${N_POS:-4}
HF_PREFIX=${HF_PREFIX:-syvb/head-nla-qwen3-8b-l24-std}
mkdir -p "$RES"
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
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok, pre = os.environ["HF_TOKEN"], os.environ["HF_PREFIX"]
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset",
                      allow_patterns=["l24/sft_data/*", "l24/meta/selection.parquet"], local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for f in Path(d, "l24/sft_data").iterdir():
    shutil.copy(f, Path("/workspace/sft", f.name))
shutil.copy(Path(d, "l24/meta/selection.parquet"), "/workspace/results/saved_selection.parquet")
for name in ("av", "ar"):
    snapshot_download(f"{pre}-{name}", local_dir=f"/workspace/ckpt/sft_{name}", token=tok)
snapshot_download(f"{pre}-rl-ckpts", allow_patterns=["iter_000400/*"], ignore_patterns=["*optim_latest.pt"],
                  local_dir="/workspace/ckpt/rl", token=tok)
print(sorted(os.listdir("/workspace/ckpt/rl/iter_000400")))
PY

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats,select

# the records must be the ones the models' data came from (same corpus, positions and stats)
stage selection_check
python - "$OUT/warmstart/selection.parquet" "$RES/saved_selection.parquet" <<'PY'
import sys, numpy as np, pyarrow.parquet as pq
cols = ["example_idx", "split", "layer", "head", "doc_id", "pos", "dev_norm"]
new, old = (pq.read_table(p, columns=cols).to_pydict() for p in sys.argv[1:])
n = len(old["example_idx"])
dn_new, dn_old = np.array(new.pop("dev_norm")[:n]), np.array(old.pop("dev_norm"))
assert {k: v[:n] for k, v in new.items()} == old, "regenerated selection differs from the saved one"
assert np.allclose(dn_new, dn_old, rtol=1e-3, atol=1e-6), "dev_norm drifted"
print(f"first {n} rows identical")
PY

stage viewer_select
python scripts/heads_viewer_build.py select --head-dir "$OUT" --layer 24 --out-ds "$VDS" --results "$RES" \
    --n-docs "$N_DOCS" --n-pos "$N_POS"
cp "$VDS/warmstart/selection.parquet" "$RES/viewer_selection.parquet"

stage readouts
python scripts/heads_soft_token_readouts.py --output-dir "$VDS" --work-dir "$RES/viewer_readouts" --device-map cuda:0
rm -f "$RES/viewer_readouts/hidden.npy"

stage explain
python scripts/heads_soft_token_explain.py --texts "$RES/viewer_readouts/texts.parquet" \
    --out "$RES/viewer_readouts/explain.parquet"

stage eval_parquet
python scripts/heads_viewer_build.py eval --out-ds "$VDS" --explain "$RES/viewer_readouts/explain.parquet" \
    --sft-dir "$SFT" --out-sft "$VSFT"

stage checks
CHK=(--sft-dir "$VSFT" --base-ckpt "$BASE" --no-inject-preserve-magnitude --batch-size 64 --max-new-tokens 256 --temperature 1)
python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$CK/rl/iter_000400" --ar-ckpt "$CK/rl/iter_000400/critic" \
    --out-dir "$RES/checks_rl_t1"
python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$CK/sft_av" --ar-ckpt "$CK/sft_ar" --out-dir "$RES/checks_sft_t1"

date -u > "$RES/DONE"
echo "=== all stages done ==="
