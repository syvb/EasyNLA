#!/usr/bin/env bash
# L24 head-NLA, standard recipe, AR retrained for the standard NUMBER OF UPDATES
# (docs/head_nla_l24_std_plan.md, "AR retrain") — executes ON the pod. Reuses the standard AV SFT
# (HF $HF_PREFIX-av) and the RL prompts (HF dataset l24/rl_data); retrains the AR on the standard
# recipe for up to $AR_EPOCHS epochs (one epoch here is 324 steps vs ~3.9k in the standard 250k-row
# split), keeping the checkpoint with the best held-out cosine on the doc-disjoint AV split; then
# pre-RL checks and RL with configs/rl_sgpu.yaml's defaults. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_std_ar2_remote.sh \
#       --secrets hf,wandb --fetch results --no-checkpoints --max-hours 8 --out <local dir>
# Markers: /workspace/results/DONE | FAILED (first failing stage); log /workspace/results/run.log.
#
# Head-data deviations that remain (each needed): the AR's input names the head ("Attention head
# L24Hh: ..."; one AR serves 32 heads — the AV prompt names none), bf16 LoRA instead of 4-bit (memory
# only), and base_fve as the only RL eval (text_judges needs document text this data lacks).
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (pip_install)

REPO=/workspace/EasyNLA
OUT=/workspace/heads_l24
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
CFG=configs/datagen/heads_qwen3_8b_l24.yaml
BASE=Qwen/Qwen3-8B
PER_HEAD_RL=${PER_HEAD_RL:-1000}   # 32k prompts; 400 steps × 64 = 25.6k draws ≈ 0.8 epoch
RL_STEPS=${RL_STEPS:-400}
RL_HOURS=${RL_HOURS:-5.0}           # trainer stops (and saves) after the step that passes this
HF_PREFIX=${HF_PREFIX:-syvb/head-nla-qwen3-8b-l24-std}
AR_EPOCHS=${AR_EPOCHS:-12}           # 12 × 324 = 3,888 steps ≈ one standard-size epoch
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
keep_ckpt() {   # keep_ckpt <iter dir> <dest>: the policy adapter + co-trained AR, minus the frozen
  mkdir -p "$2"  # "reference" adapter (a copy of the SFT AV)
  cp -r "$1" "$2/" && rm -rf "$2/$(basename "$1")/reference"
}
save_partial() {
  local latest
  latest=$(ls -d "$CK"/rl/iter_* 2>/dev/null | sort | tail -1 || true)
  if [ -n "$latest" ] && [ ! -d "$RES/rl_ckpt/$(basename "$latest")" ]; then
    keep_ckpt "$latest" "$RES/rl_ckpt_partial"
  fi
}
# copy the newest checkpoint out BEFORE writing FAILED: the launcher fetches and deletes on the marker
trap 'save_partial || true; echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, WANDB_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_PREFIX
unset PYTORCH_CUDA_ALLOC_CONF
wandb_flags() {   # wandb_flags <run name>
  if [ -n "${WANDB_API_KEY:-}" ]; then
    echo --wandb-project head-nla-qwen3-8b --wandb-group l24-std --wandb-name "$1"
  else
    echo --no-wandb
  fi
}

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate bitsandbytes
python -c "import torch, transformers, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', \
torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__, \
'bitsandbytes', bitsandbytes.__version__)"

stage download
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok, pre = os.environ["HF_TOKEN"], os.environ["HF_PREFIX"]
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset",
                      allow_patterns=["l24/sft_data/*", "l24/rl_data/*"], local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for sub in ("sft_data", "rl_data"):
    for f in Path(d, "l24", sub).iterdir():
        shutil.copy(f, Path("/workspace/sft", f.name))
snapshot_download(f"{pre}-av", local_dir="/workspace/ckpt/av_sft", token=tok)
print(sorted(os.listdir("/workspace/sft")), sorted(os.listdir("/workspace/ckpt/av_sft")))
PY

# docs/train_new_model.md §2–3: LoRA r128/α16, batch 64, one epoch (no --num-steps), warmup 50,
# clip 1.0; AV lr 1e-4 → 1e-5 with the standard norm-matched injection; AR lr 2e-5 → 2e-6 on the
# normalized (direction-only) MSE, truncated to layer_index + 1 = 25 blocks.
stage ar_sft
AR_STEPS=$(python -c "import math, pyarrow.parquet as pq; print($AR_EPOCHS * math.ceil(pq.ParquetFile('$SFT/ar_sft.parquet').metadata.num_rows / 64))")
python -m nla.train_sft --mode ar --base-ckpt "$BASE" --parquet "$SFT/ar_sft.parquet" --save-dir "$CK/ar" \
    --use-lora --lora-r 128 --lora-alpha 16 --ar-num-layers 25 --batch-size 64 --num-steps "$AR_STEPS" \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 --save-every 324 \
    --heldout-parquet "$SFT/av_sft.parquet" --heldout-rows 2000 --heldout-every 162 --heldout-head-prefix \
    $(wandb_flags l24-std-ar2-sft) --seed 0 2>&1 | tee "$RES/ar_sft.log"
# Keep the saved checkpoint with the best held-out cosine (eval at the save step itself).
AR_SFT=$(python - "$RES/ar_sft.log" "$CK/ar" <<'PY'
import re, sys
from pathlib import Path
cos = {int(m[1]) + 1: float(m[2]) for m in re.finditer(r"\[heldout@(\d+)\] mse \S+ \| FVE \S+ \| cos (-?[\d.]+)", open(sys.argv[1]).read())}
its = {int(p.name.split("_")[1]): p for p in Path(sys.argv[2]).glob("iter_*")}
scored = {n: cos[n] for n in its if n in cos}
assert scored, f"no held-out cosine at any saved step: saves {sorted(its)}, evals {sorted(cos)[:5]}…"
best = max(scored, key=scored.get)
print({n: round(c, 4) for n, c in sorted(scored.items())}, file=sys.stderr)
print(its[best])
PY
)
AV_SFT="$CK/av_sft"
echo "SFT checkpoints: AV $AV_SFT (standard, reused), AR $AR_SFT (best held-out cosine)"

stage upload_sft
python - "$AR_SFT" <<'PY'
import os, sys
from huggingface_hub import HfApi
api, tok, pre = HfApi(), os.environ["HF_TOKEN"], os.environ["HF_PREFIX"]
repo = f"{pre}-ar2"
api.create_repo(repo, private=True, exist_ok=True, token=tok)
api.upload_folder(repo_id=repo, folder_path=sys.argv[1], token=tok,
                  commit_message=f"L24 head-NLA AR, standard recipe, best held-out cosine ({os.path.basename(sys.argv[1])})")
print("uploaded", sys.argv[1], "→", repo)
PY

# Pre-RL checks for these SFT models (the bar RL must beat); both metrics are reported, direction
# first (the objective now), and every AV generation is kept for the faithfulness judge.
stage checks_sft
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$AV_SFT" --ar-ckpt "$AR_SFT" \
    --no-inject-preserve-magnitude --out-dir "$RES/checks_sft" --batch-size 64 --max-new-tokens 256

# configs/rl_sgpu.yaml defaults (AV lr 1e-4, AR lr 8e-5 co-trained, 64 prompts × 8, max 256 new
# tokens, hinged length penalty, k3 KL 0.01, direction-only reward, failure reward −2). Overrides:
# bf16 instead of 4-bit, the AR's per-row head prefix, batched rollouts and larger micro-batches for
# an 80 GB GPU, checkpoints every 50 steps (mirrored to HF), base_fve only, and a wall-clock stop.
stage rl
setsid nohup nice -n 19 python scripts/runpod/hf_ckpt_uploader.py --log "$RES/run.log" --ckpt-dir "$CK/rl" \
  --repo "$HF_PREFIX-ar2-rl-ckpts" > "$RES/hf_upload.log" 2>&1 < /dev/null &
rl_train() {   # rl_train <rollout batch> <logp micro-batch> <AR micro-batch> <eval gen batch>
  python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml \
    --base-ckpt "$BASE" --quant none --device cuda \
    --av-ckpt "$AV_SFT" --ar-ckpt "$AR_SFT" \
    --rl-parquet "$SFT/rl.parquet" --sidecar "$SFT/rl.parquet" --save-dir "$CK/rl" \
    --critic-prefix-column critic_prefix \
    --rollout-batch "$1" --logp-micro-batch "$2" --critic-micro-batch "$3" --eval-gen-batch "$4" \
    --num-steps "$RL_STEPS" --max-hours "$RL_HOURS" --save-every 50 --val-rows 2000 --evals base_fve \
    $(wandb_flags l24-std-ar2-rl) --seed 0 2>&1 | tee "$RES/rl_train.log"
}
# An out-of-memory before the first checkpoint costs only the load: retry once at half the batches
# (same recipe, smaller chunks). Any other failure, or an OOM later on, fails the stage.
if ! rl_train 256 8 16 64; then
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$RES/rl_train.log" && ! ls -d "$CK"/rl/iter_* >/dev/null 2>&1; then
    echo "OOM before the first checkpoint: retrying at half batch"
    rm -rf "$CK/rl"
    rl_train 128 4 8 32
  else
    false
  fi
fi

# Keep the last checkpoint and the one whose held-out FVE (trainer eval on 128 held-out rl prompts,
# averaged over the evals within ±10 steps of the save) was best, in results/ right away.
stage keep_ckpts
LAST=$(ls -d "$CK"/rl/iter_* | sort | tail -1)
BEST=$(python - "$RES/rl_train.log" "$CK/rl" <<'PY'
import re, sys
from pathlib import Path
ev = {int(m[1]): float(m[2]) for m in re.finditer(r"\[eval@(\d+)\] reward \S+ \| FVE (-?[\d.]+)%", open(sys.argv[1]).read())}
its = sorted(Path(sys.argv[2]).glob("iter_*"))
def score(it):
    n = int(it.name.split("_")[1])
    near = [v for s, v in ev.items() if abs(s - n) <= 10]
    return sum(near) / len(near) if near else float("-inf")
best = max(its, key=score)
if score(best) == float("-inf"):   # no parsable eval FVE: fall back to the last checkpoint
    best = its[-1]
print(best)
print({it.name: round(score(it), 2) for it in its}, file=sys.stderr)
PY
)
echo "last checkpoint: $LAST; best by held-out FVE: $BEST"
keep_ckpt "$LAST" "$RES/rl_ckpt"
if [ "$BEST" != "$LAST" ]; then keep_ckpt "$BEST" "$RES/rl_ckpt"; fi
cp "$CK/rl/run_config.yaml" "$CK/rl/nla_meta.yaml" "$RES/rl_ckpt/"

stage checks
CHK=(--sft-dir "$SFT" --base-ckpt "$BASE" --no-inject-preserve-magnitude --batch-size 64 --max-new-tokens 256)
python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$LAST" --ar-ckpt "$LAST/critic" --out-dir "$RES/checks_rl"
if [ "$BEST" != "$LAST" ]; then
  python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$BEST" --ar-ckpt "$BEST/critic" --out-dir "$RES/checks_rl_best"
fi
# Cross pairings: did the AV itself improve (scored by the SFT AR), and does the RL AR read the SFT
# AV's text? Guards against the two models co-adapting to a private code.
python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$LAST" --ar-ckpt "$AR_SFT" \
    --out-dir "$RES/checks_rlav_sftar" --max-rows 3200
python scripts/heads_sft_checks.py "${CHK[@]}" --av-ckpt "$AV_SFT" --ar-ckpt "$LAST/critic" \
    --out-dir "$RES/checks_sftav_rlar" --max-rows 3200

stage collect
cp "$RES/ar_sft.log" "$RES/ar_sft_heldout.log" || true
cp "$CK/ar/run_config.yaml" "$RES/ar_sft_run_config.yaml" || true
du -sh "$CK"/rl/* "$RES" || true
date -u > "$RES/DONE"
echo "=== all stages done ==="
