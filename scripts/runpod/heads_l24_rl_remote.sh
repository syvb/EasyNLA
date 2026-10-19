#!/usr/bin/env bash
# L24 head-NLA RL from the SFT warm-starts (docs/head_nla_l24_rl_plan.md) — executes ON the pod.
# Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_rl_remote.sh \
#       --secrets hf,wandb --fetch results --no-checkpoints --max-hours 7.5 --out <local dir>
# Markers: /workspace/results/DONE | FAILED (first failing stage); log /workspace/results/run.log.
#
# Stages: install → download the SFT checkpoints + SFT/eval data from the private HF repos →
# datagen (seeded; adds the rl split's prompts) → selection check (the first 48k rows must be the
# saved selection) → rl.parquet export → RL (nla.train_rl_self_contained with configs/rl_sgpu.yaml,
# the repo's single-GPU recipe) → pre-RL checks on the 6,360 eval rows for three AV/AR pairings →
# collect the final checkpoints into results/.
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
RL_HOURS=${RL_HOURS:-5.5}           # trainer stops (and saves) after the step that passes this
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, WANDB_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
WANDB_FLAGS=(--no-wandb)
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_FLAGS=(--wandb-project head-nla-qwen3-8b --wandb-group l24-rl --wandb-name l24-rl-sgpu)
fi

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
tok = os.environ["HF_TOKEN"]
for name in ("av", "ar"):
    snapshot_download(f"syvb/head-nla-qwen3-8b-l24-{name}", local_dir=f"/workspace/ckpt/sft_{name}", token=tok)
d = snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset",
                      allow_patterns=["l24/sft_data/*", "l24/meta/selection.parquet"],
                      local_dir="/workspace/hfdata", token=tok)
Path("/workspace/sft").mkdir(exist_ok=True)
for f in Path(d, "l24/sft_data").iterdir():
    shutil.copy(f, Path("/workspace/sft", f.name))
shutil.copy(Path(d, "l24/meta/selection.parquet"), "/workspace/results/saved_selection.parquet")
print(sorted(os.listdir("/workspace/sft")), sorted(os.listdir("/workspace/ckpt")))
PY

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats,select \
    --override "select.per_head_rl=$PER_HEAD_RL"

# The SFT models were trained, and are checked, on the saved selection's av_sft / ar_sft / eval rows;
# the RL prompts must come from the same doc split, so the regenerated warm-start rows must match.
stage selection_check
python - "$OUT/warmstart/selection.parquet" "$RES/saved_selection.parquet" <<'PY'
import collections, sys, pyarrow.parquet as pq
cols = ["example_idx", "split", "layer", "head", "doc_id", "pos"]
new, old = (pq.read_table(p, columns=cols).to_pydict() for p in sys.argv[1:])
n = len(old["example_idx"])
assert {k: v[:n] for k, v in new.items()} == old, "regenerated warm-start rows differ from the saved selection"
print(f"first {n} rows identical; then {len(new['example_idx']) - n} rows:",
      dict(collections.Counter(new["split"][n:])))
assert set(new["split"][n:]) == {"rl"}
PY

stage export_rl
python -m nla.datagen.heads.export_sft --output-dir "$OUT" --sft-dir "$SFT" --rl-only

# configs/rl_sgpu.yaml is the repo's single-GPU recipe (AV lr 1e-4, AR lr 8e-5, 64 prompts × 8,
# max 256 new tokens, hinged length penalty, k3 KL 0.01, co-trained AR). Overrides: bf16 instead of
# 4-bit (the SFT checkpoints are bf16), the head-data contract the SFT models were trained with
# (magnitude-preserving injection, raw-loss AR with √d targets, the AR's per-row head prefix),
# batched rollouts, larger micro-batches for an 80 GB GPU, base_fve only (text_judges scores against
# document text, which this data doesn't carry), and a wall-clock stop.
stage rl
python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml \
    --base-ckpt "$BASE" --quant none --device cuda \
    --av-ckpt "$CK/sft_av" --ar-ckpt "$CK/sft_ar" \
    --rl-parquet "$SFT/rl.parquet" --sidecar "$SFT/rl.parquet" --save-dir "$CK/rl" \
    --inject-preserve-magnitude --ar-raw-loss --critic-prefix-column critic_prefix \
    --rollout-batch 256 --logp-micro-batch 8 --critic-micro-batch 16 --eval-gen-batch 64 \
    --num-steps "$RL_STEPS" --max-hours "$RL_HOURS" --val-rows 2000 --evals base_fve \
    "${WANDB_FLAGS[@]}" --seed 0

stage checks
LAST=$(ls -d "$CK"/rl/iter_* | sort | tail -1)
echo "final RL checkpoint: $LAST"
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$LAST" --ar-ckpt "$LAST/critic" \
    --out-dir "$RES/checks_rl" --batch-size 64 --max-new-tokens 256
# Cross pairings: did the AV itself improve (scored by the SFT AR), and does the RL AR read the SFT
# AV's text? Guards against the two models co-adapting to a private code.
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$LAST" --ar-ckpt "$CK/sft_ar" \
    --out-dir "$RES/checks_rlav_sftar" --batch-size 64 --max-new-tokens 256 --max-rows 3200
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$CK/sft_av" --ar-ckpt "$LAST/critic" \
    --out-dir "$RES/checks_sftav_rlar" --batch-size 64 --max-new-tokens 128 --max-rows 3200

stage collect
mkdir -p "$RES/rl_ckpt"
# the final policy adapter + co-trained AR (the frozen KL "reference" adapter is the SFT AV: skipped)
rsync -a --exclude reference "$LAST/" "$RES/rl_ckpt/$(basename "$LAST")/"
cp "$CK/rl/run_config.yaml" "$CK/rl/nla_meta.yaml" "$RES/rl_ckpt/"
cp "$SFT/rl.parquet" "$SFT/rl.parquet.nla_meta.yaml" "$OUT/warmstart/selection.parquet" "$RES/"
du -sh "$CK"/rl/* "$RES" || true
date -u > "$RES/DONE"
echo "=== all stages done ==="
