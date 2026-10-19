#!/usr/bin/env bash
# SFT-only rerun of heads_l24_remote.sh with explanations made earlier — executes ON the pod.
# The first run's host went into maintenance (reboot window) after its explain stage, so this
# regenerates the seeded datagen, maps the saved explanations onto its selection by site (doc, pos,
# layer, head; ≥95% must match), and trains on them. Inputs ride along in the repo upload: tmp/l24_inputs/{explain,selection}.parquet.
# Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_sft_remote.sh \
#       --secrets hf,anthropic,wandb --fetch results,heads_l24/sft --max-hours 6 --out <local dir>
# (no --hold: the launcher fetches and deletes the pod as soon as DONE or FAILED appears)
# Markers: /workspace/results/DONE | FAILED (first failing stage); log /workspace/results/run.log.
#
# Stages: install → datagen → selection check → export → AR SFT → AV SFT → pre-RL checks.
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (pip_install)

REPO=/workspace/EasyNLA
OUT=/workspace/heads_l24
SFT=$OUT/sft
CK=/workspace/ckpt
RES=/workspace/results
INPUTS=$REPO/tmp/l24_inputs
CFG=configs/datagen/heads_qwen3_8b_l24.yaml
BASE=Qwen/Qwen3-8B
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, ANTHROPIC_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
# wandb when the launcher was given --secrets ...,wandb (WANDB_API_KEY); otherwise local logs only
WANDB_FLAGS=(--no-wandb)
if [ -n "${WANDB_API_KEY:-}" ]; then WANDB_FLAGS=(--wandb-project head-nla-qwen3-8b --wandb-group l24-explain-sft); fi

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate anthropic
python -c "import torch, transformers, peft, anthropic; print('torch', torch.__version__, 'cuda', \
torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__, \
'anthropic', anthropic.__version__)"

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats,select

stage selection_check
# The explanations belong to (doc, position, layer, head) sites; map them onto the regenerated
# selection by site (bf16 write norms steer the weighted picks, so a different GPU could flip a few).
python - "$OUT/warmstart/selection.parquet" "$INPUTS/selection.parquet" "$INPUTS/explain.parquet" \
    "$RES/explain_remap.parquet" <<'PY'
import sys, pyarrow as pa, pyarrow.parquet as pq
new_p, old_p, exp_p, out_p = sys.argv[1:]
key = lambda r: (r["doc_id"], int(r["pos"]), int(r["layer"]), int(r["head"]))
old = {r["example_idx"]: key(r) for r in pq.read_table(old_p).to_pylist()}
ex = pq.read_table(exp_p).to_pylist()
by_site = {old[r["example_idx"]]: r for r in ex}
new = pq.read_table(new_p).to_pylist()
hit = [(r["example_idx"], by_site[key(r)]) for r in new if key(r) in by_site]
same = sum(old.get(r["example_idx"]) == key(r) for r in new)
print(f"regenerated selection: {len(new)} rows, {same} identical to the saved one, {len(hit)} with an explanation")
assert len(hit) >= 0.95 * len(new), "under 95% of the regenerated selection has a saved explanation"
pq.write_table(pa.table({"example_idx": [i for i, _ in hit], "explain": [r["explain"] for _, r in hit],
                         "ok": [r["ok"] for _, r in hit]}), out_p)
PY

stage export
python -m nla.datagen.heads.export_sft --output-dir "$OUT" --sft-dir "$SFT" \
    --descriptions "$RES/explain_remap.parquet" --description-column explain
steps() {   # epochs × ceil(rows / batch)
  python -c "import math, pyarrow.parquet as pq; print($2 * math.ceil(pq.ParquetFile('$1').metadata.num_rows / $3))"
}

stage ar_sft
python -m nla.train_sft --mode ar --base-ckpt "$BASE" --parquet "$SFT/ar_sft.parquet" --save-dir "$CK/ar" \
    --use-lora --ar-raw-loss --value-head-lr-mult 10 --ar-num-layers 25 \
    --batch-size 64 --num-steps "$(steps "$SFT/ar_sft.parquet" 2 64)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 "${WANDB_FLAGS[@]}" --seed 0

stage av_sft
python -m nla.train_sft --mode av --base-ckpt "$BASE" --parquet "$SFT/av_sft.parquet" --save-dir "$CK/av" \
    --use-lora --inject-preserve-magnitude \
    --batch-size 32 --num-steps "$(steps "$SFT/av_sft.parquet" 2 32)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 "${WANDB_FLAGS[@]}" --seed 0 \
    --heldout-parquet "$SFT/eval.parquet" --heldout-rows 2000 --heldout-every 200 \
    --sample-every 200 --n-samples 4 --sample-max-new-tokens 128

stage checks
AV_CK=$(ls -d "$CK"/av/iter_* | sort | tail -1)
AR_CK=$(ls -d "$CK"/ar/iter_* | sort | tail -1)
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$AV_CK" --ar-ckpt "$AR_CK" \
    --out-dir "$RES/checks" --batch-size 64 --max-new-tokens 128

stage collect
cp "$OUT/warmstart/selection.parquet" "$OUT/splits.parquet" "$OUT/head_meta.yaml" "$OUT/head_stats.safetensors" \
    "$RES/"
du -sh "$OUT" "$SFT" "$CK" "$RES" || true
date -u > "$RES/DONE"
echo "=== all stages done ==="
