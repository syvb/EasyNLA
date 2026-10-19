#!/usr/bin/env bash
# L24 probe test + SFT on soft-token explanations (docs/head_nla_l24_plan.md) — executes ON the
# RunPod pod. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_remote.sh \
#       --secrets hf,anthropic --fetch results,heads_l24/sft --max-hours 6 --out <local dir>
# (no --hold: the launcher fetches and deletes the pod as soon as DONE or FAILED appears)
# Markers: /workspace/results/DONE | FAILED (first failing stage); log /workspace/results/run.log.
#
# Stages: install → datagen (corpus/extract/stats/select; no attention-based describer) →
# soft-token readouts → Haiku explanations (PAID, ~$16) → probe test → export → AR SFT → AV SFT →
# pre-RL checks. Warm-start text depends on the write alone.
set -eEuo pipefail   # -E: the ERR trap also fires inside functions (pip_install)

REPO=/workspace/EasyNLA
OUT=/workspace/heads_l24
SFT=$OUT/sft
CK=/workspace/ckpt
RES=/workspace/results
WORK=$RES/probe
CFG=configs/datagen/heads_qwen3_8b_l24.yaml
BASE=Qwen/Qwen3-8B
mkdir -p "$RES" "$WORK"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN, ANTHROPIC_API_KEY
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate anthropic
python -c "import torch, transformers, peft, anthropic; print('torch', torch.__version__, 'cuda', \
torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__, \
'anthropic', anthropic.__version__)"

stage datagen
python -m nla.datagen.heads.run_pipeline --config "$CFG" --stages corpus,extract,stats,select

stage readouts
python scripts/heads_soft_token_readouts.py --output-dir "$OUT" --work-dir "$WORK" --batch-size 64 \
    --device-map cuda:0

# Org limits (checked earlier): 10k requests/min, 10M input / 2M output tokens/min. 120 in flight with
# thinking stays under the output-token limit. A second pass picks up failed examples; rows still missing are dropped.
stage explain
# A pass that times out still keeps its cached answers; the coverage check below decides.
timeout 75m python scripts/heads_soft_token_explain.py --texts "$WORK/texts.parquet" \
    --out "$WORK/explain.parquet" --concurrency 120 || echo "first pass exited $?"
timeout 20m python scripts/heads_soft_token_explain.py --texts "$WORK/texts.parquet" \
    --out "$WORK/explain.parquet" --concurrency 32 || echo "second pass exited $?"
python - "$WORK" <<'PY'
import sys, pyarrow.parquet as pq
w = sys.argv[1]
t = pq.read_table(f"{w}/texts.parquet", columns=["example_idx", "head"]).to_pydict()
e = pq.read_table(f"{w}/explain.parquet").to_pydict()
n, got, ok = len(t["example_idx"]), len(e["example_idx"]), sum(e["ok"])
print(f"explained {got}/{n} ({got / n:.1%}); {got - ok} still break the rules (ok=False, dropped downstream)")
have = set(e["example_idx"])
per = {}
for x, h in zip(t["example_idx"], t["head"]):
    per[h] = per.get(h, 0) + (x in have)
print("explained per head:", dict(sorted(per.items())))
assert got >= 0.95 * n, "under 95% of examples explained"
PY

stage probe
python scripts/heads_explain_probe.py --work-dir "$WORK" --embed-model "$BASE" --device-map cuda:0 \
    2>&1 | tee "$RES/probe.log"

stage export
python -m nla.datagen.heads.export_sft --output-dir "$OUT" --sft-dir "$SFT" \
    --descriptions "$WORK/explain.parquet" --description-column explain
steps() {   # epochs × ceil(rows / batch)
  python -c "import math, pyarrow.parquet as pq; print($2 * math.ceil(pq.ParquetFile('$1').metadata.num_rows / $3))"
}

stage ar_sft
python -m nla.train_sft --mode ar --base-ckpt "$BASE" --parquet "$SFT/ar_sft.parquet" --save-dir "$CK/ar" \
    --use-lora --ar-raw-loss --value-head-lr-mult 10 --ar-num-layers 25 \
    --batch-size 64 --num-steps "$(steps "$SFT/ar_sft.parquet" 2 64)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 --no-wandb --seed 0

stage av_sft
python -m nla.train_sft --mode av --base-ckpt "$BASE" --parquet "$SFT/av_sft.parquet" --save-dir "$CK/av" \
    --use-lora --inject-preserve-magnitude \
    --batch-size 32 --num-steps "$(steps "$SFT/av_sft.parquet" 2 32)" --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-steps 20 --save-every 100000 --no-wandb --seed 0 \
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
