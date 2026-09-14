#!/usr/bin/env bash
# End-to-end CPU smoke of the future-lens pipeline on Qwen3-0.6B (no GPU, no money):
#   collect -> SFT (LoRA) -> GRPO (exact_match) -> eval (4 conditions) -> baselines -> tables
# ~10 min on 4 vCPUs. Needs the cached Qwen/Qwen3-0.6B and a local token corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CORPUS=${CORPUS:-$HOME/metamodelling/data/raw/docs.jsonl}   # .jsonl with `ids` or `text`
OUT=${OUT:-/tmp/fl_smoke}
LABEL=${LABEL:-greedy}     # text | greedy (Future Lens convention)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
rm -rf "$OUT"; mkdir -p "$OUT"

$PY -m nla.future_lens.collect --base-ckpt Qwen/Qwen3-0.6B --corpus "$CORPUS" \
    --n-train-docs 8 --n-eval-docs 4 --layers 4,8,12 --max-len 256 --positions-per-doc 5 \
    --batch-size 4 --greedy all --greedy-batch 8 --out-dir "$OUT/data"

$PY -m nla.train_sft --mode av --future-lens --base-ckpt Qwen/Qwen3-0.6B \
    --parquet "$OUT/data/train.parquet" --heldout-parquet "$OUT/data/eval.parquet" \
    --heldout-rows 16 --heldout-every 3 --heldout-gen-rows 16 --save-dir "$OUT/sft" \
    --use-lora --lora-r 8 --lora-alpha 16 --batch-size 8 --num-steps 3 --lr 1e-3 \
    --lr-warmup-steps 1 --save-every 3 --device cpu --no-wandb --no-gradient-checkpointing --label "$LABEL"

$PY -m nla.future_lens.train_rl --config configs/future_lens/smoke_cpu.yaml \
    --base-ckpt Qwen/Qwen3-0.6B --av-ckpt "$OUT/sft/iter_0000003" \
    --parquet "$OUT/data/train.parquet" --eval-parquet "$OUT/data/eval.parquet" --save-dir "$OUT/rl"

$PY -m nla.future_lens.eval --base-ckpt Qwen/Qwen3-0.6B --adapter "$OUT/rl/iter_000002" \
    --parquet "$OUT/data/eval.parquet" --out "$OUT/evals/rl.jsonl" \
    --conditions real,shuffled,none,wrong_layer --ks 1,3 --layers 8,12 --wrong-layer 4 \
    --max-rows 4 --batch-size 8 --device cpu --checkpoint-name rl_smoke --group rl_smoke \
    --dump-readouts "$OUT/evals/readouts_rl.jsonl"

$PY -m nla.future_lens.baselines --label "$LABEL" ngram --parquet "$OUT/data/eval.parquet" --out "$OUT/evals/baselines.jsonl"
$PY -m nla.future_lens.baselines --label "$LABEL" leakage --parquet "$OUT/data/eval.parquet" \
    --readouts "$OUT/evals/readouts_rl.jsonl" --out "$OUT/evals/leakage.jsonl" --order 4
$PY -m nla.future_lens.baselines --label "$LABEL" probe --train-parquet "$OUT/data/train.parquet" \
    --parquet "$OUT/data/eval.parquet" --base-ckpt Qwen/Qwen3-0.6B --layers 8 --epochs 1 \
    --batch 32 --leakage --device cpu --out "$OUT/evals/baselines.jsonl"

$PY -m nla.future_lens.plots "$OUT"/evals/rl.jsonl "$OUT"/evals/baselines.jsonl --out "$OUT/plots"
echo "SMOKE OK -> $OUT"
