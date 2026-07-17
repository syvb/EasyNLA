#!/bin/bash
# Full first-pass run order (after setup_box.sh). Each step prints a *_DONE
# sentinel; run detached and poll for sentinels / process death.
#
#   nohup bash /workspace/nla/experiments/bottleneck/run_all.sh > /workspace/run.log 2>&1 &
#
# Order of operations == README §sanity: cheap checks first, C1 pilot before
# the full Stage B matrix (abort early if the codec floors everything).
set -euo pipefail
VENV=$HOME/envs/vllm-lens
CFG=/workspace/nla/experiments/bottleneck/config.yaml
cd /workspace/nla
test -f /workspace/nlabtl/domains.parquet || {
  echo "FATAL: /workspace/nlabtl/domains.parquet missing — run prep_domains.py" \
       "on the dev box and upload/rsync it (see setup_box.sh)"; exit 1; }

run() { echo "### $(date -u +%H:%M:%S) $*"; "$VENV/bin/python" -m "$@"; }

# 1. harness sanity: custom loop == HF generate; C0' == C0; tiny C1 smoke
run experiments.bottleneck.sanity_checks --config "$CFG"

# 2. codec reproduction + norm scatter (expect FVE ~0.78; pred norms collapsed)
run experiments.bottleneck.norm_scatter --config "$CFG" \
    --parquet /workspace/nlabtl/av_sft_val.parquet

# 3. Stage A domain map (cheap dense go/no-go; ~1-2h)
run experiments.bottleneck.stage_a --config "$CFG"
run experiments.bottleneck.analysis_stage_a --results /workspace/nlabtl/results/stage_a

# 4. C1 pilot spanning answer horizons (a long-CoT-only pilot would read
#    "abort" even when short-horizon tasks show a rich graded map)
"$VENV/bin/python" - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(open("/workspace/nla/experiments/bottleneck/config.yaml"))
cfg["task_sizes"] = {"gsm8k": 24, "gsm8k_short": 24, "triviaqa": 30,
                     "mmlu_pro": 2, "fluency": 8}
cfg["out_dir"] = "/workspace/nlabtl/results_pilot"
pathlib.Path("/workspace/nlabtl/pilot.yaml").write_text(yaml.dump(cfg))
PY
for cond in clean nla; do
  run experiments.bottleneck.stage_b --config /workspace/nlabtl/pilot.yaml \
      --condition "$cond" --tasks gsm8k,gsm8k_short,triviaqa,mmlu_pro,fluency --seed 0
done
run experiments.bottleneck.score_stage_b --results /workspace/nlabtl/results_pilot/stage_b
echo "### PILOT DONE — inspect retention above before the full matrix continues"

# 5. full Stage B: C0, C0', C1 (+2 extra codec seeds on gsm8k/triviaqa).
#    Optional afterwards: --condition nla_prompt on triviaqa to quantify the
#    clean-prompt-KV bypass (see README caveats).
run experiments.bottleneck.stage_b --config "$CFG" --condition clean    --seed 0
run experiments.bottleneck.stage_b --config "$CFG" --condition identity --seed 0
run experiments.bottleneck.stage_b --config "$CFG" --condition nla      --seed 0
run experiments.bottleneck.stage_b --config "$CFG" --condition nla --seed 1 --tasks gsm8k,triviaqa
run experiments.bottleneck.stage_b --config "$CFG" --condition nla --seed 2 --tasks gsm8k,triviaqa
run experiments.bottleneck.stage_b --config "$CFG" --condition nla_prompt --seed 0 --tasks triviaqa

# 6. score
run experiments.bottleneck.score_stage_b --results /workspace/nlabtl/results/stage_b
echo "RUN_ALL_DONE"
