#!/bin/bash
# One-shot setup for the bottleneck experiment on a fresh Vast H200 box.
# Assumes: repo rsynced to /workspace/nla, HF token at /root/.hf_token,
# uv available (installs it if not). Run detached (nohup ... &) — downloads
# ~45GB and the merge loads a full 8B twice.
#
#   bash /workspace/nla/experiments/bottleneck/setup_box.sh
#
# Layout it creates (ALL on /workspace — the big volume; container disks on
# RunPod-style pods are small):
#   /workspace/envs/vllm-lens   — the pinned vllm venv (runs everything)
#   /workspace/hf_home          — HF hub cache (Qwen3-8B, datasets)
#   /workspace/nlabtl/warmstart — asher577/nla-warmstart-2x av/ (AV-SFT base)
#   /workspace/nlabtl/rl_ckpt   — asher577/nla-qwen-3-8b (av LoRA + ar critic)
#   /workspace/nlabtl/av_merged — merged AV for vLLM
#   /workspace/nlabtl/ar_ckpt   — AR critic dir
#   /workspace/nlabtl/domains.parquet, av_sft_val.parquet, results/
set -euo pipefail
# Non-interactive SSH shells on Vast boxes don't reliably carry the image's
# PATH — pin conda + uv locations up front or a detached run dies at `pip`.
export PATH=/opt/conda/bin:$HOME/.local/bin:$PATH
# Keep EVERYTHING bulky on the big /workspace volume: the default HF hub cache
# (~/.cache) and $HOME/envs sit on the small CONTAINER disk on RunPod-style
# pods — a 16GB hub-cached model or a 20GB venv there = mid-run out-of-space.
export HF_HOME=/workspace/hf_home
REPO=/workspace/nla
WORK=/workspace/nlabtl
VENV=/workspace/envs/vllm-lens
mkdir -p "$WORK" "$HF_HOME"

command -v uv >/dev/null || pip install -q uv

echo "=== [1/5] vllm-lens venv (pinned; applies injection patch) ==="
bash "$REPO/scripts/install_vllm_lens.sh" "$VENV"
# peft pinned to the version that wrote the AV adapter config; datasets pinned
# to the version every loader in tasks.py/prep_domains.py was verified against.
uv pip install --python "$VENV/bin/python" \
  "peft==0.19.1" "datasets==5.0.0" pyarrow pyyaml accelerate \
  "huggingface_hub[hf_transfer]"
# IFEval strict scorer (pip-installable port; scoring falls back to writing
# official-format JSONL if this is absent, so failure here is non-fatal)
uv pip install --python "$VENV/bin/python" \
  "git+https://github.com/josejg/instruction_following_eval" || \
  echo "WARN: ifeval scorer install failed — score_stage_b will emit JSONL instead"
uv pip install --python "$VENV/bin/python" -e "$REPO" --no-deps

echo "=== [2/5] downloads (models + val parquet + domains) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
"$VENV/bin/python" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
tok = Path("/root/.hf_token").read_text().strip()
snapshot_download("asher577/nla-warmstart-2x", allow_patterns=["av/*"],
                  local_dir="/workspace/nlabtl/warmstart", token=tok, max_workers=16)
snapshot_download("asher577/nla-qwen-3-8b",
                  local_dir="/workspace/nlabtl/rl_ckpt", token=tok, max_workers=16)
snapshot_download("Qwen/Qwen3-8B", token=tok, max_workers=16)  # hub cache (M)
hf_hub_download("asher577/easynla-warmstart-data", "av_sft_val.parquet",
                repo_type="dataset", token=tok, local_dir="/workspace/nlabtl")
hf_hub_download("asher577/easynla-warmstart-data", "av_sft_val.parquet.nla_meta.yaml",
                repo_type="dataset", token=tok, local_dir="/workspace/nlabtl")
try:
    hf_hub_download("syvb/nla-bottleneck-domains", "domains.parquet",
                    repo_type="dataset", token=tok, local_dir="/workspace/nlabtl")
except Exception as e:
    print(f"domains.parquet not on HF yet ({e}) — run prep_domains.py and rsync/upload it")
print("DOWNLOADS_DONE")
PY

echo "=== [3/5] AR critic dir (tokenizer files come from the AV side) ==="
# -T so a re-run replaces instead of nesting ar_ckpt/ar
rm -rf /workspace/nlabtl/ar_ckpt
cp -rT /workspace/nlabtl/rl_ckpt/ar /workspace/nlabtl/ar_ckpt

echo "=== [4/5] merge AV LoRA into warmstart base ==="
cd "$REPO"
"$VENV/bin/python" -m experiments.bottleneck.merge_av \
  --base-dir /workspace/nlabtl/warmstart/av \
  --lora-dir /workspace/nlabtl/rl_ckpt/av \
  --out /workspace/nlabtl/av_merged

echo "=== [5/5] smoke: codec contract loads ==="
"$VENV/bin/python" - <<'PY'
from transformers import AutoTokenizer
from nla.config import load_nla_config
tok = AutoTokenizer.from_pretrained("/workspace/nlabtl/av_merged")
cfg = load_nla_config("/workspace/nlabtl/av_merged", tok)
assert cfg.extraction_layer_index == 24 and cfg.d_model == 4096
print(f"contract OK: layer={cfg.extraction_layer_index} inj_char={cfg.injection_char!r} "
      f"mse_scale={cfg.mse_scale}")
PY

mkdir -p "$WORK/results"
echo "SETUP_DONE"
