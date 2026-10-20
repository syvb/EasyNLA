#!/usr/bin/env bash
# Verify a mid-run RL checkpoint mirrored to HF by hf_ckpt_uploader.py — executes ON the pod. Launch with:
#   python scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_l24_ckpt_verify_remote.sh \
#       --secrets hf --fetch results --no-checkpoints --max-hours 1.5 --out <local dir>
# Pulls the newest iter_* (or $CKPT_ITER) from $CKPT_REPO plus the SFT AV/AR and SFT data, then:
#   1. structure: files present, adapter/critic keys + shapes identical to the SFT checkpoints, weights
#      moved off SFT (relative change), finite, optimizer state loads with the right step;
#   2. function: heads_sft_checks.py with the checkpoint's AV + co-trained AR on the first
#      $MAX_ROWS eval rows (compared locally, paired on example_idx, against the full SFT checks).
set -eEuo pipefail

REPO=/workspace/EasyNLA
SFT=/workspace/sft
CK=/workspace/ckpt
RES=/workspace/results
BASE=Qwen/Qwen3-8B
CKPT_REPO=${CKPT_REPO:-syvb/head-nla-qwen3-8b-l24-rl-ckpts}
MAX_ROWS=${MAX_ROWS:-960}
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CKPT_REPO
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate bitsandbytes

stage download
CKPT_ITER=$(python - <<'PY'
import os
from huggingface_hub import HfApi, snapshot_download
tok, repo = os.environ["HF_TOKEN"], os.environ["CKPT_REPO"]
iters = sorted({f.split("/")[0] for f in HfApi().list_repo_files(repo, token=tok) if f.startswith("iter_")})
it = os.environ.get("CKPT_ITER") or iters[-1]
snapshot_download(repo, allow_patterns=[f"{it}/*"], local_dir="/workspace/ckpt/rl", token=tok)
for name in ("av", "ar"):
    snapshot_download(f"syvb/head-nla-qwen3-8b-l24-{name}", local_dir=f"/workspace/ckpt/sft_{name}", token=tok)
snapshot_download("syvb/head-nla-qwen3-8b", repo_type="dataset", allow_patterns=["l24/sft_data/*"], local_dir="/workspace/hfdata", token=tok)
print(it)
PY
)
CKPT_ITER=$(echo "$CKPT_ITER" | tail -1)
echo "verifying $CKPT_REPO/$CKPT_ITER"
mkdir -p "$SFT" && cp /workspace/hfdata/l24/sft_data/* "$SFT/"
ls -la "$CK/rl/$CKPT_ITER" "$CK/rl/$CKPT_ITER/critic" "$SFT"

stage structure
python - "$CK/rl/$CKPT_ITER" "$CK/sft_av" "$CK/sft_ar" "$RES/structure.json" <<'PY'
import json, re, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
it, sft_av, sft_ar, out = map(Path, sys.argv[1:])
step = int(re.search(r"iter_(\d+)", it.name).group(1))
rep = {"iter": it.name}
for f in ["adapter_config.json", "adapter_model.safetensors", "critic/ar_lora_value_head.safetensors",
          "critic/ar_meta.json", "optim_latest.pt", "run_config.yaml"]:
    assert (it / f).exists(), f"missing {f}"
assert not (it / "reference").exists(), "reference adapter should not be in the upload"

def compare(name, a_path, b_path, norm_key=lambda k: k):
    a, b = load_file(str(a_path)), load_file(str(b_path))
    a = {norm_key(k): v for k, v in a.items()}
    b = {norm_key(k): v for k, v in b.items()}
    assert set(a) == set(b), f"{name}: key mismatch, only-ckpt={sorted(set(a)-set(b))[:3]} only-sft={sorted(set(b)-set(a))[:3]}"
    num = den = 0.0
    for k in a:
        assert a[k].shape == b[k].shape, f"{name}: {k} shape {a[k].shape} vs {b[k].shape}"
        x, y = a[k].float(), b[k].float()
        assert torch.isfinite(x).all(), f"{name}: non-finite values in {k}"
        num += float(((x - y) ** 2).sum()); den += float((y ** 2).sum())
    rel = (num / den) ** 0.5
    rep[name] = {"tensors": len(a), "rel_change_vs_sft": rel}
    print(f"{name}: {len(a)} tensors, keys/shapes match SFT, finite, ‖Δ‖/‖SFT‖ = {rel:.4f}")
    assert rel > 0, f"{name}: identical to SFT — not trained"

compare("av_adapter", it / "adapter_model.safetensors", sft_av / "adapter_model.safetensors")
compare("ar_critic", it / "critic/ar_lora_value_head.safetensors", sft_ar / "ar_lora_value_head.safetensors")
cfg = json.loads((it / "adapter_config.json").read_text())
rep["adapter_config"] = {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "base_model_name_or_path")}
print("adapter_config:", rep["adapter_config"])
opt = torch.load(str(it / "optim_latest.pt"), map_location="cpu", weights_only=False)
assert opt["step"] >= step, f"optim step {opt['step']} < checkpoint step {step}"
rep["optim"] = {"step": opt["step"], "keys": sorted(opt), "actor_state_tensors": len(opt["actor_optim"]["state"])}
print("optim:", rep["optim"])
Path(out).write_text(json.dumps(rep, indent=2))
PY

stage checks
python scripts/heads_sft_checks.py --sft-dir "$SFT" --base-ckpt "$BASE" --av-ckpt "$CK/rl/$CKPT_ITER" \
    --ar-ckpt "$CK/rl/$CKPT_ITER/critic" --out-dir "$RES/checks" --batch-size 64 --max-new-tokens 128 \
    --max-rows "$MAX_ROWS"

date -u > "$RES/DONE"
echo "=== all stages done ==="
