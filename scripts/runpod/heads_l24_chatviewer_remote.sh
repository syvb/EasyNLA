#!/usr/bin/env bash
# Chat viewer — executes ON the pod. Qwen3-8B answers the prompts in scripts/heads_chat_prompts.json
# (scripts/heads_chat_gen.py, thinking off); then every token × head of each chat gets RL + SFT explanations and
# scores (scripts/heads_doc_viewer_job.py, no Haiku gold) and matryoshka explanations and scores
# (scripts/heads_doc_viewer_mtry.py). Launch with:
#   scripts/runpod/launch_with_retry.sh <out dir> heads-l24-chatviewer \
#       --remote-script scripts/runpod/heads_l24_chatviewer_remote.sh --secrets hf \
#       --fetch results --no-checkpoints --max-hours 6 --avoid-machine bdh8atjj57i7
# CHATS=name1,name2 (passed through the launcher's environment) limits the pod to those chats, to split the work
# across pods. Each chat's job output is uploaded to the HF dataset (<HF_PATH>/<chat>/) as soon as it finishes;
# the matryoshka parquets and chats.json once the last stage is done.
set -eEuo pipefail

REPO=/workspace/EasyNLA
RES=/workspace/results
CK=/workspace/ckpt
BASE=Qwen/Qwen3-8B
HF_DATA=syvb/head-nla-qwen3-8b
HF_PATH=${HF_PATH:-l24/chat_viewer}
GEN_BATCH=${GEN_BATCH:-128}
CHATS=${CHATS:-}
export STD_ITER=${STD_ITER:-iter_000400} WARM_ITER=${WARM_ITER:-iter_000440} MTRY_ITER=${MTRY_ITER:-iter_000640}
mkdir -p "$RES" /workspace/queue /workspace/smoke_queue
exec > >(tee -a "$RES/run.log") 2>&1
cd "$REPO"

STAGE=setup
trap 'echo "$STAGE" > "$RES/FAILED"; echo "!!! FAILED in stage $STAGE"' ERR
stage() { STAGE=$1; echo; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }

# shellcheck disable=SC1091
set -a; source /workspace/.secrets; set +a      # HF_TOKEN
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_DATA HF_PATH CHATS
unset PYTORCH_CUDA_ALLOC_CONF

stage install
nvidia-smi --query-gpu=name,memory.total --format=csv
pip_install() { python -m pip install -q "$@" || python -m pip install -q --break-system-packages "$@"; }
pip_install -e . scikit-learn accelerate anthropic

stage download
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok, e = os.environ["HF_TOKEN"], os.environ
snapshot_download(e["HF_DATA"], repo_type="dataset",
                  allow_patterns=["l24/sft_data/*", "l24/meta/head_meta.yaml", "l24/meta/head_stats.safetensors",
                                  "l24/meta/wo.safetensors", "l24/probe/texts.parquet"], local_dir="/workspace/hfdata", token=tok)
for name in ("av", "ar"):
    snapshot_download(f"syvb/head-nla-qwen3-8b-l24-std-{name}", local_dir=f"/workspace/ckpt/sft_{name}", token=tok)
skip = ["*optim_latest.pt"]
snapshot_download("syvb/head-nla-qwen3-8b-l24-std-rl-ckpts", allow_patterns=[f"{e['STD_ITER']}/*"], ignore_patterns=skip,
                  local_dir="/workspace/ckpt/rl", token=tok)
snapshot_download("syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts", allow_patterns=[f"{e['WARM_ITER']}/critic/*", f"{e['MTRY_ITER']}/*"],
                  ignore_patterns=skip, local_dir="/workspace/ckpt/mtry", token=tok)
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"], token=tok)
PY
ls "$CK"/*

JOB_ARGS=(--meta /workspace/hfdata/l24/meta --sft-dir /workspace/hfdata/l24/sft_data
          --texts-ref /workspace/hfdata/l24/probe/texts.parquet --sft-av "$CK/sft_av" --sft-ar "$CK/sft_ar"
          --rl-ckpt "$CK/rl/$STD_ITER" --no-gold --gen-batch "$GEN_BATCH" --idle-minutes 1)
MTRY_ARGS=(--sft-dir /workspace/hfdata/l24/sft_data --base "$BASE" --mtry-ckpt "$CK/mtry/$MTRY_ITER"
           --warm-ar "$CK/mtry/$WARM_ITER/critic" --gen-batch "$GEN_BATCH")

stage chats
python scripts/heads_chat_gen.py --prompts scripts/heads_chat_prompts.json --queue /workspace/queue --only "$CHATS"
cp /workspace/chats.json "$RES/chats.json"
touch /workspace/queue/CLOSE
ls /workspace/queue

stage smoke          # the whole path on 2 positions of the first chat, before the long stages
cp "$(ls /workspace/queue/*.json | head -1)" /workspace/smoke_queue/
touch /workspace/smoke_queue/CLOSE
# output outside $RES so the fetch doesn't pull a second copy of the first chat's writes
python scripts/heads_doc_viewer_job.py --queue /workspace/smoke_queue --out /workspace/smoke "${JOB_ARGS[@]}" --debug-positions 2
ls /workspace/smoke/*/sites.parquet          # the job catches per-document errors: fail here if it produced nothing
python scripts/heads_doc_viewer_mtry.py --viewer-dir /workspace/smoke "${MTRY_ARGS[@]}" --max-sites 64
ls /workspace/smoke/*/mtry.parquet

stage explain_rl_sft
python scripts/heads_doc_viewer_job.py --queue /workspace/queue --out "$RES/chats" "${JOB_ARGS[@]}" \
    --hf-repo "$HF_DATA" --hf-path "$HF_PATH"

stage explain_mtry
python scripts/heads_doc_viewer_mtry.py --viewer-dir "$RES/chats" "${MTRY_ARGS[@]}"

stage upload
python - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import HfApi
api, e, res = HfApi(token=os.environ["HF_TOKEN"]), os.environ, Path("/workspace/results")
def up(src, dst, msg):        # retried: another pod may be committing to the same dataset; results are fetched anyway
    for attempt in range(4):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=e["HF_DATA"], repo_type="dataset", commit_message=msg)
            print("uploaded", src, "->", dst); return
        except Exception as err:   # noqa: BLE001
            print(f"upload attempt {attempt + 1} of {src} failed: {err}"); time.sleep(30)
for f in sorted(res.glob("chats/*/mtry.parquet")):
    up(f, f"{e['HF_PATH']}/{f.parent.name}/mtry.parquet", f"chat viewer: matryoshka explanations for {f.parent.name}")
name = "chats" + ("_" + e["CHATS"].replace(",", "_") if e.get("CHATS") else "") + ".json"
up(res / "chats.json", f"{e['HF_PATH']}/{name}", "chat viewer: prompts, Qwen3-8B replies, seeds")
PY

if ls "$RES"/chats/*.FAILED >/dev/null 2>&1; then     # finished chats are uploaded; report the failed ones
    echo "!!! chats failed: $(ls "$RES"/chats/*.FAILED | xargs -n1 basename)"; false
fi
date -u > "$RES/DONE"
echo "=== all stages done ==="
