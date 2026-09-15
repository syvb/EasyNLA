"""Fetch paths from the experiment's HF dataset repo into a local dir (pods without the network volume).

    python scripts/hf_fetch.py --dest /workspace/fl --patterns "data_8b_evalu/**,data_base_v2/train.parquet*" \
        [--wait ckpts/<run>/iter_0008000/adapter_model.safetensors --wait-hours 3]

--wait polls the repo until that file exists (another pod is still producing it), then fetches.
Patterns are huggingface_hub allow_patterns globs relative to the repo root; files land at
<dest>/<path_in_repo> (same layout as the volume)."""
import argparse
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--repo", default=os.environ.get("FL_HF_REPO", "syvb/rl-future-lens-qwen3-8b"))
p.add_argument("--dest", required=True)
p.add_argument("--patterns", required=True, help="comma list of allow_patterns")
p.add_argument("--wait", default=None, help="repo file that must exist before fetching")
p.add_argument("--wait-hours", type=float, default=4.0)
a = p.parse_args()
tok = os.environ.get("HF_TOKEN")
if a.wait:
    api = HfApi(token=tok)
    t0 = time.time()
    while not api.file_exists(a.repo, a.wait, repo_type="dataset"):
        if time.time() - t0 > a.wait_hours * 3600:
            sys.exit(f"[hf_fetch] gave up waiting for {a.wait} after {a.wait_hours} h")
        print(f"[hf_fetch] waiting for {a.repo}/{a.wait} ({int(time.time() - t0)}s)", flush=True)
        time.sleep(60)
    time.sleep(30)   # let the producer's multi-file commit settle
pats = [x for x in a.patterns.split(",") if x]
snapshot_download(a.repo, repo_type="dataset", allow_patterns=pats, local_dir=a.dest, token=tok, max_workers=8)
print("[hf_fetch] fetched", pats, "->", a.dest, flush=True)
