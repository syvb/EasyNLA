"""Upload every file under a work dir that the HF dataset repo does not have yet.

    python scripts/hf_sync_missing.py --work /workspace/fl [--repo ...] [--skip hf,logs] [--dry-run]

Walks <work>/<top>/... ; a file is uploaded to <top>/<rel> unless the repo already lists that path.
Skips optimizer states (optim*.pt), the HF model cache dir, wandb dirs and __pycache__. Used once
before deleting the network volume, so nothing is lost that the per-stage uploads missed."""
import argparse
import os

from huggingface_hub import HfApi

IGNORE_DIRS = {"wandb", "__pycache__", ".cache"}

p = argparse.ArgumentParser()
p.add_argument("--work", default="/workspace/fl")
p.add_argument("--repo", default=os.environ.get("FL_HF_REPO", "syvb/rl-future-lens-qwen3-8b"))
p.add_argument("--skip", default="hf", help="comma list of top-level dirs to ignore (model cache)")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
api = HfApi(token=os.environ.get("HF_TOKEN"))
have = set(api.list_repo_files(a.repo, repo_type="dataset"))
skip = {x for x in a.skip.split(",") if x}
total = 0
for top in sorted(os.listdir(a.work)):
    root = os.path.join(a.work, top)
    if top in skip or not os.path.isdir(root):
        continue
    missing = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in IGNORE_DIRS]
        for f in files:
            if f.startswith("optim") and f.endswith(".pt"):
                continue
            rel = os.path.relpath(os.path.join(d, f), root)
            if f"{top}/{rel}" not in have:
                missing.append(rel)
    size = sum(os.path.getsize(os.path.join(root, r)) for r in missing)
    print(f"[sync] {top}: {len(missing)} missing files, {size / 1e9:.2f} GB", flush=True)
    for r in missing[:8]:
        print("   ", r)
    total += len(missing)
    if missing and not a.dry_run:
        api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=root, path_in_repo=top,
                          allow_patterns=missing, commit_message=f"sync missing files under {top}")
print(f"[sync] done: {total} files {'would be' if a.dry_run else ''} uploaded", flush=True)
