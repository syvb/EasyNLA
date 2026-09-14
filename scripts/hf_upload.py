"""Upload a folder to the experiment's private HF dataset repo (creates it if missing).

    python scripts/hf_upload.py <local_dir> <path_in_repo> [--repo syvb/rl-future-lens-qwen3-8b]

Optimizer states (optim*.pt) are skipped; everything else (parquet, sidecars, adapters,
future_lens.json, evals, logs) is mirrored. Kept as a file so the launcher's
`bash -lc '...'` wrapper needs no nested quotes, and so it survives the
huggingface-cli -> hf CLI rename."""
import argparse
import os
import sys

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("local_dir")
p.add_argument("path_in_repo")
p.add_argument("--repo", default=os.environ.get("FL_HF_REPO", "syvb/rl-future-lens-qwen3-8b"))
p.add_argument("--message", default=None)
a = p.parse_args()
if not os.path.isdir(a.local_dir):
    sys.exit(f"[hf_upload] {a.local_dir} is not a directory; nothing uploaded")
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
url = api.upload_folder(
    repo_id=a.repo, repo_type="dataset", folder_path=a.local_dir, path_in_repo=a.path_in_repo,
    ignore_patterns=["**/optim*.pt", "optim*.pt", "**/wandb/**", "**/__pycache__/**"],
    commit_message=a.message or f"upload {a.path_in_repo}",
)
print("[hf_upload]", a.local_dir, "->", f"{a.repo}/{a.path_in_repo}", url)
