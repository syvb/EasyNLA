"""Mirror pred-NLA artifacts to a private HuggingFace dataset repo, and back.

The pods that run this experiment keep no network volume (see the RunPod notes
in docs/pred_nla.md), so HuggingFace is the only durable store: every stage
uploads what it produced and the next stage pulls what it needs. Checkpoints,
the positions parquet, and the eval jsonl all go to the same repo under
predictable prefixes.

    python scripts/pred_hf_sync.py push  <repo> <local_dir> <remote_prefix>
    python scripts/pred_hf_sync.py pull  <repo> <remote_prefix> <local_dir>
    python scripts/pred_hf_sync.py wait  <repo> <remote_path> [--timeout-min 240]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        p = Path.home() / ".hf_token"
        if p.exists():
            token = p.read_text().strip()
    assert token, "set HF_TOKEN (or ~/.hf_token)"
    return HfApi(token=token), token


def push(a):
    api, _ = _api()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    local = Path(a.local_dir)
    assert local.exists(), f"{local} does not exist"
    api.upload_folder(
        folder_path=str(local), path_in_repo=a.remote_prefix.strip("/"),
        repo_id=a.repo, repo_type="dataset",
        ignore_patterns=a.ignore.split(",") if a.ignore else None,
        commit_message=f"pred-nla: {a.remote_prefix}",
    )
    print(f"[hf] pushed {local} -> {a.repo}:{a.remote_prefix}")


def pull(a):
    from huggingface_hub import snapshot_download

    _, token = _api()
    out = snapshot_download(
        repo_id=a.repo, repo_type="dataset", token=token,
        allow_patterns=[f"{a.remote_prefix.strip('/')}/*"],
        local_dir=a.local_dir,
    )
    print(f"[hf] pulled {a.repo}:{a.remote_prefix} -> {out}")


def wait(a):
    """Block until a path exists in the repo. Lets a later stage start on a
    second pod the moment an earlier one has uploaded, without a shared disk."""
    from huggingface_hub import HfApi

    api, _ = _api()
    deadline = time.time() + a.timeout_min * 60
    while time.time() < deadline:
        try:
            files = set(api.list_repo_files(a.repo, repo_type="dataset"))
            if any(f == a.remote_path or f.startswith(a.remote_path.rstrip("/") + "/")
                   for f in files):
                print(f"[hf] {a.remote_path} present")
                return
        except Exception as e:                                  # noqa: BLE001
            print(f"[hf] list failed ({type(e).__name__}); retrying", flush=True)
        time.sleep(a.poll_s)
    sys.exit(f"[hf] timed out waiting for {a.remote_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("push")
    p1.add_argument("repo"); p1.add_argument("local_dir"); p1.add_argument("remote_prefix")
    p1.add_argument("--ignore", default="optim_latest.pt,*.tmp")
    p2 = sub.add_parser("pull")
    p2.add_argument("repo"); p2.add_argument("remote_prefix"); p2.add_argument("local_dir")
    p3 = sub.add_parser("wait")
    p3.add_argument("repo"); p3.add_argument("remote_path")
    p3.add_argument("--timeout-min", type=int, default=240)
    p3.add_argument("--poll-s", type=int, default=60)
    a = ap.parse_args()
    {"push": push, "pull": pull, "wait": wait}[a.cmd](a)


if __name__ == "__main__":
    main()
