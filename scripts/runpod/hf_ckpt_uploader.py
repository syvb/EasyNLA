"""Upload RL checkpoints to HF as the trainer writes them (runs on the pod, beside the trainer).

Watches the trainer log for `[save] LoRA → <dir>` (printed only after the critic, the actor
adapter and optim_latest.pt are all written), snapshots that iter dir — minus the frozen
`reference/` adapter, a copy of the SFT AV already on HF — plus the newest optimizer state into a
staging dir, and uploads it as `<iter>/` (each iter dir gets the optimizer state current at its save,
as `<iter>/optim_latest.pt`, so any uploaded checkpoint can be resumed).
Read-only toward the training run: it never signals, locks or writes into the checkpoint dir.

  HF_TOKEN=... python scripts/runpod/hf_ckpt_uploader.py --log /workspace/results/run.log \
      --ckpt-dir /workspace/ckpt/rl --repo syvb/head-nla-qwen3-8b-l24-rl-ckpts
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi

SAVE_RE = re.compile(r"\[save\] LoRA → (\S+)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--stage", type=Path, default=Path("/workspace/hf_upload_stage"))
    p.add_argument("--done-markers", default="/workspace/results/DONE,/workspace/results/FAILED")
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    uploaded: set[str] = set()
    done_state = args.stage / "uploaded.txt"
    args.stage.mkdir(parents=True, exist_ok=True)
    if done_state.exists():
        uploaded = set(done_state.read_text().split())
    markers = [Path(m) for m in args.done_markers.split(",") if m]
    while True:
        finished = any(m.exists() for m in markers)
        text = args.log.read_text(errors="replace") if args.log.exists() else ""
        for d in (Path(m) for m in SAVE_RE.findall(text)):
            if d.name in uploaded or not d.is_dir():
                continue
            try:
                stage = args.stage / d.name
                shutil.rmtree(stage, ignore_errors=True)
                shutil.copytree(d, stage, ignore=shutil.ignore_patterns("reference"))
                for extra in ("optim_latest.pt", "run_config.yaml", "nla_meta.yaml"):
                    if (args.ckpt_dir / extra).exists():
                        shutil.copy2(args.ckpt_dir / extra, stage / extra)
                t0 = time.time()
                api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(stage),
                                  path_in_repo=d.name, commit_message=f"RL checkpoint {d.name}")
                uploaded.add(d.name)
                done_state.write_text("\n".join(sorted(uploaded)) + "\n")
                shutil.rmtree(stage, ignore_errors=True)
                print(f"[upload] {d.name} → {args.repo} in {time.time() - t0:.0f}s", flush=True)
            except Exception:   # noqa: BLE001 — retried on the next poll
                print(f"[upload] {d.name} failed (will retry):\n{traceback.format_exc()}", flush=True)
        if finished:
            print("[upload] run finished; exiting", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
