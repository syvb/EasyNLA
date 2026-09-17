"""Print the name of the checkpoint to evaluate from a train_rl save directory.

    python scripts/pred_best_ckpt.py <save_dir>   ->  iter_000250

Prefers `best.json`'s `best_ckpt` (the saved checkpoint with the best in-loop
score on the training reader), falls back to the last iter_* directory, and
exits non-zero if there is none - so a shell caller fails loudly rather than
evaluating a path that does not exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def best_checkpoint(save_dir: Path) -> str | None:
    best = save_dir / "best.json"
    if best.exists():
        name = json.loads(best.read_text()).get("best_ckpt")
        if name and (save_dir / name / "adapter_config.json").exists():
            return name
    iters = sorted(p for p in save_dir.glob("iter_*")
                   if (p / "adapter_config.json").exists())
    return iters[-1].name if iters else None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    name = best_checkpoint(Path(sys.argv[1]))
    if name is None:
        sys.exit(f"[pred_best_ckpt] no loadable iter_* checkpoint under {sys.argv[1]}")
    print(name)


if __name__ == "__main__":
    main()
