"""Stream a remote run's train_sft log lines to wandb (for runs started with --no-wandb).

    WANDB_API_KEY=... python scripts/wandb_tail_log.py --log /workspace/results/run.log \\
        --project head-nla-qwen3-8b --group l24-explain-sft --prefix l24

Follows the log from the start (so it backfills a stage already under way) and opens one wandb run
per training stage, keyed on the remote script's stage headers (`=== [hh:mm:ss] ar_sft ===`). It
parses train_sft's step lines (`step 0042 | loss 1.23 | lr 1.0e-04 | grad 0.5 | ...`) and held-out
lines (`[heldout@199] val_loss 1.75 | val_ppl 5.78 (n=2000)`). Exits after the DONE or FAILED marker
next to the log appears and the log is read to its end.
"""

import argparse
import re
import time
from pathlib import Path

import wandb

STAGES = ("ar_sft", "av_sft")
_STAGE = re.compile(r"^=== \[[0-9:]+\] ([a-z_]+) ===")
_STEP = re.compile(r"^step (\d+) \| (.*)$")
_HELD = re.compile(r"\[heldout@(\d+)\] (.*)$")
_NUM = re.compile(r"^-?[0-9.]+(e[-+]?\d+)?")


def parse_pairs(body: str) -> dict[str, float]:
    """'loss 1.23 | lr 1.0e-04 | t 0.4s | FVE -34.5% | |p|/|g| 12.3/77.3' → floats."""
    out: dict[str, float] = {}
    for part in body.split(" | "):
        part = re.sub(r"\(n=\d+\)", "", part).strip()
        if " " not in part:
            continue
        key, val = part.rsplit(" ", 1)
        key = key.strip()
        if key == "|p|/|g|" and "/" in val:
            a, b = val.split("/", 1)
            out["pred_norm"], out["gold_norm"] = float(a), float(b)
            continue
        m = _NUM.match(val)
        if m:
            out[key.replace(" ", "_")] = float(m.group(0))
    return out


def follow(path: Path, done_markers: list[Path], poll: float = 5.0):
    while not path.exists():
        time.sleep(poll)
    with path.open(errors="replace") as f:
        buf = ""
        while True:
            chunk = f.read()
            if chunk:
                buf += chunk.replace("\r", "\n")
                *lines, buf = buf.split("\n")
                yield from lines
                continue
            if any(m.exists() for m in done_markers):
                if buf:
                    yield buf
                return
            time.sleep(poll)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--project", default="head-nla-qwen3-8b")
    p.add_argument("--group", required=True)
    p.add_argument("--prefix", required=True, help="run names are <prefix>-<stage>")
    args = p.parse_args()

    log = Path(args.log)
    run = None
    for line in follow(log, [log.parent / "DONE", log.parent / "FAILED"]):
        m = _STAGE.match(line)
        if m:
            if run is not None:
                run.finish()
                run = None
            if m.group(1) in STAGES:
                run = wandb.init(project=args.project, group=args.group, name=f"{args.prefix}-{m.group(1)}",
                                 job_type=m.group(1), reinit="finish_previous")
                print(f"wandb run for {m.group(1)}: {run.url}", flush=True)
            continue
        if run is None:
            continue
        if (m := _STEP.match(line.strip())):
            run.log({f"train/{k}": v for k, v in parse_pairs(m.group(2)).items()}, step=int(m.group(1)))
        elif (m := _HELD.search(line)):
            run.log({f"heldout/{k}": v for k, v in parse_pairs(m.group(2)).items()}, step=int(m.group(1)))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
