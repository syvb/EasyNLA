"""Print the last logged value of selected metrics for every run in a wandb group.

    python scripts/fl_wandb_summary.py --group alpha_sweep
    python scripts/fl_wandb_summary.py --run sft_replace_embed_a1.0 --metrics heldout/p1_off0,heldout/p1_off1

Used for the stage gates (alpha choice = largest heldout/readout_ce_gap among true-label runs;
SFT gate = heldout/p1_off0, p1_off1 vs the bigram baseline)."""
import argparse
import os

import wandb

DEFAULT = ("heldout/readout_ce_gap", "heldout/ce_gap", "heldout/readout_loss", "heldout/val_loss",
           "heldout/exact", "heldout/p1_off0", "heldout/p1_off1", "heldout/p1_off2", "heldout/p1_off3")

p = argparse.ArgumentParser()
p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "rl-future-lens"))
p.add_argument("--entity", default=None)
p.add_argument("--group", default=None)
p.add_argument("--run", default=None, help="run name (substring match)")
p.add_argument("--metrics", default=",".join(DEFAULT))
a = p.parse_args()
metrics = a.metrics.split(",")
api = wandb.Api()
path = f"{a.entity}/{a.project}" if a.entity else a.project
filters = {}
if a.group:
    filters["group"] = a.group
runs = [r for r in api.runs(path, filters=filters or None) if not a.run or a.run in r.name]
runs.sort(key=lambda r: r.name)
short = [m.split("/")[-1] for m in metrics]
print(f'{"run":<32} {"state":<9} {"step":>6} ' + " ".join(f"{s:>14}" for s in short))
for r in runs:
    hist = r.history(keys=metrics, pandas=False, samples=10000)
    hist = [h for h in hist if any(h.get(m) is not None for m in metrics)]
    last = hist[-1] if hist else {}
    vals = [f'{last[m]:>14.4f}' if isinstance(last.get(m), (int, float)) else f'{"-":>14}' for m in metrics]
    print(f'{r.name:<32} {r.state:<9} {last.get("_step", "-"):>6} ' + " ".join(vals))
