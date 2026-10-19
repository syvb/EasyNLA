"""Log a head-NLA run's probe test and pre-RL checks to wandb as one summary run.

    WANDB_API_KEY=... python scripts/wandb_log_results.py --probe-dir <dir with probe.json> \\
        [--checks-dir <dir with checks.json, examples.md>] --group l24-explain-sft --name l24-results

Probe: mean held-out FVE over heads per feature and examples-per-head size (one line plot per
feature), plus a per-head table. Checks: every FVE condition, both check margins with their 95% CIs,
the AR-free text probe, and the examples as a table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--checks-dir", default=None)
    p.add_argument("--project", default="head-nla-qwen3-8b")
    p.add_argument("--group", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    res = json.loads((Path(args.probe_dir) / "probe.json").read_text())
    run = wandb.init(project=args.project, group=args.group, name=args.name, job_type="results")
    feats = list(next(iter(res.values())))
    sizes = sorted({k for r in res.values() for c in r.values() for k in c if k != "all"}, key=int)
    sizes = [s for s in sizes if all(s in res[h]["avg16"] for h in res)]   # sizes every head reached
    xs = [int(s) for s in sizes]
    ys = [[float(np.mean([res[h][f][s] for h in res])) for s in sizes] for f in feats]
    run.log({"probe/curves": wandb.plot.line_series(xs=xs, ys=ys, keys=feats, title="probe: held-out FVE vs examples per head",
                                                    xname="examples per head")})

    def at_all(c: dict) -> float:
        return c.get("all", c[max((k for k in c if k != "all"), key=int)])
    table = wandb.Table(columns=["head", *feats])
    for h in res:
        table.add_data(h, *[at_all(res[h][f]) for f in feats])
    run.log({"probe/per_head": table})
    run.summary.update({f"probe/{f}_all": float(np.mean([at_all(res[h][f]) for h in res])) for f in feats})

    if args.checks_dir:
        c = json.loads((Path(args.checks_dir) / "checks.json").read_text())
        run.summary.update({f"checks/fve/{k}": v for k, v in c["fve"].items()})
        run.summary.update({f"checks/fve_head_avg/{k}": v for k, v in c["fve_head_avg"].items()})
        for k in ("check1_ar_beats_empty", "check2_av_true_minus_shuf", "check1_per_head", "check2_per_head",
                  "av_distinct_fraction"):
            run.summary[f"checks/{k}"] = c[k]
        for k, (lo, hi) in c.get("ci95", {}).items():
            run.summary[f"checks/ci95/{k}_lo"], run.summary[f"checks/ci95/{k}_hi"] = lo, hi
        run.summary.update({f"checks/text_probe/{k}": v for k, v in c.get("text_probe_head_avg", {}).items()})
        md = (Path(args.checks_dir) / "examples.md").read_text().split("\n## ")[1:]
        ex = wandb.Table(columns=["example", "gold", "av_true", "av_shuf"])
        for block in md:
            head, *lines = block.split("\n")
            get = {ln.split(":**", 1)[0]: ln.split(":**", 1)[1].strip() for ln in lines if ":**" in ln}
            ex.add_data(head, *[next((v for k, v in get.items() if key in k), "") for key in
                                ("Gold", "true write", "another example")])
        run.log({"checks/examples": ex})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
