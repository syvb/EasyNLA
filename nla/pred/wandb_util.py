"""One wandb convention for every pred-NLA stage.

Every stage (continuation prep, gate, RL, eval) logs to the same project with a
`group` and a `job_type`, so one workspace shows the whole experiment: prep ->
gate -> rl -> eval. Stages that are not training still create a run, because
their outputs (the gate's matched-vs-shuffled gap, the eval tables) are the
experiment's actual results and belong next to the training curve.
"""

from __future__ import annotations


def init_run(args, *, project, name=None, group=None, job_type=None, extra_config=None):
    import wandb

    cfg = {k: v for k, v in vars(args).items() if k != "config"}
    if extra_config:
        cfg.update(extra_config)
    tags = []
    raw_tags = getattr(args, "wandb_tags", None)
    if raw_tags:
        tags = raw_tags.split(",") if isinstance(raw_tags, str) else list(raw_tags)
    run = wandb.init(
        project=project, name=name, group=group, job_type=job_type,
        tags=tags + ["pred-nla"], config=cfg,
    )
    return run


def finish_run(run):
    if run is None:
        return
    import wandb

    wandb.finish()


def log_table(run, key, columns, rows, limit=500):
    """Log a wandb Table, truncated so a big eval does not bloat the run."""
    if run is None:
        return
    import wandb

    run.log({key: wandb.Table(columns=columns, data=[list(r) for r in rows[:limit]])})
