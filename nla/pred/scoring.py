"""Turning explanations into predictive gain, shared by the gate, RL and eval.

score(position, explanation) = mean over the position's continuation branches of
the reader's summed log-probability on each bucket's tokens.

gain = score(explanation) - score(no explanation), in nats per TARGET token.

The no-explanation baseline depends only on (reader, position, branch set), never
on the explanation, so it is computed once and cached. During RL that matters:
the baseline for a position is reused across all group members and across epochs,
turning what looks like a doubling of reader cost into a few percent.

Because GRPO normalizes rewards within a prompt group, subtracting a per-position
constant cannot change the gradient - the baseline is there to make the reward
READABLE (nats/token, on the same scale as the eval tables) and to give failed
rollouts a floor that means something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nla.pred.reader import DEFAULT_BUCKETS, ScoreJob, bucket_char_ranges


def target_tokens_per_bucket(buckets=DEFAULT_BUCKETS) -> np.ndarray:
    """Denominator for nats-per-token: the TARGET model's token count per bucket.

    Identical for every reader, which is what makes Qwen and Gemma numbers
    comparable even though they tokenize the continuation differently.
    """
    return np.array([hi - lo for lo, hi in buckets], dtype=np.float64)


@dataclass
class BaselineCache:
    """Cache of no-explanation scores.

    The key covers everything the value depends on: which reader, which prompt
    template, which position, which branches and which buckets. Leaving any of
    those out would let a cached score be served to a caller measuring something
    slightly different - the kind of mistake that produces a plausible number
    rather than an error.
    """

    store: dict = field(default_factory=dict)

    def _key(self, reader, row_id, branches, buckets):
        name = getattr(reader, "model_name", reader)
        tmpl = getattr(reader, "templates", None)
        tmpl_key = hash(tuple(sorted(tmpl.as_dict().items()))) if tmpl else None
        return (name, tmpl_key, int(row_id), tuple(branches),
                tuple(tuple(b) for b in buckets))

    def get(self, reader, row_id, branches, buckets=DEFAULT_BUCKETS):
        return self.store.get(self._key(reader, row_id, branches, buckets))

    def put(self, reader, row_id, branches, value, buckets=DEFAULT_BUCKETS):
        self.store[self._key(reader, row_id, branches, buckets)] = value

    def __len__(self):
        return len(self.store)


def _run_jobs(reader, rows, explanations, branches, buckets):
    """Score (row, branch) jobs and average over branches. NaN where no job ran."""
    n, nb = len(rows), len(buckets)
    jobs = []
    for i, row in enumerate(rows):
        expl = explanations[i]
        if expl is _SKIP:
            continue
        for b in branches:
            jobs.append(ScoreJob(
                key=(i, b), explanation=expl, cont_text=row["cont_text"][b],
                char_ranges=bucket_char_ranges(list(row["cont_bounds"][b]), buckets),
            ))
    out = np.full((n, nb), np.nan, dtype=np.float64)
    if not jobs:
        return out
    acc = np.zeros((n, nb), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    for res in reader.score(jobs, buckets=buckets):
        i, _b = res.key
        acc[i] += np.asarray(res.bucket_logp, dtype=np.float64)
        cnt[i] += 1
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit][:, None]
    return out


class _Skip:
    """Sentinel: this row has no explanation to score (extraction failed)."""

    def __repr__(self):
        return "<SKIP>"


_SKIP = _Skip()


def score_explanations(
    reader, rows, explanations, *, branches=(0, 1, 2, 3), buckets=DEFAULT_BUCKETS,
    allow_none=False,
) -> np.ndarray:
    """[n_rows, n_buckets] summed log-prob, averaged over branches.

    Rows whose explanation could not be extracted must be passed as the `SKIP`
    sentinel and come back NaN, so the caller decides what a failure is worth
    rather than having it silently averaged in.

    None is REJECTED by default. One layer up, `extract_explanation` returns None
    to mean "extraction failed", while to the reader None means "use the
    no-explanation template" - forwarding one as the other would turn a failed
    rollout into a free, legitimate-looking baseline score. `baseline_scores`
    passes allow_none=True because there it is deliberate.
    """
    expl = list(explanations)
    if not allow_none:
        assert not any(e is None for e in expl), (
            "score_explanations got None, which means the NO-EXPLANATION template "
            "here but 'extraction failed' in extract_explanation. Map failures to "
            "SKIP, or pass allow_none=True if you really want the baseline prompt.")
    return _run_jobs(reader, rows, expl, branches, buckets)


def baseline_scores(
    reader, rows, *, branches=(0, 1, 2, 3), buckets=DEFAULT_BUCKETS, cache=None,
) -> np.ndarray:
    """[n_rows, n_buckets] no-explanation scores, cached per (reader, row)."""
    n, nb = len(rows), len(buckets)
    out = np.zeros((n, nb), dtype=np.float64)
    todo = []
    for i in range(n):
        hit = None if cache is None else cache.get(
            reader, rows[i]["row_id"], branches, buckets)
        if hit is None:
            todo.append(i)
        else:
            out[i] = hit
    if todo:
        sub = [rows[i] for i in todo]
        vals = _run_jobs(reader, sub, [None] * len(sub), branches, buckets)
        for k, i in enumerate(todo):
            out[i] = vals[k]
            if cache is not None:
                cache.put(reader, rows[i]["row_id"], branches, vals[k], buckets)
    return out


def gain_per_token(scores: np.ndarray, baseline: np.ndarray, buckets=DEFAULT_BUCKETS):
    """(score - baseline) / target tokens per bucket -> nats per target token."""
    return (scores - baseline) / target_tokens_per_bucket(buckets)[None, :]


SKIP = _SKIP

__all__ = [
    "SKIP", "BaselineCache", "baseline_scores", "gain_per_token",
    "score_explanations", "target_tokens_per_bucket",
]
