"""Uncertainty over POSITIONS (and documents), not over scored tokens.

Every number in this experiment is a mean over positions of a per-position
quantity, and neighbouring positions from the same document are correlated. So
the bootstrap resamples DOCUMENTS (clusters) by default; falling back to
positions would understate the interval whenever a document contributes more
than one position.

Comparisons between checkpoints are PAIRED: the same positions and the same
sampled continuations are scored under every condition, so the difference has far
less variance than the two means do, and pairing is what makes a small gain
detectable at pilot scale.
"""

from __future__ import annotations

import numpy as np


def _cluster_index(clusters):
    """Map cluster labels to an array of member-index arrays."""
    order: dict = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in order.values()]


def bootstrap_mean(
    x, clusters=None, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
):
    """(mean, lo, hi) percentile bootstrap CI of the mean of `x`.

    NaNs are dropped first (a failed extraction has no score, and averaging it in
    as a zero would be a silent lie).
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    x = x[ok]
    if clusters is not None:
        clusters = [c for c, k in zip(clusters, ok) if k]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    if clusters is None:
        draws = rng.integers(0, n, size=(n_boot, n))
        boots = x[draws].mean(axis=1)
    else:
        groups = _cluster_index(clusters)
        g = len(groups)
        boots = np.empty(n_boot, dtype=np.float64)
        sums = np.array([x[idx].sum() for idx in groups])
        counts = np.array([len(idx) for idx in groups], dtype=np.float64)
        for b in range(n_boot):
            pick = rng.integers(0, g, size=g)
            boots[b] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def paired_bootstrap_diff(
    a, b, clusters=None, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
):
    """Paired (a - b) mean difference with a CI and a two-sided bootstrap p-value.

    Positions where either side is NaN are dropped, so the pairing stays exact.
    Returns (diff, lo, hi, p, n_pairs).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    cl = None if clusters is None else [c for c, k in zip(clusters, ok) if k]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean, lo, hi = bootstrap_mean(d, clusters=cl, n_boot=n_boot, alpha=alpha, seed=seed)
    # Two-sided p: how often does the bootstrap distribution of the mean cross 0?
    rng = np.random.default_rng(seed + 1)
    n = len(d)
    if cl is None:
        boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    else:
        groups = _cluster_index(cl)
        g = len(groups)
        sums = np.array([d[idx].sum() for idx in groups])
        counts = np.array([len(idx) for idx in groups], dtype=np.float64)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, g, size=g)
            boots[i] = sums[pick].sum() / counts[pick].sum()
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return mean, lo, hi, float(min(1.0, p)), int(n)


def fmt_ci(mean, lo, hi, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


__all__ = ["bootstrap_mean", "fmt_ci", "paired_bootstrap_diff"]
