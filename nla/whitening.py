"""ZCA whitening of residual-stream activations.

Trains the NLA on x̃ = W(x − μ) instead of raw x, where W = Σ^{-1/2} of the
training activation distribution. MSE in whitened space is Mahalanobis
distance in raw space — the reward stops being dominated by the few
high-variance residual directions, so whitened FVE weights every direction
equally (and reads LOWER than raw-space FVE; the two are not comparable).

The transform is applied OFFLINE (scripts/whiten_dataset.py rewrites the
parquets) so trainers run unchanged. This module owns the numerics:

    compute_whitening_stats(parquet)  → WhiteningStats (μ, W, W⁻¹)
    whiten(X, stats) / unwhiten(X, stats)
    save_stats / load_stats           → .npz with integrity hash

Everything is float64 internally — Σ accumulation over ~10⁶ rows in float32
loses digits, and eigh of a 4096² matrix is cheap. W and W⁻¹ are the
symmetric (ZCA) roots, so whitened vectors stay maximally aligned with the
raw basis (vs PCA whitening, which permutes/rotates dimensions arbitrarily).
"""

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from nla.schema import ACTIVATION_COLUMN, NORM_WHITENED_ZCA

# Default covariance shrinkage toward the isotropic target (tr Σ / d) · I.
# Whitening amplifies direction i by 1/√λᵢ — near-null directions of the
# sample covariance would otherwise blow up numerical noise. λ = 0.01 caps
# the amplification of a zero-variance direction at 10× the mean direction.
DEFAULT_SHRINKAGE = 0.01


@dataclass
class WhiteningStats:
    mean: np.ndarray  # (d,)  float64
    w: np.ndarray  # (d, d) float64 symmetric — Σ_shrunk^{-1/2}
    w_inv: np.ndarray  # (d, d) float64 symmetric — Σ_shrunk^{+1/2}
    eigenvalues: np.ndarray  # (d,) float64 ascending — of the SHRUNK covariance
    shrinkage: float
    n_samples: int
    source: str  # parquet path (or description) the stats came from

    @property
    def d_model(self) -> int:
        return self.mean.shape[0]

    def sha256(self) -> str:
        """Content hash of the transform itself (μ + W), for sidecar pinning.
        Byte-exact over canonical float64 little-endian — save/load stable."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.mean, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.w, dtype="<f8").tobytes())
        return h.hexdigest()

    def sidecar_block(self, stats_path: str) -> dict:
        """The extraction.whitening block whiten_dataset.py writes — enough to
        locate + integrity-check the stats and re-derive how they were made."""
        return {
            "method": NORM_WHITENED_ZCA,
            "stats_path": str(stats_path),
            "stats_sha256": self.sha256(),
            "shrinkage": self.shrinkage,
            "n_samples": self.n_samples,
            "source": self.source,
        }


def iter_activation_batches(
    parquet_source, batch_size: int = 8192, max_rows: int | None = None
) -> Iterator[np.ndarray]:
    """Stream (n, d) float64 chunks of ACTIVATION_COLUMN from a parquet.

    Same flatten-not-to_pylist trick as schema.load_predict_mean_baselines —
    ListArray → flat values buffer → reshape, no per-float PyObjects.
    """
    pf = pq.ParquetFile(parquet_source)
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[ACTIVATION_COLUMN]):
        col = batch.column(ACTIVATION_COLUMN)
        flat = col.flatten().to_numpy(zero_copy_only=False)
        chunk = flat.reshape(len(col), -1).astype(np.float64)
        if max_rows is not None and n + chunk.shape[0] > max_rows:
            chunk = chunk[: max_rows - n]
        if chunk.shape[0] == 0:
            break
        yield chunk
        n += chunk.shape[0]
        if max_rows is not None and n >= max_rows:
            break


def compute_stats_from_moments(
    n: int,
    s: np.ndarray,
    outer: np.ndarray,
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    source: str = "",
) -> WhiteningStats:
    """Finish the streaming computation: moments → μ, Σ, shrink, eigh, W, W⁻¹.

    n: row count, s: Σx (d,), outer: Σ x xᵀ (d, d) — all float64.
    """
    d = s.shape[0]
    assert n >= 2, f"need ≥2 samples to estimate covariance, got {n}"
    mean = s / n
    # Unbiased sample covariance; symmetrize to kill accumulation asymmetry.
    cov = (outer - n * np.outer(mean, mean)) / (n - 1)
    cov = (cov + cov.T) / 2.0

    iso = np.trace(cov) / d
    assert iso > 0, "covariance has non-positive trace — degenerate input?"
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * iso * np.eye(d)

    eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
    assert eigvals[0] > 0, (
        f"smallest eigenvalue {eigvals[0]:.3e} ≤ 0 after shrinkage={shrinkage} "
        f"(n={n}, d={d}). Sample covariance is rank-deficient — raise --shrinkage "
        f"or use more rows (need n ≫ d)."
    )
    w = (eigvecs * (eigvals**-0.5)) @ eigvecs.T
    w_inv = (eigvecs * (eigvals**0.5)) @ eigvecs.T
    return WhiteningStats(
        mean=mean,
        w=(w + w.T) / 2.0,
        w_inv=(w_inv + w_inv.T) / 2.0,
        eigenvalues=eigvals,
        shrinkage=shrinkage,
        n_samples=n,
        source=source,
    )


def compute_stats_from_array(
    x: np.ndarray, *, shrinkage: float = DEFAULT_SHRINKAGE, source: str = ""
) -> WhiteningStats:
    """In-memory convenience path (tests, small data)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2, f"expected (n, d), got shape {x.shape}"
    return compute_stats_from_moments(
        x.shape[0], x.sum(axis=0), x.T @ x, shrinkage=shrinkage, source=source
    )


def compute_whitening_stats(
    parquet_source,
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    max_rows: int | None = None,
    batch_size: int = 8192,
    source: str | None = None,
) -> WhiteningStats:
    """Single streaming pass over the parquet: accumulate Σx and Σxxᵀ in
    float64, then finish with compute_stats_from_moments. Peak memory is the
    d² accumulator (4096² × 8B ≈ 134 MB), independent of row count."""
    n, s, outer = 0, None, None
    for chunk in iter_activation_batches(parquet_source, batch_size, max_rows):
        if s is None:
            d = chunk.shape[1]
            s = np.zeros(d, dtype=np.float64)
            outer = np.zeros((d, d), dtype=np.float64)
        assert chunk.shape[1] == s.shape[0], (
            f"row width changed mid-file: {chunk.shape[1]} vs {s.shape[0]}"
        )
        n += chunk.shape[0]
        s += chunk.sum(axis=0)
        outer += chunk.T @ chunk
    assert n > 0, f"no rows read from {parquet_source!r}"
    return compute_stats_from_moments(
        n, s, outer, shrinkage=shrinkage, source=source or str(parquet_source)
    )


def whiten(x: np.ndarray, stats: WhiteningStats) -> np.ndarray:
    """x̃ = (x − μ) W. Accepts (n, d) or (d,); float64 math, float64 out —
    caller casts to storage dtype."""
    return (np.asarray(x, dtype=np.float64) - stats.mean) @ stats.w


def unwhiten(x: np.ndarray, stats: WhiteningStats) -> np.ndarray:
    """Inverse: x = x̃ W⁻¹ + μ. Use to map a whitened-space AR prediction back
    to the raw residual stream (raw-space FVE, steering, inspection)."""
    return np.asarray(x, dtype=np.float64) @ stats.w_inv + stats.mean


def save_stats(stats: WhiteningStats, path: str) -> None:
    np.savez_compressed(
        path,
        mean=stats.mean,
        w=stats.w,
        w_inv=stats.w_inv,
        eigenvalues=stats.eigenvalues,
        shrinkage=np.float64(stats.shrinkage),
        n_samples=np.int64(stats.n_samples),
        source=np.str_(stats.source),
        sha256=np.str_(stats.sha256()),
    )


def load_stats(path: str) -> WhiteningStats:
    z = np.load(path, allow_pickle=False)
    stats = WhiteningStats(
        mean=z["mean"],
        w=z["w"],
        w_inv=z["w_inv"],
        eigenvalues=z["eigenvalues"],
        shrinkage=float(z["shrinkage"]),
        n_samples=int(z["n_samples"]),
        source=str(z["source"]),
    )
    stored = str(z["sha256"])
    live = stats.sha256()
    assert live == stored, (
        f"whitening stats corrupted: stored sha256 {stored[:12]}… != recomputed "
        f"{live[:12]}… ({path})"
    )
    return stats


def describe(stats: WhiteningStats) -> str:
    """Human-readable summary for script output / logs."""
    ev = stats.eigenvalues
    lines = [
        f"d_model      : {stats.d_model}",
        f"n_samples    : {stats.n_samples}",
        f"shrinkage    : {stats.shrinkage}",
        f"eig (shrunk) : min {ev[0]:.4g} · median {np.median(ev):.4g} · max {ev[-1]:.4g}",
        f"condition #  : {ev[-1] / ev[0]:.4g}",
        f"mean ‖μ‖     : {np.linalg.norm(stats.mean):.4g}",
        f"sha256       : {stats.sha256()[:16]}…",
    ]
    return "\n".join(lines)


def _asdict_meta(stats: WhiteningStats) -> dict:
    """Small-field view (no matrices) — for logging."""
    d = dataclasses.asdict(stats)
    for k in ("mean", "w", "w_inv", "eigenvalues"):
        d.pop(k)
    return d
