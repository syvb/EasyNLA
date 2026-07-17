"""ZCA whitening of residual-stream activations.

Trains the NLA on x̃ = W(x − μ) instead of raw x, where W = Σ^{-1/2} of the
training activation distribution. Because the reward/FVE pipeline L2-normalizes
both prediction and gold before MSE (schema.normalize_activation), the trained
quantity is effectively the ANGLE in the Mahalanobis geometry of the centered
distribution — the point being that it weights every residual direction
equally, instead of being dominated by the few high-variance ones. Whitened
FVE reads LOWER than raw-space FVE; the two are not comparable.

The transform is applied OFFLINE (scripts/whiten_dataset.py rewrites the
parquets) so trainers run unchanged. This module owns the numerics:

    compute_whitening_stats(parquet)  → WhiteningStats (μ, W, W⁻¹)
    whiten(X, stats) / unwhiten(X, stats)
    save_stats / load_stats           → .npz with integrity hash

Regularization: whitening amplifies direction i by 1/√λᵢ, so near-null sample
eigenvalues blow up estimation noise. Default is eigenvalue FLOORING at a low
quantile — directions above the floor are whitened exactly, only the extreme
tail is capped. (Shrinkage toward (trΣ/d)·I is also available, but its target
is inflated by the outlier top of the spectrum, so it under-whitens a broad
band of the low-variance directions this experiment is about.) The achieved
per-direction whitened variance λ_raw/λ_reg is recorded and reported.

Everything is float64 internally — Σ accumulation over ~10⁶ rows in float32
loses digits, and eigh of a 4096² matrix is cheap. W and W⁻¹ are the
symmetric (ZCA) roots, so whitened vectors stay maximally aligned with the
raw basis (vs PCA whitening, which permutes/rotates dimensions arbitrarily).
"""

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from nla.schema import ACTIVATION_COLUMN, NORM_WHITENED_ZCA

DEFAULT_SHRINKAGE = 0.0
DEFAULT_FLOOR_QUANTILE = 0.05


@dataclass
class WhiteningStats:
    mean: np.ndarray  # (d,)  float64
    w: np.ndarray  # (d, d) float64 symmetric — Σ_reg^{-1/2}
    w_inv: np.ndarray  # (d, d) float64 symmetric — Σ_reg^{+1/2}
    eigenvalues: np.ndarray  # (d,) float64 ascending — REGULARIZED spectrum (what W uses)
    eigenvalues_raw: np.ndarray  # (d,) float64 ascending — sample spectrum before reg
    shrinkage: float
    floor_quantile: float
    n_samples: int
    source: str  # parquet path (or description) the stats came from
    # Provenance pinned from the source sidecar — whiten_dataset.py refuses to
    # apply stats from one model/layer to another's parquets. Empty/None for
    # stats built without a sidecar (tests, ad-hoc arrays).
    base_model: str = ""
    layer_index: int | None = None

    @property
    def d_model(self) -> int:
        return self.mean.shape[0]

    def whitened_variance_spectrum(self) -> np.ndarray:
        """Per-eigendirection variance the whitened data actually gets:
        λ_raw/λ_reg, ascending in λ_raw. 1.0 = exactly whitened; < 1 =
        under-whitened by regularization (the floor/shrinkage bit)."""
        return self.eigenvalues_raw / self.eigenvalues

    def expected_whitened_variance(self) -> float:
        """Mean per-element variance of whitened TRAIN data: (1/d)·tr(WΣW).
        The distribution gate in whiten_dataset.py checks batches against
        this — ≈1 with no regularization, below 1 with it."""
        return float(self.whitened_variance_spectrum().mean())

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
            "floor_quantile": self.floor_quantile,
            "n_samples": self.n_samples,
            "source": self.source,
            "base_model": self.base_model,
            "layer_index": self.layer_index,
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
        if len(col) == 0:
            continue
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
    floor_quantile: float = DEFAULT_FLOOR_QUANTILE,
    source: str = "",
    base_model: str = "",
    layer_index: int | None = None,
) -> WhiteningStats:
    """Finish the streaming computation: moments → μ, Σ, regularize, eigh,
    W, W⁻¹. n: row count, s: Σx (d,), outer: Σ x xᵀ (d, d) — all float64.

    Regularization order: shrink toward (trΣ/d)·I first (if shrinkage > 0),
    then floor eigenvalues at the floor_quantile of the (shrunk) spectrum.
    """
    d = s.shape[0]
    assert n >= 2, f"need ≥2 samples to estimate covariance, got {n}"
    assert 0.0 <= shrinkage < 1.0, f"shrinkage must be in [0, 1), got {shrinkage}"
    assert 0.0 <= floor_quantile < 1.0, f"floor_quantile must be in [0, 1), got {floor_quantile}"
    mean = s / n
    # Unbiased sample covariance; symmetrize to kill accumulation asymmetry.
    cov = (outer - n * np.outer(mean, mean)) / (n - 1)
    cov = (cov + cov.T) / 2.0

    iso = np.trace(cov) / d
    assert iso > 0, "covariance has non-positive trace — degenerate input?"
    if shrinkage > 0:
        cov = (1.0 - shrinkage) * cov + shrinkage * iso * np.eye(d)

    eigvals_raw, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals_raw
    if floor_quantile > 0:
        floor = np.quantile(eigvals_raw, floor_quantile)
        eigvals = np.maximum(eigvals_raw, floor)
    # A floor computed from a mostly-degenerate spectrum is itself ~0 (or
    # negative) — flooring must not silently "rescue" rank-deficient data.
    assert eigvals[0] > 1e-8 * iso, (
        f"smallest regularized eigenvalue {eigvals[0]:.3e} ≲ 0 "
        f"(shrinkage={shrinkage}, floor_quantile={floor_quantile}, n={n}, d={d}). "
        f"Sample covariance is rank-deficient — use more rows (need n ≫ d) or "
        f"positive --shrinkage."
    )
    w = (eigvecs * (eigvals**-0.5)) @ eigvecs.T
    w_inv = (eigvecs * (eigvals**0.5)) @ eigvecs.T
    return WhiteningStats(
        mean=mean,
        w=(w + w.T) / 2.0,
        w_inv=(w_inv + w_inv.T) / 2.0,
        eigenvalues=eigvals,
        eigenvalues_raw=eigvals_raw,
        shrinkage=shrinkage,
        floor_quantile=floor_quantile,
        n_samples=n,
        source=source,
        base_model=base_model,
        layer_index=layer_index,
    )


def compute_stats_from_array(x: np.ndarray, *, source: str = "", **kwargs) -> WhiteningStats:
    """In-memory convenience path (tests, small data). kwargs as in
    compute_stats_from_moments (shrinkage, floor_quantile, provenance)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2, f"expected (n, d), got shape {x.shape}"
    return compute_stats_from_moments(
        x.shape[0], x.sum(axis=0), x.T @ x, source=source, **kwargs
    )


def compute_whitening_stats(
    parquet_source,
    *,
    max_rows: int | None = None,
    batch_size: int = 8192,
    source: str | None = None,
    **kwargs,
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
        n, s, outer, source=source or str(parquet_source), **kwargs
    )


def whiten(x: np.ndarray, stats: WhiteningStats) -> np.ndarray:
    """x̃ = (x − μ) W. Accepts (n, d) or (d,); float64 math, float64 out —
    caller casts to storage dtype."""
    return (np.asarray(x, dtype=np.float64) - stats.mean) @ stats.w


def unwhiten(x: np.ndarray, stats: WhiteningStats) -> np.ndarray:
    """Inverse: x = x̃ W⁻¹ + μ. Use to map a whitened-space AR prediction back
    to the raw residual stream (raw-space eval, steering, inspection)."""
    return np.asarray(x, dtype=np.float64) @ stats.w_inv + stats.mean


def save_stats(stats: WhiteningStats, path: str) -> None:
    np.savez_compressed(
        path,
        mean=stats.mean,
        w=stats.w,
        w_inv=stats.w_inv,
        eigenvalues=stats.eigenvalues,
        eigenvalues_raw=stats.eigenvalues_raw,
        shrinkage=np.float64(stats.shrinkage),
        floor_quantile=np.float64(stats.floor_quantile),
        n_samples=np.int64(stats.n_samples),
        source=np.str_(stats.source),
        base_model=np.str_(stats.base_model),
        layer_index=np.int64(-1 if stats.layer_index is None else stats.layer_index),
        sha256=np.str_(stats.sha256()),
    )


def load_stats(path: str) -> WhiteningStats:
    z = np.load(path, allow_pickle=False)
    layer = int(z["layer_index"])
    stats = WhiteningStats(
        mean=z["mean"],
        w=z["w"],
        w_inv=z["w_inv"],
        eigenvalues=z["eigenvalues"],
        eigenvalues_raw=z["eigenvalues_raw"],
        shrinkage=float(z["shrinkage"]),
        floor_quantile=float(z["floor_quantile"]),
        n_samples=int(z["n_samples"]),
        source=str(z["source"]),
        base_model=str(z["base_model"]),
        layer_index=None if layer < 0 else layer,
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
    ev, raw = stats.eigenvalues, stats.eigenvalues_raw
    v = stats.whitened_variance_spectrum()
    lines = [
        f"d_model        : {stats.d_model}",
        f"n_samples      : {stats.n_samples}",
        f"base_model     : {stats.base_model or '(unset)'} · layer {stats.layer_index}",
        f"regularization : shrinkage={stats.shrinkage} floor_quantile={stats.floor_quantile}",
        f"eig raw        : min {raw[0]:.4g} · median {np.median(raw):.4g} · max {raw[-1]:.4g}",
        f"eig regularized: min {ev[0]:.4g} · condition # {ev[-1] / ev[0]:.4g}",
        f"whitened var   : mean {v.mean():.3f} · min {v.min():.3f} · "
        f"frac dirs < 0.9: {(v < 0.9).mean():.1%}",
        f"mean ‖μ‖       : {np.linalg.norm(stats.mean):.4g}",
        f"sha256         : {stats.sha256()[:16]}…",
    ]
    return "\n".join(lines)
