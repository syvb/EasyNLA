"""Fix 1: Identity-Anchored Ridge Alignment (W_id_anc = I + delta_W).

Fits an affine alignment from Base-model residual-stream activations h_base to
Instruct/Target activations h_target strictly on raw pretraining tokens (never
chat transcripts), solving in float64:

    min_{dW, b} (1/N) sum_i || (h_t_i - h_b_i) - (dW h_b_i + b) ||_2^2 + alpha ||dW||_F^2

Why Identity-Anchored instead of Standard Ridge:
  Standard Ridge shrinks W -> 0, throwing away the large shared coordinate
  frame between Base and Instruct and underperforming W = I on held-out web
  tokens. Anchoring regularization on dW = W - I shrinks toward W = I as
  alpha -> inf, guaranteeing held-out FVE >= Identity+Bias (up to CV variance).

Note on alpha scaling:
  We solve the 1/N-normalized objective directly (matching Eq. 2.1 of the
  experiment plan) AND sweep an expanded log-grid covering both small alphas
  (1e-2..1e0, typical optimum under 1/N normalization) and large alphas
  (1e1..1e6, matching unnormalized/sklearn scale) via a single float64
  symmetric eigendecomposition of C = X^T X / N. Both `alpha` (1/N objective)
  and `alpha_sum_loss = alpha * N` (sum-of-squares objective) are persisted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# Expanded CV grid spanning both 1/N-normalized and unnormalized effective scales.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    1e-3,
    1e-2,
    1e-1,
    1.0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
)


def compute_fve(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Fraction of Variance Explained: 1 - E[||y - y_hat||^2] / E[||y - mean(y)||^2]."""
    yt = y_true.double()
    yp = y_pred.double()
    mse = (yt - yp).pow(2).sum(dim=-1).mean().item()
    var = (yt - yt.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean().item()
    if var <= 1e-18:
        return 0.0
    return float(1.0 - mse / var)


def doc_grouped_folds(
    doc_ids: Sequence[Any],
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Assign each row to a fold in [0, n_folds) so no doc_id spans two folds."""
    n = len(doc_ids)
    unique_docs = list(dict.fromkeys(doc_ids))
    if len(unique_docs) < n_folds:
        raise ValueError(
            f"Need at least n_folds={n_folds} unique doc_ids, got {len(unique_docs)}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_docs))
    doc_to_fold = {
        unique_docs[int(idx)]: int(rank % n_folds)
        for rank, idx in enumerate(perm)
    }
    folds = np.fromiter((doc_to_fold[d] for d in doc_ids), dtype=np.int64, count=n)
    return folds


def _ridge_solutions_from_cov(
    X_mean: torch.Tensor,
    Y_mean: torch.Tensor,
    cov_xx: torch.Tensor,
    cov_xy: torch.Tensor,
    alphas: Sequence[float],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve (cov_xx + alpha * I) W^T = cov_xy for all alphas via one eigh.

    All inputs are float64 on the target compute device.
    Returns list of (W [d, d], b [d]) pairs in float64 where y_hat = x @ W^T + b.
    """
    eigvals, Q = torch.linalg.eigh(cov_xx)
    eigvals = eigvals.clamp_min(0.0)
    QtM = Q.T @ cov_xy  # [d, d]
    solutions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for alpha in alphas:
        inv_eigs = 1.0 / (eigvals + float(alpha))
        W_T = Q @ (QtM * inv_eigs.unsqueeze(1))
        W = W_T.T  # [d, d], y_hat = x @ W^T + b
        b = Y_mean - (X_mean @ W_T)
        solutions.append((W, b))
    return solutions


def _centered_covariances(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (X_mean, Y_mean, X_c^T X_c / N, X_c^T Y_c / N) in float64."""
    X64 = X.double()
    Y64 = Y.double()
    n = X64.shape[0]
    assert n >= 2, f"Need at least 2 samples to fit alignment, got {n}"
    xm = X64.mean(dim=0)
    ym = Y64.mean(dim=0)
    xc = X64 - xm
    yc = Y64 - ym
    cov_xx = (xc.T @ xc) / float(n)
    cov_xy = (xc.T @ yc) / float(n)
    return xm, ym, cov_xx, cov_xy


def fit_alignment(
    H_base: torch.Tensor,
    H_target: torch.Tensor,
    *,
    mode: str = "id_anchored",
    alpha: float = 10.0,
    layer_index: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Fit a single alignment transform in float64.

    Modes:
      - "identity":       h_hat = h_base
      - "identity_bias":  h_hat = h_base + mean(h_target - h_base)
      - "id_anchored":    h_hat = h_base + dW @ h_base + b   (reg: alpha * ||dW||_F^2)
      - "standard_ridge": h_hat = W @ h_base + b             (reg: alpha * ||W||_F^2)
    """
    if device is None:
        device = H_base.device
    Hb = H_base.to(device=device, dtype=torch.float64)
    Ht = H_target.to(device=device, dtype=torch.float64)
    n, d = Hb.shape
    assert Ht.shape == (n, d), f"Shape mismatch: H_base={tuple(Hb.shape)} vs H_target={tuple(Ht.shape)}"

    artifact: dict[str, Any] = {
        "mode": mode,
        "alpha": float(alpha),
        "alpha_sum_loss": float(alpha) * n,
        "n_train": int(n),
        "d_model": int(d),
        "layer_index": layer_index,
    }

    if mode == "identity":
        artifact["delta_W"] = torch.zeros((d, d), dtype=torch.float32)
        artifact["bias"] = torch.zeros((d,), dtype=torch.float32)
    elif mode == "identity_bias":
        b = (Ht - Hb).mean(dim=0)
        artifact["delta_W"] = torch.zeros((d, d), dtype=torch.float32)
        artifact["bias"] = b.float().cpu()
    elif mode == "id_anchored":
        R = Ht - Hb
        xm, ym, cxx, cxy = _centered_covariances(Hb, R)
        (dW, b) = _ridge_solutions_from_cov(xm, ym, cxx, cxy, [alpha])[0]
        artifact["delta_W"] = dW.float().cpu()
        artifact["bias"] = b.float().cpu()
    elif mode == "standard_ridge":
        xm, ym, cxx, cxy = _centered_covariances(Hb, Ht)
        (W, b) = _ridge_solutions_from_cov(xm, ym, cxx, cxy, [alpha])[0]
        artifact["W"] = W.float().cpu()
        artifact["bias"] = b.float().cpu()
    else:
        raise ValueError(f"Unknown alignment mode: {mode!r}")

    return artifact


def predict_aligned_base(
    H_base: torch.Tensor,
    artifact: dict[str, Any],
) -> torch.Tensor:
    """Predict target-space activations h_hat from H_base using `artifact`.

    Supports 1-D [d] or 2-D [N, d] tensors. Returns float32 on H_base.device.
    """
    squeeze = H_base.ndim == 1
    Hb = H_base.unsqueeze(0).float() if squeeze else H_base.float()
    mode = artifact["mode"]
    dev = Hb.device
    if mode == "identity":
        out = Hb
    elif mode == "identity_bias":
        b = artifact["bias"].to(dev, dtype=torch.float32)
        out = Hb + b
    elif mode == "id_anchored":
        dW = artifact["delta_W"].to(dev, dtype=torch.float32)
        b = artifact["bias"].to(dev, dtype=torch.float32)
        out = Hb + (Hb @ dW.T) + b
    elif mode == "standard_ridge":
        W = artifact["W"].to(dev, dtype=torch.float32)
        b = artifact["bias"].to(dev, dtype=torch.float32)
        out = (Hb @ W.T) + b
    else:
        raise ValueError(f"Unknown alignment mode: {mode!r}")
    return out.squeeze(0) if squeeze else out


def apply_alignment(
    H_base: torch.Tensor,
    H_target: torch.Tensor,
    artifact: dict[str, Any],
) -> torch.Tensor:
    """Compute aligned diff Delta = H_target - h_hat(H_base) in float32."""
    pred = predict_aligned_base(H_base, artifact)
    return H_target.to(pred.device, dtype=torch.float32) - pred


def cv_select_alignment(
    H_base: torch.Tensor,
    H_target: torch.Tensor,
    doc_ids: Sequence[Any],
    *,
    mode: str = "id_anchored",
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    n_folds: int = 5,
    seed: int = 0,
    layer_index: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Select alpha by doc-grouped K-fold CV in float64, then refit on all rows."""
    assert mode in ("id_anchored", "standard_ridge"), f"CV only applies to ridge modes, got {mode!r}"
    if device is None:
        device = H_base.device
    Hb = H_base.to(device=device, dtype=torch.float64)
    Ht = H_target.to(device=device, dtype=torch.float64)
    n, _d = Hb.shape
    assert len(doc_ids) == n, f"len(doc_ids)={len(doc_ids)} != N={n}"

    folds = doc_grouped_folds(doc_ids, n_folds=n_folds, seed=seed)
    folds_t = torch.from_numpy(folds).to(device=device)

    alpha_list = [float(a) for a in alphas]
    fold_fves = np.zeros((n_folds, len(alpha_list)), dtype=np.float64)

    for fold_idx in range(n_folds):
        val_mask = folds_t == fold_idx
        tr_mask = ~val_mask
        Hb_tr, Ht_tr = Hb[tr_mask], Ht[tr_mask]
        Hb_va, Ht_va = Hb[val_mask], Ht[val_mask]
        Y_tr = (Ht_tr - Hb_tr) if mode == "id_anchored" else Ht_tr
        xm, ym, cxx, cxy = _centered_covariances(Hb_tr, Y_tr)
        sols = _ridge_solutions_from_cov(xm, ym, cxx, cxy, alpha_list)
        # Precompute denominator for val FVE once per fold
        var_va = (Ht_va - Ht_va.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean().item()
        for ai, (M, b) in enumerate(sols):
            if mode == "id_anchored":
                pred_va = Hb_va + (Hb_va @ M.T) + b
            else:
                pred_va = (Hb_va @ M.T) + b
            mse_va = (Ht_va - pred_va).pow(2).sum(dim=-1).mean().item()
            fold_fves[fold_idx, ai] = 1.0 - mse_va / max(var_va, 1e-18)

    mean_fves = fold_fves.mean(axis=0)
    best_idx = int(np.argmax(mean_fves))
    best_alpha = alpha_list[best_idx]

    artifact = fit_alignment(
        Hb, Ht, mode=mode, alpha=best_alpha, layer_index=layer_index, device=device,
    )
    artifact["cv_fve_by_alpha"] = {str(a): float(mean_fves[i]) for i, a in enumerate(alpha_list)}
    artifact["best_cv_fve"] = float(mean_fves[best_idx])
    artifact["n_folds"] = int(n_folds)
    return artifact


def evaluate_alignment_suite(
    H_base_train: torch.Tensor,
    H_target_train: torch.Tensor,
    train_doc_ids: Sequence[Any],
    H_base_val: torch.Tensor,
    H_target_val: torch.Tensor,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    n_folds: int = 5,
    seed: int = 0,
    layer_index: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compare Identity, Identity+Bias, Standard Ridge, and Identity-Anchored Ridge.

    Returns dict with:
      - fve_identity, fve_identity_bias, fve_std_ridge, fve_id_anchored
      - tau_web_median, tau_web_p75, delta_norm_stats
      - artifact_id_anchored, artifact_std_ridge
      - fix1_passed (bool), identity_already_tight (bool)
    """
    if device is None:
        device = H_base_train.device

    art_id = fit_alignment(
        H_base_train, H_target_train, mode="identity", layer_index=layer_index, device=device,
    )
    art_idb = fit_alignment(
        H_base_train, H_target_train, mode="identity_bias", layer_index=layer_index, device=device,
    )
    art_std = cv_select_alignment(
        H_base_train, H_target_train, train_doc_ids,
        mode="standard_ridge", alphas=alphas, n_folds=n_folds, seed=seed,
        layer_index=layer_index, device=device,
    )
    art_idanc = cv_select_alignment(
        H_base_train, H_target_train, train_doc_ids,
        mode="id_anchored", alphas=alphas, n_folds=n_folds, seed=seed,
        layer_index=layer_index, device=device,
    )

    Hb_va = H_base_val.to(device=device, dtype=torch.float32)
    Ht_va = H_target_val.to(device=device, dtype=torch.float32)

    fve_id = compute_fve(Ht_va, predict_aligned_base(Hb_va, art_id))
    fve_idb = compute_fve(Ht_va, predict_aligned_base(Hb_va, art_idb))
    fve_std = compute_fve(Ht_va, predict_aligned_base(Hb_va, art_std))
    fve_idanc = compute_fve(Ht_va, predict_aligned_base(Hb_va, art_idanc))

    # Attach held-out verification metrics onto the artifacts before saving
    art_idanc["val_fve_id_anchored"] = fve_idanc
    art_idanc["val_fve_identity"] = fve_id
    art_idanc["val_fve_identity_bias"] = fve_idb
    art_idanc["val_fve_std_ridge"] = fve_std

    art_std["val_fve_std_ridge"] = fve_std
    art_std["val_fve_identity"] = fve_id

    # Compute Fix 3 noise floor tau on held-out web val tokens
    delta_val = apply_alignment(Hb_va, Ht_va, art_idanc)
    norms = delta_val.norm(dim=-1).cpu().numpy()
    tau_median = float(np.median(norms))
    tau_p75 = float(np.percentile(norms, 75))
    tau_mean = float(np.mean(norms))
    tau_p25 = float(np.percentile(norms, 25))
    tau_p90 = float(np.percentile(norms, 90))

    art_idanc["tau_web_median"] = tau_median
    art_idanc["tau_web_p75"] = tau_p75

    # Gate 0 Fix 1 criterion:
    # FVE_id_anc(web_val) >= FVE_identity(web_val) > FVE_std_ridge(web_val)
    # Allow tiny 1e-4 numeric tolerance on >= when Identity is already near-optimal.
    fix1_strict = (fve_idanc >= fve_id - 1e-4) and (fve_id > fve_std)
    # Also track whether id_anchored beats or ties BOTH identity and std_ridge
    # (in case standard_ridge matches identity at 8B with small alpha under expanded grid!)
    fix1_best = (fve_idanc >= fve_id - 1e-4) and (fve_idanc >= fve_std - 1e-4)

    return {
        "fve_identity": fve_id,
        "fve_identity_bias": fve_idb,
        "fve_std_ridge": fve_std,
        "fve_id_anchored": fve_idanc,
        "best_alpha_id_anchored": art_idanc["alpha"],
        "best_alpha_std_ridge": art_std["alpha"],
        "cv_fve_id_anchored": art_idanc["cv_fve_by_alpha"],
        "cv_fve_std_ridge": art_std["cv_fve_by_alpha"],
        "tau_web_median": tau_median,
        "tau_web_p75": tau_p75,
        "delta_norm_stats": {
            "mean": tau_mean,
            "p25": tau_p25,
            "median": tau_median,
            "p75": tau_p75,
            "p90": tau_p90,
        },
        "identity_already_tight": bool(fve_id > 0.95),
        "fix1_strict_passed": bool(fix1_strict),
        "fix1_best_passed": bool(fix1_best),
        "artifact_id_anchored": art_idanc,
        "artifact_std_ridge": art_std,
    }


def save_alignment(path: str | Path, artifact: dict[str, Any]) -> Path:
    """Save alignment artifact dict (.pt) safely via atomic temp rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(artifact, str(tmp))
    tmp.replace(p)
    return p


def load_alignment(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load alignment artifact dict (.pt)."""
    return torch.load(str(path), map_location=map_location, weights_only=False)
