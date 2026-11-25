"""Geometry, probing, KL, leakage, verbatim quoting, and bootstrap CI metrics.

Used by Phase 0 (Geometry & Base-Wrapper Gate) and Phase 4 (Full Auditing Suite).
All functions are pure CPU/GPU numpy/torch operations with zero external API calls.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def unit_normalize(X: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Row-wise unit normalization in float32."""
    Xf = X.float()
    return Xf / (Xf.norm(dim=-1, keepdim=True) + eps)


def compute_fisher_snr(X: torch.Tensor | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    """Binary Fisher Signal-to-Noise Ratio (trace-normalized centroid separation).

        SNR = ||mu_1 - mu_0||_2^2 / (0.5 * (E_{y=0}[||x - mu_0||^2] + E_{y=1}[||x - mu_1||^2]))

    Scale-invariant and well-conditioned even when d >> N (e.g. d=4096, N=100).
    """
    arr = X.detach().float().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float64)
    arr = arr.astype(np.float64, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    x0 = arr[y == 0]
    x1 = arr[y == 1]
    assert len(x0) >= 2 and len(x1) >= 2, (
        f"Need >= 2 samples per class for Fisher SNR, got n0={len(x0)}, n1={len(x1)}"
    )
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    between = float(np.sum((mu1 - mu0) ** 2))
    var0 = float(np.mean(np.sum((x0 - mu0) ** 2, axis=1)))
    var1 = float(np.mean(np.sum((x1 - mu1) ** 2, axis=1)))
    within = 0.5 * (var0 + var1)
    if within <= 1e-18:
        return 0.0
    return float(between / within)


def compute_probe_auroc_cv(
    X: torch.Tensor | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_folds: int = 5,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 0,
) -> float:
    """5-fold stratified CV linear-probe AUROC (fast dual Ridge classifier).

    Solves in N x N dual space (X_tr X_tr^T + alpha I) so d=4096, N=200 runs
    in milliseconds on CPU with no convergence warnings.
    """
    arr = X.detach().float().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float64)
    arr = arr.astype(np.float64, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    assert set(np.unique(y)) == {0, 1}, f"Expected binary labels {{0,1}}, got {np.unique(y)}"

    n_splits = min(n_folds, int(np.sum(y == 0)), int(np.sum(y == 1)))
    assert n_splits >= 2, "Need at least 2 samples per class for CV probe"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_auroc = 0.0
    for alpha in alphas:
        oof_scores = np.zeros(len(y), dtype=np.float64)
        for tr_idx, va_idx in skf.split(arr, y):
            X_tr = arr[tr_idx]
            X_va = arr[va_idx]
            mu = X_tr.mean(axis=0, keepdims=True)
            Xc_tr = X_tr - mu
            Xc_va = X_va - mu
            y_tr = (2.0 * y[tr_idx] - 1.0)
            y_mean = y_tr.mean()
            yc_tr = y_tr - y_mean
            # Dual ridge: w = Xc_tr^T (Xc_tr Xc_tr^T + alpha I)^{-1} yc_tr
            K = Xc_tr @ Xc_tr.T
            K[np.diag_indices_from(K)] += float(alpha)
            dual = np.linalg.solve(K, yc_tr)
            scores = (Xc_va @ Xc_tr.T) @ dual + y_mean
            oof_scores[va_idx] = scores
        auroc = float(roc_auc_score(y, oof_scores))
        if auroc > best_auroc:
            best_auroc = auroc
    return best_auroc


def compute_tok1_ridge_fve(
    H: torch.Tensor | np.ndarray,
    token_ids: Sequence[int] | np.ndarray,
    *,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """Closed-form K-fold CV Ridge FVE predicting activations H from one-hot token_id.

    Because the one-hot design matrix Z has diagonal Gram matrix Z^T Z = diag(n_v),
    exact centered Ridge regression on an arbitrarily large vocabulary (e.g. 151,936)
    has closed-form solution per token v:
        y_hat(v) = mu_train + (1 / (n_v + alpha)) * sum_{i in train: tok_i == v} (h_i - mu_train)
    Runs in O(N * d) time without ever materializing a V x V matrix.
    """
    arr = H.detach().float().cpu().numpy() if isinstance(H, torch.Tensor) else np.asarray(H, dtype=np.float64)
    arr = arr.astype(np.float64, copy=False)
    toks = np.asarray(token_ids, dtype=np.int64)
    n, d = arr.shape
    assert len(toks) == n
    if n < n_folds:
        return 0.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.empty(n, dtype=np.int64)
    for rank, idx in enumerate(perm):
        folds[idx] = rank % n_folds

    best_fve = -1e9
    total_var = float(np.mean(np.sum((arr - arr.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    if total_var <= 1e-18:
        return 0.0

    for alpha in alphas:
        sq_err = 0.0
        for f in range(n_folds):
            va = folds == f
            tr = ~va
            H_tr = arr[tr]
            t_tr = toks[tr]
            mu = H_tr.mean(axis=0)
            Hc_tr = H_tr - mu

            # Accumulate centered sum & count per token id in training fold
            sum_by_tok: dict[int, np.ndarray] = {}
            cnt_by_tok: dict[int, int] = defaultdict(int)
            for i in range(len(t_tr)):
                tid = int(t_tr[i])
                cnt_by_tok[tid] += 1
                if tid in sum_by_tok:
                    sum_by_tok[tid] += Hc_tr[i]
                else:
                    sum_by_tok[tid] = Hc_tr[i].copy()

            H_va = arr[va]
            t_va = toks[va]
            for i in range(len(t_va)):
                tid = int(t_va[i])
                c = cnt_by_tok.get(tid, 0)
                if c > 0:
                    pred = mu + sum_by_tok[tid] / (c + float(alpha))
                else:
                    pred = mu
                diff = H_va[i] - pred
                sq_err += float(np.dot(diff, diff))
        mse = sq_err / n
        fve = 1.0 - mse / total_var
        if fve > best_fve:
            best_fve = fve
    return float(best_fve)


def compute_next_token_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(P || Q) in nats for logits of shape [..., V] in float32."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def compute_chat_template_amplification(
    delta_raw_user: torch.Tensor,
    delta_wrapped_user: torch.Tensor,
) -> float:
    """Ratio mean(||Delta_wrapped||_2) / mean(||Delta_raw||_2) on identical user tokens."""
    n_raw = float(delta_raw_user.float().norm(dim=-1).mean().item())
    n_wrap = float(delta_wrapped_user.float().norm(dim=-1).mean().item())
    return float(n_wrap / max(n_raw, 1e-12))


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def longest_verbatim_ngram(explanation: str, source_text: str) -> int:
    """Length (in words) of the longest contiguous verbatim substring shared by both texts."""
    e_words = _tokenize_words(explanation)
    s_words = _tokenize_words(source_text)
    if not e_words or not s_words:
        return 0
    # Dynamic programming over word tokens (explanations <= ~250 words)
    prev = [0] * (len(s_words) + 1)
    best = 0
    for ew in e_words:
        curr = [0] * (len(s_words) + 1)
        for j, sw in enumerate(s_words, start=1):
            if ew == sw:
                v = prev[j - 1] + 1
                curr[j] = v
                if v > best:
                    best = v
        prev = curr
    return int(best)


def verbatim_ngram_precision(explanation: str, source_text: str, n: int = 4) -> float:
    """Fraction of n-grams in `explanation` that appear verbatim in `source_text`."""
    e_words = _tokenize_words(explanation)
    s_words = _tokenize_words(source_text)
    if len(e_words) < n or len(s_words) < n:
        return 0.0
    src_ngrams = {tuple(s_words[i : i + n]) for i in range(len(s_words) - n + 1)}
    exp_ngrams = [tuple(e_words[i : i + n]) for i in range(len(e_words) - n + 1)]
    hits = sum(1 for g in exp_ngrams if g in src_ngrams)
    return float(hits / len(exp_ngrams))


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Non-parametric percentile bootstrap CI for a 1-D metric array."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    point = float(stat_fn(arr))
    if len(arr) == 1:
        return {"mean": point, "ci_low": point, "ci_high": point, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = np.array([stat_fn(arr[row]) for row in idx], dtype=np.float64)
    alpha = 0.5 * (1.0 - ci)
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {"mean": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(len(arr))}


def bootstrap_auroc_ci(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Stratified/paired bootstrap 95% CI for binary AUROC."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(ys)
    yt, ys = yt[valid], ys[valid]
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(len(yt))}
    point = float(roc_auc_score(yt, ys))
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = yt[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(float(roc_auc_score(yb, ys[idx])))
    if not boots:
        return {"auroc": point, "ci_low": point, "ci_high": point, "n": int(n)}
    alpha = 0.5 * (1.0 - ci)
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {"auroc": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}
