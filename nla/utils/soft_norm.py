"""Soft-clipped norm weighting (Fix 3) for Direct Diff-Target NLA training.

Single source of truth for the three Fix 3 sites across HF injection, vLLM
steering, AR critic MSE loss (SFT & RL), and post-standardization GRPO policy
advantages:

  1. Noise-floor gate:
         w_i = ||v_i||^2 / (||v_i||^2 + tau^2)   in [0, 1)
     When tau == 0.0, every helper reduces identically to baseline EasyNLA.
  2. AV injection scaling (HF hook & vllm-lens SteeringVector):
         v_scaled = v / (sqrt(||v||^2 + tau^2) + eps)
         h'_p     = h_p + ||h_p|| * v_scaled
     Prevents blind unit-normalization from amplifying near-zero web/noise
     diffs up to full residual-stream magnitude.
  3. AR critic loss & post-standardization GRPO advantages:
     Weight each sample by w_i / (mean(w) + 1e-8) so effective learning rate
     is preserved while low-norm noise diffs receive negligible gradient.
     Critically, GRPO advantage weighting MUST happen AFTER per-prompt group
     standardization (otherwise dividing by group std cancels w_i).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def soft_norm_weights(vectors: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute per-vector soft-clipping weights w_i = ||v_i||^2 / (||v_i||^2 + tau^2).

    Args:
        vectors: Raw unnormalized vectors of shape [..., d].
        tau: Soft-clipping noise floor (>= 0). When tau == 0.0, returns ones.

    Returns:
        Float32 tensor of shape [...] with values in [0, 1].
    """
    if tau < 0.0:
        raise ValueError(f"soft_norm_tau must be >= 0.0, got {tau}")
    if tau == 0.0:
        return torch.ones(vectors.shape[:-1], device=vectors.device, dtype=torch.float32)
    norm_sq = vectors.float().pow(2).sum(dim=-1)
    return norm_sq / (norm_sq + float(tau) ** 2)


def soft_norm_scale_factor(v_norm: float, tau: float, eps: float = 1e-9) -> float:
    """Scalar multiplier for vllm-lens norm_match steering.

    Because vllm-lens computes `h'_p = h_p + scale * ||h_p|| * (v / ||v||)`
    when `norm_match=True`, passing:
        scale = ||v|| / (sqrt(||v||^2 + tau^2) + eps)
    produces the exact soft-clipped injection `h'_p = h_p + ||h_p|| * v / (sqrt(||v||^2 + tau^2) + eps)`
    without requiring additional vllm-lens worker patches.
    """
    if tau < 0.0:
        raise ValueError(f"soft_norm_tau must be >= 0.0, got {tau}")
    if tau == 0.0:
        return 1.0
    if v_norm <= 1e-12:
        return 0.0
    return float(v_norm / (math.sqrt(v_norm * v_norm + tau * tau) + eps))


def soft_norm_scaled_vector(
    v: torch.Tensor,
    tau: float = 0.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Return unit-normalized (tau==0) or soft-clipped (tau>0) direction vector.

    When tau == 0.0, executes the exact legacy EasyNLA formula
    `v / (v.norm(dim=-1, keepdim=True) + eps)` in `v.dtype`.
    When tau > 0.0, computes the denominator in fp32 to avoid bf16/fp16
    overflow on large norms, then casts back to `v.dtype`.
    """
    if tau < 0.0:
        raise ValueError(f"soft_norm_tau must be >= 0.0, got {tau}")
    if tau == 0.0:
        return v / (v.norm(dim=-1, keepdim=True) + eps)
    v_f = v.float()
    denom = torch.sqrt(v_f.pow(2).sum(dim=-1, keepdim=True) + float(tau) ** 2) + eps
    return (v_f / denom).to(v.dtype)


def apply_soft_norm_to_advantages(
    adv: torch.Tensor,
    group_t: torch.Tensor,
    prompt_activations: torch.Tensor,
    tau: float,
    *,
    dist_mean_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weight standardized per-group GRPO advantages by per-prompt soft-norm gate.

    MUST be called AFTER per-prompt standardization `A_ij = (r_ij - mu_i) / (std_i + eps)`
    so `std_i` does not cancel `w_i`.

    Args:
        adv: Standardized rollout advantages of shape [N_rollouts].
        group_t: Prompt index per rollout in [0, B_prompts), shape [N_rollouts].
        prompt_activations: Raw unnormalized prompt vectors of shape [B_prompts, d].
        tau: Soft-clipping noise floor (>= 0). When 0.0, returns (adv, {}).
        dist_mean_fn: Optional callable that all-reduces (averages) the scalar
            `w_mean` tensor across DP ranks so every rank uses the identical
            global denominator.

    Returns:
        (weighted_adv, metrics_dict)
    """
    if tau < 0.0:
        raise ValueError(f"soft_norm_tau must be >= 0.0, got {tau}")
    if tau == 0.0:
        return adv, {}

    w_prompt = soft_norm_weights(prompt_activations.to(adv.device).float(), tau)
    w_mean = w_prompt.mean()
    if dist_mean_fn is not None:
        w_mean = dist_mean_fn(w_mean)
    w_scale = w_prompt / (w_mean + 1e-8)
    weighted_adv = adv * w_scale.index_select(0, group_t.to(adv.device))

    with torch.no_grad():
        w_cpu = w_prompt.detach().float()
        w_sum = w_cpu.sum().clamp_min(1e-12)
        ess_frac = float((w_sum * w_sum) / (max(1, w_cpu.numel()) * w_cpu.pow(2).sum().clamp_min(1e-12)))
        metrics = {
            "av/soft_norm_w_mean": float(w_mean.item()),
            "av/soft_norm_w_min": float(w_cpu.min().item()) if w_cpu.numel() else float("nan"),
            "av/soft_norm_w_p10": float(torch.quantile(w_cpu, 0.10).item()) if w_cpu.numel() else float("nan"),
            "av/soft_norm_w_max": float(w_cpu.max().item()) if w_cpu.numel() else float("nan"),
            "av/soft_norm_ess_frac": ess_frac,
        }
    return weighted_adv, metrics


def weighted_mse_loss(
    pred_n: torch.Tensor,
    gold_n: torch.Tensor,
    weights: torch.Tensor,
    total_weight_sum: float | torch.Tensor,
) -> torch.Tensor:
    """Micro-batch-invariant weighted per-element MSE loss for AR critic training.

    Summing this over micro-batches covering a full batch of N examples with
    `total_weight_sum = sum_{i=1}^N w_i` produces the exact full-batch weighted
    mean `sum_i (w_i * mse_i) / sum_i w_i == mean_i ((w_i / w_mean) * mse_i)`.
    When all `w_i == 1.0` and `total_weight_sum == N`, equals
    `F.mse_loss(pred_n, gold_n) * (bs / N)`.
    """
    per_row_mse = ((pred_n - gold_n) ** 2).mean(dim=-1)
    w = weights.to(device=per_row_mse.device, dtype=per_row_mse.dtype)
    denom = (
        total_weight_sum.to(device=per_row_mse.device, dtype=per_row_mse.dtype).clamp_min(1e-8)
        if isinstance(total_weight_sum, torch.Tensor)
        else max(float(total_weight_sum), 1e-8)
    )
    return (per_row_mse * w).sum() / denom
