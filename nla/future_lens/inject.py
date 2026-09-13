"""Activation injection for the future-lens decoder.

Two mechanisms, selected by `--injection`:

  replace_embed (default, NLA paper): the marker token's *input embedding* is
      overwritten with `alpha * h / ||h||` (layer 0, before any transformer
      block). Vectors handed to the hook must already be scaled — see
      `scale_for_injection`. Implemented on top of `inject_at_marked_positions`
      (the same routine EasyNLA's vLLM path uses), so the marker-neighbour
      validity check and the loud count-mismatch failure are shared.

  karvonen (EasyNLA default, Activation Oracles eq. 1): additive, norm-matched
      to the decoder's own residual at the output of block `layer_idx` (1).
      Delegates to `nla.utils.hooks.register_karvonen_hook`.

Both hooks read the activation batch from `vectors_ref[0]` ([B, d] float, one
row per sequence in the batch, batch order) and are a no-op when it is None
(e.g. the frozen-target forward used by the log-prob reward) or when the
sequence length is 1 (KV-cached decode steps).

`AffineInjector` is the paper's "weakly recommended" learnable affine map:
identity-initialised Linear(d, d) applied to the *scaled* vector before the
embedding write. Off unless `--affine`.
"""

from __future__ import annotations

import torch

from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

INJECTION_MODES = ("replace_embed", "karvonen")


class AffineInjector(torch.nn.Module):
    """Identity-initialised affine map applied to the injected vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model))
            self.proj.bias.zero_()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.proj(v.to(self.proj.weight.dtype))


def scale_for_injection(vectors: torch.Tensor, alphas: torch.Tensor | float) -> torch.Tensor:
    """`alpha_i * v_i / ||v_i||` per row. `alphas` is a scalar or a [B] tensor."""
    v = vectors.float()
    if not torch.is_tensor(alphas):
        return normalize_activation(v, float(alphas))
    a = alphas.to(v.device).float().view(-1, 1)
    unit = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return unit * a


def register_replace_embed_hook(model, vectors_ref, inj_id, left_id, right_id,
                                affine: AffineInjector | None = None):
    """Forward hook on the input-embedding module: overwrite the marker rows.

    Returns the hook handle. `vectors_ref[0]` must hold the ALREADY-SCALED
    vectors ([B, d]); pass through `scale_for_injection` first.
    """
    embed = model.get_input_embeddings()

    def embed_hook(module, args, kwargs, output):
        v = vectors_ref[0]
        if v is None:
            return output
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        if ids is None or ids.dim() != 2 or ids.shape[1] < 2:
            # KV-cached decode step (S == 1): the marker was injected at prefill.
            return output
        if torch.is_tensor(v) and v.shape[0] == 0:
            return output
        v = v.to(output.device)
        if affine is not None:
            if next(affine.parameters()).device != output.device:   # device_map=auto
                affine.to(output.device)
            v = affine(v)
        return inject_at_marked_positions(
            ids.to(output.device), output, v, inj_id, left_id, right_id,
        )

    return embed.register_forward_hook(embed_hook, with_kwargs=True)


def register_injection(model, mode: str, vectors_ref, inj_id, left_id, right_id,
                       affine: AffineInjector | None = None):
    """Dispatch on `mode`; returns the handle (replace_embed) or None (karvonen)."""
    assert mode in INJECTION_MODES, f"--injection must be one of {INJECTION_MODES}, got {mode!r}"
    if mode == "replace_embed":
        return register_replace_embed_hook(model, vectors_ref, inj_id, left_id, right_id, affine)
    assert affine is None, "--affine is only meaningful with --injection replace_embed"
    from nla.utils.hooks import register_karvonen_hook
    register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1)
    return None


def prepare_vectors(raw: torch.Tensor, alphas, mode: str) -> torch.Tensor:
    """What goes into `vectors_ref[0]` for a batch of RAW activations.

    replace_embed: alpha-scaled unit vectors. karvonen: raw (its hook norm-matches).
    """
    if mode == "replace_embed":
        return scale_for_injection(raw, alphas)
    return raw.float()
