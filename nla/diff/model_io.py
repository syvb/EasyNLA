"""Shared single-model activation & logit extraction utilities for Diff-NLA.

Loads one HF CausalLM at a time in bf16 (or fp32 on CPU), extracts Layer-K
residual-stream hidden states (matching `nla/datagen/extractors.py` and
`NLACriticModel`: output of decoder block K == `hidden_states[K + 1]`),
optionally captures exact logits or top-K next-token distributions at selected
positions, and cleanly frees VRAM between models.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config


class _CaptureStop(Exception):
    """Short-circuit forward pass right after target layer when logits are unneeded."""


@dataclass
class PositionExtraction:
    """Extracted hidden states (and optional logits / top-K next tokens) for one sequence."""

    positions: list[int]
    token_ids: list[int]
    hidden: torch.Tensor  # [len(positions), d_model], float32 CPU
    logits: torch.Tensor | None = None  # [len(positions), vocab], float32 CPU (only if need_logits)
    topk_ids: list[list[int]] | None = None
    topk_probs: list[list[float]] | None = None


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        # Avoid slow/unsupported bf16 ops on some CPU builds during tests
        return torch.float32
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str!r} (expected bf16|fp32)")


def load_causal_lm(
    model_name_or_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    adapter_path: str | None = None,
) -> Any:
    """Load an HF CausalLM (and optional PEFT LoRA adapter) in eval mode."""
    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def unload_model(model: Any) -> None:
    """Move weights off GPU, release references, and clear CUDA cache cleanly.

    Callers should assign the return value (`model = unload_model(model)`) so
    the caller's local reference is immediately dropped alongside VRAM.
    """
    if model is not None and hasattr(model, "to"):
        from contextlib import suppress

        with suppress(Exception):
            model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unwrap_for_layers(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "peft_config") else model


@torch.no_grad()
def extract_at_positions(
    model: Any,
    sequences: Sequence[Sequence[int]],
    positions_per_seq: Sequence[Sequence[int]],
    *,
    layer_index: int,
    pad_token_id: int,
    batch_size: int = 8,
    need_logits: bool = False,
    topk_logits: int = 0,
) -> list[PositionExtraction]:
    """Extract Layer-K residual states (and optional logits/top-K) at exact token positions.

    Right-pads every batch with explicit `attention_mask` so token indices are never
    shifted. Short-circuits right after decoder block `layer_index` whenever
    `need_logits=False and topk_logits == 0`, skipping upper blocks and `lm_head`.
    """
    assert len(sequences) == len(positions_per_seq)
    if not sequences:
        return []

    base_model = _unwrap_for_layers(model)
    layers = resolve_decoder_layers(base_model)
    assert 0 <= layer_index < len(layers), (
        f"layer_index={layer_index} out of range for {len(layers)} layers"
    )
    d_model = resolve_text_config(base_model.config).hidden_size
    device = next(model.parameters()).device
    short_circuit = (not need_logits) and (topk_logits <= 0)

    captured: list[torch.Tensor | None] = [None]

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured[0] = h.detach()
        if short_circuit:
            raise _CaptureStop

    handle = layers[layer_index].register_forward_hook(_hook)
    results: list[PositionExtraction] = []
    try:
        for start in range(0, len(sequences), batch_size):
            sub_seqs = sequences[start : start + batch_size]
            sub_pos = positions_per_seq[start : start + batch_size]
            bs = len(sub_seqs)
            max_len = max(len(s) for s in sub_seqs)

            input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
            for i, seq in enumerate(sub_seqs):
                L = len(seq)
                input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
                attn_mask[i, :L] = 1

            captured[0] = None
            out = None
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            except _CaptureStop:
                pass

            h_batch = captured[0]
            assert h_batch is not None, f"Hook on layer {layer_index} did not fire"
            assert h_batch.shape[-1] == d_model, (
                f"Hidden width mismatch: {h_batch.shape[-1]} != {d_model}"
            )
            logits_batch = out.logits if (out is not None and (need_logits or topk_logits > 0)) else None

            for i, (seq, pos_list) in enumerate(zip(sub_seqs, sub_pos)):
                pos_clean = [int(p) for p in pos_list]
                if not pos_clean:
                    results.append(
                        PositionExtraction(
                            positions=[],
                            token_ids=[],
                            hidden=torch.zeros((0, d_model), dtype=torch.float32),
                            logits=(
                                torch.zeros((0, logits_batch.shape[-1]), dtype=torch.float32)
                                if need_logits and logits_batch is not None
                                else None
                            ),
                            topk_ids=([] if topk_logits > 0 else None),
                            topk_probs=([] if topk_logits > 0 else None),
                        )
                    )
                    continue

                idx_t = torch.tensor(pos_clean, dtype=torch.long, device=device)
                h_sel = h_batch[i].index_select(0, idx_t).float().cpu().clone()
                tok_sel = [int(seq[p]) for p in pos_clean]

                lg_sel = None
                tk_ids = None
                tk_probs = None
                if logits_batch is not None:
                    raw_lg = logits_batch[i].index_select(0, idx_t).float()
                    if need_logits:
                        lg_sel = raw_lg.cpu().clone()
                    if topk_logits > 0:
                        probs = F.softmax(raw_lg, dim=-1)
                        k = min(topk_logits, probs.shape[-1])
                        vals, inds = probs.topk(k, dim=-1)
                        tk_ids = inds.cpu().tolist()
                        tk_probs = vals.cpu().tolist()

                results.append(
                    PositionExtraction(
                        positions=pos_clean,
                        token_ids=tok_sel,
                        hidden=h_sel,
                        logits=lg_sel,
                        topk_ids=tk_ids,
                        topk_probs=tk_probs,
                    )
                )
            del h_batch, logits_batch, out, input_ids, attn_mask
    finally:
        handle.remove()

    return results
