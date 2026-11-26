"""Whole-model CPU loader for generation when fp32 weights do not fit in RAM.

Weights stay resident in bf16 (the checkpoint dtype). The first `n_resident`
decoder blocks are cast to fp32 once; every other block and the lm_head are
cast to fp32 only for the duration of their own forward call. All arithmetic is
therefore fp32 on exactly the checkpoint values, the same numbers
`load_causal_lm(..., dtype=float32)` computes, at about half the memory.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def _make_jit(mod: torch.nn.Module) -> None:
    def pre(m: torch.nn.Module, _args: Any) -> None:
        m._jit_orig = [(p, p.data) for p in m.parameters()]  # type: ignore[attr-defined]
        for p, d in m._jit_orig:  # type: ignore[attr-defined]
            p.data = d.float()

    def post(m: torch.nn.Module, _args: Any, _out: Any) -> None:
        for p, d in m._jit_orig:  # type: ignore[attr-defined]
            p.data = d
        m._jit_orig = None  # type: ignore[attr-defined]

    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)


def load_full_jit(name: str, *, n_resident: int | None = None, keep_head: bool = True) -> Any:
    if n_resident is None:
        n_resident = int(os.environ.get("NLA_JIT_RESIDENT", "16"))
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model.eval()
    if not keep_head:
        # Activation-only use: drop the unembedding (1.25 GB in bf16, 2.5 GB in fp32).
        model.lm_head = torch.nn.Identity()
    model.model.norm.float()
    model.model.embed_tokens.register_forward_hook(lambda _m, _a, out: out.float())
    for i, layer in enumerate(model.model.layers):
        if i < n_resident:
            layer.float()
        else:
            _make_jit(layer)
    # Resident fp32 head: 1.25 GB more than bf16, but a per-step cast would peak 2.5 GB higher.
    if keep_head:
        model.lm_head.float()
    return model


def length_sorted_generate(orig_generate: Any, *, token_budget: int = 6000) -> Any:
    """Wrap Phase 2's `_generate_rollouts_hf` to batch by length under a token budget.

    The stock function left-pads each fixed-size batch to its longest prompt, which on CPU
    spends most of the time (and KV-cache memory) on padding.
    """

    def wrapped(model: Any, tokenizer: Any, prompt_token_id_lists: list[list[int]], *,
                max_new_tokens: int, temperature: float, batch_size: int) -> tuple[list[str], list[list[int]]]:
        import time

        n = len(prompt_token_id_lists)
        order = sorted(range(n), key=lambda i: len(prompt_token_id_lists[i]))
        texts: list[Any] = [None] * n
        toks: list[Any] = [None] * n
        i = 0
        t0 = time.time()
        while i < n:
            j = i + 1
            while (j < n and (j - i) < batch_size
                   and (len(prompt_token_id_lists[order[j]]) + max_new_tokens) * (j - i + 1) <= token_budget):
                j += 1
            idx = order[i:j]
            t, k = orig_generate(model, tokenizer, [prompt_token_id_lists[q] for q in idx],
                                 max_new_tokens=max_new_tokens, temperature=temperature, batch_size=len(idx))
            for q, tt, kk in zip(idx, t, k):
                texts[q], toks[q] = tt, kk
            i = j
            print(f"[jit_generate] {i}/{n} prompts ({time.time() - t0:.0f}s)", flush=True)
        return texts, toks

    return wrapped


def jit_load_causal_lm(model_name_or_path: str, *, dtype: Any = None, device: Any = None,
                       adapter_path: str | None = None) -> Any:
    assert adapter_path is None
    return load_full_jit(model_name_or_path)
