"""Activation-injection hooks for the AV actor. Shared by SFT/RL/evals.

Two injection mechanisms, same `vectors_ref` interface:

- register_karvonen_hook — the EasyNLA default. Registers (1) an embedding
  forward-hook that stashes the current input_ids and (2) a forward-hook on
  transformer block `layer_idx` that norm-match-ADDS the activation in
  `vectors_ref[0]` at the marker token (see
  nla.injection.karvonen_inject_in_residual).
- register_embed_replace_hook — the classic NLA mechanism (nanoNLA): REPLACE
  the embedding-matrix output row at the marker position with the activation
  rescaled to `injection_scale` L2 norm (see
  nla.injection.inject_at_marked_positions).

Both no-op when seq_len < 2 (the autoregressive cache steps after a rollout's
prefill) or when `vectors_ref[0]` is None. Device-aligned so they also work
under device_map="auto".
"""

from nla.injection import inject_at_marked_positions, karvonen_inject_in_residual
from nla.schema import normalize_activation


def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        # device_map="auto": this layer may live on a different GPU than where
        # the caller staged input_ids / the vector. Align to the residual.
        ids = input_ids.to(resid.device)
        # NO zero-marker early-return: every legit forward with vectors_ref set
        # contains markers (decode steps are caught by the seq_len<2 guard above),
        # so zero markers = template drift — let karvonen_inject_in_residual's
        # count-mismatch check fail LOUD instead of silently skipping injection.
        injected = karvonen_inject_in_residual(
            ids, resid, v.to(resid.device), inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    # PEFT-aware: unwrap to the raw CausalLM first, then let arch_adapters find
    # the decoder list — handles multimodal wrappers (Gemma-3 language_model)
    # and the GPT-2/Falcon `.transformer.h` shape, where the old
    # `while hasattr(.model)` walk crashed with AttributeError('layers').
    from nla.utils.arch_adapters import resolve_decoder_layers
    target = model.get_base_model() if hasattr(model, "peft_config") else model
    resolve_decoder_layers(target)[layer_idx].register_forward_hook(layer_hook)


def register_embed_replace_hook(model, vectors_ref, inj_id, left_id, right_id,
                                injection_scale):
    """Classic-NLA injection: overwrite the embedding output at the marker.

    injection_scale: L2 norm the activation is rescaled to before replacement
    (float), or None to inject the RAW vector. Unlike the Karvonen hook there
    is no residual to norm-match against, so the scale is an explicit
    hyperparameter — layer-24 activations (mean norm ~277 on Qwen3-8B) are far
    off token-embedding scale, and injecting them raw vs at sqrt(d) vs at
    ~mean-act-norm are materially different experiments.

    The embedding module's forward receives input_ids directly, so no separate
    id-stash hook is needed. Replacement (not addition) means the marker row's
    embedding weight gets no gradient from the marker position; all other
    positions train normally.
    """

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0 or ids is None or ids.shape[-1] < 2:
            return output
        ids = ids.to(output.device)
        # Same LOUD-failure policy as the Karvonen hook: zero markers on a
        # legit injected forward = template drift — let the count check raise.
        return inject_at_marked_positions(
            ids, output, normalize_activation(v.to(output.device), injection_scale),
            inj_id, left_id, right_id,
        )

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
