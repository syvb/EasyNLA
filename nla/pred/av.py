"""Loading the activation verbalizer and generating explanations from activations.

Every condition in this experiment is the SAME base model with a different LoRA
on top, which is what makes the comparison clean:

    SFT               = merged AV checkpoint, no adapter
    reconstruction RL = merged AV + the reconstruction-trained adapter
    behavioral RL     = merged AV + the adapter this experiment trains

So `--base-ckpt` is the merged AV (e.g. syvb/nanonla-qwen3-8b-L24-av) for all of
them, and the KL reference during RL is simply the adapter-disabled base, which
IS the SFT policy. Shared by the gate, the trainer and the eval so the three
cannot drift in how they build a prompt or inject a vector.
"""

from __future__ import annotations

import torch

from nla.schema import extract_explanation
from nla.utils import build_prompt_text, register_karvonen_hook


def load_av(
    base_ckpt: str,
    adapter: str | None = None,
    *,
    adapter_subfolder: str | None = None,
    quant: str = "none",
    device: str = "cuda",
    dtype=torch.bfloat16,
    inj_ids: tuple[int, int, int] | None = None,
    attn_implementation: str = "sdpa",
    trainable: bool = False,
):
    """(model, vectors_ref) with the Karvonen injection hook already registered.

    vectors_ref is the one-element list the hook reads: set vectors_ref[0] to a
    [B, d] float tensor before a forward whose prompts each carry one marker,
    and clear it to None afterwards.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = None
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=dtype, attn_implementation=attn_implementation,
        quantization_config=quant_config,
    )
    if quant_config is None:
        model = model.to(device)
    if adapter:
        from peft import PeftModel

        kw = {"subfolder": adapter_subfolder} if adapter_subfolder else {}
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, **kw)
        print(f"[av] {base_ckpt} + adapter {adapter}"
              + (f"/{adapter_subfolder}" if adapter_subfolder else ""), flush=True)
    else:
        print(f"[av] {base_ckpt} (no adapter)", flush=True)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    vectors_ref = [None]
    if inj_ids is not None:
        register_karvonen_hook(model, vectors_ref, *inj_ids, layer_idx=1)
    return model, vectors_ref


def resolve_eos_ids(model, tokenizer) -> set[int]:
    """Every id that ends a generation (Qwen3 lists two)."""
    eos = {tokenizer.eos_token_id}
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc_eos is not None:
        eos.update(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
    eos.discard(None)
    return eos


@torch.no_grad()
def generate_explanations(
    model,
    tokenizer,
    vectors_ref,
    rows,
    inject_char: str,
    *,
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    batch_size: int = 16,
    device: str = "cuda",
    activation_key: str = "activation",
    return_raw: bool = False,
):
    """Batched activation -> explanation. Returns the extracted explanation text
    per row (None when the <explanation> tags are missing/unclosed).

    Left-padded so completions align; the hook scans each row for its own marker,
    so a [B, d] vectors_ref injects each row's own activation.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [build_prompt_text(r["prompt"], inject_char, tokenizer) for r in rows]
    acts = [torch.as_tensor(r[activation_key], dtype=torch.float32) for r in rows]
    raws: list[str] = []
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for c0 in range(0, len(prompts), batch_size):
            cp = prompts[c0 : c0 + batch_size]
            ca = acts[c0 : c0 + batch_size]
            enc = tokenizer(cp, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            vectors_ref[0] = torch.stack(ca).to(device).float()
            try:
                gen = model.generate(
                    input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    **({"temperature": temperature} if temperature > 0 else {}),
                    top_p=1.0, top_k=0, repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            finally:
                vectors_ref[0] = None
            new = gen.sequences[:, enc.input_ids.shape[1] :]
            for i in range(len(cp)):
                raws.append(tokenizer.decode(new[i], skip_special_tokens=True))
    finally:
        tokenizer.padding_side = orig_side
    expls = [extract_explanation(t) for t in raws]
    if return_raw:
        return expls, raws
    return expls
