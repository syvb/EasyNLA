"""Check split-loader LoRA merging against PEFT on a random adapter.

    NLA_ACT_CACHE=/tmp/verify_cache_lora python verify_lora.py Qwen/Qwen3-0.6B 18
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.metrics import compute_next_token_kl
from nla.diff.model_io import extract_at_positions, load_causal_lm
from nla.diff.wrappers import encode_chatml_dialogue

torch.set_num_threads(4)
name, layer = sys.argv[1], int(sys.argv[2])
adir = Path("/tmp/verify_lora_adapter")
tok = AutoTokenizer.from_pretrained(name)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

torch.manual_seed(0)
full = load_causal_lm(name, dtype=torch.float32, device=torch.device("cpu"))
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, init_lora_weights=False, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
pm = get_peft_model(full, cfg)
for n, p in pm.named_parameters():
    if "lora_" in n:
        torch.nn.init.normal_(p, std=0.02)
pm.save_pretrained(str(adir))
del pm, full

encs = [encode_chatml_dialogue(tok, t, enable_thinking=False) for t in
        ("What is the capital of France?", "Give me a hint about your secret word.", "Write a haiku about rain.")]
seqs = [e.input_ids for e in encs]
pos = [[e.asst_last_idx] for e in encs]

ref_model = PeftModel.from_pretrained(load_causal_lm(name, dtype=torch.float32, device=torch.device("cpu")), str(adir))
ref_model.eval()
ref = extract_at_positions(ref_model, seqs, pos, layer_index=layer, pad_token_id=pad, need_logits=True)
plain = extract_at_positions(ref_model.unload(), seqs, pos, layer_index=layer, pad_token_id=pad, need_logits=True)
del ref_model

lazy = cached_io.lazy_peft_from_pretrained(cached_io.lazy_load_causal_lm(name), str(adir))
got = cached_io.cached_extract(lazy, seqs, pos, layer_index=layer, pad_token_id=pad, need_logits=True)

rel = max(((r.hidden - g.hidden).norm() / r.hidden.norm()).item() for r, g in zip(ref, got))
kl = max(compute_next_token_kl(r.logits, g.logits).abs().max().item() for r, g in zip(ref, got))
effect = max(compute_next_token_kl(r.logits, p.logits).abs().max().item() for r, p in zip(ref, plain))
print(f"max relative hidden error vs PEFT = {rel:.2e}; max KL(PEFT||split) = {kl:.2e} nats")
print(f"(adapter effect size for scale: KL(PEFT||no adapter) = {effect:.2f} nats)")
assert rel < 2e-3 and kl < 1e-3 and effect > 0.05
print("LORA_MERGE_OK")
