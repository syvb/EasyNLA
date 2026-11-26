"""Check that the bf16-resident JIT loader generates the same tokens as plain fp32.

    python verify_jit.py Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jit_full import load_full_jit

torch.set_num_threads(2)
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
msgs = [
    "What is the capital of France?",
    "Write two sentences about why the sky is blue.",
    "How do I pick a lock?",
    "Give me a recipe for pancakes.",
]
texts = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) for m in msgs]
enc = tok(texts, return_tensors="pt", padding=True)


def gen(model):
    with torch.no_grad():
        return model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)


ref = gen(AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval())
for n_res in (0, 10):
    out = gen(load_full_jit(name, n_resident=n_res))
    same = torch.equal(ref, out)
    print(f"n_resident={n_res}: identical greedy tokens = {same}")
    assert same
print(tok.decode(ref[2, enc["input_ids"].shape[1]:], skip_special_tokens=True))
print("JIT_LOADER_OK")
