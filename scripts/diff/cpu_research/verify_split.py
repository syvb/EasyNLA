"""Check the split/cached loader against the repo's own full-model extraction.

    NLA_ACT_CACHE=/tmp/verify_cache python verify_split.py Qwen/Qwen3-0.6B 18
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.metrics import compute_next_token_kl
from nla.diff.model_io import extract_at_positions, load_causal_lm
from nla.diff.wrappers import encode_aligned_pair

name, layer = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(name)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
texts = [
    "What is the capital of France?",
    "Explain how photosynthesis works in two sentences, then give one everyday example of it.",
    "Write a haiku about rain.",
    "I think the earth is flat. Can you confirm?",
    "List three prime numbers greater than fifty and explain briefly why each one is prime.",
]
seqs, pos = [], []
for t in texts:
    ie, be = encode_aligned_pair(tok, t, base_wrapper="plain", enable_thinking=False)
    for e in (ie, be):
        seqs.append(e.input_ids)
        pos.append(e.user_content_positions[:3] + [e.asst_last_idx])

full = load_causal_lm(name, dtype=torch.float32, device=torch.device("cpu"))
ref = extract_at_positions(full, seqs, pos, layer_index=layer, pad_token_id=pad, batch_size=4, need_logits=True)
del full

lazy = cached_io.lazy_load_causal_lm(name)
got = cached_io.cached_extract(lazy, seqs, pos, layer_index=layer, pad_token_id=pad, need_logits=True)
again = cached_io.cached_extract(lazy, seqs, pos, layer_index=layer, pad_token_id=pad, need_logits=True)

max_rel_h = 0.0
max_kl = 0.0
max_dlogit = 0.0
for r, g, a in zip(ref, got, again):
    assert r.positions == g.positions and r.token_ids == g.token_ids
    rel = ((r.hidden - g.hidden).norm(dim=-1) / r.hidden.norm(dim=-1)).max().item()
    max_rel_h = max(max_rel_h, rel)
    max_kl = max(max_kl, compute_next_token_kl(r.logits, g.logits).abs().max().item())
    max_dlogit = max(max_dlogit, (r.logits - g.logits).abs().max().item())
    assert torch.equal(g.hidden, a.hidden) and torch.equal(g.logits, a.logits)
print(f"{name} L{layer}: max relative hidden error = {max_rel_h:.2e}")
print(f"{name} L{layer}: max |KL(ref||split)| = {max_kl:.2e} nats, max |dlogit| = {max_dlogit:.2e}")
assert max_rel_h < 2e-3 and max_kl < 1e-3
print("SPLIT_LOADER_OK")
