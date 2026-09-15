---
title: Future Oracles
emoji: 🔮
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
license: apache-2.0
short_description: Read what a model will say next from one activation
models:
  - Qwen/Qwen3-0.6B-Base
  - Qwen/Qwen3-1.7B-Base
  - Qwen/Qwen3-8B-Base
datasets:
  - syvb/rl-future-lens-qwen3-8b
---

# 🔮 Future Oracles

A **Future Oracle** is a LoRA decoder trained on a Qwen3 base model that reads one residual-stream
vector (the output of a single transformer block at a single token) and verbalises the tokens the
model itself is about to produce. It is the [Future Lens](https://arxiv.org/abs/2311.04897) question
("how much of the future is already in this one vector?") answered with an activation-verbalizer
decoder: the target model, with a LoRA adapter, gets the vector written into the input embedding of
a marker token inside a short prompt and then writes the next *k* tokens.

Click any token of any text. Every selected oracle extracts that token's vector at its layer from
its own base model, reads it out, and the readout is scored against the model's own greedy
continuation from that token (the oracle's training label). The text's real continuation is shown
for reference only. Controls: **none** leaves the marker embedding untouched, **shuffled** injects
another token's vector from the same text.

Oracles are discovered automatically from the experiment's dataset repo
(`ckpts/<run>/iter_*/`): *Refresh oracles* picks up newly trained ones without a restart.

Training recipe (all sizes identical): greedy labels with the model's own top-64 next-token
distribution distilled as soft targets, LoRA r=64, 8000 steps at batch 64, injection scale = p75
activation norm of the layer × 2, layer at ~2/3 depth (24 of 36 for 8B, 19 of 28 for 0.6B/1.7B).
Code: EasyNLA `nla/future_lens`.
