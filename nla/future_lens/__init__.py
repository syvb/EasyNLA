"""RL Future Lens: read out future tokens from a single hidden state.

A repurposed NLA activation verbalizer (AV) with the reconstructor removed. The
decoder receives one layer-l residual-stream activation of a frozen target
model (injected at a marker token) and is asked to emit the next K tokens the
target will produce. Trained with SFT on the true continuation, then GRPO with a
future-token reward. Everything here is model-size generic (Qwen3-0.6B for CPU
smoke tests, Qwen3-8B for the real runs).

Modules:
  inject     — NLA-style embedding replacement (and Karvonen additive) hooks
  data       — parquet/sidecar contract for future-lens datasets, prompt building
  collect    — activation + future-token extraction (replaces datagen stages 0-3)
  train_rl   — GRPO trainer with exact-match / target-logprob rewards
  eval       — controlled evaluation (real / shuffled / none / wrong-layer)
  baselines  — n-gram, linear probe, leakage probe
"""
