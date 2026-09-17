"""Predictive-readout NLA ("pred-NLA"): train the AV for frozen-reader usefulness.

Standard NLA rewards the verbalizer (AV) by how well the reconstructor (AR)
rebuilds the activation from its words. This package replaces that reward with a
BEHAVIORAL one: a frozen, never-trained reader LM is shown the explanation and
then scored on how well it predicts the target model's own continuation from
that point. The AV is rewarded for explanations that make the reader better at
predicting what the target model actually did next.

Stages (each a CLI, each logs to wandb):
  1. nla.pred.continuations  — sample target-model continuations per activation
  2. nla.pred.gate           — pre-RL check that the score is activation-specific
  3. nla.pred.train_rl       — GRPO on the frozen-reader reward
  4. nla.pred.eval           — all checkpoints x both readers x matched/shuffled
  5. nla.pred.report         — markdown report + bootstrap CIs

See docs/pred_nla.md.
"""

from nla.pred.reader import (
    DEFAULT_BUCKETS,
    FrozenReader,
    ReaderTemplates,
    bucket_char_ranges,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "FrozenReader",
    "ReaderTemplates",
    "bucket_char_ranges",
]
