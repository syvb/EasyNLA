"""The reward seam: two objectives, one trainer, everything else identical.

The point of the experiment is a comparison between objectives, so the two
objectives have to differ in nothing but the reward. Both implementations below
are handed the same rollouts from the same policy at the same step and return a
reward per rollout, so a run of each differs only in this object.

    reader_gain : frozen reader's nats saved on the target model's own
                  continuation when shown the explanation (this experiment)
    recon       : negative reconstruction MSE from the AR (standard NLA)

The reconstruction arm here keeps the AR FROZEN, which matches the published
comparator checkpoint (the June length-penalty sweep used a frozen critic; the
co-trained recipe lives in nla/train_rl_vllm.py and needs the memory for a second
full-size optimizer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from nla.pred.reader import DEFAULT_BUCKETS, HEADLINE_BUCKET
from nla.pred.scoring import (
    SKIP,
    BaselineCache,
    baseline_scores,
    gain_per_token,
    score_explanations,
    target_tokens_per_bucket,
)


@dataclass
class RewardOut:
    reward: np.ndarray          # [n_rollouts], already floored for failures
    valid: np.ndarray           # [n_rollouts] bool - reward came from a real score
    logs: dict                  # scalars for wandb
    per_bucket: np.ndarray | None = None   # [n_rollouts, n_buckets] (reader only)


class ReaderGainReward:
    """reward = nats per target token the explanation saves the frozen reader."""

    name = "reader_gain"
    default_length_penalty = 0.001

    def __init__(self, reader, *, branches, buckets=DEFAULT_BUCKETS,
                 fail_gain=1.0, cache=None):
        self.reader = reader
        self.branches = tuple(branches)
        self.buckets = buckets
        self.hb = [tuple(b) for b in buckets].index(tuple(HEADLINE_BUCKET))
        self.fail_gain = fail_gain
        self.cache = cache if cache is not None else BaselineCache()
        self.tgt_tok = target_tokens_per_bucket(buckets)

    def score(self, *, rows, samples, explanations, truncated) -> RewardOut:
        rep_rows = [rows[s["row"]] for s in samples]
        base_unique = baseline_scores(self.reader, rows, branches=self.branches,
                                      buckets=self.buckets, cache=self.cache)
        base_rep = np.stack([base_unique[s["row"]] for s in samples])
        score_in = [e if (e is not None and not t) else SKIP
                    for e, t in zip(explanations, truncated)]
        sc = score_explanations(self.reader, rep_rows, score_in,
                                branches=self.branches, buckets=self.buckets)
        gains = gain_per_token(sc, base_rep, self.buckets)
        raw = gains[:, self.hb]
        valid = np.isfinite(raw)
        reward = np.where(valid, raw, -self.fail_gain)
        logs = {
            "reader/baseline_nats_per_tok":
                float(np.mean(base_rep[:, self.hb]) / self.tgt_tok[self.hb]),
            "reader/cache_size": len(self.cache),
        }
        for bi, (lo, hi) in enumerate(self.buckets):
            g = gains[:, bi]
            g = g[np.isfinite(g)]
            logs[f"gain_bucket/{lo}-{hi}"] = float(np.mean(g)) if g.size else float("nan")
        return RewardOut(reward=reward, valid=valid, logs=logs, per_bucket=gains)

    def eval_score(self, rows, explanations, truncated):
        """Same quantity on held-out positions (one explanation per row)."""
        score_in = [e if (e is not None and not t) else SKIP
                    for e, t in zip(explanations, truncated)]
        base = baseline_scores(self.reader, rows, branches=self.branches,
                               buckets=self.buckets, cache=self.cache)
        sc = score_explanations(self.reader, rows, score_in, branches=self.branches,
                                buckets=self.buckets)
        return gain_per_token(sc, base, self.buckets)[:, self.hb]


class ReconReward:
    """reward = -MSE(AR(explanation), activation), the standard NLA objective.

    Present so the reconstruction baseline can be run through this same trainer:
    same rollouts, same LoRA, same steps, same batch, only the reward differs.
    Without that, "behavioral beats reconstruction" could just be a difference in
    rollout engine or hyperparameters.
    """

    name = "recon"
    default_length_penalty = 0.01     # the reconstruction trainer's tuned value

    def __init__(self, critic, tokenizer, template, mse_scale, device,
                 *, fail_reward=-2.0, batch_size=32, max_len=1024,
                 fve_baseline=None):
        self.critic = critic
        self.tokenizer = tokenizer
        self.template = template
        self.mse_scale = mse_scale
        self.device = device
        self.fail_reward = fail_reward
        self.batch_size = batch_size
        self.max_len = max_len
        self.fve_baseline = fve_baseline

    @torch.no_grad()
    def _mse(self, explanations, activations):
        from nla.schema import normalize_activation
        from nla.utils import critic_predict

        n = len(explanations)
        out = [None] * n
        pad_id = self.tokenizer.eos_token_id
        ids_list = [None] * n
        for i, e in enumerate(explanations):
            if e is None:
                continue
            ids = self.tokenizer.encode(self.template.format(explanation=e),
                                        add_special_tokens=False)
            if 0 < len(ids) <= self.max_len:
                ids_list[i] = ids
        idx = [i for i in range(n) if ids_list[i] is not None]
        for c0 in range(0, len(idx), self.batch_size):
            chunk = idx[c0 : c0 + self.batch_size]
            mx = max(len(ids_list[i]) for i in chunk)
            bx = torch.full((len(chunk), mx), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((len(chunk), mx), dtype=torch.long, device=self.device)
            for r, i in enumerate(chunk):
                L = len(ids_list[i])
                bx[r, :L] = torch.tensor(ids_list[i], dtype=torch.long, device=self.device)
                attn[r, :L] = 1
            pred = critic_predict(self.critic, bx, attn, self.mse_scale)
            gold = torch.stack([activations[i].to(self.device).float() for i in chunk])
            pn = normalize_activation(pred, self.mse_scale)
            gn = normalize_activation(gold, self.mse_scale)
            mse = ((pn - gn) ** 2).mean(dim=1)
            for r, i in enumerate(chunk):
                m = float(mse[r])
                out[i] = m if math.isfinite(m) else None
        return out

    def score(self, *, rows, samples, explanations, truncated) -> RewardOut:
        acts = [torch.as_tensor(rows[s["row"]]["activation"], dtype=torch.float32)
                for s in samples]
        ex = [e if (e is not None and not t) else None
              for e, t in zip(explanations, truncated)]
        mses = self._mse(ex, acts)
        valid = np.array([m is not None for m in mses])
        reward = np.array([(-m) if m is not None else self.fail_reward for m in mses],
                          dtype=np.float64)
        vm = [m for m in mses if m is not None]
        logs = {"ar/recon_mse": float(np.mean(vm)) if vm else float("nan")}
        if self.fve_baseline:
            logs["fve_pct"] = (1.0 - (np.mean(vm) / self.fve_baseline)) * 100.0 if vm else float("nan")
        return RewardOut(reward=reward, valid=valid, logs=logs)

    def eval_score(self, rows, explanations, truncated):
        acts = [torch.as_tensor(r["activation"], dtype=torch.float32) for r in rows]
        ex = [e if (e is not None and not t) else None
              for e, t in zip(explanations, truncated)]
        mses = self._mse(ex, acts)
        return np.array([(-m) if m is not None else np.nan for m in mses])


__all__ = ["ReaderGainReward", "ReconReward", "RewardOut"]
