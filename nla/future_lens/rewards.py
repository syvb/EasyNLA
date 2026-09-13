"""Future-token rewards and group-relative advantages for the GRPO stage.

Reward conventions (spec Phase 2):
  exact_match : r = (1/K) * sum_{j<K} 1[y_j == x_{t+1+j}] over the first K readout
                tokens; readouts are truncated to K post hoc; a length violation
                (fewer than K tokens before EOS, or more than K generated) costs
                `length_penalty`. Free.
  target_logp : (1/K) [ sum_{j<len} log p_target(y_j | true prefix, y_<j)
                        - missing_token_penalty * (K - len) ]   (len = tokens before EOS, <= K)
                Future Lens's surprisal, negated, SUMMED over the K slots so that
                stopping early is never rewarded: a per-token mean would let "one
                safe token + EOS" (~ -1.5) beat an honest 9-token readout (~ -3.5).
                Each missing slot costs `missing_token_penalty` (default 4 nats,
                about the surprisal of an average natural-text token). Needs one
                batched forward of the frozen target (adapters disabled) per step.

`None` rewards (unusable rollouts) follow EasyNLA's convention and get the
`fail_value` floor before the advantage computation.
"""

from __future__ import annotations

import numpy as np
import torch

REWARD_KINDS = ("exact_match", "target_logp")


def truncate_readout(resp_ids: list[int], k: int, eos_ids: set[int]) -> tuple[list[int], bool]:
    """Strip stop tokens, cut to K. Returns (ids, length_violation)."""
    ids = [t for t in resp_ids if t not in eos_ids]
    viol = len(ids) != k
    return ids[:k], viol


def exact_match_reward(resp_ids, target_ids, k: int, *, length_penalty: float = 0.1,
                       eos_ids: set[int] | None = None) -> tuple[float, bool]:
    """Per-position exact-match fraction over K, minus a light length penalty."""
    ids, viol = truncate_readout(list(resp_ids), k, eos_ids or set())
    tgt = np.asarray(target_ids)[:k]
    hits = sum(1 for j, y in enumerate(ids) if j < len(tgt) and int(y) == int(tgt[j]))
    r = hits / max(k, 1)
    if viol:
        r -= length_penalty
    return float(r), bool(viol)


def per_offset_hits(resp_ids, target_ids, k: int, eos_ids: set[int] | None = None) -> list[int]:
    """[K] entries: 1/0 per offset; a readout shorter than K is a MISS at the missing
    offsets (same convention as eval.score_readout, so train and eval curves agree)."""
    ids, _ = truncate_readout(list(resp_ids), k, eos_ids or set())
    tgt = list(target_ids)
    return [int(j < len(ids) and j < len(tgt) and int(ids[j]) == int(tgt[j])) for j in range(k)]


@torch.no_grad()
def target_token_logps(model, prefixes: list[np.ndarray], readouts: list[list[int]], device,
                       *, pad_id: int, micro_batch: int = 16, max_prefix: int = 1024,
                       disable_adapter=True) -> list[list[float] | None]:
    """Per-token log-probs of each readout under the frozen target given its TRUE
    prefix. `model` is the (PEFT-wrapped) policy; the frozen target is the same
    weights with adapters disabled and no injection (the caller must leave
    vectors_ref[0] = None). Empty readouts -> None. Only the last (max readout + 1)
    positions' logits are materialised (`logits_to_keep`)."""
    out: list[list[float] | None] = [None] * len(readouts)
    idx = [i for i, r in enumerate(readouts) if len(r) > 0]
    ctx = model.disable_adapter() if (disable_adapter and hasattr(model, "disable_adapter")) else _nullctx()
    was_training = model.training
    model.eval()
    try:
        with ctx:
            for cs in range(0, len(idx), micro_batch):
                chunk = idx[cs: cs + micro_batch]
                seqs = []
                for i in chunk:
                    pre = list(prefixes[i][-max_prefix:])
                    seqs.append((pre, list(readouts[i])))
                L = max(len(p) + len(r) for p, r in seqs)
                ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), L), dtype=torch.long)
                for row, (pre, rd) in enumerate(seqs):
                    full = pre + rd
                    ids[row, L - len(full):] = torch.tensor(full, dtype=torch.long)   # left pad
                    attn[row, L - len(full):] = 1
                n_max = max(len(rd) for _, rd in seqs)
                logits = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                               logits_to_keep=n_max + 1).logits       # [B, n_max+1, V]: last positions
                for row, (pre, rd) in enumerate(seqs):
                    n = len(rd)
                    lg = logits[row, -(n + 1):-1].float()            # predicts rd[0..n-1]
                    lp = torch.log_softmax(lg, dim=-1)
                    tgt = torch.tensor(rd, dtype=torch.long, device=logits.device)
                    out[chunk[row]] = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist()
                del logits
    finally:
        if was_training:
            model.train()
    return out


def target_logp_reward(model, prefixes, readouts, device, **kw) -> list[float | None]:
    """Mean per-token target log-prob (the eval surprisal metric; NOT the RL reward —
    see `target_logp_sum_reward`)."""
    return [None if t is None else float(np.mean(t)) for t in target_token_logps(model, prefixes, readouts, device, **kw)]


def target_logp_sum_reward(token_logps: list[float] | None, k: int, *, missing_token_penalty: float = 4.0,
                           length_violation: bool = False, length_penalty: float = 0.1) -> float:
    """RL reward from per-token target log-probs: sum over produced slots, minus
    `missing_token_penalty` per slot the readout did not fill, divided by K."""
    t = token_logps or []
    r = (float(sum(t)) - missing_token_penalty * max(0, k - len(t))) / max(k, 1)
    if length_violation:
        r -= length_penalty
    return r


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def group_advantages(rewards: torch.Tensor, groups: torch.Tensor, ok: torch.Tensor, *,
                     n_groups: int, norm: str = "none") -> torch.Tensor:
    """Group-relative advantages. norm='none' = Dr. GRPO (mean-centred only, no
    std division -> no length/variance bias); norm='std' = original GRPO.
    Rows with ok=False are excluded from the baseline and get advantage 0."""
    adv = torch.zeros_like(rewards)
    for g in range(n_groups):
        m = (groups == g) & ok
        if m.sum() < 2:
            continue
        r = rewards[m]
        a = r - r.mean()
        if norm == "std":
            a = a / (r.std() + 1e-6)
        adv[m] = a
    return adv
