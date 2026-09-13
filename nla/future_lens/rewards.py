"""Future-token rewards and group-relative advantages for the GRPO stage.

Reward conventions (spec Phase 2):
  exact_match : r = (1/K) * sum_{j<K} 1[y_j == x_{t+1+j}] over the first K readout
                tokens; readouts are truncated to K post hoc; a length violation
                (fewer than K tokens before EOS, or more than K generated) costs
                `length_penalty`. Free.
  target_logp : mean log p_target(y_j | true prefix, y_<j) over the (truncated)
                readout — Future Lens's surprisal, negated. Needs one batched
                forward of the frozen target (adapters disabled) per step.

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


def per_offset_hits(resp_ids, target_ids, k: int, eos_ids: set[int] | None = None) -> list[int | None]:
    """[K] entries: 1/0 per offset, None where the readout is shorter than K."""
    ids, _ = truncate_readout(list(resp_ids), k, eos_ids or set())
    out: list[int | None] = []
    for j in range(k):
        out.append(None if j >= len(ids) else int(int(ids[j]) == int(target_ids[j])))
    return out


@torch.no_grad()
def target_logp_reward(model, prefixes: list[np.ndarray], readouts: list[list[int]], device,
                       *, pad_id: int, micro_batch: int = 16, max_prefix: int = 1024,
                       disable_adapter=True) -> list[float | None]:
    """Mean per-token log-prob of each readout under the frozen target given its TRUE
    prefix. `model` is the (PEFT-wrapped) policy; the frozen target is the same
    weights with adapters disabled and no injection (the caller must leave
    vectors_ref[0] = None). Empty readouts -> None."""
    out: list[float | None] = [None] * len(readouts)
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
                logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
                for row, (pre, rd) in enumerate(seqs):
                    n = len(rd)
                    pred_pos = torch.arange(L - n - 1, L - 1, device=logits.device)
                    lg = logits[row].index_select(0, pred_pos).float()
                    lp = torch.log_softmax(lg, dim=-1)
                    tgt = torch.tensor(rd, dtype=torch.long, device=logits.device)
                    out[chunk[row]] = float(lp.gather(1, tgt.unsqueeze(1)).mean())
                del logits
    finally:
        if was_training:
            model.train()
    return out


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
