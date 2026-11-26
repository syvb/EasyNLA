"""Sampled Layer-K activations of Target and Base (both Base wrappers) on a Phase 2 rollout corpus.

Encodes every conversation in a `rollouts.parquet` the way Phase 2's extract step does
(teacher-forcing the exact prompt and rollout token ids), samples up to N positions per
active role, and pulls the activations through the disk cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import cached_io
import numpy as np
import pyarrow.parquet as pq
import torch

from nla.diff.wrappers import encode_aligned_pair


@dataclass
class CorpusRows:
    ht: np.ndarray  # [N, d] Target under ChatML
    hb: dict[str, np.ndarray]  # Base under "chatml" and "plain", row-aligned with `ht`
    roles: np.ndarray  # [N] role name per row
    tokens: np.ndarray  # [N] token id at the row's position
    conv: np.ndarray  # [N] conversation index per row
    sources: list[str]  # per conversation: source corpus
    texts: list[str]  # per conversation: assistant rollout text
    seqs: dict[str, list[list[int]]]  # per conversation token ids: "chatml" (Target's) and "plain"
    pos: dict[str, list[list[int]]]  # per conversation sampled positions in each of `seqs`, row order
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # other models on the Target's token ids


def load_corpus_rows(cfg: dict, tok: Any, rollouts_path: str, *, max_pos_per_role: int = 12, seed: int = 0,
                     max_convs: int = 0, extra_models: dict[str, tuple[str, int]] | None = None) -> CorpusRows:
    """`extra_models` maps a name to (model, layer) to pull at the same ChatML positions."""
    k = int(cfg["layer_index"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    tbl = pq.read_table(rollouts_path).to_pylist()
    if max_convs:
        tbl = tbl[:max_convs]
    rng = np.random.default_rng(seed)
    plans = []
    for row in tbl:
        kw = dict(history=json.loads(row["history_json"]), user_ids=row.get("prompt_user_ids"),
                  rollout_ids=row["rollout_token_ids"], enable_thinking=False, max_total_tokens=None,
                  truncate_history=False)
        ie, _ = encode_aligned_pair(tok, row["user_text"], base_wrapper="chatml", **kw)
        _, bp = encode_aligned_pair(tok, row["user_text"], base_wrapper="plain", **kw)
        p_ids = row.get("prompt_token_ids")
        if p_ids is not None:
            # Same check as Phase 2's extract step: teacher-force after the exact prompt the rollout saw.
            assert ie.input_ids[: len(p_ids)] == list(p_ids), f"prompt prefix diverged for {row['doc_id']}"
        pos: list[tuple[str, int, int, int]] = []  # role, ChatML position, Plain position, token
        for role, ipos, ppos in (
            ("user_content", ie.user_content_positions, bp.user_content_positions),
            ("asst_last", [ie.asst_last_idx], [bp.asst_last_idx]),
            ("asst_rollout", ie.asst_rollout_positions, bp.asst_rollout_positions),
        ):
            assert len(ipos) == len(ppos)
            sel = np.arange(len(ipos))
            if len(sel) > max_pos_per_role:
                sel = np.sort(rng.choice(len(ipos), size=max_pos_per_role, replace=False))
            pos.extend((role, ipos[j], ppos[j], ie.input_ids[ipos[j]]) for j in sel.tolist())
        plans.append({"inst_ids": ie.input_ids, "plain_ids": bp.input_ids, "pos": pos})
    print(f"[corpus] {len(plans)} conversations, {sum(len(p['pos']) for p in plans)} sampled positions", flush=True)

    def pull(model_name: str, layer: int, ids_key: str, pos_idx: int) -> np.ndarray:
        m = cached_io.lazy_load_causal_lm(model_name)
        exs = cached_io.cached_extract(m, [p[ids_key] for p in plans], [[q[pos_idx] for q in p["pos"]] for p in plans],
                                       layer_index=layer, pad_token_id=pad)
        m.free()
        return torch.cat([e.hidden for e in exs], dim=0).numpy()

    ht = pull(cfg["target_model"], k, "inst_ids", 1)
    hb = {"chatml": pull(cfg["base_model"], k, "inst_ids", 1), "plain": pull(cfg["base_model"], k, "plain_ids", 2)}
    extra = {name: pull(model, layer, "inst_ids", 1) for name, (model, layer) in (extra_models or {}).items()}
    return CorpusRows(
        ht=ht, hb=hb, extra=extra,
        roles=np.asarray([q[0] for p in plans for q in p["pos"]]),
        tokens=np.asarray([q[3] for p in plans for q in p["pos"]]),
        conv=np.asarray([ci for ci, p in enumerate(plans) for _ in p["pos"]]),
        sources=[row["source_corpus"] for row in tbl],
        texts=[row["assistant_text"] for row in tbl],
        seqs={"chatml": [p["inst_ids"] for p in plans], "plain": [p["plain_ids"] for p in plans]},
        pos={"chatml": [[q[1] for q in p["pos"]] for p in plans], "plain": [[q[2] for q in p["pos"]] for p in plans]},
    )


def massive_rows(*mats: np.ndarray, mult: float = 5.0) -> np.ndarray:
    """Rows whose norm is far above the median in any of the given matrices (massive activations)."""
    out = np.zeros(len(mats[0]), dtype=bool)
    for x in mats:
        n = np.linalg.norm(x, axis=1)
        out |= n > mult * np.median(n)
    return out
