"""Fix 2 & §2.4: Chat-template masking, 1-to-1 Base/Instruct wrappers, and per-role centering.

Constructs token sequences piece-wise (`prefix_ids + user_ids + suffix_ids + rollout_ids`)
so that BPE boundaries never merge across turn headers and `user_content`,
`asst_last`, and `asst_rollout` positions align strictly 1-to-1 between:
  - Instruct(ChatML, enable_thinking=False)
  - Base(ChatML)  (identical token IDs)
  - Base(Plain)   (2-shot neutral Q&A dialogue preamble + User/Assistant turns)

Role taxonomy:
  - Masked template roles (dropped by Fix 2; kept in Arm 1 naive ablation):
      * chat_bos:    position 0
      * sys_prefix:  positions [1, user_start)
      * asst_header: positions [user_end, asst_last_idx)
  - Active training roles (retained by Fix 2, per-role mean-centered):
      * user_content: positions [user_start, user_end) (excluding pos 0 if empty prefix)
      * asst_last:    single position `header_end - 1` immediately before assistant answer
      * asst_rollout: positions [rollout_start, rollout_end) (excluding trailing EOS)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ACTIVE_ROLES: tuple[str, ...] = ("user_content", "asst_last", "asst_rollout")
TEMPLATE_ROLES: tuple[str, ...] = ("chat_bos", "sys_prefix", "asst_header")
ALL_ROLES: tuple[str, ...] = ACTIVE_ROLES + TEMPLATE_ROLES

_USER_SLOT_SENTINEL = "<<<NLA_DIFF_USER_SLOT_9F3A7C>>>"

PLAIN_FEWSHOT_PREAMBLE = (
    "User:\n"
    "What is the capital of France?\n"
    "Assistant:\n"
    "The capital of France is Paris.\n\n"
    "User:\n"
    "How many days are in a week?\n"
    "Assistant:\n"
    "There are seven days in a week.\n\n"
)


@dataclass(frozen=True)
class EncodedDialogue:
    """Piece-wise tokenized dialogue with exact role index spans."""

    input_ids: list[int]
    wrapper: str  # "chatml" | "plain" | "raw"
    user_span: tuple[int, int]  # [start, end)
    header_span: tuple[int, int]  # [start, end) — includes asst_last at end-1
    asst_last_idx: int  # single prefill trigger position (header_end - 1), or -1 for raw
    rollout_span: tuple[int, int]  # [start, end) excluding trailing EOS
    history_span: tuple[int, int] = (0, 0)  # [start, end) earlier-turn content (NOT template!)
    kept_history_start_idx: int = 0  # slice offset into input `history` after left-truncation

    @property
    def sys_prefix_positions(self) -> list[int]:
        u0 = self.user_span[0]
        h0, h1 = self.history_span
        return [p for p in range(1, u0) if not (h0 <= p < h1)]

    @property
    def history_positions(self) -> list[int]:
        h0, h1 = self.history_span
        return [p for p in range(h0, h1) if p > 0]

    @property
    def user_content_positions(self) -> list[int]:
        u0, u1 = self.user_span
        # Never sample position 0 (BOS/attention sink)
        return [p for p in range(u0, u1) if p > 0]

    @property
    def asst_header_positions(self) -> list[int]:
        h0, h1 = self.header_span
        if h1 <= h0 + 1:
            return []
        return list(range(h0, h1 - 1))

    @property
    def asst_rollout_positions(self) -> list[int]:
        r0, r1 = self.rollout_span
        return list(range(r0, r1))

    def role_at(self, pos: int) -> str:
        if pos == 0:
            return "chat_bos"
        if self.history_span[0] <= pos < self.history_span[1]:
            return "history"
        if self.user_span[0] <= pos < self.user_span[1]:
            return "user_content"
        if pos == self.asst_last_idx:
            return "asst_last"
        if self.header_span[0] <= pos < self.header_span[1]:
            return "asst_header"
        if self.rollout_span[0] <= pos < self.rollout_span[1]:
            return "asst_rollout"
        return "sys_prefix"


def _strip_trailing_eos(ids: Sequence[int], eos_ids: set[int]) -> list[int]:
    out = list(ids)
    while out and out[-1] in eos_ids:
        out.pop()
    return out


def _collect_eos_ids(tokenizer: Any) -> set[int]:
    eos: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            eos.add(v)
        elif isinstance(v, (list, tuple)):
            eos.update(int(x) for x in v if x is not None)
    # Qwen3 ChatML stop tokens
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            enc = tokenizer.encode(tok, add_special_tokens=False)
            if len(enc) == 1:
                eos.add(int(enc[0]))
        except Exception:
            pass
    return eos


def _render_chatml_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render ChatML prefill via tokenizer.apply_chat_template with graceful fallbacks."""
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    # Fallback for minimal synthetic test tokenizers without a Jinja chat_template
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if enable_thinking:
        parts.append("<|im_start|>assistant\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _split_chatml_prefix_suffix(
    tokenizer: Any,
    kept_history: Sequence[dict[str, str]],
    system_prompt: str | None,
    enable_thinking: bool,
) -> tuple[list[int], list[int], tuple[int, int]]:
    """Return (prefix_ids, suffix_ids, history_span) for ChatML with `kept_history`."""
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for m in kept_history:
        msgs.append({"role": str(m["role"]), "content": str(m["content"])})
    msgs.append({"role": "user", "content": _USER_SLOT_SENTINEL})

    rendered = _render_chatml_template(tokenizer, msgs, enable_thinking=enable_thinking)
    assert _USER_SLOT_SENTINEL in rendered, (
        "Sentinel missing from rendered chat template — cannot align user span."
    )
    prefix_str, suffix_str = rendered.split(_USER_SLOT_SENTINEL, 1)
    prefix_ids = list(tokenizer.encode(prefix_str, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(suffix_str, add_special_tokens=False))
    assert len(suffix_ids) >= 1, f"Empty assistant header suffix from template: {suffix_str!r}"

    if not kept_history:
        return prefix_ids, suffix_ids, (0, 0)

    # Locate exact history span inside prefix_ids so earlier-turn content is NOT labeled sys_prefix:
    # prefix_str = [sys_block] + [history_block] + [final_user_open_tag]
    no_hist_msgs: list[dict[str, str]] = []
    if system_prompt:
        no_hist_msgs.append({"role": "system", "content": system_prompt})
    no_hist_msgs.append({"role": "user", "content": _USER_SLOT_SENTINEL})
    no_hist_prefix = _render_chatml_template(
        tokenizer, no_hist_msgs, enable_thinking=enable_thinking
    ).split(_USER_SLOT_SENTINEL, 1)[0]
    bare_user_open = _render_chatml_template(
        tokenizer, [{"role": "user", "content": _USER_SLOT_SENTINEL}], enable_thinking=enable_thinking
    ).split(_USER_SLOT_SENTINEL, 1)[0]
    no_hist_ids = list(tokenizer.encode(no_hist_prefix, add_special_tokens=False))
    user_open_ids = list(tokenizer.encode(bare_user_open, add_special_tokens=False))
    h_start = max(0, len(no_hist_ids) - len(user_open_ids))
    h_end = max(h_start, len(prefix_ids) - len(user_open_ids))
    return prefix_ids, suffix_ids, (h_start, h_end)


def encode_chatml_dialogue(
    tokenizer: Any,
    user_text: str,
    *,
    history: Sequence[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    assistant_text: str | None = None,
    rollout_ids: Sequence[int] | None = None,
    user_ids: Sequence[int] | None = None,
    enable_thinking: bool = False,
    max_total_tokens: int | None = None,
    truncate_history: bool = True,
    min_user_tokens: int = 32,
) -> EncodedDialogue:
    """Piece-wise encode a conversation in Qwen3 ChatML format.

    If `user_ids` / `rollout_ids` are passed in, those exact token IDs are spliced
    between the template prefix and assistant header so BPE merges can never alter
    token boundaries relative to `encode_plain_dialogue`.

    When `max_total_tokens` is set and `truncate_history=True`, oldest history turns
    are dropped from the left until the template + history + at least `min_user_tokens`
    (or the full user turn if shorter) fit inside `max_total_tokens`.
    """
    u_ids = (
        list(user_ids)
        if user_ids is not None
        else list(tokenizer.encode(user_text, add_special_tokens=False))
    )
    eos_ids = _collect_eos_ids(tokenizer)
    if rollout_ids is not None:
        r_ids = _strip_trailing_eos(rollout_ids, eos_ids)
    elif assistant_text:
        r_ids = _strip_trailing_eos(
            tokenizer.encode(assistant_text, add_special_tokens=False), eos_ids
        )
    else:
        r_ids = []

    hist_list = list(history) if history else []
    kept_start = 0
    prefix_ids, suffix_ids, history_span = _split_chatml_prefix_suffix(
        tokenizer, hist_list[kept_start:], system_prompt, enable_thinking
    )

    if max_total_tokens is not None and truncate_history and hist_list:
        needed_user = min(len(u_ids), max(1, min_user_tokens))
        while (
            kept_start < len(hist_list)
            and len(prefix_ids) + len(suffix_ids) + needed_user + len(r_ids) > max_total_tokens
        ):
            # Drop oldest user+assistant pair together when aligned, else 1 turn
            if (
                kept_start + 1 < len(hist_list)
                and hist_list[kept_start].get("role") == "user"
                and hist_list[kept_start + 1].get("role") == "assistant"
            ):
                kept_start += 2
            else:
                kept_start += 1
            prefix_ids, suffix_ids, history_span = _split_chatml_prefix_suffix(
                tokenizer, hist_list[kept_start:], system_prompt, enable_thinking
            )

    if max_total_tokens is not None:
        budget = max_total_tokens - len(prefix_ids) - len(suffix_ids)
        if budget < 1:
            raise ValueError(
                f"Template overhead ({len(prefix_ids) + len(suffix_ids)}) >= "
                f"max_total_tokens ({max_total_tokens})"
            )
        if len(u_ids) + len(r_ids) > budget:
            # Keep user content first (truncate rollout first, then user tail if needed)
            if len(u_ids) <= budget:
                r_ids = r_ids[: budget - len(u_ids)]
            else:
                u_ids = u_ids[:budget]
                r_ids = []

    u_start = len(prefix_ids)
    u_end = u_start + len(u_ids)
    h_start = u_end
    h_end = h_start + len(suffix_ids)
    r_start = h_end
    r_end = r_start + len(r_ids)

    return EncodedDialogue(
        input_ids=prefix_ids + u_ids + suffix_ids + r_ids,
        wrapper="chatml",
        user_span=(u_start, u_end),
        header_span=(h_start, h_end),
        asst_last_idx=h_end - 1,
        rollout_span=(r_start, r_end),
        history_span=history_span,
        kept_history_start_idx=kept_start,
    )


def encode_plain_dialogue(
    tokenizer: Any,
    user_text: str,
    *,
    history: Sequence[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    assistant_text: str | None = None,
    rollout_ids: Sequence[int] | None = None,
    user_ids: Sequence[int] | None = None,
    fewshot_preamble: str = PLAIN_FEWSHOT_PREAMBLE,
) -> EncodedDialogue:
    """Piece-wise encode a conversation in Base-native plain dialogue format.

    Uses 2 short neutral Q&A turns + `User:\\n{user}\\nAssistant:\\n{asst}` and
    splices the EXACT `user_ids` and `rollout_ids` so every `user_content`,
    `asst_last`, and `asst_rollout` position maps 1-to-1 with ChatML.
    """
    pre_parts: list[str] = []
    if system_prompt:
        pre_parts.append(f"System:\n{system_prompt}\n\n")
    if fewshot_preamble:
        pre_parts.append(fewshot_preamble)
    pre_hist_str = "".join(pre_parts)

    hist_parts: list[str] = []
    if history:
        for m in history:
            label = "User" if m["role"] == "user" else "Assistant"
            hist_parts.append(f"{label}:\n{m['content']}\n\n")
    hist_str = "".join(hist_parts)

    user_open_str = "User:\n"
    prefix_str = pre_hist_str + hist_str + user_open_str
    suffix_str = "\nAssistant:\n"

    prefix_ids = list(tokenizer.encode(prefix_str, add_special_tokens=False))
    if hist_str:
        pre_hist_ids = list(tokenizer.encode(pre_hist_str, add_special_tokens=False))
        user_open_ids = list(tokenizer.encode(user_open_str, add_special_tokens=False))
        history_span = (len(pre_hist_ids), max(len(pre_hist_ids), len(prefix_ids) - len(user_open_ids)))
    else:
        history_span = (0, 0)

    u_ids = (
        list(user_ids)
        if user_ids is not None
        else list(tokenizer.encode(user_text, add_special_tokens=False))
    )
    suffix_ids = list(tokenizer.encode(suffix_str, add_special_tokens=False))
    assert len(suffix_ids) >= 1, "Empty plain suffix tokenization"

    eos_ids = _collect_eos_ids(tokenizer)
    if rollout_ids is not None:
        r_ids = _strip_trailing_eos(rollout_ids, eos_ids)
    elif assistant_text:
        r_ids = _strip_trailing_eos(
            tokenizer.encode(assistant_text, add_special_tokens=False), eos_ids
        )
    else:
        r_ids = []

    u_start = len(prefix_ids)
    u_end = u_start + len(u_ids)
    h_start = u_end
    h_end = h_start + len(suffix_ids)
    r_start = h_end
    r_end = r_start + len(r_ids)

    return EncodedDialogue(
        input_ids=prefix_ids + u_ids + suffix_ids + r_ids,
        wrapper="plain",
        user_span=(u_start, u_end),
        header_span=(h_start, h_end),
        asst_last_idx=h_end - 1,
        rollout_span=(r_start, r_end),
        history_span=history_span,
    )


def encode_raw_user_only(
    tokenizer: Any,
    user_text: str,
    *,
    user_ids: Sequence[int] | None = None,
) -> EncodedDialogue:
    """Encode raw unwrapped user text (for chat_template_amplification diagnostic)."""
    u_ids = (
        list(user_ids)
        if user_ids is not None
        else list(tokenizer.encode(user_text, add_special_tokens=False))
    )
    n = len(u_ids)
    return EncodedDialogue(
        input_ids=u_ids,
        wrapper="raw",
        user_span=(0, n),
        header_span=(n, n),
        asst_last_idx=-1,
        rollout_span=(n, n),
    )


def encode_aligned_pair(
    tokenizer: Any,
    user_text: str,
    *,
    base_wrapper: str = "chatml",
    history: Sequence[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    assistant_text: str | None = None,
    rollout_ids: Sequence[int] | None = None,
    user_ids: Sequence[int] | None = None,
    enable_thinking: bool = False,
    max_total_tokens: int | None = None,
    truncate_history: bool = True,
) -> tuple[EncodedDialogue, EncodedDialogue]:
    """Encode (inst_enc, base_enc) with guaranteed 1-to-1 aligned spans."""
    assert base_wrapper in ("chatml", "plain"), f"Unknown base_wrapper: {base_wrapper!r}"
    inst_enc = encode_chatml_dialogue(
        tokenizer,
        user_text,
        history=history,
        system_prompt=system_prompt,
        assistant_text=assistant_text,
        rollout_ids=rollout_ids,
        user_ids=user_ids,
        enable_thinking=enable_thinking,
        max_total_tokens=max_total_tokens,
        truncate_history=truncate_history,
    )
    # Extract the exact (possibly truncated) history, user, and rollout IDs from inst_enc
    kept_hist = list(history)[inst_enc.kept_history_start_idx :] if history else None
    u_ids = inst_enc.input_ids[inst_enc.user_span[0] : inst_enc.user_span[1]]
    r_ids = inst_enc.input_ids[inst_enc.rollout_span[0] : inst_enc.rollout_span[1]]

    if base_wrapper == "chatml":
        base_enc = inst_enc
    else:
        base_enc = encode_plain_dialogue(
            tokenizer,
            user_text,
            history=kept_hist,
            system_prompt=system_prompt,
            user_ids=u_ids,
            rollout_ids=r_ids,
        )
    verify_aligned_pair(inst_enc, base_enc)
    return inst_enc, base_enc


def verify_aligned_pair(inst_enc: EncodedDialogue, base_enc: EncodedDialogue) -> None:
    """Assert 1-to-1 token identity across user_content and asst_rollout spans."""
    iu0, iu1 = inst_enc.user_span
    bu0, bu1 = base_enc.user_span
    assert inst_enc.input_ids[iu0:iu1] == base_enc.input_ids[bu0:bu1], (
        "user_content token IDs diverged between Instruct and Base wrappers"
    )
    ir0, ir1 = inst_enc.rollout_span
    br0, br1 = base_enc.rollout_span
    assert inst_enc.input_ids[ir0:ir1] == base_enc.input_ids[br0:br1], (
        "asst_rollout token IDs diverged between Instruct and Base wrappers"
    )
    assert inst_enc.asst_last_idx >= 0 and base_enc.asst_last_idx >= 0


def map_inst_pos_to_base_pos(
    inst_pos: int,
    inst_enc: EncodedDialogue,
    base_enc: EncodedDialogue,
) -> int:
    """Map an active Instruct position (`user_content`, `asst_last`, `asst_rollout`) to Base."""
    iu0, iu1 = inst_enc.user_span
    if iu0 <= inst_pos < iu1:
        return base_enc.user_span[0] + (inst_pos - iu0)
    if inst_pos == inst_enc.asst_last_idx:
        return base_enc.asst_last_idx
    ir0, ir1 = inst_enc.rollout_span
    if ir0 <= inst_pos < ir1:
        return base_enc.rollout_span[0] + (inst_pos - ir0)
    if inst_enc.wrapper == base_enc.wrapper:
        return inst_pos
    raise ValueError(
        f"Position {inst_pos} (role={inst_enc.role_at(inst_pos)!r}) is a template token "
        f"that cannot be mapped 1-to-1 across mismatched wrappers "
        f"({inst_enc.wrapper!r} vs {base_enc.wrapper!r})."
    )


def compute_role_means(
    deltas_by_role: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute exact per-role mean vectors mu_r in float64 -> float32 (Fix 2)."""
    means: dict[str, torch.Tensor] = {}
    for role, mat in deltas_by_role.items():
        assert mat.ndim == 2 and mat.shape[0] >= 1, (
            f"Role {role!r} must have shape [N>=1, d], got {tuple(mat.shape)}"
        )
        means[role] = mat.double().mean(dim=0).float().cpu()
    return means


def apply_role_centering(
    deltas: torch.Tensor,
    roles: Sequence[str],
    role_means: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Subtract role-specific mean mu_{role(i)} from each row of `deltas` [N, d]."""
    assert deltas.ndim == 2 and len(roles) == deltas.shape[0], (
        f"Shape/length mismatch: deltas={tuple(deltas.shape)}, len(roles)={len(roles)}"
    )
    out = deltas.float().clone()
    dev = out.device
    for role in set(roles):
        if role not in role_means:
            raise KeyError(f"Role {role!r} not found in role_means (have: {sorted(role_means)})")
        mu = role_means[role].to(device=dev, dtype=torch.float32)
        idxs = [i for i, r in enumerate(roles) if r == role]
        out[idxs] = out[idxs] - mu
    return out


def save_role_means(
    path: str | Path,
    role_means: dict[str, torch.Tensor],
    *,
    tau_web_median: float | None = None,
    tau_web_p75: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist `role_means_L24.pt` atomically."""
    payload: dict[str, Any] = {k: v.float().cpu() for k, v in role_means.items()}
    if tau_web_median is not None:
        payload["tau_web_median"] = float(tau_web_median)
    if tau_web_p75 is not None:
        payload["tau_web_p75"] = float(tau_web_p75)
    if extra:
        payload.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(p)
    return p


def load_role_means(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load `role_means_L24.pt`."""
    return torch.load(str(path), map_location=map_location, weights_only=False)
