"""Target model M (stock Qwen3-8B) with a layer-l substitution tap.

The tap is a forward hook on decoder block `layer_index` (sidecar convention:
layer_index=24 hooks model.layers[24]'s OUTPUT = the residual stream entering
block 25 — same stream the AV/AR were trained on, same convention as
nla/utils/hooks.py). Modes:

  - None:          pass through (C0 clean)
  - "capture":     stash the layer output (Stage A clean forward)
  - "replace_all": overwrite the full [B, S, d] hidden states with a provided
                   tensor at masked positions (Stage A patched forward)
  - callable:      resid -> resid, set per-forward by generate() (C0′ / C1 /
                   C1-prompt); the callable does its own position slicing

Substitution semantics during generation: EVERY generated token must be
sampled from a substituted stream, so the transform fires (a) at the LAST real
prompt position during prefill — that position's stream produces the first
generated token; without this, single-token answers (MMLU-Pro letters) would
bypass the codec entirely — and (b) at the current position of every decode
step. The rest of the prefill stays clean (condition "nla_prompt" additionally
substitutes every real prompt position, to quantify the clean-prompt-KV
bypass). Prefill codec calls are logged with step < 0.

Generation is a custom batched greedy loop (left-padded prefill + DynamicCache)
rather than model.generate(): we need per-step synchronous codec calls on the
active rows only, per-token logging, and finished-row masking. Correctness of
the loop is validated by sanity_checks.py (C0 == batched model.generate greedy;
C0′ token-identical to C0) and tests/test_bottleneck_cpu.py on a tiny model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nla.utils.arch_adapters import resolve_decoder_layers

CONDITIONS = ("clean", "identity", "nla", "nla_prompt")


def identity_transform(h: torch.Tensor) -> torch.Tensor:
    """C0′: substitute h for itself through the full interception machinery —
    device→cpu fp32→device bf16 serialization round-trip + the same write path.
    bf16→fp32→bf16 is exact (no arithmetic), so C0′ must match C0 token-for-token;
    any divergence is a harness bug (dtype, indexing, off-by-one-layer)."""
    return h.detach().float().cpu().to(h.device, h.dtype)


@dataclass
class StepLog:
    """One (row, step) codec application during generation.

    step >= 0: decode step; token_id is the token FED IN at that step (the
    position whose activation was substituted — the substitution influences
    the NEXT sampled token). step < 0: prefill substitution at prompt position
    P+step (step=-1 = last prompt position); token_id is that prompt token.
    """
    row: int
    step: int
    token_id: int
    cosine: float
    h_norm: float
    pred_norm: float
    z_len: int
    z_text: str
    truncated: bool
    extract_failed: bool
    steer_verified: bool


class BottleneckModel:
    def __init__(self, m_ckpt: str = "Qwen/Qwen3-8B", layer_index: int = 24,
                 device: str = "cuda", attn_implementation: str = "sdpa"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.layer_index = layer_index
        print(f"[M] loading {m_ckpt} (tap at layers[{layer_index}] output)")
        self.model = AutoModelForCausalLM.from_pretrained(
            m_ckpt, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(m_ckpt)
        n_layers = self.model.config.num_hidden_layers
        assert 0 <= layer_index < n_layers - 1, (
            f"layer_index {layer_index} must leave layers above it ({n_layers} total)"
        )
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tokenizer.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # --- tap state ---
        self._mode = None            # None | "capture" | "replace_all" | callable
        self._captured: torch.Tensor | None = None
        self._replacement: torch.Tensor | None = None       # [B, S, d]
        self._replace_mask: torch.Tensor | None = None      # [B, S] bool
        self._handle = resolve_decoder_layers(self.model)[layer_index] \
            .register_forward_hook(self._tap)

    # ------------------------------------------------------------------ tap --
    def _tap(self, module, args, output):
        if self._mode is None:
            return output
        if isinstance(output, tuple):
            resid, rest = output[0], output[1:]
        else:
            resid, rest = output, None

        if self._mode == "capture":
            self._captured = resid.detach()
            return output
        if self._mode == "replace_all":
            assert self._replacement is not None and self._replace_mask is not None
            assert self._replacement.shape == resid.shape, (
                f"replacement {tuple(self._replacement.shape)} vs resid {tuple(resid.shape)}"
            )
            m = self._replace_mask.to(resid.device).unsqueeze(-1)
            new = torch.where(m, self._replacement.to(resid.device, resid.dtype), resid)
        else:
            assert callable(self._mode)
            new = self._mode(resid)
        if rest is None:
            return new
        return (new, *rest)

    # -------------------------------------------------------------- prompts --
    def build_chat_ids(self, messages: list[dict]) -> list[int]:
        """M's chat template, non-thinking mode (every generated token costs one
        full AV explanation; thinking traces would multiply the ~150× overhead
        and make output lengths incomparable across conditions)."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------ transform builders --
    def _last_pos_fn(self, active: torch.Tensor, codec, step: int,
                     tok_at_pos: torch.Tensor, step_logs: list[StepLog] | None,
                     identity: bool):
        """Transform resid[:, -1] for active rows. Valid for both the prefill
        forward (left-padding puts every row's last real prompt token in the
        final column) and decode steps (S=1)."""
        def fn(resid: torch.Tensor) -> torch.Tensor:
            new = resid.clone()
            rows = active.to(resid.device).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return new
            if identity:
                new[rows, -1] = identity_transform(resid[rows, -1])
                return new
            h_hat, recs = codec.roundtrip(resid[rows, -1])
            new[rows, -1] = h_hat.to(resid.device, resid.dtype)
            if step_logs is not None:
                for j, r in enumerate(rows.tolist()):
                    rec = recs[j]
                    step_logs.append(StepLog(
                        row=r, step=step, token_id=int(tok_at_pos[r]),
                        cosine=rec.cosine, h_norm=rec.h_norm, pred_norm=rec.pred_norm,
                        z_len=rec.verb.n_tokens, z_text=rec.verb.text,
                        truncated=rec.verb.truncated,
                        extract_failed=rec.verb.extract_failed,
                        steer_verified=rec.verb.steer_verified,
                    ))
            return new
        return fn

    def _all_prompt_fn(self, mask: torch.Tensor, ids: torch.Tensor, codec,
                       step_logs: list[StepLog] | None):
        """nla_prompt prefill: substitute EVERY real prompt position (quantifies
        how much information survives via clean prompt KV in plain C1)."""
        def fn(resid: torch.Tensor) -> torch.Tensor:
            new = resid.clone()
            S = resid.shape[1]
            flat = mask.bool().to(resid.device).nonzero(as_tuple=False)  # [N,2]
            h_hat, recs = codec.roundtrip(resid[flat[:, 0], flat[:, 1]])
            new[flat[:, 0], flat[:, 1]] = h_hat.to(resid.device, resid.dtype)
            if step_logs is not None:
                for j in range(flat.shape[0]):
                    r, pos = int(flat[j, 0]), int(flat[j, 1])
                    rec = recs[j]
                    step_logs.append(StepLog(
                        row=r, step=pos - S, token_id=int(ids[r, pos]),
                        cosine=rec.cosine, h_norm=rec.h_norm, pred_norm=rec.pred_norm,
                        z_len=rec.verb.n_tokens, z_text=rec.verb.text,
                        truncated=rec.verb.truncated,
                        extract_failed=rec.verb.extract_failed,
                        steer_verified=rec.verb.steer_verified,
                    ))
            return new
        return fn

    # ----------------------------------------------------------- generation --
    @torch.no_grad()
    def generate(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int,
        codec=None,
        condition: str = "clean",
        step_logs: list[StepLog] | None = None,
    ) -> list[list[int]]:
        """Batched greedy generation under a condition. Returns generated ids
        (per row, EOS included if emitted). See module docstring for the
        substitution semantics per condition."""
        assert condition in CONDITIONS, condition
        if condition in ("nla", "nla_prompt"):
            assert codec is not None
        from transformers import DynamicCache

        B = len(prompt_ids_list)
        maxp = max(len(p) for p in prompt_ids_list)
        ids = torch.full((B, maxp), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, maxp), dtype=torch.long, device=self.device)
        for r, p in enumerate(prompt_ids_list):      # left-pad
            ids[r, maxp - len(p):] = torch.tensor(p, dtype=torch.long, device=self.device)
            mask[r, maxp - len(p):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        all_rows = torch.ones(B, dtype=torch.bool, device=self.device)
        last_prompt_tok = ids[:, -1]

        cache = DynamicCache()
        if condition == "clean":
            self._mode = None
        elif condition == "nla_prompt":
            self._mode = self._all_prompt_fn(mask, ids, codec, step_logs)
        else:
            self._mode = self._last_pos_fn(all_rows, codec, -1, last_prompt_tok,
                                           step_logs, identity=condition == "identity")
        try:
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                             past_key_values=cache, use_cache=True, logits_to_keep=1)
        finally:
            self._mode = None
        next_tok = out.logits[:, -1, :].float().argmax(-1)          # [B]
        next_pos = mask.sum(-1, keepdim=True)                        # [B,1]

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        generated: list[list[int]] = [[] for _ in range(B)]

        for step in range(max_new_tokens):
            for r in range(B):
                if not finished[r]:
                    generated[r].append(int(next_tok[r]))
            newly = torch.tensor([int(next_tok[r]) in self.eos_ids for r in range(B)],
                                 device=self.device)
            finished |= newly
            if bool(finished.all()) or step == max_new_tokens - 1:
                break

            # One decode step; the tap substitutes layer-l output for active rows.
            active = ~finished
            if condition == "clean":
                self._mode = None
            else:
                self._mode = self._last_pos_fn(active, codec, step, next_tok,
                                               step_logs,
                                               identity=condition == "identity")
            step_ids = next_tok.unsqueeze(-1)                        # [B,1]
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long,
                                               device=self.device)], dim=-1)
            try:
                out = self.model(input_ids=step_ids, attention_mask=mask,
                                 position_ids=next_pos, past_key_values=cache,
                                 use_cache=True)
            finally:
                self._mode = None
            next_pos = next_pos + 1
            nxt = out.logits[:, -1, :].float().argmax(-1)
            # finished rows keep stepping (their KV grows) but their output is
            # discarded; pin them to pad so they can't re-trigger EOS logic.
            next_tok = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)

        return generated

    # -------------------------------------------------- teacher-forced (A) --
    @torch.no_grad()
    def forward_teacher_forced(
        self,
        ids: torch.Tensor,            # [B, S] right-padded
        mask: torch.Tensor,           # [B, S]
        replacement: torch.Tensor | None = None,   # [B, S, d] → patched forward
        replace_mask: torch.Tensor | None = None,  # [B, S] bool, positions to swap
        capture: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One full forward (no cache). Returns (logits [B,S,V] in model dtype —
        caller reduces immediately) and the captured layer-l stream if capture."""
        assert not (capture and replacement is not None)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        self._captured = None
        if capture:
            self._mode = "capture"
        elif replacement is not None:
            assert replace_mask is not None
            self._mode = "replace_all"
            self._replacement = replacement
            self._replace_mask = replace_mask
        try:
            out = self.model(input_ids=ids.to(self.device),
                             attention_mask=mask.to(self.device),
                             position_ids=pos.to(self.device), use_cache=False)
        finally:
            self._mode = None
            self._replacement = None
            self._replace_mask = None
        return out.logits, self._captured

    @torch.no_grad()
    def score_continuation_nll(self, prompt_ids: list[int], cont_ids: list[int]) -> float:
        """Mean NLL of cont_ids given prompt_ids under CLEAN M (fluency metric:
        PPL of a condition's outputs under the unpatched model)."""
        assert cont_ids, "empty continuation"
        full = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=self.device)
        self._mode = None
        logits = self.model(input_ids=full, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[0, len(prompt_ids) - 1:-1], dim=-1)
        tgt = torch.tensor(cont_ids, dtype=torch.long, device=self.device)
        return float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())
