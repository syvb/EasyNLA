"""Per-head capture during one forward pass — the only model-specific code here.

For each requested (doc, position, layer) we record, in a single no-grad
forward:

  resid_pre   decoder-layer input at that position (pre-hook on the layer)
  head_out    o_proj input — the concatenated per-head attention outputs,
              head h at [h*D:(h+1)*D] (pre-hook on self_attn.o_proj)
  attention   softmax(q·kᵀ·scale) over all source positions for every head,
              summarized as top-k sources + weight on the first token, on the
              current token, and entropy

Attention weights come from a registered attention implementation that wraps
SDPA (`_ATTN_IMPL`): it sees the exact post-RoPE / post-QK-norm q and k the
model uses, recomputes scores only for the sampled query rows in float32, and
then runs the normal SDPA kernel. No eager [B, H, S, S] tensor, no second
forward, and no per-architecture RoPE re-implementation. Works for
llama-family modules (q/k/v/o_proj, num_key_value_groups, layer_idx);
logit soft-capping is asserted absent.

The forward is aborted after the deepest layer that has a capture (same
sentinel trick as extractors.HFExtractor) — no upper layers, no lm_head.
"""

import contextlib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

from nla.datagen._common import load_tokenizer
from nla.datagen.extractors import _CaptureComplete
from nla.datagen.heads.store import to_bf16_bits
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config

_ATTN_IMPL = "nla_head_capture"
_ACTIVE: "_CaptureState | None" = None


def _capture_attention_forward(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                               value: torch.Tensor, attention_mask: torch.Tensor | None,
                               **kwargs: Any) -> tuple[torch.Tensor, None]:
    state = _ACTIVE
    if state is not None:
        state.on_attention(module, query, key, kwargs.get("sliding_window"))
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


# Registering the mask function too matters: masking_utils returns NO mask for
# implementations it doesn't know, which would silently drop padding masks.
AttentionInterface.register(_ATTN_IMPL, _capture_attention_forward)
AttentionMaskInterface.register(_ATTN_IMPL, sdpa_mask)


@dataclass
class _LayerSelection:
    batch_idx: torch.Tensor   # [N] long, on model device
    pos: torch.Tensor         # [N] long, on model device


class _CaptureState:
    """Holds one sub-batch's selections and collects results as layers fire."""

    def __init__(self, sel: dict[int, _LayerSelection], top_k: int):
        self.sel = sel
        self.top_k = top_k
        self.stop_layer = max(sel)
        self.resid: dict[int, np.ndarray] = {}
        self.head_out: dict[int, np.ndarray] = {}
        self.attn: dict[int, dict[str, np.ndarray]] = {}

    def on_layer_input(self, layer: int, hidden: torch.Tensor) -> None:
        s = self.sel.get(layer)
        if s is not None:
            self.resid[layer] = to_bf16_bits(hidden[s.batch_idx, s.pos])

    def on_o_proj_input(self, layer: int, z: torch.Tensor) -> None:
        s = self.sel.get(layer)
        if s is None:
            return
        self.head_out[layer] = to_bf16_bits(z[s.batch_idx, s.pos])
        if layer == self.stop_layer:
            raise _CaptureComplete

    @torch.no_grad()
    def on_attention(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                     sliding_window: int | None) -> None:
        layer = module.layer_idx
        s = self.sel.get(layer)
        if s is None:
            return
        assert getattr(module.config, "attn_logit_softcapping", None) is None, (
            "attention logit soft-capping is not reproduced by the head capture — "
            "extend _CaptureState.on_attention before using this architecture"
        )
        if sliding_window is None:
            sliding_window = getattr(module, "sliding_window", None)
        _, n_heads, seq_len, head_dim = query.shape
        n_kv = key.shape[1]
        assert key.shape[2] == seq_len, "head capture expects a cache-free full forward"
        groups = n_heads // n_kv
        src = torch.arange(seq_len, device=query.device)

        n = s.pos.shape[0]
        k = min(self.top_k, seq_len)
        top_w = torch.empty(n, n_heads, k, dtype=torch.float32, device=query.device)
        top_i = torch.empty(n, n_heads, k, dtype=torch.long, device=query.device)
        first_w = torch.empty(n, n_heads, dtype=torch.float32, device=query.device)
        self_w = torch.empty(n, n_heads, dtype=torch.float32, device=query.device)
        entropy = torch.empty(n, n_heads, dtype=torch.float32, device=query.device)
        # Loop over batch rows, not selections: key[b] stays a view (no [N, KV,
        # S, D] gather), and each row's queries share one matmul.
        for b in s.batch_idx.unique().tolist():
            rows = (s.batch_idx == b).nonzero(as_tuple=True)[0]
            p = s.pos[rows]                                                    # [n_b]
            q = query[b][:, p, :].float().view(n_kv, groups, len(rows), head_dim)
            scores = torch.matmul(q, key[b].float().transpose(-1, -2).unsqueeze(1)) * module.scaling
            scores = scores.reshape(n_heads, len(rows), seq_len).transpose(0, 1)  # [n_b, H, S]
            valid = src[None, :] <= p[:, None]
            if sliding_window is not None:
                valid &= src[None, :] > (p[:, None] - sliding_window)
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            w, i = torch.topk(probs, k, dim=-1)
            top_w[rows], top_i[rows] = w, i
            first_w[rows] = probs[:, :, 0]
            self_w[rows] = probs.gather(-1, p.view(-1, 1, 1).expand(-1, n_heads, 1)).squeeze(-1)
            entropy[rows] = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
        self.attn[layer] = {
            "attn_top_idx": top_i.to(torch.int32).cpu().numpy(),
            "attn_top_w": top_w.cpu().numpy(),
            "attn_first_w": first_w.cpu().numpy(),
            "attn_self_w": self_w.cpu().numpy(),
            "attn_entropy": entropy.cpu().numpy(),
        }

    def check_complete(self) -> None:
        for layer in self.sel:
            missing = [name for name, got in (("resid_pre", self.resid), ("head_out", self.head_out),
                                              ("attention", self.attn)) if layer not in got]
            assert not missing, (
                f"layer {layer}: capture hooks did not fire for {missing}. The model's "
                f"attention may not route through config._attn_implementation, or its "
                f"module layout differs from the llama family (self_attn.o_proj)."
            )


class HeadCaptureExtractor:
    """Tokenize + forward + per-head capture. GPU placement mirrors HFExtractor."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 4096,
        batch_size: int = 12,
    ):
        tokenizer = load_tokenizer(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=getattr(torch, dtype),
            attn_implementation=_ATTN_IMPL,
        ).eval()
        self._init(model, tokenizer, max_length, batch_size)

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any, max_length: int = 4096,
                   batch_size: int = 12) -> "HeadCaptureExtractor":
        """Wrap an already-loaded model (tests). Switches it to the capture impl."""
        self = cls.__new__(cls)
        model.set_attn_implementation(_ATTN_IMPL)
        self._init(model.eval(), tokenizer, max_length, batch_size)
        return self

    def _init(self, model: Any, tokenizer: Any, max_length: int, batch_size: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        cfg = resolve_text_config(model.config)
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.d_model = cfg.hidden_size
        self.layers = resolve_decoder_layers(model)
        attn = self.layers[0].self_attn
        assert hasattr(attn, "o_proj") and attn.o_proj.in_features == self.n_heads * self.head_dim, (
            f"{type(attn).__name__}: expected self_attn.o_proj with in_features = "
            f"n_heads*head_dim = {self.n_heads * self.head_dim}"
        )
        # Padding id for right-padded batches. Pad positions are never captured
        # (sampled positions are < each doc's length).
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        assert pad is not None, "tokenizer has neither pad nor eos token"
        self.pad_id = pad
        # Right truncation: token_ids[0] must be the document start.
        self.tokenizer.truncation_side = "right"

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, add_special_tokens=True)
        return enc["input_ids"]

    def o_proj_weight(self, layer: int) -> torch.Tensor:
        return self.layers[layer].self_attn.o_proj.weight.detach()

    @contextlib.contextmanager
    def _hooks(self, layers: list[int]) -> Iterator[None]:
        handles = []

        def layer_pre(layer: int):
            def hook(_m: torch.nn.Module, args: tuple, kwargs: dict) -> None:
                hidden = args[0] if args else kwargs["hidden_states"]
                if _ACTIVE is not None:
                    _ACTIVE.on_layer_input(layer, hidden)
            return hook

        def o_proj_pre(layer: int):
            def hook(_m: torch.nn.Module, args: tuple) -> None:
                if _ACTIVE is not None:
                    _ACTIVE.on_o_proj_input(layer, args[0])
            return hook

        try:
            for layer in layers:
                handles.append(self.layers[layer].register_forward_pre_hook(layer_pre(layer), with_kwargs=True))
                handles.append(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(o_proj_pre(layer)))
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def capture(
        self, token_ids: list[list[int]], plan: list[list[tuple[int, tuple[int, ...]]]], top_k: int,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Run the captures in `plan` and return per-layer column arrays.

        plan[i] = [(pos, (layer, ...)), ...] for document i. The result maps
        layer → {"doc": [N] index into token_ids, "pos": [N], "resid_pre":
        [N, d] uint16, "head_out": [N, H*D] uint16, "attn_*": [N, H(, K)]}.
        """
        global _ACTIVE
        wanted = sorted({lay for doc_plan in plan for _, lays in doc_plan for lay in lays})
        out: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        if not wanted:
            return {}
        device = self.model.get_input_embeddings().weight.device
        # Length-sorted sub-batches: documents vary from ~50 to max_length tokens,
        # and right-padding a 60-token doc to 4096 wastes most of a forward.
        order = sorted((i for i in range(len(token_ids)) if plan[i]), key=lambda i: len(token_ids[i]))
        with self._hooks(wanted):
            for start in range(0, len(order), self.batch_size):
                docs = order[start : start + self.batch_size]
                seq_len = max(len(token_ids[i]) for i in docs)
                input_ids = torch.full((len(docs), seq_len), self.pad_id, dtype=torch.long)
                attention_mask = torch.zeros((len(docs), seq_len), dtype=torch.long)
                per_layer: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
                for b, i in enumerate(docs):
                    ids = token_ids[i]
                    input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[b, : len(ids)] = 1
                    for pos, lays in plan[i]:
                        assert 0 <= pos < len(ids), f"doc {i}: position {pos} outside its {len(ids)} tokens"
                        for lay in lays:
                            per_layer[lay].append((b, pos, i))
                sel = {
                    lay: _LayerSelection(
                        batch_idx=torch.tensor([b for b, _, _ in rows], dtype=torch.long, device=device),
                        pos=torch.tensor([p for _, p, _ in rows], dtype=torch.long, device=device),
                    )
                    for lay, rows in per_layer.items()
                }
                state = _CaptureState(sel, top_k)
                _ACTIVE = state
                try:
                    self.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                               use_cache=False)
                except _CaptureComplete:
                    pass
                finally:
                    _ACTIVE = None
                state.check_complete()
                for lay, rows in per_layer.items():
                    cols = out[lay]
                    cols["doc"].append(np.array([i for _, _, i in rows], dtype=np.int64))
                    cols["pos"].append(np.array([p for _, p, _ in rows], dtype=np.int32))
                    cols["resid_pre"].append(state.resid[lay])
                    cols["head_out"].append(state.head_out[lay])
                    for name, arr in state.attn[lay].items():
                        cols[name].append(arr)
        return {lay: {k: np.concatenate(v) for k, v in cols.items()} for lay, cols in out.items()}
