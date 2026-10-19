"""Self-descriptions: inject a head's write into the model itself as a soft token and let it describe it.

The no-Claude alternative for warm-start descriptions (scripts/heads_compare_descriptions.py
scores both). The injected vector is the head's write deviation v = W_h (z_h − z̄_h); the
average write is a per-head constant and says nothing about the example. Two variants:

  l1_add      the verbalizer's own injection (as nla.injection.karvonen_inject_in_residual):
              at the marker, h ← h + ‖h‖·v/‖v‖ on the output of decoder layer 1
  same_layer  Patchscopes-style: the marker's residual ENTERING the head's own layer L is
              replaced by v rescaled to that residual's norm, so the write is read in the
              space it was written into (examples are batched per layer)

Both are norm-matched, so only v's direction reaches the model — an idle head's near-zero
write goes in at full strength. That is inherent to zero-shot injection, and part of what
the comparison measures.

The prompt mirrors the verbalizer's (marker between <concept> tags) and ends with an open
<description> tag. Every example shares the prompt — only the vector differs — so batches
need no padding, and greedy decoding makes the output a function of the vector alone.
"""

import argparse
import contextlib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from nla.datagen._common import load_tokenizer
from nla.datagen.heads.prompts import DESCRIPTION_CLOSE, DESCRIPTION_OPEN
from nla.datagen.heads.store import (
    atomic_write_table,
    fixed_numpy,
    from_bf16_bits,
    load_stats,
    load_wo,
    read_meta,
    read_records,
)
from nla.datagen.injection_tokens import _INJECTION_RANGE
from nla.utils.arch_adapters import resolve_decoder_layers

VARIANTS = ("l1_add", "same_layer")

# Kept free of quotable phrasing: an earlier version said the vector was "the information
# that head moved there from earlier tokens", and small models echoed that clause verbatim.
PROMPT = (
    "You are a meticulous AI researcher investigating a vector taken from inside a language model: "
    "one attention head's contribution to one token.\n\n"
    "Here is the vector:\n\n<concept>{marker}</concept>\n\n"
    "In one or two short clauses, name the specific words, entities or facts it carries."
)


def pick_marker(tokenizer: Any) -> tuple[str, int]:
    """First single-token char in the injection range — injection_tokens' rule, without
    writing to its committed cache."""
    lo, hi = _INJECTION_RANGE
    for cp in range(lo, hi + 1):
        ids = tokenizer(chr(cp), add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return chr(cp), ids[0]
    raise AssertionError(f"no single-token character in U+{lo:04X}–U+{hi:04X} for this tokenizer")


def resolve_style(style: str, model_name: str, tokenizer: Any) -> str:
    if style != "auto":
        return style
    if "base" in model_name.lower() or not getattr(tokenizer, "chat_template", None):
        return "completion"
    return "chat"


def head_write_deviations(out_dir: Path, rows: list[dict]) -> torch.Tensor:
    """v = W_h (z_h − z̄_h) for each {layer, head, doc_id, pos} row → [N, d_model] float32."""
    meta = read_meta(out_dir)
    stats = load_stats(out_dir)
    h_all, d = meta.n_heads, meta.head_dim
    out = torch.empty(len(rows), meta.d_model)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_layer[int(r["layer"])].append(i)
    for layer, idxs in by_layer.items():
        rec = read_records(out_dir, layer, columns=["doc_id", "pos", "head_out"])
        row_of = {(dd, p): j for j, (dd, p) in enumerate(zip(rec.column("doc_id").to_pylist(),
                                                              rec.column("pos").to_pylist(), strict=True))}
        take = [row_of[(rows[i]["doc_id"], int(rows[i]["pos"]))] for i in idxs]
        z = from_bf16_bits(fixed_numpy(rec.column("head_out").take(pa.array(take)), np.uint16, h_all * d))
        z = z.view(-1, h_all, d)
        wo = load_wo(out_dir, layer)
        z_bar = stats.mean_head_out[stats.layer_pos(layer)]
        heads = torch.tensor([int(rows[i]["head"]) for i in idxs])
        idx_t = torch.tensor(idxs)
        # one matmul per head: per-row matvecs on a many-core pod spent minutes in thread-pool overhead
        for h in heads.unique().tolist():
            sel = (heads == h).nonzero(as_tuple=True)[0]
            out[idx_t[sel]] = (z[sel, h] - z_bar[h]) @ wo[:, h * d:(h + 1) * d].T
    return out


class SelfDescriber:
    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = "bfloat16",
                 style: str = "auto", max_new_tokens: int = 48, batch_size: int = 16):
        self.tokenizer = load_tokenizer(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, dtype=getattr(torch, dtype)).eval()
        self.layers = resolve_decoder_layers(self.model)
        self.style = resolve_style(style, model_name, self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        marker, marker_id = pick_marker(self.tokenizer)
        content = PROMPT.format(marker=marker)
        if self.style == "chat":
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        else:
            text = content + "\n\n"
        self.prompt_ids = self.tokenizer(text + DESCRIPTION_OPEN, add_special_tokens=False)["input_ids"]
        hits = [i for i, t in enumerate(self.prompt_ids) if t == marker_id]
        assert len(hits) == 1, f"marker {marker!r} appears {len(hits)}× in the self-description prompt"
        self.marker_pos = hits[0]
        self.prompt_text = text + DESCRIPTION_OPEN

    @contextlib.contextmanager
    def _inject(self, variant: str, vecs: torch.Tensor, layer: int) -> Iterator[None]:
        n_prompt, p = len(self.prompt_ids), self.marker_pos

        def is_prefill(h: torch.Tensor) -> bool:
            return h.shape[1] == n_prompt   # decode steps run one position at a time

        if variant == "l1_add":
            def hook(_m, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                if not is_prefill(h):
                    return None
                h = h.clone()
                v = vecs.to(h.dtype)
                h[:, p] = h[:, p] + h[:, p].norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)
                return (h, *output[1:]) if isinstance(output, tuple) else h
            handle = self.layers[1].register_forward_hook(hook)
        elif variant == "same_layer":
            def pre_hook(_m, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                if not is_prefill(h):
                    return None
                h = h.clone()
                v = vecs.to(h.dtype)
                h[:, p] = v * (h[:, p].norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True))
                return ((h, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})
            handle = self.layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        else:
            raise AssertionError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def describe(self, vecs: torch.Tensor, layers: list[int], variant: str) -> list[str]:
        """Raw generations (text after the open <description> tag), one per vector."""
        device = self.model.get_input_embeddings().weight.device
        out: list[str] = [""] * len(vecs)
        # same_layer hooks the example's own layer, so batches never mix layers.
        groups: dict[int, list[int]] = defaultdict(list)
        for i, layer in enumerate(layers):
            groups[layer if variant == "same_layer" else -1].append(i)
        todo = [(g, idx[s : s + self.batch_size]) for g, idx in groups.items()
                for s in range(0, len(idx), self.batch_size)]
        for g, batch in tqdm(todo, desc=f"self-describe ({variant})", leave=False):
            ids = torch.tensor([self.prompt_ids] * len(batch), device=device)
            with self._inject(variant, vecs[batch].to(device), g):
                gen = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            texts = self.tokenizer.batch_decode(gen[:, len(self.prompt_ids):], skip_special_tokens=True)
            for i, t in zip(batch, texts, strict=True):
                out[i] = t
        return out


def clean_generation(raw: str) -> str:
    """Text up to </description> or the first line break."""
    return " ".join(raw.split(DESCRIPTION_CLOSE)[0].split("\n")[0].split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="head dataset (extract + stats done)")
    p.add_argument("--selection", default=None, help="parquet with example_idx, layer, head, doc_id, pos; "
                                                      "default warmstart/selection.parquet")
    p.add_argument("--variant", default="same_layer", choices=VARIANTS)
    p.add_argument("--model", default=None, help="default: the dataset's base model")
    p.add_argument("--style", default="auto", choices=["auto", "chat", "completion"])
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None, help="default warmstart/self_descriptions_<variant>.parquet")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    model = args.model or read_meta(out_dir).base_model
    sel = pq.read_table(args.selection or out_dir / "warmstart" / "selection.parquet").to_pylist()
    vecs = head_write_deviations(out_dir, sel)
    sd = SelfDescriber(model, args.device_map, args.dtype, args.style, args.max_new_tokens, args.batch_size)
    raw = sd.describe(vecs, [int(r["layer"]) for r in sel], args.variant)
    atomic_write_table(pa.table({
        "example_idx": [r["example_idx"] for r in sel],
        "variant": [args.variant] * len(sel),
        "description": [clean_generation(t) for t in raw],
        "raw_text": raw,
    }), Path(args.out or out_dir / "warmstart" / f"self_descriptions_{args.variant}.parquet"))
    print(f"self-described {len(sel)} examples ({args.variant}, {sd.style} prompt) with {model}")


if __name__ == "__main__":
    main()
