"""Soft-token readouts of a head's write: warm-start descriptions that are a function of the
write alone.

The write deviation v = W_h (z_h − z̄_h), normalized per head (÷ dev_rms, capped at
--max-norm, as export_sft), is added to the model's own attention output at the head's
own layer, at a placeholder token in a fixed few-shot prompt. That is exactly where and
how the head's write enters the residual stream, so the MLP of that layer and every later
layer read it the way they read real head writes; the placeholder's own state is the
neutral carrier. Then the model continues the prompt greedily, or its next-token distribution
at the end of the prompt is listed (readouts "topk" and "delta").

Nothing else about the example reaches the model: not the document, not the attention
pattern, not the residual. The prompt is identical for every example and decoding is
deterministic, so the description is a function of v (and the head's layer, which picks
the injection site).

Injected size is β · ‖carrier residual at the site‖ · v — v's own size is kept, so an idle
head (‖v‖ ≈ 0) leaves the carrier's default reading.

Carrier state ("carrier_state"): "token" keeps the placeholder token's own residual, whose
identity can dominate the reading (8B reads " X" as "a mathematical variable" whatever the
write); "mean" replaces it, entering the site layer, with that layer's average residual over
the corpus — one fixed vector per layer, the same for every example — so the placeholder is
an average token rather than a word.
"""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from nla.datagen._common import load_tokenizer
from nla.utils.arch_adapters import resolve_decoder_layers

MARK = "{m}"


@dataclass(frozen=True)
class Recipe:
    name: str
    prompt: str          # completion-style text with one or more {m} placeholder slots
    beta: float          # injected norm per unit ‖v‖, as a fraction of the carrier residual's norm
    carrier: str = " X"  # text each {m} becomes; the write is added at its last token
    site: str = "own"    # "own" = the head's layer; "own-<k>" = k layers earlier; "L<k>" = fixed layer k
    carrier_state: str = "token"   # "token" | "mean" (see module docstring)
    readout: str = "greedy"   # "greedy" | "topk" | "delta" (see SoftTokenDescriber.describe)
    max_new_tokens: int = 20
    top_k: int = 8
    delta_pool: int = 64      # "delta" ranks only the injected run's top delta_pool tokens


# Few-shot prompts end at the placeholder, so the first generated token reads it directly.
PROMPTS = {
    "identity": "cat → cat\n1135 → 1135\nhello → hello\nParis → Paris\nrunning → running\n{m} →",
    "meaning": ("Short meanings of words and phrases:\n"
                "violin: a string instrument\n"
                "Paris: the capital of France\n"
                "running: moving quickly on foot\n"
                "1984: a year\n"
                "{m}:"),
    "assoc": ("Words that come to mind:\n"
              "ocean: water, waves, sea, blue\n"
              "guitar: music, strings, band, song\n"
              "1990s: decade, grunge, internet, nineties\n"
              "Tokyo: Japan, city, sushi, capital\n"
              "ate: eat, food, past, meal\n"
              "{m}:"),
    # the placeholder ends the text, so the continuation is what the write makes come next
    "continue": "Here is some text:\n\n{m}",
    # four placeholder copies, each carrying the write
    "assoc4": ("Words that come to mind:\n"
               "ocean: water, waves, sea, blue\n"
               "guitar: music, strings, band, song\n"
               "1990s: decade, grunge, internet, nineties\n"
               "Tokyo: Japan, city, sushi, capital\n"
               "ate: eat, food, past, meal\n"
               "{m}{m}{m}{m}:"),
    "continue4": "Here is some text:\n\n{m}{m}{m}{m}",
}


def recipe_grid(prompts: list[str], betas: list[float], readouts: list[str], carrier: str = " X",
                site: str = "own", carrier_state: str = "token") -> list[Recipe]:
    tag = ("" if site == "own" else f"_{site}") + ("" if carrier == " X" else f"_c{carrier.strip() or 'sp'}") + \
          ("" if carrier_state == "token" else f"_{carrier_state}")
    return [Recipe(name=f"{p}_b{b:g}_{r}{tag}", prompt=PROMPTS[p], beta=b, carrier=carrier, site=site,
                   carrier_state=carrier_state, readout=r)
            for p in prompts for b in betas for r in readouts]


class SoftTokenDescriber:
    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = "bfloat16", batch_size: int = 32,
                 mean_resid: dict[int, torch.Tensor] | None = None):
        """mean_resid: layer → average residual entering it (for carrier_state "mean")."""
        self.mean_resid = mean_resid or {}
        self.tokenizer = load_tokenizer(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, dtype=getattr(torch, dtype)).eval()
        self.layers = resolve_decoder_layers(self.model)
        self.batch_size = batch_size
        self.device = self.model.get_input_embeddings().weight.device
        self._carrier_norm: dict[tuple[str, str, int], float] = {}

    def encode(self, recipe: Recipe) -> tuple[list[int], list[int]]:
        """Prompt ids and the positions to inject at (last token of each placeholder).
        Pieces are tokenized separately so the placeholder positions are exact."""
        ids: list[int] = []
        marks: list[int] = []
        pieces = recipe.prompt.split(MARK)
        for i, piece in enumerate(pieces):
            if piece:
                ids += self.tokenizer(piece, add_special_tokens=False)["input_ids"]
            if i < len(pieces) - 1:
                ids += self.tokenizer(recipe.carrier, add_special_tokens=False)["input_ids"]
                marks.append(len(ids) - 1)
        assert marks, f"recipe {recipe.name} has no {MARK} slot"
        return ids, marks

    def site_layer(self, recipe: Recipe, layer: int) -> int:
        if recipe.site == "own":
            return layer
        if recipe.site.startswith("own-"):
            return max(0, layer - int(recipe.site.removeprefix("own-")))
        return int(recipe.site.removeprefix("L"))

    @torch.no_grad()
    def carrier_norm(self, recipe: Recipe, site: int) -> float:
        """‖residual entering layer `site`‖ at the (first) placeholder of the clean prompt."""
        if recipe.carrier_state == "mean":
            return float(self.mean_resid[site].float().norm())
        key = (recipe.prompt, recipe.carrier, site)
        if key not in self._carrier_norm:
            ids, marks = self.encode(recipe)
            out = self.model(torch.tensor([ids], device=self.device), output_hidden_states=True)
            self._carrier_norm[key] = float(out.hidden_states[site][0, marks[0]].float().norm())
        return self._carrier_norm[key]

    @contextmanager
    def _inject(self, site: int, marks: list[int], n_prompt: int, delta: torch.Tensor,
                carrier: torch.Tensor | None = None) -> Iterator[None]:
        """Add delta [B, d] to the attention output of layer `site` at the marks; with `carrier`
        [d], first set the marks' residual entering that layer to it. Prefill only: decode steps
        see the edited states through the KV cache."""
        def hook(_m: Any, _args: Any, out: torch.Tensor) -> torch.Tensor | None:
            if out.shape[1] != n_prompt:
                return None
            out = out.clone()
            out[:, marks] += delta.to(out.dtype)[:, None, :]
            return out

        def pre_hook(_m: Any, args: tuple, kwargs: dict) -> tuple | None:
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] != n_prompt:
                return None
            h = h.clone()
            h[:, marks] = carrier.to(h.device, h.dtype)
            return ((h, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})
        handles = [self.layers[site].self_attn.o_proj.register_forward_hook(hook)]
        if carrier is not None:
            handles.append(self.layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def describe(self, recipe: Recipe, vecs: torch.Tensor, layers: list[int]) -> list[str]:
        """vecs: [N, d_model] normalized write deviations (typical norm 1); layers: each one's layer.

        greedy  the greedy continuation's first line
        topk    the top_k next tokens at the end of the prompt
        delta   the top_k tokens whose log-prob the write raised most over the clean prompt's,
                among the injected run's delta_pool most likely (so rare tokens can't win on Δ alone)
        """
        ids, marks = self.encode(recipe)
        clean_logp: dict[int, torch.Tensor] = {}   # per site: the carrier's own (uninjected) reading
        out = [""] * len(vecs)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, layer in enumerate(layers):
            groups[self.site_layer(recipe, layer)].append(i)
        todo = [(s, idx[a : a + self.batch_size]) for s, idx in groups.items()
                for a in range(0, len(idx), self.batch_size)]
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for site, batch in tqdm(todo, desc=recipe.name, leave=False):
            scale = recipe.beta * self.carrier_norm(recipe, site)
            delta = vecs[batch].to(self.device, torch.float32) * scale
            carrier = self.mean_resid[site] if recipe.carrier_state == "mean" else None
            x = torch.tensor([ids] * len(batch), device=self.device)
            if recipe.readout == "delta" and site not in clean_logp:
                with self._inject(site, marks, len(ids), torch.zeros_like(delta[:1]), carrier):
                    clean_logp[site] = self.model(x[:1]).logits[0, -1].float().log_softmax(-1)
            with self._inject(site, marks, len(ids), delta, carrier):
                if recipe.readout in ("topk", "delta"):
                    logp = self.model(x).logits[:, -1].float().log_softmax(-1)
                    if recipe.readout == "topk":
                        top = logp.topk(recipe.top_k, dim=-1).indices
                    else:
                        pool = logp.topk(recipe.delta_pool, dim=-1).indices
                        gain = logp.gather(1, pool) - clean_logp[site][pool]
                        top = pool.gather(1, gain.topk(recipe.top_k, dim=-1).indices)
                    texts = [", ".join(repr(self.tokenizer.decode([t])) for t in row) for row in top.tolist()]
                else:
                    gen = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                              max_new_tokens=recipe.max_new_tokens, do_sample=False,
                                              temperature=None, top_p=None, top_k=None, pad_token_id=pad)
                    texts = [clean(t) for t in self.tokenizer.batch_decode(gen[:, len(ids):],
                                                                            skip_special_tokens=True)]
            for i, t in zip(batch, texts, strict=True):
                out[i] = t
        return out


def clean(raw: str) -> str:
    """First line of the continuation, whitespace-collapsed."""
    return " ".join(raw.lstrip(" ").split("\n")[0].split())
