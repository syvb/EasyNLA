"""Future Oracle inference (self-contained port of nla.future_lens.eval's readout path).

A Future Oracle is a LoRA adapter on a Qwen3 base model that reads a single residual-stream
vector (block-`layer` output at token t) and verbalises the tokens the model will produce next.
Recipe (identical to training/eval in EasyNLA `nla/future_lens`):
  1. activation  h = hidden_states[layer + 1][t]  of the bare model on the token ids (no prompt)
  2. ground truth = the bare model's greedy continuation of ids[:t+1] (the oracle's label convention)
  3. decoder prompt = template.format(layer, k, MARKER) + "\\n"   (prompt_format "plain")
  4. the marker token's *input embedding* is replaced by  alpha_layer * alpha_mult * h / ||h||
     (alpha_layer = p75 activation norm of that layer, from the collection sidecar)
  5. greedy decode with the adapter on, cut at EOS, truncated to k tokens.

Registry: every `ckpts/<run>/iter_*/adapter_config.json` in the HF dataset repo is an oracle
(latest iter per run; sweep/ablation runs skipped). Alpha per layer + template come from the
matching `data*/train.parquet.nla_meta.yaml` sidecar (same base model, layers covered).
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = os.environ.get("FL_HF_REPO", "syvb/rl-future-lens-qwen3-8b")
TOKEN = os.environ.get("HF_TOKEN")
SKIP_RUN = re.compile(r"^(sweep_|ablation_|sft_filt|sft_unf)")
# base checkpoints served (the obsolete chat-model/text-label oracle would add a second 16 GB 8B)
BASES = [x for x in os.environ.get("FO_BASES", "Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-8B-Base").split(",") if x]


@dataclass
class Sidecar:
    data_dir: str
    base_model: str
    template: str
    prompt_format: str
    marker: str
    marker_id: int
    left_id: int
    right_id: int
    alpha_by_layer: dict[int, float]


@dataclass
class Oracle:
    run: str
    iter_dir: str          # repo path ckpts/<run>/iter_XXXXXXX
    step: int
    base_model: str
    layers: list[int]
    alpha_mult: float
    injection: str
    label: str
    sidecar: Sidecar
    local_dir: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def adapter_name(self) -> str:   # torch module names can't contain "."
        return self.run.replace(".", "_")

    @property
    def size(self) -> str:
        m = re.search(r"Qwen3-([\d.]+B)", self.base_model)
        return m.group(1) if m else self.base_model

    @property
    def name(self) -> str:
        kind = "RL" if self.run.startswith("rl_") else "SFT"
        lay = f"layer {self.layers[0]}" if len(self.layers) == 1 else f"layers {','.join(map(str, self.layers))}"
        return f"Future Oracle · Qwen3-{self.size} · {lay} · {kind} {self.run} @ {self.step}"


def _dl(path: str, local_dir: str) -> str:
    return hf_hub_download(REPO, path, repo_type="dataset", token=TOKEN, local_dir=local_dir)


def load_sidecar(data_dir: str, cache: str) -> Sidecar | None:
    for split in ("train", "eval"):
        try:
            p = _dl(f"{data_dir}/{split}.parquet.nla_meta.yaml", cache)
        except Exception:
            continue
        m = yaml.safe_load(open(p))
        fl, tk = m.get("future_lens", {}), m.get("tokens", {})
        if not fl or not tk:
            continue
        pf = (fl.get("extra") or {}).get("prompt_format") or ("plain" if "Base" in m["extraction"]["base_model"] else "chat")
        return Sidecar(
            data_dir=data_dir, base_model=m["extraction"]["base_model"], template=fl["template"], prompt_format=pf,
            marker=tk["injection_char"], marker_id=int(tk["injection_token_id"]),
            left_id=int(tk["injection_left_neighbor_id"]), right_id=int(tk["injection_right_neighbor_id"]),
            alpha_by_layer={int(k): float(v) for k, v in fl["injection_scale_by_layer"].items()},
        )
    return None


def scan_registry(cache: str = "/tmp/fo_meta") -> list[Oracle]:
    """Every finished oracle adapter in the repo, newest run first."""
    api = HfApi(token=TOKEN)
    files = [s.rfilename for s in api.dataset_info(REPO).siblings]
    sidecars: list[Sidecar] = []
    for d in sorted({f.split("/")[0] for f in files if f.startswith("data") and f.endswith(".nla_meta.yaml")}):
        if d == "data_unf":      # filter-ablation split: not a training split of any kept oracle
            continue
        sc = load_sidecar(d, cache)
        if sc:
            sidecars.append(sc)
    iters: dict[str, list[tuple[int, str]]] = {}
    for f in files:
        m = re.match(r"ckpts/([^/]+)/iter_(\d+)/adapter_config\.json$", f)
        if m and not SKIP_RUN.match(m.group(1)) and f"ckpts/{m.group(1)}/iter_{m.group(2)}/future_lens.json" in files:
            iters.setdefault(m.group(1), []).append((int(m.group(2)), f"ckpts/{m.group(1)}/iter_{m.group(2)}"))
    oracles: list[Oracle] = []
    for run, its in iters.items():
        step, idir = max(its)
        cfg = json.load(open(_dl(f"{idir}/adapter_config.json", cache)))
        fl = json.load(open(_dl(f"{idir}/future_lens.json", cache)))
        base = cfg["base_model_name_or_path"]
        if base not in BASES:
            continue
        layers = [int(x) for x in str(fl.get("layers", "")).split(",") if x]
        cands = [s for s in sidecars if s.base_model == base and set(layers) <= set(s.alpha_by_layer)]
        if not cands:
            continue
        sfx = re.search(r"_(\d+(?:\.\d+)?b)_L(\d+)$", run)
        pref = f"data_{sfx.group(1)}_L{sfx.group(2)}" if sfx else None
        sc = next((s for s in cands if s.data_dir == pref), None) or sorted(cands, key=lambda s: s.data_dir)[-1]
        oracles.append(Oracle(run=run, iter_dir=idir, step=step, base_model=base, layers=layers,
                              alpha_mult=float(fl.get("alpha_mult", 1.0)), injection=fl.get("injection", "replace_embed"),
                              label=fl.get("label", "greedy"), sidecar=sc, extra=fl))
    # small models first, then by run name
    def key(o):
        m = re.search(r"([\d.]+)B", o.size)
        return (float(m.group(1)) if m else 99, o.run)
    return sorted(oracles, key=key)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

class Bank:
    """Base models (bf16) with all oracle adapters attached as named PEFT adapters.

    ZeroGPU: everything is assembled on CPU at import time and moved to CUDA once per base model
    (`to_device`), which ZeroGPU packs for its workers. Attaching an adapter later (Refresh) moves
    that base back to CPU, attaches, and moves it to CUDA again."""

    def __init__(self, device: str = "cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.tok = None
        self.models: dict[str, torch.nn.Module] = {}     # base id -> PeftModel (or bare base)
        self.on_device: set[str] = set()
        self.loaded: set[str] = set()                     # oracle runs attached
        self.lock = threading.Lock()

    def to_device(self):
        for b, m in self.models.items():
            if b not in self.on_device:
                m.to(self.device)
                self.on_device.add(b)

    def tokenizer(self, base: str):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(base)   # Qwen3 sizes share one tokenizer
        return self.tok

    def base(self, base: str):
        if base not in self.models:
            self.tokenizer(base)
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(base, dtype=self.dtype, attn_implementation="sdpa")
            self.models[base] = m.eval()          # CPU until to_device()
        return self.models[base]

    def attach(self, o: Oracle):
        if o.run in self.loaded:
            return
        with self.lock:
            if o.run in self.loaded:
                return
            if o.local_dir is None:
                o.local_dir = os.path.join(snapshot_download(REPO, repo_type="dataset", token=TOKEN,
                                                             allow_patterns=[f"{o.iter_dir}/*"],
                                                             local_dir="/tmp/fo_ckpts"), o.iter_dir)
            from peft import PeftModel
            m = self.base(o.base_model)
            back = o.base_model in self.on_device
            if back:                                 # late attach: assemble on CPU, then re-pack
                m.to("cpu"); self.on_device.discard(o.base_model)
            # torch_device="cpu": PEFT otherwise infers "cuda" from ZeroGPU's patched availability flag
            # and maps the safetensors there, which is forbidden outside a GPU call
            if isinstance(m, PeftModel):
                m.load_adapter(o.local_dir, adapter_name=o.adapter_name, torch_device="cpu")
            else:
                self.models[o.base_model] = PeftModel.from_pretrained(m, o.local_dir, adapter_name=o.adapter_name,
                                                                      torch_device="cpu").eval()
            self.loaded.add(o.run)
            if back:
                self.to_device()

    def model(self, o: Oracle):
        self.attach(o)
        m = self.models[o.base_model]
        m.set_adapter(o.adapter_name)
        return m


def stop_ids(tok, model) -> set[int]:
    ids = {tok.eos_token_id}
    gc = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc is not None:
        ids.update(gc if isinstance(gc, (list, tuple)) else [gc])
    ids.discard(None)
    return ids


@torch.no_grad()
def read_future(bank: Bank, o: Oracle, layer: int, ids: list[int], t: int, k: int,
                control: str = "real", alpha_mult: float | None = None, shuffle_t: int | None = None) -> dict:
    """One oracle readout at token t of `ids`. control: real | none | shuffled (vector from
    position shuffle_t of the same text) | wrong_layer (layer given, adapter untrained on it)."""
    assert 0 <= t < len(ids)
    dev = bank.device
    model = bank.model(o)
    tok = bank.tokenizer(o.base_model)
    sc = o.sidecar
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    # ---- target model: activation + its own greedy continuation (adapter OFF) ----
    with model.disable_adapter():
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
        src_t = shuffle_t if control == "shuffled" else t
        h = out.hidden_states[layer + 1][0, src_t].float()
        top1 = int(out.logits[0, t].argmax())
        gen = model.generate(input_ids=x[:, : t + 1], attention_mask=torch.ones_like(x[:, : t + 1]),
                             max_new_tokens=k, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        greedy = gen[0, t + 1:].tolist()
        p_next = torch.log_softmax(out.logits[0, t].float(), -1)
    # ---- decoder prompt with the marker ----
    prompt = sc.template.format(layer=int(layer), k=int(k), injection_char=sc.marker)
    if sc.prompt_format == "plain":
        text = prompt + "\n"
    else:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(text, add_special_tokens=False)
    pos = [i for i in range(1, len(p_ids) - 1)
           if p_ids[i] == sc.marker_id and p_ids[i - 1] == sc.left_id and p_ids[i + 1] == sc.right_id]
    assert len(pos) == 1, f"marker not found exactly once in the decoder prompt: {pos}"
    alpha = sc.alpha_by_layer[int(layer)] * (o.alpha_mult if alpha_mult is None else alpha_mult)
    vec = None if control == "none" else (h / h.norm().clamp_min(1e-12)) * alpha
    embed = model.get_input_embeddings()

    def hook(module, args, kwargs, output):
        if vec is None:
            return output
        inp = kwargs.get("input") if kwargs else None
        if inp is None and args:
            inp = args[0]
        if inp is None or inp.dim() != 2 or inp.shape[1] < 2:
            return output          # KV-cached decode steps
        out2 = output.clone()
        out2[0, pos[0]] = vec.to(out2.dtype)
        return out2

    hd = embed.register_forward_hook(hook, with_kwargs=True)
    try:
        px = torch.tensor([p_ids], dtype=torch.long, device=dev)
        g = model.generate(input_ids=px, attention_mask=torch.ones_like(px), max_new_tokens=k + 3, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        hd.remove()
    resp = g[0, len(p_ids):].tolist()
    eos = stop_ids(tok, model)
    cut = next((i for i, x_ in enumerate(resp) if x_ in eos), len(resp))
    readout = resp[:cut][:k]
    actual = ids[t + 1: t + 1 + k]
    return {
        "oracle": o.name, "run": o.run, "size": o.size, "layer": int(layer), "t": t, "k": k, "control": control,
        "alpha": alpha, "act_norm": float(h.norm()),
        "readout": readout, "readout_text": tok.decode(readout),
        "greedy": greedy, "greedy_text": tok.decode(greedy),
        "actual": actual, "actual_text": tok.decode(actual),
        "hits": [i < len(readout) and i < len(greedy) and readout[i] == greedy[i] for i in range(k)],
        "top1_correct": (top1 == ids[t + 1]) if t + 1 < len(ids) else None,   # None: no next token to compare
        "p_actual_next": float(p_next[ids[t + 1]].exp()) if t + 1 < len(ids) else None,
    }
