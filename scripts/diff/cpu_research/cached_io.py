"""Disk-cached, split-model activation extraction for CPU-only Diff-NLA research.

Why this exists
---------------
`nla.diff.model_io.load_causal_lm` loads a whole model in fp32 on CPU. Qwen3-8B
in fp32 is ~33 GB, which does not fit a 31 GB orb. This module loads only the
decoder blocks a request needs:

  * "bottom": embedding + blocks [0, K]            -> Layer-K residual stream
  * "top":    blocks (K, N) + final norm + lm_head -> exact next-token logits,
              fed the cached Layer-K residual stream through `inputs_embeds`

and caches every whole-sequence Layer-K activation on disk, keyed by
(model, layer, token ids). `cached_extract` is a drop-in replacement for
`nla.diff.model_io.extract_at_positions`, so the unmodified repo scripts can be
re-run with different seeds / pair counts at no extra model cost
(see `run_patched.py`).

Numerics: forward passes are fp32; cached activations are stored as fp16
(relative rounding ~5e-4, far below the bf16 noise of the GPU pipeline).
Sequences that need logits also keep an fp32 copy to feed the top half.
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from nla.diff.model_io import PositionExtraction

CACHE_ROOT = Path(os.environ.get("NLA_ACT_CACHE", "/home/user/act_cache"))
TOKEN_BUDGET = int(os.environ.get("NLA_TOKEN_BUDGET", "2048"))
MAX_BATCH = int(os.environ.get("NLA_MAX_BATCH", "32"))
RECORD_PATH = os.environ.get("NLA_RECORD", "")
# 16 logical threads measured 1.7x faster than torch's default of 8 on the 8B bottom half.
torch.set_num_threads(int(os.environ.get("NLA_THREADS", str(os.cpu_count() or 8))))

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def seq_key(seq: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(list(seq), dtype=np.int32).tobytes()).hexdigest()


def _snapshot_dir(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    return Path(snapshot_download(name, allow_patterns=["*.safetensors", "*.json"]))


_LORA_RE = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def _merge_lora(sd: dict[str, torch.Tensor], adapter: str, lo: int, hi: int) -> int:
    """Fold a plain LoRA adapter into the fp32 block weights in `sd`: W += (alpha / r) * B @ A."""
    import json

    adir = _snapshot_dir(adapter)
    acfg = json.loads((adir / "adapter_config.json").read_text())
    assert acfg.get("peft_type") == "LORA" and not acfg.get("use_dora") and not acfg.get("use_rslora"), (
        "only plain LoRA adapters are supported"
    )
    assert not acfg.get("rank_pattern") and not acfg.get("alpha_pattern") and not acfg.get("modules_to_save")
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    with safe_open(str(adir / "adapter_model.safetensors"), "pt") as sf:
        for key in sf.keys():  # noqa: SIM118
            m = _LORA_RE.match(key)
            assert m, f"unsupported adapter key: {key}"
            i = int(m.group(1))
            if lo <= i < hi:
                parts.setdefault((i, m.group(2)), {})[m.group(3)] = sf.get_tensor(key).to(torch.float32)
    for (i, mod), ab in parts.items():
        tgt = f"model.layers.{i - lo}.{mod}.weight"
        assert tgt in sd and set(ab) == {"A", "B"}, f"adapter target missing: {tgt}"
        sd[tgt] = sd[tgt] + scale * (ab["B"] @ ab["A"])
    return len(parts)


def load_partial(name: str, lo: int, hi: int, *, embed: bool, head: bool, adapter: str | None = None) -> Any:
    """Load decoder blocks [lo, hi) of `name` in fp32, renumbered from 0.

    embed: keep the input embedding. head: keep the final norm and lm_head.
    Dropped parts are replaced so they cost no memory. `adapter` merges a LoRA.
    """
    from transformers.modeling_utils import no_init_weights

    cfg = AutoConfig.from_pretrained(name)
    n_total = int(cfg.num_hidden_layers)
    assert 0 <= lo < hi <= n_total, f"bad block range [{lo}, {hi}) for {n_total} layers"
    cfg.num_hidden_layers = hi - lo

    # Build the skeleton without touching its (uninitialized) weight memory.
    orig_lin = torch.nn.Linear.reset_parameters
    orig_emb = torch.nn.Embedding.reset_parameters
    torch.nn.Linear.reset_parameters = lambda self: None  # type: ignore[method-assign]
    torch.nn.Embedding.reset_parameters = lambda self: None  # type: ignore[method-assign]
    try:
        with no_init_weights():
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    finally:
        torch.nn.Linear.reset_parameters = orig_lin  # type: ignore[method-assign]
        torch.nn.Embedding.reset_parameters = orig_emb  # type: ignore[method-assign]

    if not embed:
        model.model.embed_tokens = None
    if not head:
        model.lm_head = torch.nn.Identity()
        model.model.norm = torch.nn.Identity()
    gc.collect()

    sd: dict[str, torch.Tensor] = {}
    embed_w: torch.Tensor | None = None
    for f in sorted(_snapshot_dir(name).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():  # noqa: SIM118
                m = _LAYER_RE.match(k)
                if m:
                    i = int(m.group(1))
                    if lo <= i < hi:
                        sd[f"model.layers.{i - lo}.{m.group(2)}"] = sf.get_tensor(k).to(torch.float32)
                elif k == "model.embed_tokens.weight":
                    if embed or head:
                        embed_w = sf.get_tensor(k).to(torch.float32)
                        if embed:
                            sd[k] = embed_w
                elif k == "model.norm.weight" or k == "lm_head.weight":
                    if head:
                        sd[k] = sf.get_tensor(k).to(torch.float32)
    if head and "lm_head.weight" not in sd:
        assert bool(getattr(cfg, "tie_word_embeddings", False)) and embed_w is not None, (
            "lm_head.weight missing from checkpoint and embeddings are not tied"
        )
        sd["lm_head.weight"] = embed_w
    del embed_w
    if adapter is not None:
        n_merged = _merge_lora(sd, adapter, lo, hi)
        assert n_merged > 0, f"adapter {adapter} touched no block in [{lo}, {hi})"

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # A tied skeleton reports the embedding as missing when only lm_head was assigned (top half).
    allowed_missing = set() if embed else {"model.embed_tokens.weight"}
    real_missing = [k for k in missing if k not in allowed_missing]
    assert not real_missing, f"missing keys after partial load: {real_missing[:5]}"
    model.eval()
    return model


class LazyLM:
    """Stand-in for a loaded CausalLM: loads the needed half only on a cache miss."""

    def __init__(self, name: str, adapter: str | None = None) -> None:
        self.name = str(name)
        self.adapter = adapter
        self.tag = self.name.strip("/").replace("/", "__")
        if adapter is not None:
            self.tag += "+" + str(adapter).strip("/").replace("/", "__")
        self._part: Any = None
        self._kind: tuple[str, int] | None = None
        self.n_layers = int(AutoConfig.from_pretrained(self.name).num_hidden_layers)

    def ensure(self, kind: str, layer_index: int) -> Any:
        if self._kind == (kind, layer_index):
            return self._part
        self.free()
        t0 = time.time()
        if kind == "bottom":
            self._part = load_partial(self.name, 0, layer_index + 1, embed=True, head=False, adapter=self.adapter)
        elif kind == "top":
            assert layer_index + 1 < self.n_layers, "no blocks above the capture layer"
            self._part = load_partial(self.name, layer_index + 1, self.n_layers, embed=False, head=True,
                                      adapter=self.adapter)
        else:
            raise ValueError(kind)
        self._kind = (kind, layer_index)
        print(f"[cached_io] loaded {kind} half of {self.tag} @L{layer_index} in {time.time() - t0:.0f}s", flush=True)
        return self._part

    def free(self) -> None:
        self._part = None
        self._kind = None
        gc.collect()

    def unload(self) -> LazyLM:
        """Mirror `PeftModel.unload()`: hand back the adapter-free model."""
        self.free()
        return LazyLM(self.name)

    # Minimal nn.Module surface used by the repo scripts.
    def eval(self) -> LazyLM:
        return self

    def to(self, *_a: Any, **_k: Any) -> LazyLM:
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def lazy_load_causal_lm(model_name_or_path: str, *, dtype: Any = None, device: Any = None,
                        adapter_path: str | None = None) -> LazyLM:
    assert adapter_path is None, "LoRA adapters are not supported by the split CPU loader"
    return LazyLM(model_name_or_path)


def lazy_peft_from_pretrained(model: Any, adapter_id: str, **_kw: Any) -> LazyLM:
    """Stand-in for `PeftModel.from_pretrained(model, adapter_id)` on a LazyLM handle."""
    assert isinstance(model, LazyLM) and model.adapter is None
    model.free()
    return LazyLM(model.name, adapter=str(adapter_id))


def lazy_unload_model(model: Any) -> None:
    if isinstance(model, LazyLM):
        model.free()
    gc.collect()


def _dirs(tag: str, layer_index: int) -> tuple[Path, Path, Path]:
    root = CACHE_ROOT / tag / f"L{layer_index}"
    h16, h32, lg = root / "h16", root / "h32", root / "logits"
    for d in (h16, h32, lg):
        d.mkdir(parents=True, exist_ok=True)
    return h16, h32, lg


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    tmp.replace(path)


def _batches(items: list[tuple[str, list[int]]]) -> list[list[tuple[str, list[int]]]]:
    """Length-sorted batches under a padded-token budget."""
    items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)
    out: list[list[tuple[str, list[int]]]] = []
    cur: list[tuple[str, list[int]]] = []
    cur_max = 0
    for kv in items:
        new_max = max(cur_max, len(kv[1]))
        if cur and (new_max * (len(cur) + 1) > TOKEN_BUDGET or len(cur) >= MAX_BATCH):
            out.append(cur)
            cur, new_max = [], len(kv[1])
        cur.append(kv)
        cur_max = new_max
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def _fill_hidden(model: LazyLM, miss: dict[str, list[int]], layer_index: int, pad_id: int,
                 keep32: set[str]) -> None:
    h16_dir, h32_dir, _ = _dirs(model.tag, layer_index)
    part = model.ensure("bottom", layer_index)
    n_tok = sum(len(s) for s in miss.values())
    done = 0
    t0 = time.time()
    last = t0
    for batch in _batches(list(miss.items())):
        max_len = len(batch[0][1])
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (_, s) in enumerate(batch):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        h = part.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        for i, (k, s) in enumerate(batch):
            arr = h[i, : len(s)].float().numpy()
            assert np.isfinite(arr).all(), f"non-finite activations for {k}"
            if k in keep32:
                _save_npy(h32_dir / f"{k}.npy", arr.astype(np.float32))
            _save_npy(h16_dir / f"{k}.npy", arr.astype(np.float16))
            done += len(s)
        now = time.time()
        if now - last > 30:
            rate = done / (now - t0)
            print(
                f"[cached_io] {model.tag} L{layer_index}: {done}/{n_tok} tok "
                f"({rate:.1f} tok/s, eta {(n_tok - done) / max(rate, 1e-9) / 60:.1f} min)",
                flush=True,
            )
            last = now
    print(f"[cached_io] {model.tag} L{layer_index}: filled {len(miss)} seqs / {n_tok} tok "
          f"in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def _fill_logits(model: LazyLM, want: dict[str, tuple[list[int], set[int]]], layer_index: int) -> None:
    _, h32_dir, lg_dir = _dirs(model.tag, layer_index)
    part = model.ensure("top", layer_index)
    for batch in _batches([(k, v[0]) for k, v in want.items()]):
        max_len = len(batch[0][1])
        hs = [torch.from_numpy(np.load(h32_dir / f"{k}.npy")) for k, _ in batch]
        d = hs[0].shape[-1]
        emb = torch.zeros((len(batch), max_len, d), dtype=torch.float32)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, h in enumerate(hs):
            emb[i, : h.shape[0]] = h
            mask[i, : h.shape[0]] = 1
        final = part.model(inputs_embeds=emb, attention_mask=mask, use_cache=False).last_hidden_state
        for i, (k, _) in enumerate(batch):
            for p in sorted(want[k][1]):
                lg = part.lm_head(final[i, p]).float().numpy()
                _save_npy(lg_dir / f"{k}_{p}.npy", lg.astype(np.float32))


@torch.no_grad()
def iter_final_hidden(model: LazyLM, items: Sequence[tuple[str, Sequence[int]]], *, layer_index: int,
                      residual_from: LazyLM | None = None):
    """Yield (item index, fp32 [n_pos, d]): the post-norm final hidden state at the given positions.

    `items` are (cache key, positions). The top half runs on the cached fp16 Layer-K residuals, and
    `model.ensure("top", layer_index).lm_head` turns a yielded state into exact next-token logits.
    For bulk next-token statistics, where saving full logits (0.6 MB per position) is not an option.
    `check_fp16_feed` measures what the fp16 rounding of the residual costs.

    `residual_from` feeds this model's top half the cached residuals of another model (same hidden
    size, same token sequences): a cross-model patch of the whole residual stream at Layer K.
    """
    h16_dir, _, _ = _dirs((residual_from or model).tag, layer_index)
    lengths = [int(np.load(h16_dir / f"{k}.npy", mmap_mode="r").shape[0]) for k, _ in items]
    part = model.ensure("top", layer_index)
    todo = [(str(i), [0] * n) for i, n in enumerate(lengths) if len(items[i][1]) > 0]
    n_tok = sum(len(s) for _, s in todo)
    done = 0
    t0 = last = time.time()
    for batch in _batches(todo):
        max_len = len(batch[0][1])
        hs = [np.load(h16_dir / f"{items[int(i)][0]}.npy").astype(np.float32) for i, _ in batch]
        emb = torch.zeros((len(batch), max_len, hs[0].shape[-1]), dtype=torch.float32)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for r, h in enumerate(hs):
            emb[r, : h.shape[0]] = torch.from_numpy(h)
            mask[r, : h.shape[0]] = 1
        final = part.model(inputs_embeds=emb, attention_mask=mask, use_cache=False).last_hidden_state
        for r, (i, _) in enumerate(batch):
            yield int(i), final[r, [int(p) for p in items[int(i)][1]]].float().clone()
        done += sum(h.shape[0] for h in hs)
        if time.time() - last > 30:
            print(f"[cached_io] {model.tag} top half: {done}/{n_tok} tok ({done / (time.time() - t0):.0f} tok/s)",
                  flush=True)
            last = time.time()


@torch.no_grad()
def check_fp16_feed(model: LazyLM, *, layer_index: int, n_seqs: int = 16) -> dict[str, float]:
    """Compare logits from `iter_final_hidden` with logits this cache already holds from fp32 residuals."""
    h16_dir, _, lg_dir = _dirs(model.tag, layer_index)
    want: dict[str, list[int]] = {}
    for f in sorted(lg_dir.glob("*.npy")):
        k, p = f.stem.rsplit("_", 1)
        if (h16_dir / f"{k}.npy").exists() and (k in want or len(want) < n_seqs):
            want.setdefault(k, []).append(int(p))
    if not want:
        return {"n_positions": 0}
    items = list(want.items())
    head = model.ensure("top", layer_index).lm_head
    kls, diffs = [], []
    for i, fin in iter_final_hidden(model, items, layer_index=layer_index):
        k, pl = items[i]
        ref = torch.from_numpy(np.stack([np.load(lg_dir / f"{k}_{p}.npy") for p in pl]))
        lp_ref, lp_new = F.log_softmax(ref, dim=-1), F.log_softmax(head(fin).float(), dim=-1)
        kls.extend((lp_ref.exp() * (lp_ref - lp_new)).sum(-1).tolist())
        diffs.append(float((lp_ref.exp() * (lp_ref - lp_new).abs()).sum(-1).max()))
    return {"n_positions": len(kls), "max_kl_nats": float(max(kls)), "mean_kl_nats": float(np.mean(kls)),
            "max_expected_abs_logprob_diff": float(max(diffs))}


def _record_and_fake(model: LazyLM, seqs: list[list[int]], pos: list[list[int]], layer_index: int,
                     need_logits: bool, topk_logits: int) -> list[PositionExtraction]:
    """Record mode: log the request and return correctly-shaped random outputs.

    The orb's 29.5 GiB memory limit cannot hold an 8B half plus the tensors a repo script
    accumulates. So a script is first run in record mode (no model, throwaway output dir),
    `warm_cache.py` then fills the cache from the manifest with nothing else in memory, and
    the real run finds every activation on disk.
    """
    import json

    with open(RECORD_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": model.name, "adapter": model.adapter, "layer": layer_index,
                             "seqs": seqs, "pos": pos, "want_logits": bool(need_logits or topk_logits > 0)}) + "\n")
    cfg = AutoConfig.from_pretrained(model.name)
    d, v = int(cfg.hidden_size), int(cfg.vocab_size)
    g = torch.Generator().manual_seed(len(seqs))
    out: list[PositionExtraction] = []
    for s, pl in zip(seqs, pos):
        lg = torch.randn((len(pl), v), generator=g) if (need_logits or topk_logits > 0) else None
        tk_ids = tk_probs = None
        if topk_logits > 0 and lg is not None:
            vals, inds = F.softmax(lg, dim=-1).topk(topk_logits, dim=-1)
            tk_ids, tk_probs = inds.tolist(), vals.tolist()
        out.append(PositionExtraction(positions=pl, token_ids=[s[p] for p in pl],
                                      hidden=torch.randn((len(pl), d), generator=g),
                                      logits=(lg if need_logits else None), topk_ids=tk_ids, topk_probs=tk_probs))
    return out


def warm_from_manifest(path: str, pad_token_id: int) -> None:
    """Fill the cache for every request in a record-mode manifest, one model half at a time."""
    import json

    groups: dict[tuple[str, str | None, int], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            g = groups.setdefault((r["name"], r["adapter"], int(r["layer"])), {"seqs": {}, "logits": {}})
            for s, pl in zip(r["seqs"], r["pos"]):
                if not pl:
                    continue
                k = seq_key(s)
                g["seqs"][k] = s
                if r["want_logits"]:
                    g["logits"].setdefault(k, set()).update(pl)
    for (name, adapter, layer), g in groups.items():
        model = LazyLM(name, adapter=adapter)
        h16_dir, h32_dir, lg_dir = _dirs(model.tag, layer)
        logit_need = {k: (g["seqs"][k], {p for p in ps if not (lg_dir / f"{k}_{p}.npy").exists()})
                      for k, ps in g["logits"].items()}
        logit_need = {k: v for k, v in logit_need.items() if v[1]}
        miss = {k: s for k, s in g["seqs"].items()
                if not (h16_dir / f"{k}.npy").exists() or (k in logit_need and not (h32_dir / f"{k}.npy").exists())}
        print(f"[warm] {model.tag} L{layer}: {len(g['seqs'])} seqs, {len(miss)} hidden misses, "
              f"{len(logit_need)} seqs need logits", flush=True)
        if miss:
            _fill_hidden(model, miss, layer, pad_token_id, keep32=set(logit_need))
        if logit_need:
            _fill_logits(model, logit_need, layer)
        model.free()


def lean_centered_covariances(
    X: torch.Tensor, Y: torch.Tensor, chunk: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same result as `nla.diff.alignment._centered_covariances`, accumulated in row chunks.

    The stock version materialises centred float64 copies of X and Y (5 GB per CV fold at
    d=4096, N=80k), which pushes the 8B alignment fit past this orb's memory limit. Only the
    float64 summation order differs.
    """
    n = X.shape[0]
    assert n >= 2, f"Need at least 2 samples to fit alignment, got {n}"
    d_x, d_y = X.shape[1], Y.shape[1]
    xm = torch.zeros(d_x, dtype=torch.float64)
    ym = torch.zeros(d_y, dtype=torch.float64)
    for s in range(0, n, chunk):
        xm += X[s : s + chunk].double().sum(dim=0)
        ym += Y[s : s + chunk].double().sum(dim=0)
    xm /= n
    ym /= n
    cov_xx = torch.zeros((d_x, d_x), dtype=torch.float64)
    cov_xy = torch.zeros((d_x, d_y), dtype=torch.float64)
    for s in range(0, n, chunk):
        xc = X[s : s + chunk].double() - xm
        yc = Y[s : s + chunk].double() - ym
        cov_xx += xc.T @ xc
        cov_xy += xc.T @ yc
    return xm, ym, cov_xx / float(n), cov_xy / float(n)


def cached_extract(
    model: Any,
    sequences: Sequence[Sequence[int]],
    positions_per_seq: Sequence[Sequence[int]],
    *,
    layer_index: int,
    pad_token_id: int,
    batch_size: int = 8,
    need_logits: bool = False,
    topk_logits: int = 0,
) -> list[PositionExtraction]:
    """Drop-in for `nla.diff.model_io.extract_at_positions`, backed by the disk cache."""
    assert isinstance(model, LazyLM), "cached_extract needs a LazyLM handle"
    assert len(sequences) == len(positions_per_seq)
    seqs = [[int(t) for t in s] for s in sequences]
    pos = [[int(p) for p in pl] for pl in positions_per_seq]
    want_logits = need_logits or topk_logits > 0
    if RECORD_PATH:
        return _record_and_fake(model, seqs, pos, layer_index, need_logits, topk_logits)
    keys = [seq_key(s) for s in seqs]
    h16_dir, h32_dir, lg_dir = _dirs(model.tag, layer_index)

    logit_need: dict[str, tuple[list[int], set[int]]] = {}
    if want_logits:
        for k, s, pl in zip(keys, seqs, pos):
            missing_p = {p for p in pl if not (lg_dir / f"{k}_{p}.npy").exists()}
            if missing_p:
                logit_need.setdefault(k, (s, set()))[1].update(missing_p)

    miss: dict[str, list[int]] = {}
    for k, s, pl in zip(keys, seqs, pos):
        if not pl:
            continue
        if not (h16_dir / f"{k}.npy").exists() or (k in logit_need and not (h32_dir / f"{k}.npy").exists()):
            miss[k] = s
    if miss:
        _fill_hidden(model, miss, layer_index, pad_token_id, keep32=set(logit_need))
    if logit_need:
        _fill_logits(model, logit_need, layer_index)

    out: list[PositionExtraction] = []
    for k, s, pl in zip(keys, seqs, pos):
        if not pl:
            first = np.load(h16_dir / f"{keys[0]}.npy", mmap_mode="r") if (h16_dir / f"{keys[0]}.npy").exists() else None
            d = int(first.shape[-1]) if first is not None else int(AutoConfig.from_pretrained(model.name).hidden_size)
            out.append(PositionExtraction(positions=[], token_ids=[], hidden=torch.zeros((0, d)),
                                          topk_ids=([] if topk_logits > 0 else None),
                                          topk_probs=([] if topk_logits > 0 else None)))
            continue
        h = np.load(h16_dir / f"{k}.npy", mmap_mode="r")
        assert h.shape[0] == len(s), f"cache length mismatch for {k}"
        hidden = torch.from_numpy(np.asarray(h[pl], dtype=np.float32))
        lg_sel = tk_ids = tk_probs = None
        if want_logits:
            raw = torch.from_numpy(np.stack([np.load(lg_dir / f"{k}_{p}.npy") for p in pl]))
            if need_logits:
                lg_sel = raw
            if topk_logits > 0:
                probs = F.softmax(raw, dim=-1)
                vals, inds = probs.topk(min(topk_logits, probs.shape[-1]), dim=-1)
                tk_ids, tk_probs = inds.tolist(), vals.tolist()
        out.append(PositionExtraction(positions=pl, token_ids=[s[p] for p in pl], hidden=hidden,
                                      logits=lg_sel, topk_ids=tk_ids, topk_probs=tk_probs))
    return out


def full_sequence_hidden(model: LazyLM, sequences: Sequence[Sequence[int]], *, layer_index: int,
                         pad_token_id: int) -> list[np.ndarray]:
    """Return fp32 [L, d] Layer-K activations for every position of each sequence."""
    seqs = [[int(t) for t in s] for s in sequences]
    exs = cached_extract(model, seqs, [list(range(len(s))) for s in seqs],
                         layer_index=layer_index, pad_token_id=pad_token_id)
    return [e.hidden.numpy() for e in exs]
