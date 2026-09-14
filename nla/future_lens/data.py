"""Future-lens dataset contract: parquet columns, sidecar extension, prompt building.

A future-lens parquet is an EasyNLA `av_sft`-shaped parquet (so `train_sft.py`
can load it unchanged) plus the columns the future-token task needs. One row
per (position, layer):

    prompt             list<struct{role,content}>  user turn, `<INJECT>` placeholder,
                                                   layer + K already substituted
    response           str                         decoded target_ids[:k] (inspection only)
    activation_vector  fixed_size_list<float16,d>  RAW residual at layer `activation_layer`
    activation_layer   int64
    doc_id             str
    n_raw_tokens       int64                       t + 1 (EasyNLA convention)
    target_ids         fixed_size_list<int64,nf>   x_{t+1} .. x_{t+nf}   (the readout labels)
    k                  int64                       readout length drawn for this row (K = N+1)
    doc_idx            int64                       index into docs.parquet
    t                  int64                       position of the activation
    p_top1             float32                     target's p(x_{t+1} | x_<=t)   (calibration)
    target_top5        fixed_size_list<int64,5nf>  teacher-forced top-5 at t..t+nf-1, row-major
    target_logp        fixed_size_list<float32,nf> log p(x_{t+1+j} | x_<=t+j)
    prev_ids           fixed_size_list<int64,np>   x_{t-np+1} .. x_t   (NEVER fed to the decoder)
    greedy_ids         fixed_size_list<int64,nf>   target's greedy continuation from x_<=t (-1 if absent)
    greedy_top5        fixed_size_list<int64,5nf>  target's top-5 at each greedy step (its own prefix)
    greedy_logp        fixed_size_list<float32,nf> log p of each greedy token (its own prefix)
    greedy_topk_ids    fixed_size_list<int32,nf*K>  target's top-K ids at each greedy step (distillation)
    greedy_topk_logp   fixed_size_list<float16,nf*K> their log-probs (K = sidecar `topk`)

Labels. `label="text"` reads out the corpus continuation (target_*); `label="greedy"`
(Future Lens, Pal et al. 2023: "the generated tokens outputs of GPT-J through greedy
decoding") swaps target_ids/top5/logp for greedy_ids/top5/logp at load time, so every
consumer keeps using the target_* names. The corpus ids stay available as `text_ids`.

`docs.parquet` (doc_idx, doc_id, split, ids) holds each document's token ids once;
the target-logprob reward and the surprisal metric read prefixes from it.

The sidecar is EasyNLA's `nla_dataset` sidecar (tokens, prompt_templates.actor for
the neighbour check) plus a `future_lens:` block — see `FLMeta`.

Readout convention (spec decision 1): the readout is K = N+1 tokens starting at
x_{t+1}; "N" indexes the offset of the last readout token. Precision@1 at N is
correctness of readout token N (0-based), i.e. of x_{t+1+N}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nla.schema import INJECT_PLACEHOLDER, sidecar_path_for

DEFAULT_TEMPLATE = (
    "Here is an activation vector from layer {layer} of a language model: "
    "<concept>{injection_char}</concept>\n"
    "Output the next {k} tokens the model will produce after this point."
)
DEFAULT_K_CHOICES = (1, 2, 3, 4, 5, 9)  # N in {0,1,2,3,4,8}: spec's {1,2,4,8} + N=0 sanity + N=3 (success criterion)
DEFAULT_N_FUTURE = 9
DEFAULT_N_PREV = 32
DEFAULT_TOPK = 64     # stored target top-K per greedy step (distillation targets)
FL_SIDECAR_KEY = "future_lens"
PROMPT_FORMATS = ("chat", "plain")
# Process-wide default prompt format, set from the dataset sidecar by load_fl_meta() so that
# collect / SFT / RL / eval all render prompts the same way without threading a flag through
# every call site. "chat" = the tokenizer's chat template with thinking off (post-trained
# checkpoints); "plain" = the template text + "\n", then the readout (base checkpoints, for
# which the chat template is meaningless).
_PROMPT_FORMAT = {"fmt": "chat"}


def set_prompt_format(fmt: str) -> None:
    assert fmt in PROMPT_FORMATS, fmt
    _PROMPT_FORMAT["fmt"] = fmt


def get_prompt_format() -> str:
    return _PROMPT_FORMAT["fmt"]


@dataclass
class FLMeta:
    layer_indices: list[int]
    n_future: int
    n_prev: int
    k_choices: list[int]
    template: str
    norm_quantiles: dict[int, dict[str, float]]
    injection_scale_by_layer: dict[int, float]
    docs_parquet: str | None = None
    discard_fraction: float | None = None
    d_model: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def prompt_format(self) -> str:
        return str(self.extra.get("prompt_format", "chat"))

    @property
    def topk(self) -> int:
        return int(self.extra.get("topk", 0))

    def alpha(self, layer: int, mult: float = 1.0) -> float:
        return float(self.injection_scale_by_layer[int(layer)]) * mult

    def to_dict(self) -> dict:
        return {
            "layer_indices": [int(x) for x in self.layer_indices],
            "n_future": int(self.n_future),
            "n_prev": int(self.n_prev),
            "k_choices": [int(x) for x in self.k_choices],
            "template": self.template,
            "norm_quantiles": {int(k): {q: float(v) for q, v in d.items()}
                               for k, d in self.norm_quantiles.items()},
            "injection_scale_by_layer": {int(k): float(v)
                                         for k, v in self.injection_scale_by_layer.items()},
            "docs_parquet": self.docs_parquet,
            "discard_fraction": self.discard_fraction,
            "d_model": self.d_model,
            **self.extra,
        }


def load_fl_meta(sidecar_source: str | Path) -> FLMeta:
    """Read the `future_lens:` block from a dataset sidecar (parquet path or ckpt dir)."""
    meta = yaml.safe_load(sidecar_path_for(sidecar_source).read_text())
    assert FL_SIDECAR_KEY in meta, (
        f"{sidecar_path_for(sidecar_source)} has no `{FL_SIDECAR_KEY}` block — not a "
        f"future-lens dataset (built by nla.future_lens.collect)?"
    )
    fl = dict(meta[FL_SIDECAR_KEY])
    known = {"layer_indices", "n_future", "n_prev", "k_choices", "template",
             "norm_quantiles", "injection_scale_by_layer", "docs_parquet",
             "discard_fraction", "d_model"}
    set_prompt_format(str(fl.get("prompt_format", "chat")))
    return FLMeta(
        layer_indices=[int(x) for x in fl["layer_indices"]],
        n_future=int(fl["n_future"]),
        n_prev=int(fl["n_prev"]),
        k_choices=[int(x) for x in fl["k_choices"]],
        template=fl["template"],
        norm_quantiles={int(k): dict(v) for k, v in fl["norm_quantiles"].items()},
        injection_scale_by_layer={int(k): float(v)
                                  for k, v in fl["injection_scale_by_layer"].items()},
        docs_parquet=fl.get("docs_parquet"),
        discard_fraction=fl.get("discard_fraction"),
        d_model=fl.get("d_model") or meta.get("extraction", {}).get("d_model"),
        extra={k: v for k, v in fl.items() if k not in known},
    )


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

def fill_template(template: str, layer: int, k: int, injection_char: str = INJECT_PLACEHOLDER) -> str:
    return template.format(layer=int(layer), k=int(k), injection_char=injection_char)


def build_prompt_messages(template: str, layer: int, k: int) -> list[dict]:
    """The parquet `prompt` value: one user turn with the `<INJECT>` placeholder."""
    return [{"role": "user", "content": fill_template(template, layer, k)}]


def chat_prompt_text(tokenizer, messages: list[dict], inject_char: str, fmt: str | None = None) -> str:
    """Render a prompt with the real marker char. fmt (default: the dataset's sidecar value
    via set_prompt_format): "chat" = chat template with thinking OFF (the decoder must read
    the state out directly, not reason about it); "plain" = the user text + newline, for base
    checkpoints. The readout token ids are appended after this string's tokens."""
    fmt = fmt or get_prompt_format()
    msgs = [
        {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
        if isinstance(m.get("content"), str) else m
        for m in messages
    ]
    if fmt == "plain":
        assert len(msgs) == 1 and msgs[0].get("role") == "user", msgs
        return msgs[0]["content"] + "\n"
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def encode_prompt(tokenizer, messages: list[dict], inject_char: str) -> list[int]:
    return tokenizer.encode(chat_prompt_text(tokenizer, messages, inject_char),
                            add_special_tokens=False)


# ----------------------------------------------------------------------------
# Parquet schema + IO
# ----------------------------------------------------------------------------

_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def fl_schema(d_model: int, n_future: int, n_prev: int, topk: int = DEFAULT_TOPK) -> pa.Schema:
    return pa.schema([
        ("prompt", _PROMPT_STRUCT),
        ("response", pa.string()),
        ("activation_vector", pa.list_(pa.float16(), d_model)),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("target_ids", pa.list_(pa.int64(), n_future)),
        ("k", pa.int64()),
        ("doc_idx", pa.int64()),
        ("t", pa.int64()),
        ("p_top1", pa.float32()),
        ("target_top5", pa.list_(pa.int64(), 5 * n_future)),
        ("target_logp", pa.list_(pa.float32(), n_future)),
        ("prev_ids", pa.list_(pa.int64(), n_prev)),
        ("greedy_ids", pa.list_(pa.int64(), n_future)),
        ("greedy_top5", pa.list_(pa.int64(), 5 * n_future)),
        ("greedy_logp", pa.list_(pa.float32(), n_future)),
    ] + ([("greedy_topk_ids", pa.list_(pa.int32(), topk * n_future)),
          ("greedy_topk_logp", pa.list_(pa.float16(), topk * n_future))] if topk > 0 else []))


def docs_schema() -> pa.Schema:
    return pa.schema([
        ("doc_idx", pa.int64()),
        ("doc_id", pa.string()),
        ("split", pa.string()),
        ("ids", pa.list_(pa.int64())),
    ])


def _fixed_col(rg, name):
    col = rg.column(name).combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1)


ROW_COLUMNS = ["prompt", "response", "activation_vector", "activation_layer", "doc_id",
               "n_raw_tokens", "target_ids", "k", "doc_idx", "t", "p_top1", "target_top5",
               "target_logp", "prev_ids", "greedy_ids", "greedy_top5", "greedy_logp"]
LABELS = ("text", "greedy")
TOPK_COLUMNS = ["greedy_topk_ids", "greedy_topk_logp"]   # not in ROW_COLUMNS: request explicitly (SFT --distill, eval tf_kl)
_GREEDY_FOR = {"target_ids": "greedy_ids", "target_top5": "greedy_top5", "target_logp": "greedy_logp"}


def load_fl_rows(parquet_path: str | Path, n_max: int | None = None, *,
                 layers: list[int] | None = None, keep_activations: bool = True,
                 columns: list[str] | None = None, label: str = "text",
                 drop_label_ids: set[int] | None = None) -> list[dict]:
    """Row-group-streamed load. Activations stay float16 numpy (half the RAM of
    fp32; every consumer converts per batch). `layers` filters by activation_layer.
    `label="greedy"` swaps target_ids/top5/logp for the target's own greedy continuation
    (see module doc); the corpus ids are kept as `text_ids`. Under greedy labels, rows
    whose greedy first token disagrees with the corpus token (a bf16 near-tie flip between
    the teacher-forced filter pass and the batched generate) and rows whose greedy
    continuation contains one of `drop_label_ids` (EOS: unreachable for the decoder) are
    dropped and counted."""
    assert label in LABELS, f"label must be one of {LABELS}, got {label!r}"
    pf = pq.ParquetFile(str(parquet_path))
    cols = list(columns or ROW_COLUMNS)
    if not keep_activations and "activation_vector" in cols:
        cols.remove("activation_vector")
    if layers is not None and "activation_layer" not in cols:
        cols.append("activation_layer")
    avail = set(pf.schema_arrow.names)
    if label == "greedy":
        for tc, gc in _GREEDY_FOR.items():
            if tc in cols and gc not in cols:
                cols.append(gc)
        missing = [gc for tc, gc in _GREEDY_FOR.items() if tc in cols and gc not in avail]
        assert not missing, f"{parquet_path}: label=greedy needs columns {missing} (re-run collect with --greedy all)"
    cols = [c for c in cols if c in avail]
    rows: list[dict] = []
    dropped = {"greedy_first_token_mismatch": 0, "greedy_contains_stop_id": 0}
    nf_schema = pf.schema_arrow.field("target_ids").type.list_size if "target_ids" in avail else None
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n = rg.num_rows
        layer_col = rg.column("activation_layer").to_numpy() if "activation_layer" in cols else None
        keep = np.ones(n, dtype=bool)
        if layers is not None and layer_col is not None:
            keep &= np.isin(layer_col, np.asarray(layers))
        if n_max is not None:
            budget = n_max - len(rows)
            kept_idx = np.flatnonzero(keep)
            if len(kept_idx) > budget:
                keep[kept_idx[budget:]] = False
        fixed = {}
        for name, dt in (("activation_vector", np.float16), ("target_ids", np.int64),
                         ("target_top5", np.int64), ("target_logp", np.float32),
                         ("prev_ids", np.int64), ("greedy_ids", np.int64),
                         ("greedy_top5", np.int64), ("greedy_logp", np.float32),
                         ("greedy_topk_ids", np.int32), ("greedy_topk_logp", np.float16)):
            if name in cols:
                arr = _fixed_col(rg, name).astype(dt, copy=False)
                # Compact to the kept rows: per-row slices are VIEWS into the row-group
                # buffer, so without this a --layers filter still pins every layer's
                # activations in RAM (~10 GB on the 8B train split).
                fixed[name] = arr[keep] if not keep.all() else np.ascontiguousarray(arr)
        scalars = {c: rg.column(c).to_pylist() for c in cols if c not in fixed}
        j = 0
        for i in range(n):
            if not keep[i]:
                continue
            row = {c: scalars[c][i] for c in scalars}
            for name, arr in fixed.items():
                row[name] = arr[j]
            j += 1
            if label == "greedy" and "greedy_ids" in row:
                g = row["greedy_ids"]
                assert int(g[0]) >= 0, "greedy_ids absent for this row (collect --greedy all)"
                if "target_ids" in row:
                    if int(g[0]) != int(row["target_ids"][0]):
                        dropped["greedy_first_token_mismatch"] += 1
                        continue
                    row["text_ids"] = row["target_ids"]
                if drop_label_ids and any(int(x) in drop_label_ids for x in g):
                    dropped["greedy_contains_stop_id"] += 1
                    continue
                for tc, gc in _GREEDY_FOR.items():
                    if gc in row:
                        row[tc] = row.pop(gc)
            if "target_top5" in row:
                row["target_top5"] = row["target_top5"].reshape(-1, 5)
            if "greedy_topk_ids" in row:
                row["greedy_topk_ids"] = row["greedy_topk_ids"].reshape(nf_schema, -1)
                row["greedy_topk_logp"] = row["greedy_topk_logp"].reshape(nf_schema, -1)
            rows.append(row)
    if any(dropped.values()):
        print(f"[data] {parquet_path}: dropped rows under label=greedy: {dropped} (kept {len(rows)})", flush=True)
    return rows


def subsample_positions(rows: list[dict], n_rows: int, seed: int) -> list[dict]:
    """Seeded random subsample of POSITIONS (doc_idx, t), shared across layers, sized so that
    about `n_rows` rows survive. Parquet order is doc-major, so "first n rows" would be a
    handful of documents repeated at every layer."""
    if n_rows is None or len(rows) <= n_rows:
        return rows
    keys = sorted({(int(r["doc_idx"]), int(r["t"])) for r in rows})
    n_layers = max(1, len({int(r["activation_layer"]) for r in rows}))
    n_pos = max(1, int(np.ceil(n_rows / n_layers)))
    rng = np.random.default_rng(seed)
    pick = {keys[i] for i in rng.choice(len(keys), size=min(n_pos, len(keys)), replace=False)}
    return [r for r in rows if (int(r["doc_idx"]), int(r["t"])) in pick]


def load_docs(docs_parquet: str | Path) -> dict[int, np.ndarray]:
    """doc_idx -> int64 token-id array."""
    tbl = pq.read_table(str(docs_parquet), columns=["doc_idx", "ids"])
    out = {}
    for di, ids in zip(tbl.column("doc_idx").to_pylist(), tbl.column("ids").to_pylist()):
        out[int(di)] = np.asarray(ids, dtype=np.int64)
    return out


def resolve_docs_path(parquet_path: str | Path, fl: FLMeta) -> Path | None:
    """docs.parquet lives next to the split parquet (sidecar records its basename)."""
    if fl.docs_parquet is None:
        return None
    p = Path(fl.docs_parquet)
    if not p.is_absolute():
        p = Path(str(parquet_path).split("@[")[0]).parent / p
    return p if p.exists() else None


def shuffle_activations(rows: list[dict], seed: int = 0, within_layer: bool = True) -> None:
    """In place: give every row another row's activation (same layer by default).
    The shuffled-activation control: any remaining skill is context-free prior."""
    rng = np.random.default_rng(seed)
    if within_layer:
        groups: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(int(r["activation_layer"]), []).append(i)
    else:
        groups = {0: list(range(len(rows)))}
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        perm = rng.permutation(len(idxs))
        # one cycle over the permuted order: every row receives a DIFFERENT row's vector
        orig = [rows[i]["activation_vector"] for i in idxs]
        for j in range(len(idxs)):
            rows[idxs[perm[j]]]["activation_vector"] = orig[perm[(j + 1) % len(idxs)]]
