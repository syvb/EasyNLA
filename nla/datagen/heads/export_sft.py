"""Head stage `export`: descriptions → AV/AR SFT parquets in train_sft.py's format (+ eval set).

The recipe's warm-start targets, per head h with o_proj slice W_h:

    v = cap(W_h (z − z̄) / dev_rms_h)          the normalized write ("subtract the average
                                               write, scale by its typical size, cap outliers";
                                               the cap is the smaller of the head's
                                               q`cap_quantile` deviation and --max-norm)

  av_sft.parquet   prompt   the fixed actor template, marker between <concept> tags (the same
                            for every row: the write is the only input that varies)
                   response <explanation>\\n{description}\\n</explanation>
                   activation_vector  v            (train with --inject-preserve-magnitude:
                                                   norm ~1 typical, ~0 for an idle head)
  ar_sft.parquet   prompt   critic template around "Attention head L12H7: {description}"
                   activation_vector  √d · v        (train with --ar-raw-loss: the critic
                                                   normalizes its backbone output to √d before
                                                   the identity-init value head, so √d-scaled
                                                   targets start it at the right magnitude)
  eval.parquet     the eval split, AV-format prompt + v + the description, for the pre-RL checks

Descriptions come from warmstart/descriptions.parquet (rows with ok=True; rule-based "little
or nothing" rows included), or any parquet with example_idx + description columns
(--descriptions, e.g. a self_describe output). The actor prompt names no head, so the
verbalizer sees only the write (the critic's text still names the head).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import load_tokenizer
from nla.datagen.heads.self_describe import head_write_deviations
from nla.datagen.heads.store import atomic_write_table, load_stats, read_meta
from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, serialize_sidecar
from nla.schema import INJECT_PLACEHOLDER, sidecar_path_for, wrap_explanation

ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the information that one attention head added to one token.

We will pass the head's contribution, as a vector enclosed in <concept> tags, into your context. You must then describe it in one or two sentences, enclosed within <explanation> tags.

<concept>{injection_char}</concept>

Please provide an explanation."""
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def actor_content() -> str:
    return ACTOR_TEMPLATE.replace("{injection_char}", INJECT_PLACEHOLDER)


def critic_explanation(layer: int, head: int, description: str) -> str:
    return f"Attention head {head_name(layer, head)}: {description}"


def normalized_writes(out_dir: Path, rows: list[dict], cap_quantile: float, max_norm: float) -> np.ndarray:
    """v for each row (see module docstring) → [N, d_model] float32."""
    stats = load_stats(out_dir)
    q_levels = stats.quantile_levels.tolist()
    assert any(abs(q - cap_quantile) < 1e-6 for q in q_levels), (   # levels are stored as float32
        f"--cap-quantile {cap_quantile} not among the stats quantiles {q_levels} — rerun stats with it"
    )
    qi = min(range(len(q_levels)), key=lambda i: abs(q_levels[i] - cap_quantile))
    w = head_write_deviations(out_dir, rows).numpy()
    out = np.empty_like(w)
    for i, r in enumerate(rows):
        li = stats.layer_pos(int(r["layer"]))
        rms = float(stats.dev_rms[li, int(r["head"])])
        # spiky heads' q0.999 can sit far above 4×rms; the hard cap keeps the AV injection
        # (‖h‖·‖v‖) and the AR target norm in one bounded range
        cap = min(float(stats.dev_quantiles[li, int(r["head"]), qi]) / rms, max_norm)
        v = w[i] / rms
        n = float(np.linalg.norm(v))
        out[i] = v * min(1.0, cap / n) if n > 0 else v
    return out.astype(np.float32)


def _sidecar_meta(meta, stage: str, n_rows: int, tokens, layer_index: int) -> NLADatasetMeta:
    return NLADatasetMeta(
        dataset_id=f"{stage}_heads_{meta.base_model.split('/')[-1]}",
        stage=stage,
        row_count=n_rows,
        extraction=NLAExtractionMeta(
            base_model=meta.base_model, d_model=meta.d_model,
            # The AR's depth contract (train_sft asserts --ar-num-layers == layer_index + 1).
            # Head writes from several layers share one critic; it reads text, not a layer.
            layer_index=layer_index, norm="none", corpus=meta.corpus,
            corpus_slice=meta.corpus_slice, positions_per_doc=meta.positions_per_doc,
        ),
        tokens=tokens,
        prompt_templates={"actor": ACTOR_TEMPLATE, "critic": CRITIC_TEMPLATE},
        created_by="nla.datagen.heads.export_sft",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="head dataset")
    p.add_argument("--descriptions", default=None, help="default warmstart/descriptions.parquet")
    p.add_argument("--description-column", default="description",
                   help="column of --descriptions holding the text (heads_soft_token_explain.py: explain)")
    p.add_argument("--sft-dir", required=True, help="where av_sft / ar_sft / eval parquets go")
    p.add_argument("--cap-quantile", type=float, default=0.999)
    p.add_argument("--max-norm", type=float, default=4.0, help="hard cap on ‖v‖ (units of the head's RMS)")
    p.add_argument("--seed", type=int, default=0, help="row-order shuffle")
    args = p.parse_args()

    out_dir, sft_dir = Path(args.output_dir), Path(args.sft_dir)
    meta = read_meta(out_dir)
    sel = {r["example_idx"]: r for r in pq.read_table(out_dir / "warmstart" / "selection.parquet").to_pylist()}
    desc_tab = pq.read_table(args.descriptions or out_dir / "warmstart" / "descriptions.parquet")
    keep = desc_tab.column("ok").to_pylist() if "ok" in desc_tab.column_names else [True] * desc_tab.num_rows
    rows = []
    for ex, d, ok in zip(desc_tab.column("example_idx").to_pylist(), desc_tab.column(args.description_column).to_pylist(),
                         keep, strict=True):
        if ok and d:
            rows.append({**sel[ex], "description": d})
    print(f"{len(rows)}/{desc_tab.num_rows} described examples usable")

    tok = load_tokenizer(meta.base_model)
    tokens = build_token_meta(tok, ACTOR_TEMPLATE, critic_template=CRITIC_TEMPLATE)

    # Shuffled row order: train_sft takes its logged FVE baseline and --heldout-rows from the
    # FIRST rows, which in selection order would all be the first heads of the first layer.
    rows = [rows[i] for i in np.random.default_rng(args.seed).permutation(len(rows))]
    v = normalized_writes(out_dir, rows, args.cap_quantile, args.max_norm)
    root_d = math.sqrt(meta.d_model)
    layer_index = max(meta.layers)
    by_split: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_split.setdefault(r["split"], []).append(i)
    common = {
        "doc_id": lambda ix: pa.array([rows[i]["doc_id"] for i in ix], type=pa.string()),
        "n_raw_tokens": lambda ix: pa.array([rows[i]["pos"] + 1 for i in ix], type=pa.int64()),
        "activation_layer": lambda ix: pa.array([rows[i]["layer"] for i in ix], type=pa.int64()),
        "head": lambda ix: pa.array([rows[i]["head"] for i in ix], type=pa.int16()),
        "example_idx": lambda ix: pa.array([rows[i]["example_idx"] for i in ix], type=pa.int64()),
    }

    def vec(ix, scale=1.0):
        return pa.FixedSizeListArray.from_arrays(pa.array((v[ix] * scale).reshape(-1)), meta.d_model)

    def av_cols(ix):
        return {
            "prompt": pa.array([[{"role": "user", "content": actor_content()}]
                                for i in ix], type=_PROMPT_STRUCT),
            "response": pa.array([wrap_explanation(rows[i]["description"]) for i in ix], type=pa.string()),
            "activation_vector": vec(ix),
        }

    written = {}
    for stage, split in (("av_sft", "av_sft"), ("ar_sft", "ar_sft"), ("eval", "eval")):
        ix = by_split.get(split, [])
        if not ix:
            print(f"  {stage}: no rows (split {split!r} absent)")
            continue
        if stage == "ar_sft":
            cols = {
                "prompt": pa.array([CRITIC_TEMPLATE.format(explanation=critic_explanation(
                    rows[i]["layer"], rows[i]["head"], rows[i]["description"])) for i in ix], type=pa.string()),
                "activation_vector": vec(ix, root_d),
            }
        else:
            cols = av_cols(ix)
            if stage == "eval":
                cols["description"] = pa.array([rows[i]["description"] for i in ix], type=pa.string())
                cols["pos"] = pa.array([rows[i]["pos"] for i in ix], type=pa.int32())
                cols["dev_rel"] = pa.array([rows[i]["dev_rel"] for i in ix], type=pa.float32())
        cols.update({k: f(ix) for k, f in common.items()})
        path = sft_dir / f"{stage}.parquet"
        atomic_write_table(pa.table(cols), path)
        # eval reuses the AV contract (same prompts/markers); training stages get their own
        side_stage = "av_sft" if stage == "eval" else stage
        sidecar_path_for(path).write_text(serialize_sidecar(_sidecar_meta(meta, side_stage, len(ix), tokens,
                                                                          layer_index)))
        written[stage] = len(ix)
    norms = np.linalg.norm(v, axis=1)
    print(f"wrote {written} → {sft_dir}")
    print(f"  ‖v‖: median {np.median(norms):.2f}  p90 {np.quantile(norms, 0.9):.2f}  max {norms.max():.2f} "
          f"(AR targets are √d = {root_d:.0f} × v); AR critic depth: --ar-num-layers {layer_index + 1}")


if __name__ == "__main__":
    main()
