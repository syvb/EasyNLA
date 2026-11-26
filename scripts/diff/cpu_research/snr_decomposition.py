"""Why the cleaned diff has lower Fisher SNR than h_target at `asst_last`.

Fisher SNR is between-class separation over within-class spread. For each audit
task and Base wrapper this splits the diff's SNR into those two parts, relative
to h_target:

  * between_base / between_target: how much of the class signal the aligned Base
    model already carries (and its direction, as a cosine),
  * between_delta / between_target: what is left of the signal after subtraction,
  * within_delta / within_target: whether subtracting Base removes prompt-to-prompt
    variation (< 1) or adds Base's own (> 1).

    python snr_decomposition.py --align stock=P0 --align robust=P0_robust --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.alignment import load_alignment, predict_aligned_base
from nla.diff.benchmarks import (
    flatten_binary_pairs,
    load_eval_awareness_pairs,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.wrappers import encode_aligned_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--target-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer-index", type=int, default=24)
    ap.add_argument("--align", action="append", required=True, help="name=phase0_dir (repeatable)")
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    k = args.layer_index
    tok = AutoTokenizer.from_pretrained(args.target_model)
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    tgt, base = cached_io.LazyLM(args.target_model), cached_io.LazyLM(args.base_model)
    arts = {}
    for spec in args.align:
        name, d = spec.split("=", 1)
        arts[name] = load_alignment(Path(d) / f"align_L{k}_id_anchored.pt")

    out: dict = {}
    for name, pairs in (("refusal", load_refusal_pairs(args.n_pairs, source="auto")),
                        ("eval_awareness", load_eval_awareness_pairs(args.n_pairs)),
                        ("sycophancy", load_sycophancy_pairs(args.n_pairs))):
        prompts, labels, _ = flatten_binary_pairs(pairs)
        y = np.asarray(labels)
        for w in ("chatml", "plain"):
            encs = [encode_aligned_pair(tok, t, base_wrapper=w, enable_thinking=False, max_total_tokens=512) for t in prompts]
            ht = torch.cat([e.hidden for e in cached_io.cached_extract(
                tgt, [e[0].input_ids for e in encs], [[e[0].asst_last_idx] for e in encs],
                layer_index=k, pad_token_id=pad)]).double()
            hb = torch.cat([e.hidden for e in cached_io.cached_extract(
                base, [e[1].input_ids for e in encs], [[e[1].asst_last_idx] for e in encs],
                layer_index=k, pad_token_id=pad)]).double()

            def parts(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
                m1, m0 = x[y == 1].mean(0), x[y == 0].mean(0)
                within = 0.5 * ((x[y == 1] - m1).pow(2).sum(1).mean() + (x[y == 0] - m0).pow(2).sum(1).mean())
                return m1 - m0, float((m1 - m0).pow(2).sum()), float(within)

            dt, bt, wt = parts(ht)
            row: dict = {"snr_h_target": bt / wt}
            for an, art in arts.items():
                pb = predict_aligned_base(hb.float(), art).double()
                db, bb, wb = parts(pb)
                dd, bd, wd = parts(ht - pb)
                row[an] = {
                    "between_base_over_target": bb / bt,
                    "cos_class_direction_base_vs_target": float(torch.nn.functional.cosine_similarity(dt, db, dim=0)),
                    "between_delta_over_target": bd / bt,
                    "within_base_over_target": wb / wt,
                    "within_delta_over_target": wd / wt,
                    "snr_delta_over_snr_target": (bd / wd) / (bt / wt),
                }
            out[f"{name}/{w}"] = row
            print(name, w, json.dumps({kk: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()})
                                       for kk, v in row.items()}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
