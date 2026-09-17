"""Shared evaluation machinery for the gate and the final evaluation.

Two phases, in this order for a reason:

  PHASE 1 (verbalizer resident, readers not yet loaded)
      Generate explanations for every checkpoint on the same positions.
  PHASE 2 (verbalizer freed, one reader resident at a time)
      Score every (checkpoint, condition) against the SAME stored continuations.

Doing it this way keeps peak memory at one 8B model or one 4B model rather than
all three at once, and - more importantly - guarantees that every condition is
scored against identical continuations and identical reader tokenizations, so the
differences are paired.

THE SHUFFLED CONDITION. To ask whether an explanation is about THIS activation,
we score checkpoint C's explanation for position j against position i's
continuation. Since the AV prompt is a fixed template whose only per-position
content is the injected vector, "the explanation generated from a mismatched
activation" and "another position's explanation" are the same object - so the
shuffled condition reuses the generated texts under a derangement instead of
generating again. That also removes sampling noise between the matched and
shuffled arms: the two conditions draw from one identical pool of explanation
texts, and differ only in which continuation each is paired with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from nla.pred.data import shuffled_partner
from nla.pred.reader import DEFAULT_BUCKETS, HEADLINE_BUCKET
from nla.pred.scoring import (
    SKIP,
    BaselineCache,
    baseline_scores,
    gain_per_token,
    score_explanations,
)
from nla.pred.stats import bootstrap_mean, paired_bootstrap_diff

CONDITIONS = ("matched", "shuffled")


@dataclass
class CheckpointSpec:
    name: str
    adapter: str | None = None
    subfolder: str | None = None

    @classmethod
    def parse(cls, spec: str) -> "CheckpointSpec":
        """NAME=ADAPTER[#SUBFOLDER]; an empty adapter means the bare base (SFT)."""
        assert "=" in spec, (
            f"--checkpoint {spec!r}: expected NAME=ADAPTER (adapter may be empty "
            f"for the un-adapted base, and may carry #subfolder)"
        )
        name, _, rest = spec.partition("=")
        rest = rest.strip()
        if not rest:
            return cls(name=name.strip())
        adapter, _, sub = rest.partition("#")
        return cls(name=name.strip(), adapter=adapter, subfolder=sub or None)


@dataclass
class ExplanationSet:
    """Generated explanations for one checkpoint, aligned with `rows`."""

    name: str
    texts: list          # str | None (None = extraction failed)
    raw: list            # raw model output, for the qualitative review
    n_tokens: list       # explanation length in AV tokens

    @property
    def extraction_rate(self) -> float:
        return float(np.mean([t is not None for t in self.texts])) if self.texts else 0.0


def generate_all(
    rows, specs, *, base_ckpt, sidecar, quant="none", device="cuda",
    max_new_tokens=192, temperature=1.0, batch_size=16, seed=0, dtype=torch.bfloat16,
) -> tuple[dict[str, ExplanationSet], object]:
    """Phase 1: explanations for every checkpoint, on the same positions."""
    from transformers import AutoTokenizer

    from nla.config import load_nla_config
    from nla.pred.av import generate_explanations, load_av

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    cfg = load_nla_config(sidecar, tokenizer)
    inj_ids = (cfg.injection_token_id, cfg.injection_left_neighbor_id,
               cfg.injection_right_neighbor_id)
    out: dict[str, ExplanationSet] = {}
    for spec in specs:
        torch.manual_seed(seed)   # same sampling noise budget for every checkpoint
        model, vectors_ref = load_av(
            base_ckpt, spec.adapter, adapter_subfolder=spec.subfolder,
            quant=quant, device=device, dtype=dtype, inj_ids=inj_ids,
        )
        expls, raws = generate_explanations(
            model, tokenizer, vectors_ref, rows, cfg.injection_char,
            max_new_tokens=max_new_tokens, temperature=temperature,
            batch_size=batch_size, device=device, return_raw=True,
        )
        n_tok = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) if t else 0
                 for t in expls]
        out[spec.name] = ExplanationSet(spec.name, expls, raws, n_tok)
        print(f"[gen] {spec.name}: extraction {out[spec.name].extraction_rate:.1%}, "
              f"median {int(np.median(n_tok))} tokens", flush=True)
        del model, vectors_ref
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out, cfg


def score_all(
    rows, expl_sets, reader_names, *, buckets=DEFAULT_BUCKETS, branches=(0, 1, 2, 3),
    device="cuda", dtype="bfloat16", templates=None, seed=0, conditions=CONDITIONS,
    max_batch_rows=32, max_batch_tokens=32768,
):
    """Phase 2: every (checkpoint, condition) under every reader.

    Returns records: list of dicts, one per (row, checkpoint, reader, condition).
    """
    from nla.pred.reader import FrozenReader

    rng = np.random.default_rng(seed)
    partner = shuffled_partner(rows, rng)
    records = []
    for reader_name in reader_names:
        reader = FrozenReader.load(
            reader_name, device=device, dtype=dtype, templates=templates,
            max_batch_rows=max_batch_rows, max_batch_tokens=max_batch_tokens,
        )
        cache = BaselineCache()
        base = baseline_scores(reader, rows, branches=branches, buckets=buckets,
                               cache=cache)
        for name, es in expl_sets.items():
            for cond in conditions:
                if cond == "matched":
                    texts = [t if t is not None else SKIP for t in es.texts]
                elif cond == "shuffled":
                    texts = [es.texts[partner[i]] if es.texts[partner[i]] is not None
                             else SKIP for i in range(len(rows))]
                else:
                    raise ValueError(f"unknown condition {cond!r}")
                sc = score_explanations(reader, rows, texts, branches=branches,
                                        buckets=buckets)
                gains = gain_per_token(sc, base, buckets)
                for i, row in enumerate(rows):
                    records.append({
                        "row_id": int(row["row_id"]), "doc_id": row["doc_id"],
                        "checkpoint": name, "reader": reader_name, "condition": cond,
                        "partner_row_id": int(rows[partner[i]]["row_id"]),
                        "gain": [float(g) for g in gains[i]],
                        "score": [float(s) for s in sc[i]],
                        "baseline": [float(b) for b in base[i]],
                        "scored": bool(np.isfinite(sc[i]).all()),
                    })
        del reader
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return records, partner


def _bucket_index(buckets, target=HEADLINE_BUCKET) -> int:
    for i, b in enumerate(buckets):
        if tuple(b) == tuple(target):
            return i
    return len(buckets) - 1


def summarize(records, expl_sets, *, buckets=DEFAULT_BUCKETS, n_boot=10000, seed=0):
    """Aggregate to the headline table: per (checkpoint, reader) gains + CIs, and
    the paired matched-minus-shuffled difference that says the explanation is
    about THIS activation."""
    hb = _bucket_index(buckets)
    by = {}
    for r in records:
        by.setdefault((r["checkpoint"], r["reader"], r["condition"]), []).append(r)
    checkpoints = sorted({r["checkpoint"] for r in records})
    readers = sorted({r["reader"] for r in records})
    summary = {
        "buckets": [list(b) for b in buckets],
        "headline_bucket": list(buckets[hb]),
        "n_positions": len({r["row_id"] for r in records}),
        "extraction_rate": {n: es.extraction_rate for n, es in expl_sets.items()},
        "explanation_tokens_median": {
            n: float(np.median([t for t in es.n_tokens if t > 0]) if any(es.n_tokens) else 0)
            for n, es in expl_sets.items()
        },
        "cells": [],
    }
    for ck in checkpoints:
        for rd in readers:
            m = by.get((ck, rd, "matched"), [])
            s = by.get((ck, rd, "shuffled"), [])
            if not m:
                continue
            m = sorted(m, key=lambda r: r["row_id"])
            s = sorted(s, key=lambda r: r["row_id"])
            docs = [r["doc_id"] for r in m]
            cell = {"checkpoint": ck, "reader": rd, "n": len(m), "per_bucket": []}
            for bi in range(len(buckets)):
                gm = [r["gain"][bi] for r in m]
                mean, lo, hi = bootstrap_mean(gm, clusters=docs, n_boot=n_boot, seed=seed)
                entry = {"bucket": list(buckets[bi]), "gain": mean,
                         "gain_lo": lo, "gain_hi": hi}
                if s:
                    gs = [r["gain"][bi] for r in s]
                    d, dlo, dhi, p, npair = paired_bootstrap_diff(
                        gm, gs, clusters=docs, n_boot=n_boot, seed=seed)
                    entry.update({
                        "shuffled_gain": float(np.nanmean(gs)),
                        "matched_minus_shuffled": d, "mms_lo": dlo, "mms_hi": dhi,
                        "mms_p": p, "mms_n": npair,
                    })
                cell["per_bucket"].append(entry)
            cell["headline"] = cell["per_bucket"][hb]
            summary["cells"].append(cell)
    return summary


def compare_checkpoints(records, a: str, b: str, *, buckets=DEFAULT_BUCKETS,
                        condition="matched", n_boot=10000, seed=0):
    """Paired (checkpoint a - checkpoint b) gain per reader, headline bucket."""
    hb = _bucket_index(buckets)
    out = {}
    readers = sorted({r["reader"] for r in records})
    for rd in readers:
        ga = {r["row_id"]: r["gain"][hb] for r in records
              if r["checkpoint"] == a and r["reader"] == rd and r["condition"] == condition}
        gb = {r["row_id"]: r["gain"][hb] for r in records
              if r["checkpoint"] == b and r["reader"] == rd and r["condition"] == condition}
        docs = {r["row_id"]: r["doc_id"] for r in records}
        keys = sorted(set(ga) & set(gb))
        d, lo, hi, p, n = paired_bootstrap_diff(
            [ga[k] for k in keys], [gb[k] for k in keys],
            clusters=[docs[k] for k in keys], n_boot=n_boot, seed=seed)
        out[rd] = {"diff": d, "lo": lo, "hi": hi, "p": p, "n": n}
    return out


def write_outputs(out_dir, rows, expl_sets, records, summary, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "explanations.jsonl", "w") as f:
        for name, es in expl_sets.items():
            for i, row in enumerate(rows):
                f.write(json.dumps({
                    "row_id": int(row["row_id"]), "doc_id": row["doc_id"],
                    "checkpoint": name, "extracted": es.texts[i] is not None,
                    "n_tokens": int(es.n_tokens[i]),
                    "explanation": es.texts[i],
                    "raw": es.raw[i][:2000] if es.texts[i] is None else None,
                    "prefix_tail": (row.get("prefix_text") or "")[-400:],
                    "continuation_0": row["cont_text"][0],
                }) + "\n")
    with open(out / "scores.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    payload = dict(summary)
    if extra:
        payload.update(extra)
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    return out


def format_table(summary) -> str:
    """The headline table as markdown (nats per target token, 95% CI)."""
    hbl = summary["headline_bucket"]
    readers = sorted({c["reader"] for c in summary["cells"]})
    lines = [
        f"Predictive gain, nats per target token, bucket {hbl[0]}-{hbl[1]} "
        f"(16-token bridge), n={summary['n_positions']} positions",
        "",
        "| checkpoint | reader | gain [95% CI] | shuffled | matched - shuffled [95% CI] | p |",
        "|---|---|---|---|---|---|",
    ]
    for cell in summary["cells"]:
        h = cell["headline"]
        mms = h.get("matched_minus_shuffled")
        lines.append(
            f"| {cell['checkpoint']} | {cell['reader'].split('/')[-1]} "
            f"| {h['gain']:+.4f} [{h['gain_lo']:+.4f}, {h['gain_hi']:+.4f}] "
            f"| {h.get('shuffled_gain', float('nan')):+.4f} "
            + (f"| {mms:+.4f} [{h['mms_lo']:+.4f}, {h['mms_hi']:+.4f}] | {h['mms_p']:.3f} |"
               if mms is not None else "| n/a | n/a |")
        )
    del readers
    return "\n".join(lines)


__all__ = [
    "CONDITIONS", "CheckpointSpec", "ExplanationSet", "compare_checkpoints",
    "format_table", "generate_all", "score_all", "summarize", "write_outputs",
]
