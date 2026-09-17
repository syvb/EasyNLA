"""Shared evaluation machinery for the gate and the final evaluation.

Two phases, in this order for a reason:

  PHASE 1 (verbalizer resident, readers not yet loaded)
      Generate explanations for every checkpoint on the same positions.
  PHASE 2 (verbalizer freed, one reader resident at a time)
      Score every (checkpoint, condition, context setting) against the SAME
      stored continuations.

Doing it this way keeps peak memory at one 8B model or one 4B model rather than
all three at once, and - more importantly - guarantees that every condition is
scored against identical continuations and identical reader tokenizations, so
the differences are paired.

CONDITIONS
  matched    the position's own explanation
  shuffled   another DOCUMENT's explanation, under a derangement. Easy to beat:
             a wrong-document explanation actively misleads the reader, so this
             gap rewards specificity of any kind, including naming the topic.
  same_doc   the explanation of the OTHER position from the same document.
             Holds topic, genre and register constant and isolates what is
             specific to THIS position - the informative control.

CONTEXT SETTINGS (context_words)
  0          the reader sees no document text (the plan's setting). Its
             baseline is a base LM predicting web text from nothing, so ANY
             information about the document is rewarded, and a verbalizer that
             quotes what the model was reading scores well, transfers across
             readers, and beats shuffled - without saying anything about what
             the model was about to do.
  N > 0      both prompts also show the last N words of the document. The
             explanation must then add something the text does not already
             say. This is the number that speaks to "predicts what the model
             does next"; the context-free number speaks to "reconstructs what
             the model was reading".

REFERENCE "CHECKPOINT"
  tail_quote  a synthetic explanation that is just the last K words of the
              document, scored like any checkpoint. A real checkpoint that does
              not clearly beat it in the context-free setting is doing context
              recovery, not explanation.

THE SHUFFLED CONDITION reuses generated texts under a permutation rather than
generating from mismatched activations: the AV prompt is a fixed template whose
only per-position content is the injected vector, so the two are the same
object, and reusing texts removes sampling noise between the arms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nla.pred.data import same_doc_partner, shuffled_partner
from nla.pred.reader import DEFAULT_BUCKETS, HEADLINE_BUCKET
from nla.pred.scoring import (
    SKIP,
    BaselineCache,
    baseline_scores,
    gain_per_token,
    score_explanations,
)
from nla.pred.stats import bootstrap_mean, paired_bootstrap_diff

CONDITIONS = ("matched", "shuffled", "same_doc")
TAIL_QUOTE = "tail_quote"


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
    kind: str = "checkpoint"   # or "reference" (synthetic, e.g. tail_quote)

    @property
    def extraction_rate(self) -> float:
        return float(np.mean([t is not None for t in self.texts])) if self.texts else 0.0


def tail_quote_set(rows, tokenizer, n_words: int) -> ExplanationSet:
    """The last n_words of each document prefix, presented as an explanation."""
    texts = [" ".join((r.get("prefix_text") or "").split()[-n_words:]) or None for r in rows]
    n_tok = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) if t else 0
             for t in texts]
    return ExplanationSet(TAIL_QUOTE, texts, [t or "" for t in texts], n_tok,
                          kind="reference")


def generate_all(
    rows, specs, *, base_ckpt, sidecar, quant="none", device="cuda",
    max_new_tokens=192, temperature=1.0, batch_size=16, seed=0, dtype=torch.bfloat16,
    tail_quote_words=40,
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
    if tail_quote_words > 0:
        out[TAIL_QUOTE] = tail_quote_set(rows, tokenizer, tail_quote_words)
        print(f"[gen] {TAIL_QUOTE}: last {tail_quote_words} words of the prefix, "
              f"median {int(np.median(out[TAIL_QUOTE].n_tokens))} tokens", flush=True)
    return out, cfg


def _condition_texts(cond, es, partner, partner_ok, sd_partner, sd_ok, n):
    if cond == "matched":
        return [t if t is not None else SKIP for t in es.texts]
    if cond == "shuffled":
        return [es.texts[partner[i]] if (partner_ok[i] and es.texts[partner[i]] is not None)
                else SKIP for i in range(n)]
    if cond == "same_doc":
        return [es.texts[sd_partner[i]]
                if (sd_ok[i] and es.texts[sd_partner[i]] is not None) else SKIP
                for i in range(n)]
    raise ValueError(f"unknown condition {cond!r}")


def score_all(
    rows, expl_sets, reader_names, *, buckets=DEFAULT_BUCKETS, branches=(0, 1, 2, 3),
    device="cuda", dtype="bfloat16", templates=None, seed=0, conditions=CONDITIONS,
    context_words_list=(0,), reliability=False, max_batch_rows=32,
    max_batch_tokens=32768,
):
    """Phase 2: every (checkpoint, condition, context setting) under every reader.

    reliability=True also scores each condition on two disjoint halves of the
    branches, so `summarize` can report the split-half correlation of the
    per-position gain: if the score of one explanation does not agree with
    itself across futures, no optimizer can see through the noise.

    Returns (records, partner, reader_diag).
    """
    from nla.pred.reader import FrozenReader

    rng = np.random.default_rng(seed)
    partner, partner_ok = shuffled_partner(rows, rng)
    sd_partner, sd_ok = same_doc_partner(rows)
    n = len(rows)
    halves = None
    if reliability and len(branches) >= 2:
        h = len(branches) // 2
        halves = (tuple(branches[:h]), tuple(branches[h:]))
    records = []
    reader_diag: dict = {}
    for reader_name in reader_names:
        reader = FrozenReader.load(
            reader_name, device=device, dtype=dtype, templates=templates,
            max_batch_rows=max_batch_rows, max_batch_tokens=max_batch_tokens,
        )
        cache = BaselineCache()
        for ctx in context_words_list:
            base = baseline_scores(reader, rows, branches=branches, buckets=buckets,
                                   cache=cache, context_words=ctx)
            base_h = (None if halves is None else
                      [baseline_scores(reader, rows, branches=hb, buckets=buckets,
                                       cache=cache, context_words=ctx) for hb in halves])
            for name, es in expl_sets.items():
                for cond in conditions:
                    texts = _condition_texts(cond, es, partner, partner_ok,
                                             sd_partner, sd_ok, n)
                    sc = score_explanations(reader, rows, texts, branches=branches,
                                            buckets=buckets, context_words=ctx)
                    gains = gain_per_token(sc, base, buckets)
                    gh = None
                    if halves is not None:
                        gh = [gain_per_token(
                            score_explanations(reader, rows, texts, branches=hb,
                                               buckets=buckets, context_words=ctx),
                            base_h[k], buckets) for k, hb in enumerate(halves)]
                    for i, row in enumerate(rows):
                        rec = {
                            "row_id": int(row["row_id"]), "doc_id": row["doc_id"],
                            "checkpoint": name, "reader": reader_name,
                            "condition": cond, "context_words": int(ctx),
                            "partner_row_id": int(rows[partner[i]]["row_id"]),
                            "gain": [float(g) for g in gains[i]],
                            "score": [float(s) for s in sc[i]],
                            "baseline": [float(b) for b in base[i]],
                            "scored": bool(np.isfinite(sc[i]).all()),
                        }
                        if gh is not None:
                            rec["gain_h1"] = [float(g) for g in gh[0][i]]
                            rec["gain_h2"] = [float(g) for g in gh[1][i]]
                        records.append(rec)
        if reader.n_nonfinite or reader.n_empty_buckets:
            print(f"[reader] {reader_name}: {reader.n_nonfinite} non-finite "
                  f"log-probs and {reader.n_empty_buckets} empty buckets were "
                  f"scored as MISSING (not as zero)", flush=True)
        reader_diag[reader_name] = {"n_nonfinite": reader.n_nonfinite,
                                    "n_empty_buckets": reader.n_empty_buckets}
        del reader
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return records, partner, reader_diag


def _bucket_index(buckets, target=HEADLINE_BUCKET) -> int:
    for i, b in enumerate(buckets):
        if tuple(b) == tuple(target):
            return i
    # Silently falling back to the last bucket would report a different span than
    # the one RL optimized, under the headline's name.
    raise AssertionError(
        f"headline bucket {tuple(target)} is not among {[tuple(b) for b in buckets]}")


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _paired_block(gm, gs, docs, n_boot, seed):
    """Everything about matched-vs-<control> on the paired subset."""
    d, dlo, dhi, p, npair = paired_bootstrap_diff(gm, gs, clusters=docs,
                                                  n_boot=n_boot, seed=seed)
    pair_ok = [np.isfinite(a) and np.isfinite(b) for a, b in zip(gm, gs)]
    gs_p = [b for b, k in zip(gs, pair_ok) if k]
    return {"diff": d, "lo": dlo, "hi": dhi, "p": p, "n": npair,
            "control_gain": float(np.mean(gs_p)) if gs_p else float("nan")}


def summarize(records, expl_sets, *, buckets=DEFAULT_BUCKETS, n_boot=10000, seed=0):
    """Aggregate to the headline tables: per (checkpoint, reader, context) gains
    with CIs, the paired matched-minus-shuffled and matched-minus-same_doc
    differences, and split-half reliability where it was measured.

    Every number in a cell's row is computed on ONE row set - the positions
    where both matched and shuffled scored - so the printed columns subtract to
    the printed difference. `gain_all` keeps the all-matched-rows figure.
    """
    hb = _bucket_index(buckets)
    by: dict = {}
    for r in records:
        by.setdefault((r["checkpoint"], r["reader"], r["context_words"], r["condition"]),
                      []).append(r)
    checkpoints = sorted({r["checkpoint"] for r in records})
    readers = sorted({r["reader"] for r in records})
    ctxs = sorted({r["context_words"] for r in records})
    summary = {
        "buckets": [list(b) for b in buckets],
        "headline_bucket": list(buckets[hb]),
        "n_positions": len({r["row_id"] for r in records}),
        "context_words_list": ctxs,
        "conditions": sorted({r["condition"] for r in records}),
        "extraction_rate": {n: es.extraction_rate for n, es in expl_sets.items()},
        "explanation_tokens_median": {
            n: float(np.median([t for t in es.n_tokens if t > 0]) if any(es.n_tokens) else 0)
            for n, es in expl_sets.items()
        },
        "reference_sets": [n for n, es in expl_sets.items() if es.kind == "reference"],
        "cells": [],
    }
    for ctx in ctxs:
        for ck in checkpoints:
            for rd in readers:
                m = sorted(by.get((ck, rd, ctx, "matched"), []), key=lambda r: r["row_id"])
                if not m:
                    continue
                s = sorted(by.get((ck, rd, ctx, "shuffled"), []), key=lambda r: r["row_id"])
                sd = sorted(by.get((ck, rd, ctx, "same_doc"), []), key=lambda r: r["row_id"])
                docs = [r["doc_id"] for r in m]
                cell = {"checkpoint": ck, "reader": rd, "context_words": ctx,
                        "n": len(m), "per_bucket": []}
                for bi in range(len(buckets)):
                    gm = [r["gain"][bi] for r in m]
                    mean, lo, hi = bootstrap_mean(gm, clusters=docs, n_boot=n_boot, seed=seed)
                    entry = {"bucket": list(buckets[bi]), "gain_all": mean,
                             "gain_all_lo": lo, "gain_all_hi": hi,
                             "gain": mean, "gain_lo": lo, "gain_hi": hi,
                             "n_paired": len([g for g in gm if np.isfinite(g)])}
                    if s:
                        gs = [r["gain"][bi] for r in s]
                        blk = _paired_block(gm, gs, docs, n_boot, seed)
                        pair_ok = [np.isfinite(a) and np.isfinite(b) for a, b in zip(gm, gs)]
                        gm_p = [a for a, k in zip(gm, pair_ok) if k]
                        docs_p = [c for c, k in zip(docs, pair_ok) if k]
                        pm, plo, phi = bootstrap_mean(gm_p, clusters=docs_p,
                                                      n_boot=n_boot, seed=seed)
                        entry.update({
                            "gain": pm, "gain_lo": plo, "gain_hi": phi,
                            "n_paired": len(gm_p),
                            "shuffled_gain": blk["control_gain"],
                            "matched_minus_shuffled": blk["diff"], "mms_lo": blk["lo"],
                            "mms_hi": blk["hi"], "mms_p": blk["p"], "mms_n": blk["n"],
                        })
                    if sd:
                        gd = [r["gain"][bi] for r in sd]
                        blk = _paired_block(gm, gd, docs, n_boot, seed)
                        entry.update({
                            "same_doc_gain": blk["control_gain"],
                            "matched_minus_same_doc": blk["diff"], "msd_lo": blk["lo"],
                            "msd_hi": blk["hi"], "msd_p": blk["p"], "msd_n": blk["n"],
                        })
                    if m and "gain_h1" in m[0]:
                        h1 = [r["gain_h1"][bi] for r in m]
                        h2 = [r["gain_h2"][bi] for r in m]
                        entry["reliability_gain"] = _corr(h1, h2)
                        if s and "gain_h1" in s[0]:
                            d1 = [a - b for a, b in zip(h1, [r["gain_h1"][bi] for r in s])]
                            d2 = [a - b for a, b in zip(h2, [r["gain_h2"][bi] for r in s])]
                            entry["reliability_mms"] = _corr(d1, d2)
                    cell["per_bucket"].append(entry)
                cell["headline"] = cell["per_bucket"][hb]
                summary["cells"].append(cell)
    return summary


def compare_checkpoints(records, a: str, b: str, *, buckets=DEFAULT_BUCKETS,
                        condition="matched", context_words=0, n_boot=10000, seed=0):
    """Paired (checkpoint a - checkpoint b) gain per reader, headline bucket."""
    hb = _bucket_index(buckets)
    out = {}
    readers = sorted({r["reader"] for r in records})
    docs = {r["row_id"]: r["doc_id"] for r in records}

    def sel(ck, rd):
        return {r["row_id"]: r["gain"][hb] for r in records
                if r["checkpoint"] == ck and r["reader"] == rd
                and r["condition"] == condition and r["context_words"] == context_words}

    for rd in readers:
        ga, gb = sel(a, rd), sel(b, rd)
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
                    "checkpoint": name, "kind": es.kind,
                    "extracted": es.texts[i] is not None,
                    "n_tokens": int(es.n_tokens[i]),
                    "explanation": es.texts[i],
                    "raw": es.raw[i][:2000] if es.texts[i] is None else None,
                    # Long enough for the verbatim-overlap diagnostic in report.py.
                    "prefix_tail": (row.get("prefix_text") or "")[-1500:],
                    "continuations": list(row["cont_text"]),
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


def _ci(d, lo, hi):
    return "n/a" if d is None else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _pv(p):
    return "n/a" if p is None else f"{p:.3f}"


def format_table(summary) -> str:
    """The headline tables as markdown (nats per target token, 95% CI), one per
    context setting."""
    hbl = summary["headline_bucket"]
    lines = []
    for ctx in summary.get("context_words_list", [0]):
        lines += [
            f"Predictive gain, nats per target token, bucket {hbl[0]}-{hbl[1]} "
            f"({hbl[0]}-token bridge), "
            + ("reader sees NO document text" if ctx == 0
               else f"reader sees the last {ctx} words of the document")
            + f", n={summary['n_positions']} positions",
            "",
            "| checkpoint | reader | gain [95% CI] | shuffled | matched - shuffled [95% CI] "
            "| p | same_doc | matched - same_doc [95% CI] | p | split-half r |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for cell in summary["cells"]:
            if cell["context_words"] != ctx:
                continue
            h = cell["headline"]
            rel = h.get("reliability_gain")
            lines.append(
                f"| {cell['checkpoint']} | {cell['reader'].split('/')[-1]} "
                f"| {_ci(h['gain'], h['gain_lo'], h['gain_hi'])} "
                f"| {h.get('shuffled_gain', float('nan')):+.4f} "
                f"| {_ci(h.get('matched_minus_shuffled'), h.get('mms_lo'), h.get('mms_hi'))} "
                f"| {_pv(h.get('mms_p'))} "
                f"| {h.get('same_doc_gain', float('nan')):+.4f} "
                f"| {_ci(h.get('matched_minus_same_doc'), h.get('msd_lo'), h.get('msd_hi'))} "
                f"| {_pv(h.get('msd_p'))} "
                f"| {'n/a' if rel is None else f'{rel:.2f}'} |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "CONDITIONS", "TAIL_QUOTE", "CheckpointSpec", "ExplanationSet",
    "compare_checkpoints", "format_table", "generate_all", "score_all", "summarize",
    "tail_quote_set", "write_outputs",
]
