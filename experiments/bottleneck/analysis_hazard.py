"""Hazard-rate analysis of error-kind onsets in existing C1 transcripts.

For each error kind, the per-step hazard = P(first onset at step t | at risk at
t, still generating). Flat hazard => constant per-step risk (long tasks suffer
by exposure); rising hazard => genuine compounding. Clean condition = detector
false-positive baseline. Plus: (a) does loop onset follow low-cosine windows?
(b) do operand errors precede loops?
"""
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/debian/.claude/jobs/3fcbf94e/tmp")
BASE = ROOT / "hfraw" / "initial_run" / "stage_b"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def load(task, cond):
    return pq.read_table(BASE / task / f"{cond}_seed0.parquet").to_pylist()


def char_to_tok(ids, char_pos):
    """Map a char offset in decode(ids) to a token index (binary search)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=False)) < char_pos:
            lo = mid + 1
        else:
            hi = mid
    return max(0, lo - 1)


# ------------------------------ detectors --------------------------------
def loop_onset(ids):
    """First token index whose incoming trigram was already seen."""
    seen = set()
    for i in range(2, len(ids)):
        tri = (ids[i - 2], ids[i - 1], ids[i])
        if tri in seen:
            return i
        seen.add(tri)
    return None


REGISTER_RE = re.compile(r"</think>|\bOkay,")

EQ_RE = re.compile(
    r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*([+\-*x×/÷])\s*(\d[\d,]*(?:\.\d+)?)"
    r"\s*=\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])")
PCT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%\s*of\s*(\d[\d,]*(?:\.\d+)?)"
                    r"\s*(?:=|is)\s*(\d[\d,]*(?:\.\d+)?)")


def _num(s):
    return float(s.replace(",", ""))


def first_bad_equation_char(text):
    """Char pos of the first arithmetically WRONG simple equation."""
    best = None
    for m in EQ_RE.finditer(text):
        a, op, b, c = _num(m.group(1)), m.group(2), _num(m.group(3)), _num(m.group(4))
        if max(a, b, c) > 1e7:
            continue
        try:
            val = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "×": a * b,
                   "/": a / b if b else None, "÷": a / b if b else None}[op]
        except Exception:
            continue
        if val is None:
            continue
        if abs(val - c) > 0.01 * max(1.0, abs(val)):
            best = m.start() if best is None else min(best, m.start())
    for m in PCT_RE.finditer(text):
        a, b, c = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        if max(a, b, c) > 1e7:
            continue
        if abs(a / 100 * b - c) > 0.01 * max(1.0, abs(a / 100 * b)):
            best = m.start() if best is None else min(best, m.start())
    return best


def onsets(rows, kinds=("loop", "register", "operand")):
    """Per transcript: {kind: onset_step_or_None}, plus length."""
    out = []
    for r in rows:
        ids = json.loads(r["output_ids"])
        text = tok.decode(ids, skip_special_tokens=False)
        d = {"len": len(ids), "pid": r["pid"]}
        if "loop" in kinds:
            d["loop"] = loop_onset(ids)
        if "register" in kinds:
            m = REGISTER_RE.search(text)
            d["register"] = char_to_tok(ids, m.start()) if m else None
        if "operand" in kinds:
            c = first_bad_equation_char(text)
            d["operand"] = char_to_tok(ids, c) if c is not None else None
        out.append(d)
    return out


def hazard(onsets_list, kind, bins=(0, 16, 32, 64, 128, 256)):
    """Binned hazard: events in bin / transcript-steps at risk in bin."""
    rates = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        ev = at_risk = 0
        for d in onsets_list:
            o, L = d[kind], d["len"]
            if o is not None and o < b0:
                continue                      # already failed before bin
            upper = min(b1, L if o is None else min(L, o + 1))
            if upper > b0:
                at_risk += upper - b0
                if o is not None and b0 <= o < b1:
                    ev += 1
        rates.append(ev / at_risk * 100 if at_risk else float("nan"))
    return rates  # events per 100 at-risk steps


def frac_with(onsets_list, kind):
    return np.mean([d[kind] is not None for d in onsets_list])


BINS = (0, 16, 32, 64, 128, 256)
BIN_LABELS = [f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:])]

print("=" * 78)
print("HAZARD CURVES — events per 100 at-risk generation steps, by step bin")
print(f"bins: {BIN_LABELS}")
for task, kinds in [("gsm8k", ("loop", "register", "operand")),
                    ("fluency", ("loop", "register")),
                    ("triviaqa", ("loop", "register"))]:
    for cond in ("clean", "nla"):
        o = onsets(load(task, cond), kinds)
        for k in kinds:
            hz = hazard(o, k)
            print(f"{task:9s} {cond:6s} {k:9s} onset-frac {frac_with(o, k):.2f}  "
                  f"hazard: " + "  ".join(f"{h:5.2f}" for h in hz))
    print()

# ---------------- cosine-conditioning: loop onset after low-cosine window ----
print("=" * 78)
print("CONDITIONING: is loop onset preceded by a low-cosine window? (gsm8k C1)")
sl = pq.read_table(ROOT / "hfres/initial_run/stage_b/gsm8k/nla_seed0_steplogs.parquet",
                   columns=["pid", "step", "cosine"]).to_pylist()
cos = {}
for r in sl:
    if r["step"] >= 0:
        cos.setdefault(r["pid"], {})[r["step"]] = r["cosine"]
o_nla = onsets(load("gsm8k", "nla"), ("loop", "operand"))
W = 8
onset_win, control_win = [], []
for d in o_nla:
    c = cos.get(d["pid"], {})
    if d["loop"] is not None and d["loop"] >= W:
        w = [c[s] for s in range(d["loop"] - W, d["loop"]) if s in c]
        if len(w) >= 4:
            onset_win.append(np.mean(w))
    # control: same-position windows from transcripts NOT looping there
    for t0 in (32, 96, 160):
        if (d["loop"] is None or d["loop"] > t0 + W) and d["len"] > t0 + W:
            w = [c[s] for s in range(t0, t0 + W) if s in c]
            if len(w) >= 4:
                control_win.append(np.mean(w))
print(f"mean cosine in the {W} steps BEFORE loop onset: "
      f"{np.mean(onset_win):.4f} (n={len(onset_win)})")
print(f"mean cosine in matched non-onset windows:       "
      f"{np.mean(control_win):.4f} (n={len(control_win)})")
d_ = np.mean(onset_win) - np.mean(control_win)
se = np.sqrt(np.var(onset_win)/len(onset_win) + np.var(control_win)/len(control_win))
print(f"difference {d_:+.4f} (~{d_/se:.1f} SE)")

# ---------------- ordering: operand error vs loop onset ----------------------
print("\nORDERING (gsm8k C1, transcripts with both events):")
both = [d for d in o_nla if d["loop"] is not None and d["operand"] is not None]
before = np.mean([d["operand"] < d["loop"] for d in both])
print(f"n={len(both)}; operand error precedes loop onset in {before:.0%} "
      f"(median operand step {np.median([d['operand'] for d in both]):.0f}, "
      f"median loop step {np.median([d['loop'] for d in both]):.0f})")
only_loop = sum(1 for d in o_nla if d["loop"] is not None and d["operand"] is None)
only_op = sum(1 for d in o_nla if d["loop"] is None and d["operand"] is not None)
print(f"loop-only: {only_loop}, operand-only: {only_op}, "
      f"neither: {sum(1 for d in o_nla if d['loop'] is None and d['operand'] is None)}")
