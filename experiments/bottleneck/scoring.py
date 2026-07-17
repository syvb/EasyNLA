"""Offline scoring for Stage B outputs (CPU-ok; no GPU deps).

Extraction must stay in sync with the answer-format instructions in tasks.py.
Code tasks EXECUTE model-generated code in a subprocess — run scoring on a
disposable box or accept the standard humaneval-style risk knowingly.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ numeric --
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").rstrip(".").lstrip("$")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_final_number(text: str) -> str | None:
    """'#### x' if present (the instructed format), else the last number."""
    m = re.findall(r"####\s*([^\n]+)", text)
    hay = m[-1] if m else text
    nums = _NUM_RE.findall(hay)
    if not nums and m:
        nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} with balanced braces."""
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    s = starts[-1]
    depth, i = 1, s
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[s:i - 1] if depth == 0 else text[s:]


def _norm_math(s: str) -> str:
    s = s.strip().strip("$ ").replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\!|\\,|\\;", "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    return s.rstrip(".")


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if _norm_math(pred) == _norm_math(gold):
        return True
    try:  # optional exact checker
        from math_verify import parse, verify  # type: ignore
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        pass
    try:
        return abs(float(normalize_number(pred)) - float(normalize_number(gold))) < 1e-6
    except ValueError:
        return False


# --------------------------------------------------------------------- code --
def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    if "def " in text:  # raw code without fences
        return text[text.index("def "):]
    return None


def run_python(program: str, timeout: float = 10.0) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def score_humaneval(gold: dict, output: str) -> float:
    code = extract_code(output)
    if code is None:
        return 0.0
    # The model was shown the signature and asked for the complete function;
    # if it emitted only a body-less continuation, prepend the prompt.
    program = code if f"def {gold['entry_point']}" in code else gold["prompt"] + code
    program += "\n\n" + gold["test"] + f"\n\ncheck({gold['entry_point']})\n"
    return float(run_python(program))


def score_mbpp(gold: dict, output: str) -> float:
    code = extract_code(output)
    if code is None:
        return 0.0
    program = (gold.get("test_setup_code") or "") + "\n" + code + "\n" \
        + "\n".join(gold["test_list"]) + "\n"
    return float(run_python(program))


# ----------------------------------------------------------------- short QA --
def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred: str, aliases: list[str]) -> tuple[float, float]:
    # Answer = first line of the output (SHORT_INSTR asks for just the answer).
    pred_n = normalize_text(pred.strip().split("\n")[0])
    best_f1 = 0.0
    em = 0.0
    for a in aliases:
        a_n = normalize_text(a)
        if not a_n:
            continue
        if pred_n == a_n:
            em = 1.0
        pt, at = pred_n.split(), a_n.split()
        common = Counter(pt) & Counter(at)
        n_same = sum(common.values())
        if n_same:
            prec, rec = n_same / len(pt), n_same / len(at)
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return em, best_f1


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([A-J])\b", text.strip())
    return m.group(1) if m else None


# ------------------------------------------------------------------ fluency --
def repetition_rate(text: str, n: int = 3) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# ------------------------------------------------------------------- driver --
def score_row(task: str, gold: dict, output: str) -> dict:
    if task in ("gsm8k", "mgsm"):
        pred = extract_final_number(output)
        ok = pred is not None and normalize_number(pred) == normalize_number(gold["answer"])
        return {"score": float(ok)}
    if task == "math500":
        return {"score": float(math_equal(extract_boxed(output), gold["answer"]))}
    if task == "humaneval":
        return {"score": score_humaneval(gold, output)}
    if task == "mbpp":
        return {"score": score_mbpp(gold, output)}
    if task in ("triviaqa", "popqa"):
        em, f1 = em_f1(output, gold["aliases"])
        return {"score": em, "f1": f1}
    if task == "mmlu_pro":
        return {"score": float(extract_letter(output) == gold["answer"])}
    if task == "fluency":
        return {"score": float("nan"), "rep3": repetition_rate(output)}
    if task == "ifeval":
        return {"score": float("nan")}  # scored by score_ifeval() over the full set
    raise ValueError(task)


def score_ifeval(rows: list[dict]) -> list[float] | None:
    """Strict-prompt IFEval scoring. Tries the reference implementation
    (pip `instruction-following-eval`, module `instruction_following_eval`);
    if unavailable, returns None — the caller then writes the official-format
    JSONL so google-research/instruction_following_eval can be run separately."""
    try:
        from instruction_following_eval import evaluation_lib as el  # type: ignore
    except ImportError:
        return None
    scores = []
    for r in rows:
        gold = json.loads(r["gold"])
        inp = el.InputExample(
            key=0, instruction_id_list=gold["instruction_id_list"],
            prompt=gold["prompt"],
            kwargs=[{k: v for k, v in kw.items() if v is not None}
                    for kw in json.loads(gold["kwargs"])],
        )
        out = el.test_instruction_following_strict(inp, {gold["prompt"]: r["output"]})
        scores.append(float(out.follow_all_instructions))
    return scores


def bootstrap_mean_ci(scores: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    means = rng.choice(scores, size=(n_boot, n), replace=True).mean(axis=1)
    return float(scores.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_retention_ci(cond: np.ndarray, clean: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Retention = mean(cond)/mean(clean) with a paired bootstrap over problems
    (both arrays aligned by pid)."""
    assert len(cond) == len(clean)
    rng = np.random.default_rng(seed)
    n = len(cond)
    idx = rng.integers(0, n, size=(n_boot, n))
    c, k = cond[idx].mean(axis=1), clean[idx].mean(axis=1)
    ratio = np.divide(c, k, out=np.full_like(c, np.nan), where=k > 0)
    point = cond.mean() / clean.mean() if clean.mean() > 0 else float("nan")
    return float(point), float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))
