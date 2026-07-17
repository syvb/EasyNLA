"""Stage B benchmark task definitions: loaders + prompt builders.

Each task yields Problems with chat messages for M (non-thinking mode) and a
`gold` payload for offline scoring (scoring.py). Answer-format instructions
live HERE, extraction lives in scoring.py — keep them in sync.

All loaders use `datasets` (installed on the GPU box) with fixed-seed
subsampling so every condition/seed sees the identical problem set.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Problem:
    task: str
    pid: str
    messages: list[dict]
    gold: dict = field(default_factory=dict)
    max_new_tokens: int = 256


GSM_INSTR = ("Solve the following math problem. Think step by step, then write "
             "your final numeric answer on the last line in the form '#### <answer>'.")
MATH_INSTR = ("Solve the following math problem. Think step by step, then give "
              "your final answer inside \\boxed{}.")
CODE_INSTR = ("Output only the complete solution inside a single ```python code block.")
SHORT_INSTR = ("Answer the following question directly and as briefly as possible, "
               "with just the answer.")
MC_INSTR = ("Answer with only the letter of the correct option.")

# MGSM languages: two high-resource non-English + one CJK + one low-resource.
MGSM_LANGS = ("fr", "zh", "ru", "sw")

FLUENCY_PROMPTS = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "Explain how a refrigerator works to a curious ten-year-old.",
    "Write a persuasive paragraph arguing that public libraries still matter.",
    "Describe your ideal city of the future in vivid detail.",
    "Write a polite email declining a job offer while keeping the relationship warm.",
    "Explain the difference between weather and climate.",
    "Write a recipe for a comforting soup, with a brief story about why you love it.",
    "Describe the experience of watching a thunderstorm from indoors.",
    "Write a product review for a fictional noise-cancelling headphone.",
    "Explain why the sky is blue.",
    "Write a dialogue between two strangers stuck in an elevator.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Write a cover letter for a junior data analyst position.",
    "Describe how to change a bicycle tire, step by step.",
    "Write a poem about the first day of autumn.",
    "Explain what inflation is and why it matters to ordinary people.",
    "Write a travel guide paragraph about a small coastal town.",
    "Describe a childhood memory involving food.",
    "Explain how vaccines work in simple terms.",
    "Write a eulogy for a beloved fictional pet turtle.",
    "Describe the inside of an old bookshop.",
    "Write instructions for teaching someone to swim.",
    "Explain the rules of chess to a complete beginner.",
    "Write a news brief about a local bakery winning a national award.",
    "Describe a sunrise over mountains without using the word 'beautiful'.",
    "Write a letter to your future self, ten years from now.",
    "Explain why exercise is good for mental health.",
    "Write a short scene where a detective interviews a nervous witness.",
    "Describe how the internet gets a web page to your screen.",
    "Write an apology note for missing a friend's birthday party.",
    "Explain what photosynthesis is and why it matters.",
    "Write a toast for a small wedding reception.",
    "Describe a bustling street market using all five senses.",
    "Explain the concept of compound interest with an example.",
    "Write a short fable with a moral about patience.",
    "Describe what makes a good teacher.",
    "Write a set of tips for a first-time public speaker.",
    "Explain how tides work.",
    "Write a diary entry from the perspective of a museum night guard.",
    "Describe the process of making bread from scratch.",
    "Explain the difference between a virus and a bacterium.",
    "Write a short speech welcoming new students to a school.",
    "Describe an abandoned amusement park at dusk.",
    "Explain why we dream, according to current science.",
    "Write a haiku sequence (three haikus) about city life.",
    "Describe how to plan a budget-friendly week of meals.",
    "Explain what machine learning is to a skeptical grandparent.",
    "Write a scene where two old friends reunite after twenty years.",
    "Describe the sound of rain on different surfaces.",
    "Explain how a seed becomes a tree.",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _user(task: str, pid: str, content: str, gold: dict, max_new: int) -> Problem:
    return Problem(task=task, pid=pid, gold=gold, max_new_tokens=max_new,
                   messages=[{"role": "user", "content": content}])


def load_task(name: str, n: int, seed: int = 0) -> list[Problem]:
    import datasets as hfd

    r = _rng(seed)
    if name == "gsm8k":
        ds = hfd.load_dataset("openai/gsm8k", "main", split="test")
        idx = r.sample(range(len(ds)), min(n, len(ds)))
        return [_user("gsm8k", str(i), f"{GSM_INSTR}\n\n{ds[i]['question']}",
                      {"answer": ds[i]["answer"].split("####")[-1].strip()}, 256)
                for i in idx]

    if name == "math500":
        ds = hfd.load_dataset("HuggingFaceH4/MATH-500", split="test")
        idx = r.sample(range(len(ds)), min(n, len(ds)))
        return [_user("math500", str(i), f"{MATH_INSTR}\n\n{ds[i]['problem']}",
                      {"answer": ds[i]["answer"]}, 512)
                for i in idx]

    if name == "humaneval":
        ds = hfd.load_dataset("openai/openai_humaneval", split="test")
        return [_user("humaneval", row["task_id"],
                      f"Complete the following Python function. {CODE_INSTR}\n\n"
                      f"```python\n{row['prompt']}```",
                      {"prompt": row["prompt"], "test": row["test"],
                       "entry_point": row["entry_point"]}, 512)
                for row in ds]

    if name == "mbpp":
        ds = hfd.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        rows = list(ds)[:n]
        return [_user("mbpp", str(row["task_id"]),
                      f"{row['prompt']}\nYour code should satisfy these tests:\n"
                      + "\n".join(row["test_list"]) + f"\n{CODE_INSTR}",
                      {"test_list": row["test_list"],
                       "test_setup_code": row.get("test_setup_code", "")}, 256)
                for row in rows]

    if name == "triviaqa":
        ds = hfd.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        idx = r.sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idx:
            row = ds[i]
            aliases = list(row["answer"]["aliases"]) + list(row["answer"]["normalized_aliases"])
            out.append(_user("triviaqa", row["question_id"],
                             f"{SHORT_INSTR}\n\nQ: {row['question']}\nA:",
                             {"aliases": aliases}, 64))
        return out

    if name == "popqa":
        ds = hfd.load_dataset("akariasai/PopQA", split="test")
        idx = r.sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idx:
            row = ds[i]
            aliases = json.loads(row["possible_answers"])
            out.append(_user("popqa", str(row["id"]),
                             f"{SHORT_INSTR}\n\nQ: {row['question']}\nA:",
                             {"aliases": aliases}, 64))
        return out

    if name == "mmlu_pro":
        # Stratified: n = per-category count (plan: ~40/category; note the
        # per-category CIs at 40 are ±15pp — aggregate to clusters in analysis).
        ds = hfd.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        by_cat: dict[str, list[int]] = {}
        for i, row in enumerate(ds):
            by_cat.setdefault(row["category"], []).append(i)
        out = []
        letters = "ABCDEFGHIJ"
        for cat, idxs in sorted(by_cat.items()):
            for i in r.sample(idxs, min(n, len(idxs))):
                row = ds[i]
                opts = "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(row["options"]))
                out.append(_user("mmlu_pro", str(row["question_id"]),
                                 f"{row['question']}\n\n{opts}\n\n{MC_INSTR}",
                                 {"answer": row["answer"], "category": cat}, 32))
        return out

    if name == "mgsm":
        out = []
        per_lang = n
        for lang in MGSM_LANGS:
            ds = hfd.load_dataset("juletxara/mgsm", lang, split="test")
            idx = _rng(seed + hash(lang) % 1000).sample(range(len(ds)), min(per_lang, len(ds)))
            for i in idx:
                row = ds[i]
                out.append(_user("mgsm", f"{lang}/{i}",
                                 f"{GSM_INSTR}\n\n{row['question']}",
                                 {"answer": str(row["answer_number"]), "lang": lang}, 256))
        return out

    if name == "ifeval":
        ds = hfd.load_dataset("google/IFEval", split="train")
        idx = r.sample(range(len(ds)), min(n, len(ds)))
        return [_user("ifeval", str(ds[i]["key"]), ds[i]["prompt"],
                      {"prompt": ds[i]["prompt"],
                       "instruction_id_list": list(ds[i]["instruction_id_list"]),
                       "kwargs": json.dumps(list(ds[i]["kwargs"]))}, 384)
                for i in idx]

    if name == "fluency":
        return [_user("fluency", str(i), p, {}, 256)
                for i, p in enumerate(FLUENCY_PROMPTS[:n])]

    raise ValueError(f"unknown task {name!r}")


# Default first-pass sizes (plan §3). For mmlu_pro the number is PER CATEGORY.
DEFAULT_SIZES = {
    "gsm8k": 200, "math500": 150, "humaneval": 164, "mbpp": 150,
    "triviaqa": 300, "popqa": 300, "mmlu_pro": 40, "mgsm": 100,
    "ifeval": 150, "fluency": 50,
}
ALL_TASKS = tuple(DEFAULT_SIZES)
