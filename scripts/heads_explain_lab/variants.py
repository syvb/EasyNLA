"""Prompt variants tried in the explanation lab; v26 won and is heads_soft_token_explain.SYSTEM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INTRO = """\
You explain what a vector from inside a language model does, using only readouts of it.

Readout A lists words the model makes MORE likely when the vector is placed into it as a stand-in for a word, \
in the prompts "Words that come to mind: <vector>:" and "Here is some text: <vector>", ranked by how much \
the vector raised them. It is the main evidence. Some entries are generic filler that shows up for many \
vectors (e.g. "it", "the", "\\n", punctuation); give those little weight unless they dominate.

Readout B lists the output tokens the vector points at directly. It is noisy (subword fragments, other \
languages) and is often uninformative; use it only when it agrees with A or is itself clearly coherent \
(e.g. many casings of one word).
"""

INTRO_C = INTRO + """
Readout C is like A but from the first prompt only, with 24 entries.
"""

RULES = """\
Hard rules:
- Never quote any word, and never use quotation marks.
- Never give examples of words (no "such as X, Y and Z" with words from the evidence).
- Never mention readouts, evidence, tokens, words, casings, spellings or languages.
- Be as specific as the evidence allows without speculating beyond it. When it is diffuse, describe the \
broad tendency rather than saying there is no theme.
"""

VARIED = """\
Vary your phrasing from vector to vector the way a person writing notes would. Do not open with a stock \
phrase such as "Pushes the model toward", "This vector", "It pushes" or "Promotes"; lead with the substance \
itself, and choose whatever sentence structure fits it.

Good answers look like:
- Doors, windows and other openings in buildings, along with where or when something happens in them.
- Obligation and possibility: it sets up a statement about what someone must, may or can do.
- A sense that something has gone wrong or is out of place, as when describing a problem or a fault.
- Stands in for an unnamed thing or person, a placeholder waiting for something to attach to it.
- Reported speech and past events, leading into what someone said or what happened next.
- Mostly grammatical: a slot right after a subject where a verb of being or feeling belongs.
"""

TASK = """\
Write one or two plain sentences (at most 35 words) explaining what the vector does: what meaning, \
topic, entity, grammatical role, sentiment or kind of continuation it carries or pushes the model toward. \
Describe its function in your own words, as concepts, never as a list of the words it promotes.
"""

TASK_LONG = """\
Write two or three plain sentences (at most 60 words) explaining what the vector does: what meaning, \
topic, entity, grammatical role, sentiment or kind of continuation it carries or pushes the model toward, \
and, if the evidence shows one, a secondary tendency. Describe its function in your own words, as concepts, \
never as a list of the words it promotes.
"""

END = "\nReply with the explanation only."

# the prompt before the lab (round 1 baseline)
V0 = INTRO.replace("using only readouts of it", "using only two readouts of it") + "\n" + TASK + "\n" + RULES + """
Good answers look like:
- Pushes the model toward talk of doors, windows and other openings in buildings.
- Signals obligation or possibility, setting up a statement about what someone must, may or can do.
- Carries a sense that something has gone wrong or is out of place, as when describing a problem or a fault.
- Points vaguely at an unspecified thing or person, the way a placeholder stands in for something unnamed.
""" + END

V1 = INTRO + "\n" + TASK + "\n" + RULES + "\n" + VARIED + END

V2 = INTRO + """
Write a short note (at most 30 words) naming what the vector carries, in the style of a researcher's \
feature label followed by a brief gloss: lead with a noun phrase for the concept, then, after a colon or \
dash, say what it does in context. Describe it as concepts, never as a list of the words it promotes.

""" + RULES + """
Good answers look like:
- Openings in buildings: doors, windows and passageways, often where something happens.
- Modality — sets up what someone must, may or can do.
- Something gone wrong: problems, faults and things out of place.
- Unnamed referent: a placeholder for a thing or person not yet specified.
""" + END

V6 = INTRO + "\n" + TASK_LONG + "\n" + RULES + "\n" + VARIED + END

V7 = INTRO + """
Think of the vector as a piece of information written into one word's representation, which the rest of \
the model then reads. Write one or two plain sentences (at most 35 words) saying what that information \
is: what it tells the model about the word or what should come next. Describe it in your own words, as \
concepts, never as a list of the words it promotes.

""" + RULES + "\n" + VARIED + END

VARIANTS = {
    "v0": {"system": V0},
    "v1": {"system": V1},
    "v2": {"system": V2},
    "v3": {"system": V1, "thinking": "adaptive", "effort": "low"},
    "v4": {"system": INTRO_C + "\n" + TASK + "\n" + RULES + "\n" + VARIED + END,
           "inputs": ("avg16", "lens", "delta24")},
    "v5": {"system": V1, "inputs": ("avg16",)},
    "v6": {"system": V6},
    "v7": {"system": V7},
}


INTRO_D = """\
You explain what a vector from inside a language model does, using only readouts of it.

Readout A lists words the model makes MORE likely when the vector is placed into it as a stand-in for a word, \
in the prompts "Words that come to mind: <vector>:" and "Here is some text: <vector>". It is the main evidence. \
It comes in two parts. The distinctive part holds the words that are rare across vectors in general, so they \
say the most about this one. The common filler part holds words that show up for a large share of all vectors \
(vague nouns for things and people, reporting verbs, line breaks); on their own they mean little, so lean on \
them only when the distinctive part is thin or agrees with them.

Readout B lists the output tokens the vector points at directly. It is noisy (subword fragments, other \
languages) and is often uninformative; use it only when it agrees with A or is itself clearly coherent \
(e.g. many casings of one word).
"""

VARIED2 = """\
Write like a researcher's notes, varying the phrasing from vector to vector. Do not open with a stock \
phrase (for example "Pushes the model toward", "This vector", "It carries", "It pushes", "Promotes", \
"A slot", "A placeholder"); lead with the substance and pick whatever sentence structure fits it.

These examples show the range of styles; do not reuse their wording:
- Doors, windows and other openings in buildings, along with where or when something happens in them.
- Obligation and possibility, setting up a statement about what someone must, may or can do.
- Something has gone wrong or is out of place, as when a problem or a fault is being described.
- Reported speech and past events lead the continuation: what someone said, or what happened next.
- Mostly grammatical, marking the point right after a subject where a verb of being or feeling belongs.
- Counting and measurement of time, especially yearly cycles and anniversaries.
"""

FOCUS = """\
Lead with whatever is most distinctive about this vector. Fall back on a vague or generic reading only \
when nothing more specific is supported.
"""

V8 = INTRO_D + "\n" + TASK + "\n" + FOCUS + "\n" + RULES + "\n" + VARIED2 + END
V9 = INTRO + "\n" + TASK + "\n" + FOCUS + "\n" + RULES + "\n" + VARIED2 + END
V10 = INTRO_D + """
Write a short note (at most 30 words) naming what the vector carries, in the style of a researcher's \
feature label followed by a brief gloss: lead with a noun phrase for the concept, then, after a colon or \
dash, say what it does in context. Describe it as concepts, never as a list of the words it promotes.

""" + FOCUS + "\n" + RULES + """
These examples show the range of styles; do not reuse their wording:
- Openings in buildings: doors, windows and passageways, often where something happens.
- Modality — sets up what someone must, may or can do.
- Things going wrong: problems, faults and anything out of place.
- Yearly time: anniversaries, annual cycles and the passing of a year.
""" + END
V13 = INTRO_D + "\n" + TASK_LONG + "\n" + FOCUS + "\n" + RULES + "\n" + VARIED2 + END

D = ("avg16d8", "lens")
VARIANTS |= {
    "v8": {"system": V8, "inputs": D},
    "v9": {"system": V9},
    "v10": {"system": V10, "inputs": D},
    "v11": {"system": V8, "inputs": D, "thinking": "adaptive", "effort": "low"},
    "v12": {"system": V8, "inputs": D, "model": "claude-sonnet-5-5", "thinking": "between_tools", "effort": "low"},
    "v13": {"system": V13, "inputs": D},
}

import re

from heads_soft_token_explain import leaks

STOCK = re.compile(r"^(this|the|a) (vector|representation)\b|^(it )?(pushes|promotes|steers|nudges)\b|^it carries\b",
                   re.IGNORECASE)


def check2(text: str, ev: str) -> bool:
    return leaks(text, ev) or bool(STOCK.search(text))


FIX2 = ("Rewrite it. Explain only what the vector does, in your own words, as concepts: no quoted or example words, "
        "no mention of readouts, tokens, words or how reliable anything is, and do not open with a stock phrase "
        "such as This vector, It pushes, Promotes or It carries; lead with the substance. "
        "Reply with the explanation only.")

SILENT = """\
Ignore weak or noisy evidence silently: never comment on what is noisy, unclear, weak or uninformative.
"""

V14 = INTRO_D + "\n" + TASK + "\n" + FOCUS + SILENT + "\n" + RULES + "\n" + VARIED2 + END
V15 = V10.replace(FOCUS, FOCUS + SILENT)
VARIANTS |= {
    "v14": {"system": V14, "inputs": D, "check": check2, "fix": FIX2},
    "v15": {"system": V15, "inputs": D, "check": check2, "fix": FIX2},
    "v17": {"system": V14, "inputs": ("avg16d12", "lens"), "check": check2, "fix": FIX2},
    "v18": {"system": V14, "inputs": D, "check": check2, "fix": FIX2, "retries": 4},
}

TASK_RICH = """\
Write one or two plain sentences (at most 45 words) explaining what the vector does: the main meaning, \
topic, entity, grammatical role, sentiment or kind of continuation it carries or pushes the model toward, \
and a secondary tendency if one is clearly supported. Describe its function in your own words, as \
concepts, never as a list of the words it promotes.
"""
V20 = INTRO_D + "\n" + TASK_RICH + "\n" + FOCUS + SILENT + "\n" + RULES + "\n" + VARIED2 + END
SUFFIX = "\n\n(Explain the vector; lead with the substance, not with a stock opening.)"
C2 = {"check": check2, "fix": FIX2, "retries": 4, "inputs": D}
VARIANTS |= {
    "v20": {"system": V20, **C2},
    "v21": {"system": V14, **C2, "thinking": "adaptive", "effort": "medium"},
    "v22": {"system": V14, **C2, "suffix": SUFFIX},
    "v23": {"system": V20, **C2, "suffix": SUFFIX},
}

VARIED3 = VARIED2.replace('''lead with the substance and pick whatever sentence structure fits it.''', '''lead with the substance and pick whatever sentence structure fits it. Avoid opening with a bare verb \
(such as Leans, Sets up, Tilts, Points, Primes, Marks, Pulls, Favors); most explanations should open with \
the concept itself or with the situation it applies to.''')
assert VARIED3 != VARIED2
V26 = INTRO_D + "\n" + TASK_RICH + "\n" + FOCUS + SILENT + "\n" + RULES + "\n" + VARIED3 + END
VARIANTS |= {
    "v24": {"system": V20, **C2, "thinking": "adaptive", "effort": "medium"},
    "v25": {"system": V20, **C2, "thinking": "adaptive", "effort": "low"},
    "v26": {"system": V26, **C2, "thinking": "adaptive", "effort": "medium"},
}
