"""Warm-start description prompts for head contributions, and response parsing.

The describer sees what a person could see — never the vectors: the text up
to the current token, and where this head's attention went (top sources with
weights and a little surrounding text). It is asked for one or two clauses on
what information the head carried from those tokens to here. The
instructions are a fixed system prompt (cacheable, ≥ the 512-token minimum);
everything per-example goes in the user turn.

Rendering is pure (decode function in, strings out) so it is unit-testable
without a tokenizer download.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

INSTRUCTIONS = """You are helping interpret a single attention head inside a transformer language model.

At every token, an attention head looks back over the earlier tokens (and the current one), weights them, and moves a little information from the tokens it weights most into the current token's representation. You will be shown one head at one token:
- the text up to and including the current token, which is the final token and is marked ⟦like this⟧;
- where the head's attention went: its most-attended source tokens with their attention weights, each shown with a little surrounding text.

Your job: in one or two short clauses, say what information this head most likely carried from the attended tokens to the current token.

Rules:
1. Say what information was carried and which tokens it came from: for example an attribute of an entity, the identity of an earlier word, the subject a verb must agree with, or the token that followed an earlier occurrence of the current one.
2. Quote only literal tokens, exactly as they appear in the text, in single quotes, e.g. 'Colosseum'. Never put paraphrases or inferred facts inside quotes.
3. Never describe the document, its topic, its style, or its genre. Describe only what this head moved.
4. If most of the attention sits on the first token of the text, the head is idling: answer exactly <description>little or nothing</description>. If attention is split between the first token and other tokens, describe only what came from the other tokens.
5. Keep it under 25 words. No preamble and nothing outside the tags.

Answer with only <description>...</description>.

Examples of good answers (each for a different head and text):

Current token ⟦ was⟧; attention 0.64 on ⟦ Colosseum⟧ in "…visited the ancient ⟦ Colosseum⟧ in the heart of…"
<description>location attributes of 'Colosseum': Rome, Italy</description>

Current token ⟦ York⟧; attention 0.93 on the previous token ⟦ New⟧
<description>the identity of the previous token, 'New'</description>

Current token ⟦ Harriet⟧; attention 0.58 on ⟦ Vane⟧ in "…said ⟦ Vane⟧. Harriet…", where ' Vane' followed an earlier ' Harriet'
<description>'Vane', the token that followed the earlier 'Harriet', as the likely next token</description>

Current token ⟦ are⟧; attention 0.47 on ⟦ keys⟧ in "…The ⟦ keys⟧ to the old cabinet…"
<description>plural number of the subject 'keys', for verb agreement</description>

Current token ⟦,⟧; attention 0.88 on ⟦ However⟧ in "…⟦ However⟧ the committee…"
<description>that the clause began with the contrastive 'However'</description>

Current token ⟦ the⟧; attention 0.96 on the first token
<description>little or nothing</description>"""

DESCRIPTION_OPEN, DESCRIPTION_CLOSE = "<description>", "</description>"
LITTLE = "little or nothing"
_DESC_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL)
# A quote opens at a ' not preceded by a word char or ' (so "head's" does not
# open one) and closes at a ' not followed by a word char.
_QUOTE_RE = re.compile(r"(?<![\w'])'([^'\n]+?)'(?!\w)")
_DOC_WORDS_RE = re.compile(
    r"\b(document|article|passage|genre|essay|blog|webpage|web page|the text is|this text)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PromptConfig:
    context_tokens: int = 1024      # text shown before the current token (inclusive)
    window_tokens: int = 12         # tokens on each side of an attended source
    max_sources: int = 6
    min_source_weight: float = 0.02  # top-1 is always shown


def _mark(s: str) -> str:
    return f"⟦{s}⟧"


def _inline(s: str) -> str:
    """One-line rendering for windows and token labels."""
    return s.replace("\r", "").replace("\n", "⏎").replace("\t", "⇥")


def render_user_prompt(
    decode: Callable[[Sequence[int]], str],
    token_ids: Sequence[int],
    pos: int,
    layer: int,
    head: int,
    n_layers: int,
    n_heads: int,
    top_idx: Sequence[int],
    top_w: Sequence[float],
    first_w: float,
    self_w: float,
    cfg: PromptConfig,
) -> str:
    assert 0 <= pos < len(token_ids), f"pos {pos} outside {len(token_ids)} tokens"
    ctx_start = max(0, pos + 1 - cfg.context_tokens)
    before = decode(token_ids[ctx_start:pos])
    current = decode(token_ids[pos : pos + 1])
    lines = [
        f"Head: L{layer}H{head} (layer {layer} of {n_layers}, head {head} of {n_heads})",
        "",
        "<text>",
        ("…" if ctx_start > 0 else "") + before + _mark(current),
        "</text>",
        "",
        f"Current token: {_mark(_inline(current))} (position {pos})",
        "",
        f"Where this head's attention went (top sources; weights sum to 1 over all {pos + 1} tokens):",
    ]
    shown = 0.0
    n = 0
    for src, w in zip(top_idx, top_w, strict=True):
        if n >= cfg.max_sources or (n > 0 and w < cfg.min_source_weight):
            break
        src = int(src)
        tok = _inline(decode(token_ids[src : src + 1]))
        if src == pos:
            what = f"the current token itself {_mark(tok)}"
            ctx = ""
        else:
            label = "the first token " if src == 0 else ("the previous token " if src == pos - 1 else "")
            lo, hi = max(0, src - cfg.window_tokens), min(pos + 1, src + 1 + cfg.window_tokens)
            left = ("…" if lo > 0 else "") + _inline(decode(token_ids[lo:src]))
            right = _inline(decode(token_ids[src + 1 : hi])) + ("…" if hi < pos + 1 else "")
            ctx = f' in "{left}{_mark(tok)}{right}"'
            what = f"{label}{_mark(tok)} (position {src})"
        lines.append(f"{n + 1}. {w:.2f} on {what}{ctx}")
        shown += float(w)
        n += 1
    lines += [
        "",
        (f"Weight on the first token: {first_w:.2f}. Weight on the current token: {self_w:.2f}. "
         f"Weight on the sources listed: {shown:.2f}."),
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedDescription:
    description: str | None
    fail_reason: str | None
    is_little: bool
    n_words: int
    quotes: tuple[str, ...]
    quote_ok: bool             # every quoted span appears verbatim in the prompt
    mentions_document: bool    # heuristic flag for rule 3


def parse_response(text: str, prompt_text: str, max_words: int = 40) -> ParsedDescription:
    m = _DESC_RE.search(text or "")
    if m is None:
        return ParsedDescription(None, "no_tags", False, 0, (), False, False)
    desc = " ".join(m.group(1).split())
    if not desc:
        return ParsedDescription(None, "empty", False, 0, (), False, False)
    n_words = len(desc.split())
    quotes = tuple(q for q in _QUOTE_RE.findall(desc))
    is_little = desc.lower().rstrip(" .") == LITTLE
    return ParsedDescription(
        description=desc,
        fail_reason="too_long" if n_words > max_words else None,
        is_little=is_little,
        n_words=n_words,
        quotes=quotes,
        quote_ok=all(q in prompt_text for q in quotes),
        mentions_document=bool(_DOC_WORDS_RE.search(desc)),
    )
