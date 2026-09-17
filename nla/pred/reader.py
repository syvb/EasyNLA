"""Frozen-reader scoring: how much does an explanation help a reader LM predict
the target model's continuation?

THE MEASUREMENT
---------------
A position gives us K continuation branches sampled from the frozen target model,
each T tokens long (default K=4, T=24). A reader LM is shown

    <framing> <explanation> <framing> <continuation>

and we take the teacher-forced log-probability it assigns to the continuation
tokens. Because teacher forcing scores every continuation token in ONE forward,
we get the "bridge" structure for free: token j is scored conditioned on the
explanation plus continuation tokens 0..j-1. Splitting the T tokens into buckets
therefore gives several bridge lengths from a single forward:

    bucket [0,8)   — no bridge      (explanation only)
    bucket [8,16)  —  8-token bridge
    bucket [16,24) — 16-token bridge   <- the headline span

`predictive gain` is always a DIFFERENCE between two prefixes over the exact same
continuation tokens:

    gain(z) = score(explanation z) - score(no explanation)

reported in nats per TARGET token (denominator = the target model's token count
for the bucket, identical for every reader), so Qwen and Gemma numbers are
directly comparable.

TWO CORRECTNESS DECISIONS
-------------------------
1. `prefix_ids + cont_ids`, tokenized SEPARATELY. The continuation is tokenized
   on its own and concatenated after the prefix's ids, rather than tokenizing the
   joined string. That costs a little naturalness at the seam (the reader may see
   a token split the tokenizer would not have chosen there) but buys the thing
   that matters: the scored token set is BIT-IDENTICAL across conditions, so a
   paired difference is attributable purely to the prefix. Tokenizing the joined
   string would let the explanation's last character change how the first
   continuation token is split, quietly changing what is being compared.

2. Bucket membership is decided by CHARACTER offsets inside the continuation
   text, not token counts. Readers from different families tokenize the same
   continuation differently; character offsets are the only shared coordinate
   system. A reader token belongs to the bucket its FIRST character falls in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Target-token index ranges scored as separate buckets. The last one is the
# headline span from the experiment plan (16-token bridge, 8 scored tokens); the
# earlier ones come free from the same forward and say whether the explanation's
# value survives being handed more real context.
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = ((0, 8), (8, 16), (16, 24))

# Bucket carrying the RL reward and the headline table entry.
HEADLINE_BUCKET = (16, 24)


@dataclass(frozen=True)
class ReaderTemplates:
    """Reader prompts. `with_expl` and `without` are deliberately parallel: both
    tell the reader it is seeing text from the middle of a document that a model
    was reading, and differ ONLY in whether a description is supplied. The gain
    therefore measures the description's content, not the framing.

    Both MUST end with a newline so the continuation starts at a line boundary.
    """

    with_expl: str = (
        "A language model was reading a document. Here is a description of what "
        "the model was representing internally at one point in the text:\n"
        "\n"
        "{explanation}\n"
        "\n"
        "Here is the text of the document continuing from that exact point:\n"
    )
    without: str = (
        "A language model was reading a document. No description of what the "
        "model was representing internally is available.\n"
        "\n"
        "Here is the text of the document continuing from that exact point:\n"
    )

    def render(self, explanation: str | None) -> str:
        if explanation is None:
            return self.without
        return self.with_expl.format(explanation=explanation.strip())

    def as_dict(self) -> dict:
        return {"with_expl": self.with_expl, "without": self.without}


def token_char_bounds(tokenizer, token_ids: list[int]) -> list[int]:
    """Character offsets of each target token boundary within the decoded text.

    Returns K+1 offsets: bounds[j] is the character index in
    `tokenizer.decode(token_ids)` at which target token j starts (bounds[0] == 0,
    bounds[K] == len(text)).

    Computed by cumulative decode, which is exact for the incremental-decode
    property BPE tokenizers have in practice, and asserted to be monotone. If a
    decode is NOT prefix-monotone (rare normalization edge cases), the boundary is
    clamped forward, which can only move a token into a later bucket - never
    silently mis-score one.
    """
    text = tokenizer.decode(token_ids)
    bounds = [0]
    for j in range(1, len(token_ids) + 1):
        piece = tokenizer.decode(token_ids[:j])
        b = len(piece) if text.startswith(piece) else bounds[-1]
        bounds.append(max(b, bounds[-1]))
    bounds[-1] = len(text)
    return bounds


def bucket_char_ranges(
    bounds: list[int], buckets=DEFAULT_BUCKETS
) -> list[tuple[int, int]]:
    """Map target-token buckets to character ranges using `token_char_bounds`."""
    n = len(bounds) - 1
    out = []
    for lo, hi in buckets:
        assert 0 <= lo < hi <= n, f"bucket ({lo},{hi}) outside 0..{n} target tokens"
        out.append((bounds[lo], bounds[hi]))
    return out


@dataclass
class ScoreJob:
    """One (prefix, continuation) pair to score.

    key: caller-supplied identifier echoed back on the result (row id, condition,
         branch index - whatever the caller needs to reassemble).
    """

    key: object
    explanation: str | None
    cont_text: str
    char_ranges: list[tuple[int, int]]


@dataclass
class ScoreResult:
    key: object
    # Per bucket: summed log-prob (nats) of the reader tokens in that bucket.
    bucket_logp: list[float]
    # Per bucket: how many READER tokens were scored (diagnostic; the reported
    # per-token normalization uses TARGET token counts, which are fixed).
    bucket_ntok: list[int]
    n_prefix_tokens: int = 0


@dataclass
class FrozenReader:
    """A frozen reader LM. Never trained, never receives gradients."""

    model_name: str
    model: object
    tokenizer: object
    device: str = "cuda"
    templates: ReaderTemplates = field(default_factory=ReaderTemplates)
    max_batch_tokens: int = 65536
    max_batch_rows: int = 64
    # caches (bounded by callers reusing the same explanations/continuations)
    _prefix_cache: dict = field(default_factory=dict, repr=False)
    _cont_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        templates: ReaderTemplates | None = None,
        attn_implementation: str = "sdpa",
        **kw,
    ) -> "FrozenReader":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(
            f"[reader] {model_name} -> {type(model).__name__} "
            f"dtype={dtype} device={device} vocab={len(tok)}",
            flush=True,
        )
        return cls(
            model_name=model_name, model=model, tokenizer=tok, device=device,
            templates=templates or ReaderTemplates(), **kw,
        )

    # ---- tokenization ------------------------------------------------------

    def prefix_ids(self, explanation: str | None) -> list[int]:
        """Token ids for the framing (+ explanation). Special tokens ON, so
        Gemma gets its required <bos> and Qwen gets whatever its config says."""
        key = explanation if explanation is not None else "\x00NONE"
        hit = self._prefix_cache.get(key)
        if hit is None:
            text = self.templates.render(explanation)
            hit = self.tokenizer(text, add_special_tokens=True)["input_ids"]
            if len(self._prefix_cache) < 100_000:
                self._prefix_cache[key] = hit
        return hit

    def cont_tokens(self, cont_text: str) -> tuple[list[int], list[int]]:
        """(ids, char_start_per_id) for the continuation, tokenized ALONE.

        Special tokens OFF - the continuation is a suffix, not a fresh sequence.
        """
        hit = self._cont_cache.get(cont_text)
        if hit is not None:
            return hit
        enc = self.tokenizer(
            cont_text, add_special_tokens=False, return_offsets_mapping=True,
        )
        ids = enc["input_ids"]
        offs = enc.get("offset_mapping")
        if offs is None:  # slow tokenizer fallback: cumulative decode
            bounds = token_char_bounds(self.tokenizer, ids)
            starts = bounds[:-1]
        else:
            starts = [int(s) for s, _ in offs]
        hit = (ids, starts)
        if len(self._cont_cache) < 200_000:
            self._cont_cache[cont_text] = hit
        return hit

    # ---- scoring -----------------------------------------------------------

    @torch.no_grad()
    def score(self, jobs: list[ScoreJob], buckets=DEFAULT_BUCKETS) -> list[ScoreResult]:
        """Teacher-forced log-probs of each job's continuation, bucketed.

        Length-sorted batching keeps padding waste low; results come back in the
        caller's original order.
        """
        if not jobs:
            return []
        nb = len(buckets)
        prepared = []
        for i, job in enumerate(jobs):
            p_ids = self.prefix_ids(job.explanation)
            c_ids, c_starts = self.cont_tokens(job.cont_text)
            assert len(job.char_ranges) == nb, (
                f"job has {len(job.char_ranges)} char ranges, expected {nb}"
            )
            # Reader token -> bucket, by the token's first character.
            bucket_of = []
            for s in c_starts:
                b = -1
                for bi, (lo, hi) in enumerate(job.char_ranges):
                    if lo <= s < hi:
                        b = bi
                        break
                bucket_of.append(b)
            prepared.append((i, p_ids, c_ids, bucket_of))
        results: list[ScoreResult | None] = [None] * len(jobs)
        order = sorted(range(len(prepared)), key=lambda k: len(prepared[k][1]) + len(prepared[k][2]))
        batch: list[int] = []
        batch_max = 0

        def flush(batch, batch_max):
            if not batch:
                return
            rows = [prepared[k] for k in batch]
            bs = len(rows)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id or 0
            ids = torch.full((bs, batch_max), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((bs, batch_max), dtype=torch.long, device=self.device)
            for r, (_, p_ids, c_ids, _) in enumerate(rows):
                seq = p_ids + c_ids
                ids[r, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attn[r, : len(seq)] = 1
            # Only the continuation positions are ever read, and the vocab is
            # large (151k for Qwen, 262k for Gemma), so a full [B, L, V] logits
            # tensor is several GB of pure waste. Ask for just the window that
            # covers every row's continuation. Positions needed by row i are
            # [np_i - 1, np_i + nc_i - 1], so the window starts at min(np_) - 1.
            min_np = min(len(r[1]) for r in rows)
            keep = batch_max - min_np + 1
            try:
                out = self.model(input_ids=ids, attention_mask=attn,
                                 logits_to_keep=keep)
                offset = batch_max - out.logits.shape[1]
            except TypeError:
                # Older/unusual forward signatures: fall back to full logits.
                out = self.model(input_ids=ids, attention_mask=attn)
                offset = 0
            logits = out.logits
            for r, (job_i, p_ids, c_ids, bucket_of) in enumerate(rows):
                np_, nc = len(p_ids), len(c_ids)
                # logits[j-1] predicts token j; continuation tokens live at
                # absolute positions np_ .. np_+nc-1, shifted by the kept window.
                pred_idx = torch.arange(np_ - 1 - offset, np_ + nc - 1 - offset,
                                        device=self.device)
                assert int(pred_idx[0]) >= 0, (
                    f"logits window too small: need absolute {np_ - 1}, window "
                    f"starts at {offset}")
                sel = logits[r].index_select(0, pred_idx).float()      # [nc, V]
                tgt = torch.tensor(c_ids, dtype=torch.long, device=self.device)
                lse = torch.logsumexp(sel, dim=-1)
                lp = sel.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse   # [nc]
                lp_l = lp.tolist()
                b_logp = [0.0] * nb
                b_ntok = [0] * nb
                for t, b in enumerate(bucket_of):
                    if b < 0:
                        continue
                    v = lp_l[t]
                    if not math.isfinite(v):
                        v = -20.0  # a reader assigning ~0 probability; keep the sum finite
                    b_logp[b] += v
                    b_ntok[b] += 1
                results[job_i] = ScoreResult(
                    key=jobs[job_i].key, bucket_logp=b_logp, bucket_ntok=b_ntok,
                    n_prefix_tokens=np_,
                )
            del logits, out

        for k in order:
            seq_len = len(prepared[k][1]) + len(prepared[k][2])
            new_max = max(batch_max, seq_len)
            if batch and (
                new_max * (len(batch) + 1) > self.max_batch_tokens
                or len(batch) >= self.max_batch_rows
            ):
                flush(batch, batch_max)
                batch, batch_max = [], 0
                new_max = seq_len
            batch.append(k)
            batch_max = new_max
        flush(batch, batch_max)
        assert all(r is not None for r in results), "internal: unscored job"
        return results  # type: ignore[return-value]
