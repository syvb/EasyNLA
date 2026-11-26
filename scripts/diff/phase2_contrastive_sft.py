"""Phase 2b — Contrastive (Arms 1 & 2) & Single-Model (Arm 0) SFT Gold Explanations.

Generates 30-80 word gold explanations for the 20,000 SFT warm-start rows using
Claude Sonnet 4.6 via the Anthropic Message Batches API (50% discount) — or sync
API / offline deterministic mock provider for CPU tests.

Features:
  - Exact contrastive teacher prompt from §3.2 item 3 (shows prompt prefix ending
    at position t, role, Base top-5 next tokens + short continuation, and Instruct
    top-5 next tokens + short continuation; asks for 2-3 concise bullet points
    describing the post-training shift inside `<explanation>...</explanation>`).
  - Cross-arm explanation cache (`--cache-from`): Arm 1 reuses Arm 2's contrastive
    explanations for all overlapping `(doc_id, position)` rows so you never pay twice.
  - Crash-safe per-chunk checkpointing (`<output>.chunks/chunk_NNNNNNNN.parquet`).
  - Strict `<explanation>...</explanation>` extraction (truncations/refusals dropped).

Example (GPU / Cloud):
    # Arm 2 (and cache for Arm 1):
    python scripts/diff/phase2_contrastive_sft.py \
        --input /data/diff_nla/corpus/arm2/base_sft.parquet \
        --output /data/diff_nla/corpus/arm2/sft_explained.parquet \
        --mode contrastive --provider batch --model claude-sonnet-4-6

    # Arm 1 (reuses overlapping rows from Arm 2 cache):
    python scripts/diff/phase2_contrastive_sft.py \
        --input /data/diff_nla/corpus/arm1/base_sft.parquet \
        --output /data/diff_nla/corpus/arm1/sft_explained.parquet \
        --mode contrastive --provider batch --model claude-sonnet-4-6 \
        --cache-from /data/diff_nla/corpus/arm2/sft_explained.parquet

    # Arm 0 (matched single-model chat SFT; skip if using asher577/easynla-warmstart-data):
    python scripts/diff/phase2_contrastive_sft.py \
        --input /data/diff_nla/corpus/arm0/base_sft.parquet \
        --output /data/diff_nla/corpus/arm0/sft_explained.parquet \
        --mode standard --provider batch --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nla.datagen.providers import AnthropicProvider, CompletionProvider
from nla.diff.resumable_batch import ResumableBatchAnthropicProvider
from nla.schema import extract_explanation

CONTRASTIVE_PROMPT_TEMPLATE = """You are analyzing the post-training residual-stream activation difference (Instruct minus Base) of Qwen3-8B at a single token position in a conversation.

Token role: {role}

Prompt prefix ending at this exact token (tail shown if long):
<prefix>
{prefix_tail}
</prefix>

Base model (Qwen3-8B-Base) next-token behavior at this position:
- Top-5 next tokens: {base_top5}
- Short greedy continuation: {base_cont}

Instruct model (Qwen3-8B) next-token behavior at this position:
- Top-5 next tokens: {inst_top5}
- Short greedy continuation: {inst_cont}

Describe in 2–3 concise bullet points (30–80 words total) what post-training cognitive, behavioral, persona, safety, or conversational shift separates the Instruct model's internal state from the Base model at this exact token. Do NOT summarize or quote the prompt text unless necessary to explain the behavioral shift. Wrap your response in <explanation>...</explanation>."""

STANDARD_PROMPT_TEMPLATE = """You are analyzing the internal residual-stream activation of Qwen3-8B at a single token position in a conversation.

Token role: {role}

Prompt prefix ending at this exact token (tail shown if long):
<prefix>
{prefix_tail}
</prefix>

Model (Qwen3-8B) next-token behavior at this position:
- Top-5 next tokens: {inst_top5}
- Short greedy continuation: {inst_cont}

Describe in 2–3 concise bullet points (30–80 words total) the most important semantic, conversational, behavioral, and next-token features represented in the model's internal state at this exact token. Wrap your response in <explanation>...</explanation>."""


class MockExplanationProvider(CompletionProvider):
    """Deterministic offline provider for CPU tests (zero API cost)."""

    def __init__(self, mode: str = "contrastive"):
        self.mode = mode
        self.model = f"mock-{mode}"
        self.max_tokens = 200
        self.temperature = 0.0

    def complete(self, prompts: list[str]) -> list[str | None]:
        out: list[str | None] = []
        for i, p in enumerate(prompts):
            role = "unknown"
            for line in p.splitlines():
                if line.startswith("Token role:"):
                    role = line.split(":", 1)[1].strip()
                    break
            if self.mode == "contrastive":
                txt = (
                    f"<explanation>\n"
                    f"- Shifts from base text-continuation prior to structured assistant-persona framing at role {role}.\n"
                    f"- Modulates helpfulness, safety-boundary checking, and concise conversational tone (sample {i}).\n"
                    f"</explanation>"
                )
            else:
                txt = (
                    f"<explanation>\n"
                    f"- Tracks active conversation context and topical constraints at role {role}.\n"
                    f"- Prepares coherent next-token continuation matching user intent (sample {i}).\n"
                    f"</explanation>"
                )
            out.append(txt)
        return out


def _build_prompt(row: dict[str, Any], mode: str, max_prefix_chars: int = 1800) -> str:
    prefix = str(row["detokenized_text_truncated"] or "")
    if len(prefix) > max_prefix_chars:
        prefix = "... " + prefix[-max_prefix_chars:]
    tmpl = CONTRASTIVE_PROMPT_TEMPLATE if mode == "contrastive" else STANDARD_PROMPT_TEMPLATE
    return tmpl.format(
        role=row["role"],
        prefix_tail=prefix,
        base_top5=row["base_top5"] or "(unavailable)",
        base_cont=repr((row["base_cont"] or "")[:200]),
        inst_top5=row["inst_top5"] or "(unavailable)",
        inst_cont=repr((row["inst_cont"] or "")[:200]),
    )


def _load_pos_cache(cache_paths: list[str]) -> dict[tuple[str, int], str]:
    cache: dict[tuple[str, int], str] = {}
    for cp in cache_paths:
        p = Path(cp)
        if not p.exists():
            continue
        tbl = pq.read_table(p, columns=["doc_id", "position", "api_explanation"])
        for did, pos, expl in zip(
            tbl.column("doc_id").to_pylist(),
            tbl.column("position").to_pylist(),
            tbl.column("api_explanation").to_pylist(),
        ):
            if expl:
                cache[(str(did), int(pos))] = str(expl)
    return cache


def _make_provider(args: argparse.Namespace, *, batch_state_dir: Path | None = None) -> CompletionProvider:
    if args.provider == "mock":
        return MockExplanationProvider(mode=args.mode)
    if args.provider == "batch":
        return ResumableBatchAnthropicProvider(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_batch_size=args.max_batch_size,
            poll_interval_s=args.poll_interval_s,
            state_dir=batch_state_dir,
        )
    if args.provider == "sync":
        return AnthropicProvider(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
    raise ValueError(f"Unknown provider: {args.provider!r}")


def run_contrastive_sft(args: argparse.Namespace) -> Path:
    in_path = Path(args.input)
    out_path = Path(args.output)
    assert in_path.exists(), f"Missing input parquet: {in_path}"
    if out_path.exists() and not args.overwrite:
        print(f"[phase2b] {out_path} already exists (pass --overwrite to regenerate)", flush=True)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(in_path)
    out_schema = table.schema.append(pa.field("api_explanation", pa.string()))
    cache = _load_pos_cache(args.cache_from) if args.cache_from else {}
    if cache:
        print(f"[phase2b] loaded {len(cache)} cached explanations from {args.cache_from}", flush=True)

    chunks_dir = Path(f"{out_path}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    provider = _make_provider(args, batch_state_dir=chunks_dir / "_batch_state")

    chunk_starts = list(range(0, table.num_rows, args.chunk_size))
    chunk_paths: list[Path] = []
    dropped_total = 0

    for cs in tqdm(chunk_starts, desc="phase2b_chunks"):
        cp = chunks_dir / f"chunk_{cs:08d}.parquet"
        chunk_paths.append(cp)
        if cp.exists() and not args.overwrite:
            continue

        sub = table.slice(cs, args.chunk_size)
        n_sub = sub.num_rows
        dids = sub.column("doc_id").to_pylist()
        positions = sub.column("position").to_pylist()
        roles = sub.column("role").to_pylist()
        texts = sub.column("detokenized_text_truncated").to_pylist()
        b_top5 = sub.column("base_top5").to_pylist()
        i_top5 = sub.column("inst_top5").to_pylist()
        b_cont = sub.column("base_cont").to_pylist()
        i_cont = sub.column("inst_cont").to_pylist()

        cached_expls: list[str | None] = [cache.get((str(d), int(p))) for d, p in zip(dids, positions)]
        miss_idx = [i for i, e in enumerate(cached_expls) if e is None]
        miss_prompts = [
            _build_prompt(
                {
                    "role": roles[i],
                    "detokenized_text_truncated": texts[i],
                    "base_top5": b_top5[i],
                    "inst_top5": i_top5[i],
                    "base_cont": b_cont[i],
                    "inst_cont": i_cont[i],
                },
                mode=args.mode,
            )
            for i in miss_idx
        ]
        raw_out = provider.complete(miss_prompts) if miss_prompts else []
        assert len(raw_out) == len(miss_prompts)

        miss_parsed: dict[int, str | None] = {}
        for idx_row, raw in zip(miss_idx, raw_out):
            if not raw:
                miss_parsed[idx_row] = None
                continue
            expl = extract_explanation(raw)
            miss_parsed[idx_row] = expl if (expl and len(expl) >= 10) else None

        keep_mask: list[bool] = []
        expls_kept: list[str] = []
        for i in range(n_sub):
            e = cached_expls[i] if cached_expls[i] is not None else miss_parsed[i]
            if e is None:
                keep_mask.append(False)
                dropped_total += 1
            else:
                keep_mask.append(True)
                expls_kept.append(e)

        if not all(keep_mask):
            sub = sub.filter(pa.array(keep_mask, type=pa.bool_()))
        chunk_out = sub.append_column("api_explanation", pa.array(expls_kept, type=pa.string()))
        tmp = cp.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.replace(cp)

    total_rows = sum(pq.read_metadata(cp).num_rows for cp in chunk_paths)
    assert total_rows > 0, "All rows dropped during Phase 2b explanation extraction!"

    tmp_out = out_path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(pa.OSFile(str(tmp_out), "wb"), out_schema) as w:
        for cp in chunk_paths:
            t_c = pq.read_table(cp)
            if t_c.num_rows > 0:
                w.write_table(t_c)
    tmp_out.replace(out_path)
    print(f"[phase2b] wrote {total_rows} explained rows -> {out_path} (dropped={dropped_total})", flush=True)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="base_sft.parquet from phase2_build_corpus.py")
    p.add_argument("--output", required=True, help="Output sft_explained.parquet")
    p.add_argument("--mode", choices=["contrastive", "standard"], default="contrastive")
    p.add_argument("--provider", choices=["batch", "sync", "mock"], default="batch")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--max-batch-size", type=int, default=10000)
    p.add_argument("--poll-interval-s", type=float, default=30.0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--cache-from", action="append", default=[],
                   help="Existing sft_explained.parquet to reuse explanations for matching (doc_id, position)")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_contrastive_sft(parse_args(argv))


if __name__ == "__main__":
    main()
