"""Phase 2a & 2c — Chat Rollout Generation, Dual-Model L24 Extraction, & Arm Materialization.

Three resumable subcommands (`--step rollouts|extract|materialize|all`):

  1. `rollouts`:
       Samples 60,000 diverse chat prompts (30k WildChat-1M EN + 20k UltraChat-200k +
       10k PKU-SafeRLHF / HH-RLHF), strictly deduplicates against ALL Phase 4 eval
       benchmarks, assigns deterministic doc-level splits (SFT / RL / Val), and
       generates on-policy Qwen3-8B completions (`enable_thinking=False`,
       `max_new_tokens=96`, `temperature=0.7`) via vLLM (or HF fallback for CPU tests).
       Writes `<out_dir>/rollouts.parquet`.

  2. `extract`:
       Reads `phase0_config.json` + Fix 1 alignments (`align_L{K}_id_anchored.pt`,
       `align_L{K}_std_ridge.pt`). Loads Target (Instruct) then Base ONE AT A TIME
       in bounded chunks. Samples 3 active positions/conv (`user_content`, `asst_last`,
       `asst_rollout`) -> 180k rows total (20k SFT, 155k RL, 5k Val).
       For the 20k SFT split, also records Base & Instruct top-5 next tokens and
       short greedy continuations for Phase 2b contrastive SFT explanation generation.
       Fits exact training-split Fix 2 role means (`role_means_L{K}_train.pt`, strictly
       excluding `val`) and writes raw base parquets for:
         - `arm0/`: raw h_target (Fix 2 active roles, tau=0)
         - `arm1/`: naive Delta_std_ridge (includes ChatML template tokens, no centering, tau=0)
         - `arm2/`: cleaned Delta_tilde = Delta_id_anc - mu_role (Fixes 1+2+3, tau=tau_web_median)

  3. `materialize`:
       Combines `base_{rl,val}.parquet` and Phase 2b `sft_explained.parquet` into
       EasyNLA-contract parquets (`av_sft_train.parquet`, `ar_sft_train.parquet`,
       `av_sft_val.parquet`, `rl_shuf.parquet`, `val.parquet`) + verified
       `.nla_meta.yaml` sidecars for each arm.

Example (GPU):
    python scripts/diff/phase2_build_corpus.py --step rollouts \
        --phase0-dir /data/diff_nla/phase0 --output-dir /data/diff_nla/corpus --backend vllm --tp-size 4

    python scripts/diff/phase2_build_corpus.py --step extract \
        --phase0-dir /data/diff_nla/phase0 --output-dir /data/diff_nla/corpus

    # Then run phase2_contrastive_sft.py for arm2, arm1, (and arm0 if not using HF warmstart)

    python scripts/diff/phase2_build_corpus.py --step materialize \
        --phase0-dir /data/diff_nla/phase0 --output-dir /data/diff_nla/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar_local
from nla.datagen.stage3_build import (
    _DEFAULT_ACTOR_TEMPLATE,
    _DEFAULT_CRITIC_TEMPLATE,
    _INJECT_PLACEHOLDER,
    _PROMPT_STRUCT,
)
from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.benchmarks import (
    NEUTRAL_QUERIES,
    TABOO_ELICITATION_PROMPTS,
    flatten_binary_pairs,
    load_eval_awareness_pairs,
    load_historical_period_items,
    load_math_items,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.model_io import (
    extract_at_positions,
    load_causal_lm,
    resolve_device,
    resolve_dtype,
    unload_model,
)
from nla.diff.wrappers import (
    ACTIVE_ROLES,
    apply_role_centering,
    encode_aligned_pair,
    encode_chatml_dialogue,
    map_inst_pos_to_base_pos,
    save_role_means,
)
from nla.schema import wrap_explanation


def _norm_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _collect_eval_blocklist(source: str = "auto") -> set[str]:
    """Normalized strings from ALL Phase 4 eval benchmarks to prevent train-eval leakage."""
    blocked: set[str] = set()
    for q in NEUTRAL_QUERIES:
        blocked.add(_norm_text(q))
    for q in TABOO_ELICITATION_PROMPTS:
        blocked.add(_norm_text(q))
    # Always blocklist built-in sets AND (when source=='auto') the full HF eval sets (XSTest/JBB + MATH-500)
    ref_sources = ("builtin", "auto") if source == "auto" else ("builtin",)
    math_sources = ("builtin", "auto") if source == "auto" else ("builtin",)
    for rs in ref_sources:
        try:
            for p in flatten_binary_pairs(load_refusal_pairs(250, source=rs))[0]:
                blocked.add(_norm_text(p))
        except Exception as exc:
            print(f"[blocklist] skipping refusal source={rs}: {exc}", flush=True)
    for loader in (
        lambda: flatten_binary_pairs(load_eval_awareness_pairs(100))[0],
        lambda: flatten_binary_pairs(load_sycophancy_pairs(100))[0],
    ):
        for p in loader():
            blocked.add(_norm_text(p))
    for it in load_historical_period_items(300):
        blocked.add(_norm_text(it["text"]))
    for ms in math_sources:
        try:
            for it in load_math_items(500, source=ms):
                blocked.add(_norm_text(it["problem"]))
        except Exception as exc:
            print(f"[blocklist] skipping math source={ms}: {exc}", flush=True)
    return blocked


def _sample_chat_prompts(
    *,
    n_wildchat: int,
    n_ultrachat: int,
    n_safety: int,
    custom_jsonl: str | None,
    seed: int,
    blocklist_source: str = "auto",
) -> list[dict[str, Any]]:
    """Sample diverse single/multi-turn prompts, deduplicated against eval benchmarks."""
    blocked = _collect_eval_blocklist(source=blocklist_source)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []

    def _try_add(source: str, user_text: str, history: list[dict[str, str]]) -> bool:
        ut = (user_text or "").strip()
        if len(ut) < 8 or len(ut) > 4000:
            return False
        key = _norm_text(ut)
        if key in blocked or key in seen:
            return False
        seen.add(key)
        doc_hash = hashlib.sha1(f"{source}:{len(records)}:{ut[:200]}".encode()).hexdigest()[:12]
        records.append({
            "doc_id": f"{source}_{doc_hash}",
            "source_corpus": source,
            "history_json": json.dumps(history, ensure_ascii=False),
            "user_text": ut,
        })
        return True

    if custom_jsonl is not None:
        raw = [
            json.loads(line)
            for line in Path(custom_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for i, row in enumerate(raw):
            ut = row.get("user_text") or row.get("prompt") or ""
            hist = row.get("history") or []
            _try_add(str(row.get("source_corpus", "custom")), ut, hist)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(records))
        return [records[int(i)] for i in idx]

    from datasets import load_dataset

    # 1. WildChat-1M (English filtered, streaming)
    if n_wildchat > 0:
        print(f"[rollouts] streaming up to {n_wildchat} prompts from allenai/WildChat-1M...", flush=True)
        added = 0
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for row in ds:
            if added >= n_wildchat:
                break
            lang = str(row.get("language", "")).lower()
            if lang and lang != "english":
                continue
            conv = row.get("conversation") or []
            # Keep 1-turn or 2-turn prefixes ending in user
            clean_turns = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", "")).strip()}
                for m in conv
                if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
            ]
            if not clean_turns or clean_turns[0]["role"] != "user":
                continue
            if len(clean_turns) >= 3 and clean_turns[2]["role"] == "user":
                hist = clean_turns[:2]
                ut = clean_turns[2]["content"]
            else:
                hist = []
                ut = clean_turns[0]["content"]
            if _try_add("wildchat", ut, hist):
                added += 1
        print(f"[rollouts] collected {added} WildChat prompts", flush=True)

    # 2. UltraChat-200k (streaming)
    if n_ultrachat > 0:
        print(f"[rollouts] streaming up to {n_ultrachat} prompts from HuggingFaceH4/ultrachat_200k...", flush=True)
        added = 0
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for row in ds:
            if added >= n_ultrachat:
                break
            msgs = row.get("messages") or []
            clean_turns = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", "")).strip()}
                for m in msgs
                if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
            ]
            if not clean_turns or clean_turns[0]["role"] != "user":
                continue
            if len(clean_turns) >= 3 and clean_turns[2]["role"] == "user" and (added % 3 == 0):
                hist = clean_turns[:2]
                ut = clean_turns[2]["content"]
            else:
                hist = []
                ut = clean_turns[0]["content"]
            if _try_add("ultrachat", ut, hist):
                added += 1
        print(f"[rollouts] collected {added} UltraChat prompts", flush=True)

    # 3. Safety / refusal / preference mix (PKU-SafeRLHF, fallback Anthropic/hh-rlhf)
    if n_safety > 0:
        print(f"[rollouts] streaming up to {n_safety} prompts from PKU-Alignment/PKU-SafeRLHF...", flush=True)
        added = 0
        try:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True)
            for row in ds:
                if added >= n_safety:
                    break
                ut = str(row.get("prompt", "")).strip()
                if _try_add("pku_saferlhf", ut, []):
                    added += 1
        except Exception as exc:
            print(f"[rollouts] PKU-SafeRLHF unavailable ({exc}); falling back to Anthropic/hh-rlhf", flush=True)
            ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
            for row in ds:
                if added >= n_safety:
                    break
                chosen = str(row.get("chosen", ""))
                # Extract first Human turn
                if "Human:" in chosen and "Assistant:" in chosen:
                    ut = chosen.split("Human:", 1)[1].split("Assistant:", 1)[0].strip()
                    if _try_add("hh_rlhf", ut, []):
                        added += 1
        print(f"[rollouts] collected {added} safety-mix prompts", flush=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(records))
    return [records[int(i)] for i in perm]


def _assign_splits(
    records: list[dict[str, Any]],
    *,
    sft_frac: float,
    val_frac: float,
) -> list[dict[str, Any]]:
    """Assign doc-disjoint splits: sft / val / rl."""
    n = len(records)
    n_sft = max(1, int(round(n * sft_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_sft + n_val >= n:
        n_sft = max(1, n // 3)
        n_val = max(1, n // 6)
    for i, r in enumerate(records):
        if i < n_sft:
            r["split"] = "sft"
        elif i < n_sft + n_val:
            r["split"] = "val"
        else:
            r["split"] = "rl"
    return records


def _generate_rollouts_vllm(
    model_name: str,
    prompt_token_id_lists: list[list[int]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tp_size: int,
    gpu_mem_util: float,
    max_model_len: int,
    seed: int,
) -> tuple[list[str], list[list[int]]]:
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        seed=seed,
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_id_lists]
    outs = llm.generate(prompts, sp)
    texts = [o.outputs[0].text for o in outs]
    tok_ids = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts, tok_ids


@torch.no_grad()
def _generate_rollouts_hf(
    model: Any,
    tokenizer: Any,
    prompt_token_id_lists: list[list[int]],
    *,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> tuple[list[str], list[list[int]]]:
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    texts: list[str] = []
    tok_ids: list[list[int]] = []
    do_sample = temperature > 0.0
    for start in range(0, len(prompt_token_id_lists), batch_size):
        sub = prompt_token_id_lists[start : start + batch_size]
        max_len = max(len(s) for s in sub)
        # Left-pad for batched autoregressive generation
        input_ids = torch.full((len(sub), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(sub), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(sub):
            L = len(seq)
            input_ids[i, max_len - L :] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[i, max_len - L :] = 1
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        out = model.generate(**gen_kwargs)
        for i in range(len(sub)):
            new_ids = out[i, max_len:].cpu().tolist()
            # Trim at first EOS if present
            if tokenizer.eos_token_id in new_ids:
                new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
            tok_ids.append(new_ids)
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return texts, tok_ids


def _format_top5(tok_ids: list[int], probs: list[float], tokenizer: Any) -> str:
    parts = []
    for tid, p in zip(tok_ids, probs):
        tok_s = repr(tokenizer.decode([tid]))
        parts.append(f"{tok_s} ({p:.3f})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Step 1: Rollouts
# ---------------------------------------------------------------------------

def run_step_rollouts(args: argparse.Namespace, p0_cfg: dict[str, Any]) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.parquet"
    if rollouts_path.exists() and not args.overwrite:
        print(f"[rollouts] {rollouts_path} already exists (pass --overwrite to regenerate)", flush=True)
        return rollouts_path

    target_model = p0_cfg["target_model"]
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _sample_chat_prompts(
        n_wildchat=args.n_wildchat,
        n_ultrachat=args.n_ultrachat,
        n_safety=args.n_safety,
        custom_jsonl=args.custom_prompts_jsonl,
        seed=args.seed,
        blocklist_source=args.eval_blocklist_source,
    )

    kept_records: list[dict[str, Any]] = []
    prefill_id_lists: list[list[int]] = []
    prompt_user_id_lists: list[list[int]] = []
    kept_hist_jsons: list[str] = []
    skipped_too_long = 0
    for r in records:
        hist = json.loads(r["history_json"])
        try:
            enc = encode_chatml_dialogue(
                tokenizer,
                r["user_text"],
                history=hist,
                enable_thinking=False,
                max_total_tokens=args.max_prompt_tokens,
                truncate_history=True,
            )
        except ValueError:
            skipped_too_long += 1
            continue
        u_ids = enc.input_ids[enc.user_span[0] : enc.user_span[1]]
        if not u_ids:
            skipped_too_long += 1
            continue
        kept_records.append(r)
        prefill_id_lists.append(enc.input_ids)
        prompt_user_id_lists.append(u_ids)
        kept_hist_jsons.append(json.dumps(hist[enc.kept_history_start_idx:], ensure_ascii=False))

    if skipped_too_long:
        print(f"[rollouts] skipped {skipped_too_long} conversation(s) that could not fit max_prompt_tokens={args.max_prompt_tokens}", flush=True)
    records = _assign_splits(kept_records, sft_frac=args.sft_frac, val_frac=args.val_frac)
    print(
        f"[rollouts] total={len(records)} "
        f"(sft={sum(r['split']=='sft' for r in records)}, "
        f"rl={sum(r['split']=='rl' for r in records)}, "
        f"val={sum(r['split']=='val' for r in records)})",
        flush=True,
    )

    t0 = time.time()
    if args.backend == "vllm":
        asst_texts, asst_tok_ids = _generate_rollouts_vllm(
            target_model,
            prefill_id_lists,
            max_new_tokens=args.rollout_max_new_tokens,
            temperature=args.rollout_temperature,
            top_p=0.95,
            tp_size=args.tp_size,
            gpu_mem_util=args.vllm_gpu_mem,
            max_model_len=args.max_prompt_tokens + args.rollout_max_new_tokens + 64,
            seed=args.seed,
        )
    else:
        device = resolve_device(args.device)
        dtype = resolve_dtype(args.dtype, device)
        model = load_causal_lm(target_model, dtype=dtype, device=device)
        asst_texts, asst_tok_ids = _generate_rollouts_hf(
            model,
            tokenizer,
            prefill_id_lists,
            max_new_tokens=args.rollout_max_new_tokens,
            temperature=args.rollout_temperature,
            batch_size=args.batch_size,
        )
        unload_model(model)

    tbl = pa.table({
        "doc_id": pa.array([r["doc_id"] for r in records], type=pa.string()),
        "split": pa.array([r["split"] for r in records], type=pa.string()),
        "source_corpus": pa.array([r["source_corpus"] for r in records], type=pa.string()),
        "history_json": pa.array(kept_hist_jsons, type=pa.string()),
        "user_text": pa.array([r["user_text"] for r in records], type=pa.string()),
        "prompt_token_ids": pa.array(prefill_id_lists, type=pa.list_(pa.int64())),
        "prompt_user_ids": pa.array(prompt_user_id_lists, type=pa.list_(pa.int64())),
        "assistant_text": pa.array(asst_texts, type=pa.string()),
        "rollout_token_ids": pa.array(asst_tok_ids, type=pa.list_(pa.int64())),
    })
    tmp = rollouts_path.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp)
    tmp.replace(rollouts_path)
    print(f"[rollouts] wrote {len(records)} rollouts -> {rollouts_path} in {time.time() - t0:.1f}s", flush=True)
    return rollouts_path


# ---------------------------------------------------------------------------
# Step 2: Dual-Model L24 Extraction -> Arm 0 / Arm 1 / Arm 2 base parquets
# ---------------------------------------------------------------------------

def _base_schema(d_model: int) -> pa.Schema:
    return pa.schema([
        ("doc_id", pa.string()),
        ("split", pa.string()),
        ("role", pa.string()),
        ("position", pa.int64()),
        ("token_id", pa.int64()),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("detokenized_text_truncated", pa.string()),
        ("base_top5", pa.string()),
        ("inst_top5", pa.string()),
        ("base_cont", pa.string()),
        ("inst_cont", pa.string()),
    ])


def _sample_positions_for_conv(
    inst_enc: Any,
    rng: np.random.Generator,
    *,
    include_template_pos: bool,
) -> tuple[list[int], list[str]]:
    """Sample 3 active positions (user_content, asst_last, asst_rollout) + optional 1 template pos."""
    positions: list[int] = []
    roles: list[str] = []

    u_pos = inst_enc.user_content_positions
    if u_pos:
        positions.append(int(rng.choice(u_pos)))
        roles.append("user_content")

    positions.append(int(inst_enc.asst_last_idx))
    roles.append("asst_last")

    r_pos = inst_enc.asst_rollout_positions
    if r_pos:
        positions.append(int(rng.choice(r_pos)))
        roles.append("asst_rollout")
    elif len(u_pos) >= 2:
        # Fallback if model generated empty completion
        other_u = [p for p in u_pos if p not in positions]
        if other_u:
            positions.append(int(rng.choice(other_u)))
            roles.append("user_content")

    if include_template_pos:
        t_pool = inst_enc.sys_prefix_positions + inst_enc.asst_header_positions
        if t_pool:
            tp = int(rng.choice(t_pool))
            positions.append(tp)
            roles.append(inst_enc.role_at(tp))

    return positions, roles


def run_step_extract(args: argparse.Namespace, p0_cfg: dict[str, Any]) -> None:
    out_dir = Path(args.output_dir)
    rollouts_path = out_dir / "rollouts.parquet"
    assert rollouts_path.exists(), f"Missing {rollouts_path} — run --step rollouts first"

    layer_idx = int(p0_cfg["layer_index"])
    d_model = int(p0_cfg["d_model"])
    base_wrapper = str(p0_cfg["base_wrapper"])
    base_model_name = str(p0_cfg["base_model"])
    target_model_name = str(p0_cfg["target_model"])
    tau_med = float(p0_cfg["tau_web_median"])

    art_idanc = load_alignment(p0_cfg["align_id_anchored_path"])
    art_std = load_alignment(p0_cfg["align_std_ridge_path"])

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tbl = pq.read_table(rollouts_path)
    n_convs = tbl.num_rows
    doc_ids = tbl.column("doc_id").to_pylist()
    splits = tbl.column("split").to_pylist()
    hist_jsons = tbl.column("history_json").to_pylist()
    user_texts = tbl.column("user_text").to_pylist()
    roll_ids_all = tbl.column("rollout_token_ids").to_pylist()
    prompt_toks_all = (
        tbl.column("prompt_token_ids").to_pylist()
        if "prompt_token_ids" in tbl.schema.names
        else [None] * n_convs
    )
    prompt_user_ids_all = (
        tbl.column("prompt_user_ids").to_pylist()
        if "prompt_user_ids" in tbl.schema.names
        else [None] * n_convs
    )

    rng = np.random.default_rng(args.seed)
    chunk_convs = max(1, args.chunk_convs)
    chunks_dir = out_dir / "_extract_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Precompute deterministic encodings & sampled positions for all conversations
    print(f"[extract] encoding {n_convs} conversations (base_wrapper={base_wrapper})...", flush=True)
    conv_plans: list[dict[str, Any]] = []
    for i in range(n_convs):
        hist = json.loads(hist_jsons[i])
        p_user_ids = prompt_user_ids_all[i]
        p_tok_ids = prompt_toks_all[i]
        inst_enc, base_win_enc = encode_aligned_pair(
            tokenizer,
            user_texts[i],
            base_wrapper=base_wrapper,
            history=hist,
            user_ids=p_user_ids,
            rollout_ids=roll_ids_all[i],
            enable_thinking=False,
            max_total_tokens=(
                None
                if p_user_ids is not None
                else args.max_prompt_tokens + args.rollout_max_new_tokens + 64
            ),
            truncate_history=(p_user_ids is None),
        )
        if p_tok_ids is not None:
            assert inst_enc.input_ids[: len(p_tok_ids)] == list(p_tok_ids), (
                f"Teacher-forcing prompt prefix diverged from rollout prompt for doc_id={doc_ids[i]}!"
            )
        pos_list, role_list = _sample_positions_for_conv(inst_enc, rng, include_template_pos=True)
        # Active positions (always first <=3) vs optional 4th template position (Arm 1 training only)
        active_pos = [p for p, r in zip(pos_list, role_list) if r in ACTIVE_ROLES]
        active_roles = [r for r in role_list if r in ACTIVE_ROLES]
        tmpl_pos = [p for p, r in zip(pos_list, role_list) if r not in ACTIVE_ROLES]
        tmpl_roles = [r for r in role_list if r not in ACTIVE_ROLES]

        base_win_active_pos = [
            map_inst_pos_to_base_pos(p, inst_enc, base_win_enc) for p in active_pos
        ]
        # Arm 1 naive ablation: swap 1 of the 3 active slots for a static template token on ~50%
        # of TRAINING convs (sft/rl) so Arm 1 trains on unmasked ChatML template tokens while
        # keeping held-out val.parquet on the EXACT same active positions across all arms!
        arm1_inst_pos = list(active_pos)
        arm1_roles = list(active_roles)
        if splits[i] != "val" and tmpl_pos and len(arm1_inst_pos) >= 3 and (i % 2) == 0:
            arm1_inst_pos[0] = tmpl_pos[0]
            arm1_roles[0] = tmpl_roles[0]

        conv_plans.append({
            "doc_id": doc_ids[i],
            "split": splits[i],
            "inst_ids": inst_enc.input_ids,
            "base_win_ids": base_win_enc.input_ids,
            "active_inst_pos": active_pos,
            "active_base_win_pos": base_win_active_pos,
            "active_roles": active_roles,
            "arm1_pos": arm1_inst_pos,
            "arm1_roles": arm1_roles,
            # Union of all Instruct/ChatML positions we need in a single pass
            "union_chatml_pos": sorted(set(active_pos) | set(arm1_inst_pos)),
        })

    # ---- Pass A: Target (Instruct) over chunks (crash-safe) ----
    need_target = [
        cs for cs in range(0, n_convs, chunk_convs)
        if not (chunks_dir / f"target_{cs:07d}.pt").exists() or args.overwrite
    ]
    if need_target:
        print(f"[extract] Pass A (Target: {target_model_name}) on {len(need_target)} chunk(s)...", flush=True)
        target_model = load_causal_lm(target_model_name, dtype=dtype, device=device)
        for cs in tqdm(need_target, desc="target_chunks"):
            sub = conv_plans[cs : cs + chunk_convs]
            seqs = [p["inst_ids"] for p in sub]
            pos_union = [p["union_chatml_pos"] for p in sub]
            # SFT rows need top-5 logits; split sub-chunk into SFT vs non-SFT so RL/Val short-circuit!
            sft_idx = [j for j, p in enumerate(sub) if p["split"] == "sft"]
            other_idx = [j for j, p in enumerate(sub) if p["split"] != "sft"]

            extracted: list[Any] = [None] * len(sub)
            if other_idx:
                res_o = extract_at_positions(
                    target_model,
                    [seqs[j] for j in other_idx],
                    [pos_union[j] for j in other_idx],
                    layer_index=layer_idx, pad_token_id=pad_id,
                    batch_size=args.batch_size, need_logits=False, topk_logits=0,
                )
                for j, ex in zip(other_idx, res_o):
                    extracted[j] = ex
            if sft_idx:
                res_s = extract_at_positions(
                    target_model,
                    [seqs[j] for j in sft_idx],
                    [pos_union[j] for j in sft_idx],
                    layer_index=layer_idx, pad_token_id=pad_id,
                    batch_size=args.batch_size, need_logits=False, topk_logits=5,
                )
                for j, ex in zip(sft_idx, res_s):
                    extracted[j] = ex

            # Short greedy continuations for SFT positions (optional, enabled by default)
            inst_cont_map: dict[tuple[int, int], str] = {}
            if sft_idx and args.sft_cont_tokens > 0:
                cont_prefixes: list[list[int]] = []
                cont_keys: list[tuple[int, int]] = []
                for j in sft_idx:
                    for pos in sub[j]["union_chatml_pos"]:
                        cont_prefixes.append(sub[j]["inst_ids"][: pos + 1])
                        cont_keys.append((j, pos))
                cont_texts, _ = _generate_rollouts_hf(
                    target_model, tokenizer, cont_prefixes,
                    max_new_tokens=args.sft_cont_tokens, temperature=0.0,
                    batch_size=args.sft_cont_batch_size,
                )
                for key, ct in zip(cont_keys, cont_texts):
                    inst_cont_map[key] = ct

            payload = []
            for j, ex in enumerate(extracted):
                top5_by_pos: dict[int, str] = {}
                if ex.topk_ids is not None and ex.topk_probs is not None:
                    for pos, tids, tps in zip(ex.positions, ex.topk_ids, ex.topk_probs):
                        top5_by_pos[pos] = _format_top5(tids, tps, tokenizer)
                payload.append({
                    "positions": ex.positions,
                    "hidden": ex.hidden,  # [k, d] fp32 CPU
                    "top5_by_pos": top5_by_pos,
                    "cont_by_pos": {pos: inst_cont_map.get((j, pos), "") for pos in ex.positions},
                })
            tmp = chunks_dir / f"target_{cs:07d}.pt.tmp"
            torch.save(payload, str(tmp))
            tmp.replace(chunks_dir / f"target_{cs:07d}.pt")
        unload_model(target_model)

    # ---- Pass B: Base model over chunks (crash-safe) ----
    need_base = [
        cs for cs in range(0, n_convs, chunk_convs)
        if not (chunks_dir / f"base_{cs:07d}.pt").exists() or args.overwrite
    ]
    if need_base:
        print(f"[extract] Pass B (Base: {base_model_name}) on {len(need_base)} chunk(s)...", flush=True)
        base_model = load_causal_lm(base_model_name, dtype=dtype, device=device)
        for cs in tqdm(need_base, desc="base_chunks"):
            sub = conv_plans[cs : cs + chunk_convs]
            sft_idx = [j for j, p in enumerate(sub) if p["split"] == "sft"]
            other_idx = [j for j, p in enumerate(sub) if p["split"] != "sft"]

            # 1) ChatML pass (needed for Arm 1 always, and also covers Arm 2 when base_wrapper == "chatml")
            chatml_seqs = [p["inst_ids"] for p in sub]
            chatml_pos = [p["union_chatml_pos"] for p in sub]
            ex_chatml: list[Any] = [None] * len(sub)
            if other_idx:
                for j, ex in zip(
                    other_idx,
                    extract_at_positions(
                        base_model,
                        [chatml_seqs[j] for j in other_idx],
                        [chatml_pos[j] for j in other_idx],
                        layer_index=layer_idx, pad_token_id=pad_id,
                        batch_size=args.batch_size,
                    ),
                ):
                    ex_chatml[j] = ex
            if sft_idx:
                for j, ex in zip(
                    sft_idx,
                    extract_at_positions(
                        base_model,
                        [chatml_seqs[j] for j in sft_idx],
                        [chatml_pos[j] for j in sft_idx],
                        layer_index=layer_idx, pad_token_id=pad_id,
                        batch_size=args.batch_size, topk_logits=5,
                    ),
                ):
                    ex_chatml[j] = ex

            # 2) If winning wrapper is "plain", also extract Base(Plain) at active_base_win_pos
            ex_plain: list[Any] | None = None
            if base_wrapper == "plain":
                ex_plain = [None] * len(sub)
                plain_seqs = [p["base_win_ids"] for p in sub]
                plain_pos = [p["active_base_win_pos"] for p in sub]
                if other_idx:
                    for j, ex in zip(
                        other_idx,
                        extract_at_positions(
                            base_model,
                            [plain_seqs[j] for j in other_idx],
                            [plain_pos[j] for j in other_idx],
                            layer_index=layer_idx, pad_token_id=pad_id,
                            batch_size=args.batch_size,
                        ),
                    ):
                        ex_plain[j] = ex
                if sft_idx:
                    for j, ex in zip(
                        sft_idx,
                        extract_at_positions(
                            base_model,
                            [plain_seqs[j] for j in sft_idx],
                            [plain_pos[j] for j in sft_idx],
                            layer_index=layer_idx, pad_token_id=pad_id,
                            batch_size=args.batch_size, topk_logits=5,
                        ),
                    ):
                        ex_plain[j] = ex

            # Short greedy Base continuations for SFT split (under winning wrapper for Arm 2, and ChatML for Arm 1 template pos)
            base_cont_win: dict[tuple[int, int], str] = {}
            base_cont_chatml: dict[tuple[int, int], str] = {}
            if sft_idx and args.sft_cont_tokens > 0:
                c_pref, c_keys = [], []
                for j in sft_idx:
                    for pos in sub[j]["union_chatml_pos"]:
                        c_pref.append(sub[j]["inst_ids"][: pos + 1])
                        c_keys.append((j, pos))
                c_txts, _ = _generate_rollouts_hf(
                    base_model, tokenizer, c_pref,
                    max_new_tokens=args.sft_cont_tokens, temperature=0.0,
                    batch_size=args.sft_cont_batch_size,
                )
                for key, ct in zip(c_keys, c_txts):
                    base_cont_chatml[key] = ct
                if base_wrapper == "plain":
                    p_pref, p_keys = [], []
                    for j in sft_idx:
                        for inst_p, bp in zip(sub[j]["active_inst_pos"], sub[j]["active_base_win_pos"]):
                            p_pref.append(sub[j]["base_win_ids"][: bp + 1])
                            p_keys.append((j, inst_p))
                    p_txts, _ = _generate_rollouts_hf(
                        base_model, tokenizer, p_pref,
                        max_new_tokens=args.sft_cont_tokens, temperature=0.0,
                        batch_size=args.sft_cont_batch_size,
                    )
                    for key, ct in zip(p_keys, p_txts):
                        base_cont_win[key] = ct
                else:
                    base_cont_win = base_cont_chatml

            payload = []
            for j in range(len(sub)):
                ec = ex_chatml[j]
                top5_chatml: dict[int, str] = {}
                if ec.topk_ids is not None and ec.topk_probs is not None:
                    for pos, tids, tps in zip(ec.positions, ec.topk_ids, ec.topk_probs):
                        top5_chatml[pos] = _format_top5(tids, tps, tokenizer)
                item: dict[str, Any] = {
                    "chatml_positions": ec.positions,
                    "chatml_hidden": ec.hidden,
                    "top5_chatml": top5_chatml,
                    "cont_chatml": {pos: base_cont_chatml.get((j, pos), "") for pos in ec.positions},
                }
                if ex_plain is not None:
                    ep = ex_plain[j]
                    top5_win: dict[int, str] = {}
                    if ep.topk_ids is not None and ep.topk_probs is not None:
                        for inst_p, tids, tps in zip(sub[j]["active_inst_pos"], ep.topk_ids, ep.topk_probs):
                            top5_win[inst_p] = _format_top5(tids, tps, tokenizer)
                    item["win_hidden"] = ep.hidden  # aligned to active_inst_pos order
                    item["top5_win"] = top5_win
                    item["cont_win"] = {pos: base_cont_win.get((j, pos), "") for pos in sub[j]["active_inst_pos"]}
                payload.append(item)
            tmp = chunks_dir / f"base_{cs:07d}.pt.tmp"
            torch.save(payload, str(tmp))
            tmp.replace(chunks_dir / f"base_{cs:07d}.pt")
        unload_model(base_model)

    # ---- Pass C: Compute exact TRAINING-split Fix 2 role means (sft + rl ONLY, never val!) ----
    print("[extract] Pass C: computing training-split Fix 2 per-role means...", flush=True)
    role_sums: dict[str, torch.Tensor] = {r: torch.zeros(d_model, dtype=torch.float64) for r in ACTIVE_ROLES}
    role_counts: dict[str, int] = {r: 0 for r in ACTIVE_ROLES}

    for cs in range(0, n_convs, chunk_convs):
        sub = conv_plans[cs : cs + chunk_convs]
        t_chunk = torch.load(str(chunks_dir / f"target_{cs:07d}.pt"), weights_only=False)
        b_chunk = torch.load(str(chunks_dir / f"base_{cs:07d}.pt"), weights_only=False)
        ht_by_role: dict[str, list[torch.Tensor]] = {r: [] for r in ACTIVE_ROLES}
        hb_by_role: dict[str, list[torch.Tensor]] = {r: [] for r in ACTIVE_ROLES}
        for plan, td, bd in zip(sub, t_chunk, b_chunk):
            if plan["split"] == "val":
                continue  # NEVER leak held-out val into role means
            t_pos_map = {p: idx for idx, p in enumerate(td["positions"])}
            b_pos_map = {p: idx for idx, p in enumerate(bd["chatml_positions"])}
            for k, (inst_p, role) in enumerate(zip(plan["active_inst_pos"], plan["active_roles"])):
                ht_v = td["hidden"][t_pos_map[inst_p]]
                if base_wrapper == "plain":
                    hb_v = bd["win_hidden"][k]
                else:
                    hb_v = bd["chatml_hidden"][b_pos_map[inst_p]]
                ht_by_role[role].append(ht_v)
                hb_by_role[role].append(hb_v)
        for role in ACTIVE_ROLES:
            if ht_by_role[role]:
                ht_m = torch.stack(ht_by_role[role], dim=0)
                hb_m = torch.stack(hb_by_role[role], dim=0)
                d_m = apply_alignment(hb_m, ht_m, art_idanc)
                role_sums[role] += d_m.double().sum(dim=0)
                role_counts[role] += d_m.shape[0]

    train_role_means: dict[str, torch.Tensor] = {}
    for role in ACTIVE_ROLES:
        if role_counts[role] > 0:
            train_role_means[role] = (role_sums[role] / float(role_counts[role])).float()
        else:
            train_role_means[role] = torch.zeros(d_model, dtype=torch.float32)
    train_rm_path = save_role_means(
        out_dir / f"role_means_L{layer_idx}_train.pt",
        train_role_means,
        tau_web_median=tau_med,
        tau_web_p75=float(p0_cfg.get("tau_web_p75", tau_med)),
        extra={"role_counts": role_counts, "base_wrapper": base_wrapper, "layer_index": layer_idx},
    )
    print(f"[extract] saved training-split role means -> {train_rm_path} (counts={role_counts})", flush=True)

    # ---- Pass D: Stream-write base_{sft,rl,val}.parquet for arm0, arm1, arm2 ----
    schema = _base_schema(d_model)
    writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    row_counts: dict[tuple[str, str], int] = {}
    for arm in ("arm0", "arm1", "arm2"):
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        for sp in ("sft", "rl", "val"):
            p = out_dir / arm / f"base_{sp}.parquet"
            writers[(arm, sp)] = pq.ParquetWriter(pa.OSFile(str(p), "wb"), schema)
            row_counts[(arm, sp)] = 0

    # Track per-arm training-split role means (both raw and unit-sphere, excluding val!)
    # for cross-arm comparable within-role reconstruction metrics (Finding 10),
    # plus Arm 2 Fix 3 norm/tau/weight ESS diagnostics (Finding 7).
    arm_unit_sums: dict[str, dict[str, torch.Tensor]] = {
        a: {r: torch.zeros(d_model, dtype=torch.float64) for r in ACTIVE_ROLES}
        for a in ("arm0", "arm1", "arm2")
    }
    arm_raw_sums: dict[str, dict[str, torch.Tensor]] = {
        a: {r: torch.zeros(d_model, dtype=torch.float64) for r in ACTIVE_ROLES}
        for a in ("arm0", "arm1", "arm2")
    }
    arm_role_counts: dict[str, dict[str, int]] = {
        a: {r: 0 for r in ACTIVE_ROLES} for a in ("arm0", "arm1", "arm2")
    }
    arm2_train_norms_by_role: dict[str, list[np.ndarray]] = {r: [] for r in ACTIVE_ROLES}

    def _flush_rows(arm: str, sp: str, buf: list[dict[str, Any]]) -> None:
        if not buf:
            return
        acts_np = np.stack([r["activation_vector"] for r in buf], axis=0).astype(np.float32, copy=False)
        if sp != "val":
            roles_b = [r["role"] for r in buf]
            acts_f64 = torch.from_numpy(acts_np).double()
            norms_b = acts_f64.norm(dim=-1, keepdim=True)
            unit_b = acts_f64 / (norms_b + 1e-9)
            for rname in ACTIVE_ROLES:
                idxs = [idx for idx, rv in enumerate(roles_b) if rv == rname]
                if idxs:
                    idx_t = torch.tensor(idxs, dtype=torch.long)
                    arm_raw_sums[arm][rname] += acts_f64.index_select(0, idx_t).sum(dim=0)
                    arm_unit_sums[arm][rname] += unit_b.index_select(0, idx_t).sum(dim=0)
                    arm_role_counts[arm][rname] += len(idxs)
                    if arm == "arm2":
                        arm2_train_norms_by_role[rname].append(
                            norms_b.squeeze(-1).index_select(0, idx_t).numpy().astype(np.float32)
                        )
        flat_arr = pa.array(acts_np.reshape(-1), type=pa.float32())
        av_arr = pa.FixedSizeListArray.from_arrays(flat_arr, d_model)
        batch_tbl = pa.table({
            "doc_id": pa.array([r["doc_id"] for r in buf], type=pa.string()),
            "split": pa.array([r["split"] for r in buf], type=pa.string()),
            "role": pa.array([r["role"] for r in buf], type=pa.string()),
            "position": pa.array([r["position"] for r in buf], type=pa.int64()),
            "token_id": pa.array([r["token_id"] for r in buf], type=pa.int64()),
            "n_raw_tokens": pa.array([r["n_raw_tokens"] for r in buf], type=pa.int64()),
            "activation_layer": pa.array([layer_idx] * len(buf), type=pa.int64()),
            "activation_vector": av_arr,
            "detokenized_text_truncated": pa.array([r["text"] for r in buf], type=pa.string()),
            "base_top5": pa.array([r["base_top5"] for r in buf], type=pa.string()),
            "inst_top5": pa.array([r["inst_top5"] for r in buf], type=pa.string()),
            "base_cont": pa.array([r["base_cont"] for r in buf], type=pa.string()),
            "inst_cont": pa.array([r["inst_cont"] for r in buf], type=pa.string()),
        }, schema=schema)
        writers[(arm, sp)].write_table(batch_tbl)
        row_counts[(arm, sp)] += len(buf)

    try:
        for cs in tqdm(range(0, n_convs, chunk_convs), desc="write_arms"):
            sub = conv_plans[cs : cs + chunk_convs]
            t_chunk = torch.load(str(chunks_dir / f"target_{cs:07d}.pt"), weights_only=False)
            b_chunk = torch.load(str(chunks_dir / f"base_{cs:07d}.pt"), weights_only=False)

            bufs: dict[tuple[str, str], list[dict[str, Any]]] = {
                (a, s): [] for a in ("arm0", "arm1", "arm2") for s in ("sft", "rl", "val")
            }
            for plan, td, bd in zip(sub, t_chunk, b_chunk):
                sp = plan["split"]
                t_pos_map = {p: idx for idx, p in enumerate(td["positions"])}
                b_pos_map = {p: idx for idx, p in enumerate(bd["chatml_positions"])}

                # Arm 0 (raw h_target) & Arm 2 (cleaned id-anchored + role centered)
                if plan["active_inst_pos"]:
                    ht_list = [td["hidden"][t_pos_map[p]] for p in plan["active_inst_pos"]]
                    if base_wrapper == "plain":
                        hb_list = [bd["win_hidden"][k] for k in range(len(plan["active_inst_pos"]))]
                    else:
                        hb_list = [bd["chatml_hidden"][b_pos_map[p]] for p in plan["active_inst_pos"]]
                    ht_m = torch.stack(ht_list, dim=0)
                    hb_m = torch.stack(hb_list, dim=0)
                    d_idanc = apply_alignment(hb_m, ht_m, art_idanc)
                    d_clean = apply_role_centering(d_idanc, plan["active_roles"], train_role_means)

                    for k, (inst_p, role) in enumerate(zip(plan["active_inst_pos"], plan["active_roles"])):
                        prefix_ids = plan["inst_ids"][: inst_p + 1]
                        prefix_txt = tokenizer.decode(prefix_ids, skip_special_tokens=False)
                        b_top5 = (
                            bd["top5_win"].get(inst_p, "")
                            if base_wrapper == "plain"
                            else bd["top5_chatml"].get(inst_p, "")
                        )
                        b_cont = (
                            bd["cont_win"].get(inst_p, "")
                            if base_wrapper == "plain"
                            else bd["cont_chatml"].get(inst_p, "")
                        )
                        common = {
                            "doc_id": plan["doc_id"],
                            "split": sp,
                            "role": role,
                            "position": inst_p,
                            "token_id": int(plan["inst_ids"][inst_p]),
                            "n_raw_tokens": len(prefix_ids),
                            "text": prefix_txt,
                            "base_top5": b_top5,
                            "inst_top5": td["top5_by_pos"].get(inst_p, ""),
                            "base_cont": b_cont,
                            "inst_cont": td["cont_by_pos"].get(inst_p, ""),
                        }
                        bufs[("arm0", sp)].append({**common, "activation_vector": ht_m[k].numpy()})
                        bufs[("arm2", sp)].append({**common, "activation_vector": d_clean[k].numpy()})

                # Arm 1 (Naive Diff: standard Ridge, includes template tokens on train, NO role centering)
                if plan["arm1_pos"]:
                    ht1 = torch.stack([td["hidden"][t_pos_map[p]] for p in plan["arm1_pos"]], dim=0)
                    hb1 = torch.stack([bd["chatml_hidden"][b_pos_map[p]] for p in plan["arm1_pos"]], dim=0)
                    d_naive = apply_alignment(hb1, ht1, art_std)
                    for k, (inst_p, role) in enumerate(zip(plan["arm1_pos"], plan["arm1_roles"])):
                        prefix_ids = plan["inst_ids"][: inst_p + 1]
                        prefix_txt = tokenizer.decode(prefix_ids, skip_special_tokens=False)
                        bufs[("arm1", sp)].append({
                            "doc_id": plan["doc_id"],
                            "split": sp,
                            "role": role,
                            "position": inst_p,
                            "token_id": int(plan["inst_ids"][inst_p]),
                            "n_raw_tokens": len(prefix_ids),
                            "text": prefix_txt,
                            "base_top5": bd["top5_chatml"].get(inst_p, ""),
                            "inst_top5": td["top5_by_pos"].get(inst_p, ""),
                            "base_cont": bd["cont_chatml"].get(inst_p, ""),
                            "inst_cont": td["cont_by_pos"].get(inst_p, ""),
                            "activation_vector": d_naive[k].numpy(),
                        })

            for (arm, sp), buf in bufs.items():
                _flush_rows(arm, sp, buf)
    finally:
        for w in writers.values():
            w.close()

    # Save per-arm training-split role means (raw & unit-sphere) for cross-arm comparable eval
    by_arm_payload: dict[str, Any] = {
        "layer_index": layer_idx,
        "tau_web_median": tau_med,
        "arms": {},
    }
    for arm in ("arm0", "arm1", "arm2"):
        by_arm_payload["arms"][arm] = {
            "raw": {
                r: (arm_raw_sums[arm][r] / max(1, arm_role_counts[arm][r])).float()
                for r in ACTIVE_ROLES
            },
            "unit": {
                r: (arm_unit_sums[arm][r] / max(1, arm_role_counts[arm][r])).float()
                for r in ACTIVE_ROLES
            },
            "counts": dict(arm_role_counts[arm]),
        }
    by_arm_path = out_dir / f"role_means_L{layer_idx}_train_by_arm.pt"
    torch.save(by_arm_payload, str(by_arm_path))

    # Save Fix 3 norm/tau/weight ESS diagnostics on Arm 2 training rows
    def _norm_w_stats(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {"n": 0}
        nsq = arr.astype(np.float64) ** 2
        w = nsq / (nsq + max(tau_med, 1e-12) ** 2)
        w_sum = float(w.sum())
        ess = float((w_sum * w_sum) / (arr.size * max(float((w * w).sum()), 1e-12)))
        w_min = float(w.min())
        w_max = float(w.max())
        return {
            "n": int(arr.size),
            "median_norm": float(np.median(arr)),
            "median_norm_over_tau": float(np.median(arr) / max(tau_med, 1e-12)),
            "w_median": float(np.median(w)),
            "w_p10": float(np.percentile(w, 10)),
            "w_p90": float(np.percentile(w, 90)),
            "w_min": w_min,
            "w_max": w_max,
            "w_max_over_min": float(w_max / max(w_min, 1e-12)),
            "ess_frac": ess,
        }

    all_a2_norms = (
        np.concatenate([np.concatenate(v) for v in arm2_train_norms_by_role.values() if v])
        if any(arm2_train_norms_by_role.values())
        else np.array([], dtype=np.float32)
    )
    diag = {
        "layer_index": layer_idx,
        "base_wrapper": base_wrapper,
        "tau_web_median": tau_med,
        "row_counts": {f"{a}/{s}": c for (a, s), c in row_counts.items()},
        "arm2_fix3_train_diagnostics": {
            "overall": _norm_w_stats(all_a2_norms),
            "by_role": {
                r: _norm_w_stats(np.concatenate(v) if v else np.array([], dtype=np.float32))
                for r, v in arm2_train_norms_by_role.items()
            },
        },
    }
    diag_path = out_dir / "phase2_extract_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[extract] saved per-arm role means -> {by_arm_path}", flush=True)
    print(
        f"[extract] saved Fix 3 diagnostics -> {diag_path} "
        f"(overall ESS={diag['arm2_fix3_train_diagnostics']['overall'].get('ess_frac', float('nan')):.2%})",
        flush=True,
    )
    print(f"[extract] complete! Row counts: {row_counts}", flush=True)


# ---------------------------------------------------------------------------
# Step 3: Materialize EasyNLA Training Parquets + Sidecars
# ---------------------------------------------------------------------------

def _write_stage_parquet_and_sidecar(
    *,
    rows_tbl: pa.Table,
    stage: str,
    out_path: Path,
    arm: str,
    p0_cfg: dict[str, Any],
    tokenizer: Any,
    token_meta: Any,
    actor_template: str,
    critic_template: str,
    shuffle_seed: int | None,
) -> int:
    d_model = int(p0_cfg["d_model"])
    layer_idx = int(p0_cfg["layer_index"])
    n = rows_tbl.num_rows
    if n == 0:
        raise RuntimeError(f"Empty table for {arm}/{stage} -> {out_path}")

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(n)
        rows_tbl = rows_tbl.take(pa.array(perm, type=pa.int64()))

    actor_prompt_content = actor_template.format(injection_char=_INJECT_PLACEHOLDER)
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    av_type = pa.list_(pa.float32(), d_model)

    # Extra metadata preserved on val.parquet for Phase 4 Eval 1 (role + token_id!)
    extra_cols = [
        ("role", pa.string()),
        ("token_id", pa.int64()),
        ("position", pa.int64()),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()),
    ]
    if stage == "av_sft":
        core_fields = [
            ("prompt", _PROMPT_STRUCT),
            ("response", pa.string()),
            ("activation_vector", av_type),
        ]
        expls = rows_tbl.column("api_explanation").to_pylist()
        cols_dict: dict[str, Any] = {
            "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
            "response": pa.array([wrap_explanation(e) for e in expls], type=pa.string()),
            "activation_vector": rows_tbl.column("activation_vector"),
        }
    elif stage == "ar_sft":
        core_fields = [
            ("prompt", pa.string()),
            ("activation_vector", av_type),
        ]
        expls = rows_tbl.column("api_explanation").to_pylist()
        suffix_ids = token_meta.critic_suffix_ids
        assert suffix_ids is not None
        n_suf = len(suffix_ids)
        critic_prompts: list[str] = []
        keep_mask: list[bool] = []
        dropped_suffix = 0
        for expl in expls:
            cp = critic_template.format(explanation=expl)
            ids = tokenizer.encode(cp, add_special_tokens=False)
            if len(ids) >= n_suf and ids[-n_suf:] == suffix_ids:
                critic_prompts.append(cp)
                keep_mask.append(True)
            else:
                dropped_suffix += 1
                keep_mask.append(False)
        if dropped_suffix:
            print(f"[materialize] {arm}/{stage}: dropped {dropped_suffix} rows failing critic_suffix_ids check")
            rows_tbl = rows_tbl.filter(pa.array(keep_mask, type=pa.bool_()))
            n = rows_tbl.num_rows
        cols_dict = {
            "prompt": pa.array(critic_prompts, type=pa.string()),
            "activation_vector": rows_tbl.column("activation_vector"),
        }
    elif stage == "rl":
        core_fields = [
            ("prompt", _PROMPT_STRUCT),
            ("activation_vector", av_type),
        ]
        cols_dict = {
            "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
            "activation_vector": rows_tbl.column("activation_vector"),
        }
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    for cname, _ctype in extra_cols:
        cols_dict[cname] = rows_tbl.column(cname)

    out_schema = pa.schema(core_fields + extra_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(pa.OSFile(str(tmp), "wb"), out_schema) as w:
        w.write_table(pa.table(cols_dict, schema=out_schema))
    tmp.replace(out_path)

    tau_val = float(p0_cfg["tau_web_median"]) if arm == "arm2" else 0.0
    meta = NLADatasetMeta(
        dataset_id=f"diff_{arm}_{stage}_L{layer_idx}",
        stage=stage,
        row_count=n,
        extraction=NLAExtractionMeta(
            base_model=str(p0_cfg["target_model"]),
            d_model=d_model,
            layer_index=layer_idx,
            norm="none",
            corpus="diff_chat_60k",
            corpus_slice={"start": 0, "length": n},
            positions_per_doc=3,
            injection_scale="raw",
            mse_scale="sqrt_d_model",
            soft_norm_tau=tau_val,
        ),
        tokens=token_meta,
        prompt_templates={"actor": actor_template, "critic": critic_template},
        created_by="scripts.diff.phase2_build_corpus",
    )
    write_sidecar_local(out_path, meta)
    return n


def run_step_materialize(args: argparse.Namespace, p0_cfg: dict[str, Any]) -> None:
    out_dir = Path(args.output_dir)
    target_model = str(p0_cfg["target_model"])
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    actor_template = _DEFAULT_ACTOR_TEMPLATE
    critic_template = _DEFAULT_CRITIC_TEMPLATE
    token_meta_full = build_token_meta(tokenizer, actor_template, critic_template=critic_template)

    default_arms = ["arm0", "arm1", "arm2"]
    if (out_dir / "arm0c" / "sft_explained.parquet").exists():
        default_arms.append("arm0c")
    arms = args.arms if args.arms else default_arms
    for arm in arms:
        arm_dir = out_dir / arm
        if not arm_dir.exists():
            print(f"[materialize] skipping missing {arm_dir}", flush=True)
            continue

        # 1. RL & Val parquets (arm0c shares raw h_target base_rl/base_val with arm0!)
        base_vec_dir = (out_dir / "arm0") if arm == "arm0c" else arm_dir
        for src_name, dst_name, shuf_seed in (
            ("base_rl.parquet", "rl_shuf.parquet", args.seed),
            ("base_val.parquet", "val.parquet", None),
        ):
            src_p = base_vec_dir / src_name
            if src_p.exists():
                tbl = pq.read_table(src_p)
                n_w = _write_stage_parquet_and_sidecar(
                    rows_tbl=tbl,
                    stage="rl",
                    out_path=arm_dir / dst_name,
                    arm=arm,
                    p0_cfg=p0_cfg,
                    tokenizer=tokenizer,
                    token_meta=token_meta_full,
                    actor_template=actor_template,
                    critic_template=critic_template,
                    shuffle_seed=shuf_seed,
                )
                print(f"[materialize] {arm}/{dst_name}: {n_w} rows", flush=True)

        # 2. SFT parquets (requires Phase 2b sft_explained.parquet)
        sft_expl_path = arm_dir / "sft_explained.parquet"
        if not sft_expl_path.exists():
            print(
                f"[materialize] note: {sft_expl_path} not found yet — skipping SFT parquets for {arm} "
                f"(run phase2_contrastive_sft.py first).",
                flush=True,
            )
            continue

        sft_tbl = pq.read_table(sft_expl_path)
        doc_ids = sft_tbl.column("doc_id").to_pylist()
        unique_docs = list(dict.fromkeys(doc_ids))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(unique_docs)

        n_val_docs = max(1, round(len(unique_docs) * args.sft_val_frac))
        if n_val_docs >= len(unique_docs):
            n_val_docs = max(1, len(unique_docs) // 5)
        val_doc_set = set(unique_docs[:n_val_docs])
        train_docs = unique_docs[n_val_docs:]

        if args.av_ar_split == "disjoint" and len(train_docs) >= 2:
            mid = len(train_docs) // 2
            av_doc_set = set(train_docs[:mid])
            ar_doc_set = set(train_docs[mid:])
        else:
            av_doc_set = set(train_docs)
            ar_doc_set = set(train_docs)

        av_tr = sft_tbl.filter(pa.array([d in av_doc_set for d in doc_ids], type=pa.bool_()))
        ar_tr = sft_tbl.filter(pa.array([d in ar_doc_set for d in doc_ids], type=pa.bool_()))
        av_va = sft_tbl.filter(pa.array([d in val_doc_set for d in doc_ids], type=pa.bool_()))

        for sub_tbl, stage, fname, seed_s in (
            (av_tr, "av_sft", "av_sft_train.parquet", args.seed + 1),
            (ar_tr, "ar_sft", "ar_sft_train.parquet", args.seed + 2),
            (av_va, "av_sft", "av_sft_val.parquet", None),
        ):
            n_w = _write_stage_parquet_and_sidecar(
                rows_tbl=sub_tbl,
                stage=stage,
                out_path=arm_dir / fname,
                arm=arm,
                p0_cfg=p0_cfg,
                tokenizer=tokenizer,
                token_meta=token_meta_full,
                actor_template=actor_template,
                critic_template=critic_template,
                shuffle_seed=seed_s,
            )
            print(f"[materialize] {arm}/{fname}: {n_w} rows", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", required=True, choices=["rollouts", "extract", "materialize", "all"])
    p.add_argument("--phase0-dir", required=True, help="Directory containing phase0_config.json")
    p.add_argument("--output-dir", required=True, help="Root output directory for Phase 2 corpus")
    p.add_argument("--arms", nargs="*", default=None, help="Subset of arms for materialize (default: arm0 arm1 arm2 [+ arm0c if present])")
    # Prompt sampling counts (default: 60k total = 30k WildChat + 20k UltraChat + 10k safety mix)
    p.add_argument("--n-wildchat", type=int, default=30000)
    p.add_argument("--n-ultrachat", type=int, default=20000)
    p.add_argument("--n-safety", type=int, default=10000)
    p.add_argument("--custom-prompts-jsonl", default=None, help="Local JSONL prompts for offline/CPU tests")
    p.add_argument("--eval-blocklist-source", choices=["auto", "builtin"], default="auto",
                   help="Blocklist HF eval sets (XSTest/JBB + MATH-500) + builtins ('auto') or builtins only ('builtin')")
    p.add_argument("--sft-frac", type=float, default=20000 / 180000, help="Doc fraction for SFT split (~11.1%)")
    p.add_argument("--val-frac", type=float, default=5000 / 180000, help="Doc fraction for held-out val (~2.8%)")
    p.add_argument("--sft-val-frac", type=float, default=0.05, help="Fraction of SFT docs held out as av_sft_val")
    p.add_argument("--av-ar-split", choices=["shared", "disjoint"], default="shared",
                   help="'shared' trains AV & AR SFT on the same ~19k train docs (with 1k doc-disjoint val); "
                        "'disjoint' splits ~19k into ~9.5k AV / ~9.5k AR.")
    # Rollout settings
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--rollout-max-new-tokens", type=int, default=96)
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    # Extraction settings
    p.add_argument("--chunk-convs", type=int, default=5000, help="Conversations per checkpointed chunk")
    p.add_argument("--sft-cont-tokens", type=int, default=20, help="Greedy continuation length for SFT rows (0=off)")
    p.add_argument("--sft-cont-batch-size", type=int, default=32, help="Batch size for short SFT greedy continuations")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    p0_cfg = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text(encoding="utf-8"))
    if args.step in ("rollouts", "all"):
        run_step_rollouts(args, p0_cfg)
    if args.step in ("extract", "all"):
        run_step_extract(args, p0_cfg)
    if args.step in ("materialize", "all"):
        run_step_materialize(args, p0_cfg)


if __name__ == "__main__":
    main()
