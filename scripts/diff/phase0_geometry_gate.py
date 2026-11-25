"""Phase 0 — Qwen3-8B L24 Geometry, Fix 1 Alignment, Base-Wrapper A/B, & Taboo <think> Gate.

Standalone Go/No-Go gate executed BEFORE any big data generation or RL spend (~45 min GPU).

What it measures and saves:
  1. Pretraining Alignment (§2.1, Fix 1):
       Extracts Layer-K activations from Base & Target on raw FineWeb-Edu passages
       (skipping pos 0 BOS/sink), compares Identity, Identity+Bias, Standard Ridge
       (5-fold doc-grouped CV), and Identity-Anchored Ridge (W = I + dW, 5-fold CV).
       Computes noise floor tau = Median_web_val(||Delta_id_anc||_2) and P75.
       Saves `align_L{K}_id_anchored.pt` and `align_L{K}_std_ridge.pt`.
  2. Base-Wrapper A/B Test (§2.4 & Fix 2):
       Compares `Base(ChatML)` vs `Base(Plain)` against `Instruct(ChatML, enable_thinking=False)`
       on neutral queries + Refusal / Eval-Awareness / Sycophancy / Historical Era pairs:
       - Neutral KL(Instruct || Base) at `asst_last`
       - `chat_template_amplification` on identical `user_content` tokens (raw vs wrapped)
       - `tok1` Ridge leakage on `user_content` tokens (`h_target`, raw `Delta_id_anc`,
         Fix 2 role-centered `Unit(Delta_id_anc)`)
       - Fisher SNR & 5-fold CV probe AUROC across Refusal, Eval-Awareness, Sycophancy,
         and Historical Era (negative pretraining control)
       Selects winning `base_wrapper` and saves starter `role_means_L{K}.pt`.
  3. Taboo LoRA `<think>` Mode Check (§2.4):
       Evaluates open Qwen3-8B Taboo LoRAs (`snow`, `blue_50_mix`) on neutral queries
       under `enable_thinking=False` vs `enable_thinking=True`, verifying
       KL(Org || Inst) < 3.0 nats under matched mode.
  4. Gate 0 Decision:
       Writes `phase0_report.json` and `phase0_config.json`. Exits 0 on PASS, 2 on FAIL
       (unless `--no-fail-exit` is passed).

Memory discipline:
  Loads ONE model at a time (Target -> optional Taboo LoRAs -> unload -> Base -> unload).

Example (GPU):
    python scripts/diff/phase0_geometry_gate.py \
        --base-model Qwen/Qwen3-8B-Base \
        --target-model Qwen/Qwen3-8B \
        --layer-index 24 \
        --output-dir /data/diff_nla/phase0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nla.diff.alignment import (
    DEFAULT_ALPHA_GRID,
    apply_alignment,
    evaluate_alignment_suite,
    save_alignment,
)
from nla.diff.benchmarks import (
    NEUTRAL_QUERIES,
    PHASE0_TABOO_ORGANISMS,
    flatten_binary_pairs,
    load_eval_awareness_pairs,
    load_fineweb_passages,
    load_historical_period_items,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.metrics import (
    compute_chat_template_amplification,
    compute_fisher_snr,
    compute_next_token_kl,
    compute_probe_auroc_cv,
    compute_tok1_ridge_fve,
    unit_normalize,
)
from nla.diff.model_io import (
    extract_at_positions,
    load_causal_lm,
    resolve_device,
    resolve_dtype,
    unload_model,
)
from nla.diff.wrappers import (
    apply_role_centering,
    compute_role_means,
    encode_aligned_pair,
    encode_chatml_dialogue,
    encode_raw_user_only,
    save_role_means,
)


def _stack_role_hidden(
    extractions: list[Any],
    *,
    expect_single: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Concatenate hidden tensors & token IDs across a list of PositionExtraction objects."""
    mats = []
    toks: list[int] = []
    for ex in extractions:
        if expect_single:
            assert ex.hidden.shape[0] == 1, f"Expected 1 position, got {ex.hidden.shape[0]}"
        if ex.hidden.shape[0] > 0:
            mats.append(ex.hidden)
            toks.extend(ex.token_ids)
    assert mats, "No activations extracted!"
    return torch.cat(mats, dim=0), toks


def _evaluate_binary_task(
    h_target: torch.Tensor,
    delta_raw: torch.Tensor,
    labels: list[int],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare raw h_target, raw Delta_id_anc, and Fix 2 role-centered Unit(Delta_id_anc)."""
    mu_role = delta_raw.double().mean(dim=0, keepdim=True).float()
    delta_centered_unit = unit_normalize(delta_raw - mu_role)
    h_target_unit = unit_normalize(h_target)
    delta_raw_unit = unit_normalize(delta_raw)

    snr_ht = compute_fisher_snr(h_target, labels)
    snr_ht_u = compute_fisher_snr(h_target_unit, labels)
    snr_dr = compute_fisher_snr(delta_raw, labels)
    snr_dr_u = compute_fisher_snr(delta_raw_unit, labels)
    snr_cu = compute_fisher_snr(delta_centered_unit, labels)

    auroc_ht = compute_probe_auroc_cv(h_target, labels, seed=seed)
    auroc_dr = compute_probe_auroc_cv(delta_raw, labels, seed=seed)
    auroc_cu = compute_probe_auroc_cv(delta_centered_unit, labels, seed=seed)

    # Constant mean-offset energy fraction: ||mu||^2 / E[||delta||^2]
    total_energy = float(delta_raw.float().pow(2).sum(dim=-1).mean().item())
    mean_energy = float(mu_role.pow(2).sum().item())
    mean_offset_frac = mean_energy / max(total_energy, 1e-12)

    return {
        "n_samples": len(labels),
        "fisher_snr": {
            "h_target": snr_ht,
            "h_target_unit": snr_ht_u,
            "delta_raw": snr_dr,
            "delta_raw_unit": snr_dr_u,
            "delta_role_centered_unit": snr_cu,
            "ratio_centered_unit_over_h_target": snr_cu / max(snr_ht, 1e-12),
            "ratio_centered_unit_over_h_target_unit": snr_cu / max(snr_ht_u, 1e-12),
        },
        "probe_auroc_5fold": {
            "h_target": auroc_ht,
            "delta_raw": auroc_dr,
            "delta_role_centered_unit": auroc_cu,
        },
        "mean_offset_energy_frac": mean_offset_frac,
    }


def run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[phase0] device={device} dtype={dtype} layer={args.layer_index}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    # ------------------------------------------------------------------
    # 1. Prepare all inputs deterministically up front
    # ------------------------------------------------------------------
    n_web_total = args.n_web_train + args.n_web_val
    print(f"[phase0] loading {n_web_total} raw pretraining passages...", flush=True)
    web_passages = load_fineweb_passages(
        n_passages=n_web_total,
        dataset_name=args.web_dataset,
        dataset_config=(None if args.web_dataset_config in ("", "none", "None") else args.web_dataset_config),
        custom_path=args.web_custom_path,
    )
    assert len(web_passages) >= args.n_web_train + 5, (
        f"Loaded only {len(web_passages)} web passages; need > {args.n_web_train}"
    )
    web_train_docs = web_passages[: args.n_web_train]
    web_val_docs = web_passages[args.n_web_train : args.n_web_train + args.n_web_val]

    def _encode_web_split(docs: list[dict[str, str]]) -> tuple[list[list[int]], list[list[int]], list[str]]:
        seqs: list[list[int]] = []
        pos_lists: list[list[int]] = []
        doc_ids_flat: list[str] = []
        for d in docs:
            ids = tokenizer.encode(d["text"], add_special_tokens=False)[: args.web_max_tokens]
            # Skip position 0 (BOS / attention sink)
            if len(ids) < 2:
                continue
            pos = list(range(1, len(ids)))
            seqs.append(ids)
            pos_lists.append(pos)
            doc_ids_flat.extend([d["doc_id"]] * len(pos))
        return seqs, pos_lists, doc_ids_flat

    web_tr_seqs, web_tr_pos, web_tr_dids = _encode_web_split(web_train_docs)
    web_va_seqs, web_va_pos, web_va_dids = _encode_web_split(web_val_docs)
    print(
        f"[phase0] web tokens: train={len(web_tr_dids)} ({len(web_tr_seqs)} docs), "
        f"val={len(web_va_dids)} ({len(web_va_seqs)} docs)",
        flush=True,
    )

    # Diagnostic chat suites
    neutral_prompts = list(NEUTRAL_QUERIES[: args.n_neutral])
    refusal_pairs = load_refusal_pairs(args.n_diag_pairs, source=args.refusal_source, seed=args.seed)
    eval_pairs = load_eval_awareness_pairs(args.n_diag_pairs)
    syc_pairs = load_sycophancy_pairs(args.n_diag_pairs)
    hist_items = load_historical_period_items(min(60, args.n_diag_pairs * 2))

    ref_prompts, ref_labels, _ = flatten_binary_pairs(refusal_pairs)
    ea_prompts, ea_labels, _ = flatten_binary_pairs(eval_pairs)
    syc_prompts, syc_labels, _ = flatten_binary_pairs(syc_pairs)
    hist_prompts = [
        f"Read the following historical excerpt and identify its historical era:\n\n{it['text']}"
        for it in hist_items
    ]
    hist_labels = [int(it["binary_label"]) for it in hist_items]

    diag_tasks: dict[str, tuple[list[str], list[int]]] = {
        "refusal": (ref_prompts, ref_labels),
        "eval_awareness": (ea_prompts, ea_labels),
        "sycophancy": (syc_prompts, syc_labels),
        "historical_era": (hist_prompts, hist_labels),
    }

    # Build aligned encodings for EVERY chat prompt under both Base wrappers
    # We keep a flat list of all unique prompt groups so extraction runs cleanly in batches.
    def _encode_group(prompts: list[str]) -> dict[str, Any]:
        inst_encs = []
        base_chatml_encs = []
        base_plain_encs = []
        raw_encs = []
        for txt in prompts:
            ie, be_c = encode_aligned_pair(
                tokenizer, txt, base_wrapper="chatml", enable_thinking=False,
                max_total_tokens=args.chat_max_tokens,
            )
            _, be_p = encode_aligned_pair(
                tokenizer, txt, base_wrapper="plain", enable_thinking=False,
                max_total_tokens=args.chat_max_tokens,
            )
            u_ids = ie.input_ids[ie.user_span[0] : ie.user_span[1]]
            re_u = encode_raw_user_only(tokenizer, txt, user_ids=u_ids)
            inst_encs.append(ie)
            base_chatml_encs.append(be_c)
            base_plain_encs.append(be_p)
            raw_encs.append(re_u)
        return {
            "inst": inst_encs,
            "base_chatml": base_chatml_encs,
            "base_plain": base_plain_encs,
            "raw": raw_encs,
        }

    neutral_enc = _encode_group(neutral_prompts)
    diag_enc = {name: _encode_group(pr) for name, (pr, _) in diag_tasks.items()}

    # For Taboo <think> mode check: encode neutral queries under enable_thinking=False & True
    taboo_neutral_nothink = [
        encode_chatml_dialogue(tokenizer, txt, enable_thinking=False, max_total_tokens=args.chat_max_tokens)
        for txt in neutral_prompts
    ]
    taboo_neutral_think = [
        encode_chatml_dialogue(tokenizer, txt, enable_thinking=True, max_total_tokens=args.chat_max_tokens)
        for txt in neutral_prompts
    ]

    # Sample up to 4 user_content positions per prompt for user_content role diagnostics
    rng = np.random.default_rng(args.seed)

    def _sample_user_pos(encs: list[Any], max_per_prompt: int = 4) -> list[list[int]]:
        """Pick identical relative user_content offsets per prompt across wrappers."""
        out = []
        for e in encs:
            up = e.user_content_positions
            if len(up) <= max_per_prompt:
                rel = list(range(len(up)))
            else:
                rel = sorted(rng.choice(len(up), size=max_per_prompt, replace=False).tolist())
            out.append(rel)
        return out

    # Compute relative offsets ONCE per task so Instruct, Base(ChatML), Base(Plain), Raw align 1-to-1
    diag_user_rel_offsets: dict[str, list[list[int]]] = {
        name: _sample_user_pos(enc_map["inst"]) for name, enc_map in diag_enc.items()
    }
    neutral_user_rel_offsets = _sample_user_pos(neutral_enc["inst"])

    def _abs_positions_for_wrapper(
        encs: list[Any],
        rel_offsets: list[list[int]],
        *,
        include_asst_last: bool,
    ) -> list[list[int]]:
        pos_lists = []
        for e, rel in zip(encs, rel_offsets):
            up = e.user_content_positions
            abs_u = [up[r] for r in rel if r < len(up)]
            if include_asst_last:
                pos_lists.append(abs_u + [e.asst_last_idx])
            else:
                # For raw user-only: map relative offset inside user_span (skipping pos 0!)
                u0 = e.user_span[0]
                u_all = list(range(u0, e.user_span[1]))
                abs_raw = [u_all[r] for r in rel if r < len(u_all) and u_all[r] > 0]
                pos_lists.append(abs_raw)
        return pos_lists

    def _split_user_and_asst_last(
        extractions: list[Any],
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        user_h, user_toks, asst_h = [], [], []
        for ex in extractions:
            assert ex.hidden.shape[0] >= 1
            if ex.hidden.shape[0] > 1:
                user_h.append(ex.hidden[:-1])
                user_toks.extend(ex.token_ids[:-1])
            asst_h.append(ex.hidden[-1:])
        return torch.cat(user_h, dim=0), user_toks, torch.cat(asst_h, dim=0)

    # ------------------------------------------------------------------
    # 2. PASS 1: Target model (Qwen3-8B Instruct) + optional Taboo LoRAs
    # ------------------------------------------------------------------
    print(f"[phase0] Pass 1: loading target model {args.target_model}...", flush=True)
    target_model = load_causal_lm(args.target_model, dtype=dtype, device=device)

    t_web_tr_ex = extract_at_positions(
        target_model, web_tr_seqs, web_tr_pos,
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    t_web_va_ex = extract_at_positions(
        target_model, web_va_seqs, web_va_pos,
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    Ht_web_tr, _ = _stack_role_hidden(t_web_tr_ex)
    Ht_web_va, _ = _stack_role_hidden(t_web_va_ex)

    # Neutral queries: hidden + logits at asst_last (enable_thinking=False)
    t_neut_asst_ex = extract_at_positions(
        target_model,
        [e.input_ids for e in neutral_enc["inst"]],
        [[e.asst_last_idx] for e in neutral_enc["inst"]],
        layer_index=args.layer_index, pad_token_id=pad_id,
        batch_size=args.batch_size, need_logits=True,
    )
    inst_neutral_logits = torch.cat([ex.logits for ex in t_neut_asst_ex], dim=0)
    inst_neutral_asst_h = torch.cat([ex.hidden for ex in t_neut_asst_ex], dim=0)

    # Neutral raw vs wrapped user_content for chat_template_amplification
    t_neut_user_wrap_ex = extract_at_positions(
        target_model,
        [e.input_ids for e in neutral_enc["inst"]],
        _abs_positions_for_wrapper(neutral_enc["inst"], neutral_user_rel_offsets, include_asst_last=False),
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    t_neut_user_raw_ex = extract_at_positions(
        target_model,
        [e.input_ids for e in neutral_enc["raw"]],
        _abs_positions_for_wrapper(neutral_enc["raw"], neutral_user_rel_offsets, include_asst_last=False),
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    Ht_neut_user_wrap, _ = _stack_role_hidden(t_neut_user_wrap_ex)
    Ht_neut_user_raw, _ = _stack_role_hidden(t_neut_user_raw_ex)

    # Diagnostic tasks on Target
    Ht_diag_asst: dict[str, torch.Tensor] = {}
    Ht_diag_user: dict[str, torch.Tensor] = {}
    diag_user_toks: dict[str, list[int]] = {}
    for name, enc_map in diag_enc.items():
        ex_list = extract_at_positions(
            target_model,
            [e.input_ids for e in enc_map["inst"]],
            _abs_positions_for_wrapper(enc_map["inst"], diag_user_rel_offsets[name], include_asst_last=True),
            layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
        )
        uh, utoks, ah = _split_user_and_asst_last(ex_list)
        Ht_diag_user[name] = uh
        diag_user_toks[name] = utoks
        Ht_diag_asst[name] = ah

    # Optional Taboo LoRA <think> Mode Check (reuses loaded Target model!)
    taboo_report: dict[str, Any] = {"skipped": bool(args.skip_taboo), "loras": {}}
    if not args.skip_taboo and args.taboo_loras:
        t_think_ex = extract_at_positions(
            target_model,
            [e.input_ids for e in taboo_neutral_think],
            [[e.asst_last_idx] for e in taboo_neutral_think],
            layer_index=args.layer_index, pad_token_id=pad_id,
            batch_size=args.batch_size, need_logits=True,
        )
        inst_think_logits = torch.cat([ex.logits for ex in t_think_ex], dim=0)
        inst_think_h = torch.cat([ex.hidden for ex in t_think_ex], dim=0)

        from peft import PeftModel
        for lora_id in args.taboo_loras:
            print(f"[phase0] Taboo mode check: {lora_id}...", flush=True)
            peft_m = PeftModel.from_pretrained(target_model, lora_id)
            peft_m.eval()
            org_nt = extract_at_positions(
                peft_m,
                [e.input_ids for e in taboo_neutral_nothink],
                [[e.asst_last_idx] for e in taboo_neutral_nothink],
                layer_index=args.layer_index, pad_token_id=pad_id,
                batch_size=args.batch_size, need_logits=True,
            )
            org_th = extract_at_positions(
                peft_m,
                [e.input_ids for e in taboo_neutral_think],
                [[e.asst_last_idx] for e in taboo_neutral_think],
                layer_index=args.layer_index, pad_token_id=pad_id,
                batch_size=args.batch_size, need_logits=True,
            )
            nt_logits = torch.cat([ex.logits for ex in org_nt], dim=0)
            nt_h = torch.cat([ex.hidden for ex in org_nt], dim=0)
            th_logits = torch.cat([ex.logits for ex in org_th], dim=0)
            th_h = torch.cat([ex.hidden for ex in org_th], dim=0)

            kl_nt = float(compute_next_token_kl(nt_logits, inst_neutral_logits).mean().item())
            kl_th = float(compute_next_token_kl(th_logits, inst_think_logits).mean().item())
            act_shift_nt = float(
                ((nt_h - inst_neutral_asst_h).norm(dim=-1) / (inst_neutral_asst_h.norm(dim=-1) + 1e-9)).mean().item()
            )
            act_shift_th = float(
                ((th_h - inst_think_h).norm(dim=-1) / (inst_think_h.norm(dim=-1) + 1e-9)).mean().item()
            )
            best_mode = "non_thinking" if kl_nt <= kl_th else "thinking"
            best_kl = min(kl_nt, kl_th)
            taboo_report["loras"][lora_id] = {
                "kl_enable_thinking_false": kl_nt,
                "kl_enable_thinking_true": kl_th,
                "act_shift_enable_thinking_false": act_shift_nt,
                "act_shift_enable_thinking_true": act_shift_th,
                "recommended_mode": best_mode,
                "min_kl_nats": best_kl,
                "passes_kl_lt_3": bool(best_kl < 3.0),
            }
            # Restore clean base target_model without PEFT wrappers
            target_model = peft_m.unload()
            del peft_m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    unload_model(target_model)

    # ------------------------------------------------------------------
    # 3. PASS 2: Base model (Qwen3-8B-Base)
    # ------------------------------------------------------------------
    print(f"[phase0] Pass 2: loading base model {args.base_model}...", flush=True)
    base_model = load_causal_lm(args.base_model, dtype=dtype, device=device)

    b_web_tr_ex = extract_at_positions(
        base_model, web_tr_seqs, web_tr_pos,
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    b_web_va_ex = extract_at_positions(
        base_model, web_va_seqs, web_va_pos,
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    Hb_web_tr, _ = _stack_role_hidden(b_web_tr_ex)
    Hb_web_va, _ = _stack_role_hidden(b_web_va_ex)

    # Neutral raw user_content on Base
    b_neut_user_raw_ex = extract_at_positions(
        base_model,
        [e.input_ids for e in neutral_enc["raw"]],
        _abs_positions_for_wrapper(neutral_enc["raw"], neutral_user_rel_offsets, include_asst_last=False),
        layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
    )
    Hb_neut_user_raw, _ = _stack_role_hidden(b_neut_user_raw_ex)

    base_by_wrapper: dict[str, dict[str, Any]] = {}
    for wname, key in (("chatml", "base_chatml"), ("plain", "base_plain")):
        print(f"[phase0] extracting Base under wrapper={wname}...", flush=True)
        b_neut_asst_ex = extract_at_positions(
            base_model,
            [e.input_ids for e in neutral_enc[key]],
            [[e.asst_last_idx] for e in neutral_enc[key]],
            layer_index=args.layer_index, pad_token_id=pad_id,
            batch_size=args.batch_size, need_logits=True,
        )
        b_neut_logits = torch.cat([ex.logits for ex in b_neut_asst_ex], dim=0)
        b_neut_user_wrap_ex = extract_at_positions(
            base_model,
            [e.input_ids for e in neutral_enc[key]],
            _abs_positions_for_wrapper(neutral_enc[key], neutral_user_rel_offsets, include_asst_last=False),
            layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
        )
        b_neut_user_wrap, _ = _stack_role_hidden(b_neut_user_wrap_ex)

        diag_asst_b: dict[str, torch.Tensor] = {}
        diag_user_b: dict[str, torch.Tensor] = {}
        for name, enc_map in diag_enc.items():
            ex_list = extract_at_positions(
                base_model,
                [e.input_ids for e in enc_map[key]],
                _abs_positions_for_wrapper(enc_map[key], diag_user_rel_offsets[name], include_asst_last=True),
                layer_index=args.layer_index, pad_token_id=pad_id, batch_size=args.batch_size,
            )
            uh, _, ah = _split_user_and_asst_last(ex_list)
            diag_user_b[name] = uh
            diag_asst_b[name] = ah

        base_by_wrapper[wname] = {
            "neutral_logits": b_neut_logits,
            "neutral_user_wrap": b_neut_user_wrap,
            "diag_asst": diag_asst_b,
            "diag_user": diag_user_b,
        }

    unload_model(base_model)

    # ------------------------------------------------------------------
    # 4. Fit Fix 1 Alignment Suite on FineWeb (GPU if available -> <2s)
    # ------------------------------------------------------------------
    print("[phase0] fitting Fix 1 alignment suite (float64 CV)...", flush=True)
    align_res = evaluate_alignment_suite(
        Hb_web_tr, Ht_web_tr, web_tr_dids,
        Hb_web_va, Ht_web_va,
        alphas=DEFAULT_ALPHA_GRID,
        n_folds=5,
        seed=args.seed,
        layer_index=args.layer_index,
        device=device,
    )
    art_idanc = align_res["artifact_id_anchored"]
    art_std = align_res["artifact_std_ridge"]

    align_idanc_path = save_alignment(out_dir / f"align_L{args.layer_index}_id_anchored.pt", art_idanc)
    align_std_path = save_alignment(out_dir / f"align_L{args.layer_index}_std_ridge.pt", art_std)

    # ------------------------------------------------------------------
    # 5. Evaluate Base(ChatML) vs Base(Plain) & Pick Winner
    # ------------------------------------------------------------------
    delta_neut_raw = apply_alignment(Hb_neut_user_raw, Ht_neut_user_raw, art_idanc)
    all_user_ht = torch.cat([Ht_diag_user[k] for k in ("refusal", "eval_awareness", "sycophancy")], dim=0)
    all_user_toks = (
        diag_user_toks["refusal"] + diag_user_toks["eval_awareness"] + diag_user_toks["sycophancy"]
    )
    tok1_fve_ht = compute_tok1_ridge_fve(all_user_ht, all_user_toks, seed=args.seed)

    wrapper_reports: dict[str, Any] = {}
    wrapper_role_means: dict[str, dict[str, torch.Tensor]] = {}
    for wname in ("chatml", "plain"):
        bw = base_by_wrapper[wname]
        kl_neutral = float(compute_next_token_kl(inst_neutral_logits, bw["neutral_logits"]).mean().item())
        delta_neut_wrap = apply_alignment(bw["neutral_user_wrap"], Ht_neut_user_wrap, art_idanc)
        amp_ratio = compute_chat_template_amplification(delta_neut_raw, delta_neut_wrap)

        # User-content deltas for tok1 leakage & role mean
        all_user_hb = torch.cat([bw["diag_user"][k] for k in ("refusal", "eval_awareness", "sycophancy")], dim=0)
        all_user_delta = apply_alignment(all_user_hb, all_user_ht, art_idanc)
        all_asst_hb = torch.cat([bw["diag_asst"][k] for k in ("refusal", "eval_awareness", "sycophancy")], dim=0)
        all_asst_ht = torch.cat([Ht_diag_asst[k] for k in ("refusal", "eval_awareness", "sycophancy")], dim=0)
        all_asst_delta = apply_alignment(all_asst_hb, all_asst_ht, art_idanc)

        rm = compute_role_means({"user_content": all_user_delta, "asst_last": all_asst_delta})
        wrapper_role_means[wname] = rm

        user_delta_cu = unit_normalize(
            apply_role_centering(all_user_delta, ["user_content"] * len(all_user_delta), rm)
        )
        tok1_fve_delta_raw = compute_tok1_ridge_fve(all_user_delta, all_user_toks, seed=args.seed)
        tok1_fve_delta_cu = compute_tok1_ridge_fve(user_delta_cu, all_user_toks, seed=args.seed)
        tok1_reduction_frac = 1.0 - (tok1_fve_delta_cu / max(tok1_fve_ht, 1e-9))

        task_metrics: dict[str, Any] = {}
        for tname, (_, labels) in diag_tasks.items():
            d_asst = apply_alignment(bw["diag_asst"][tname], Ht_diag_asst[tname], art_idanc)
            task_metrics[tname] = _evaluate_binary_task(
                Ht_diag_asst[tname], d_asst, labels, seed=args.seed,
            )

        audit_names = ("refusal", "eval_awareness", "sycophancy")
        avg_snr_ht = float(np.mean([task_metrics[k]["fisher_snr"]["h_target"] for k in audit_names]))
        avg_snr_cu = float(np.mean([task_metrics[k]["fisher_snr"]["delta_role_centered_unit"] for k in audit_names]))
        avg_snr_ratio = avg_snr_cu / max(avg_snr_ht, 1e-12)
        avg_auroc_ht = float(np.mean([task_metrics[k]["probe_auroc_5fold"]["h_target"] for k in audit_names]))
        avg_auroc_cu = float(np.mean([task_metrics[k]["probe_auroc_5fold"]["delta_role_centered_unit"] for k in audit_names]))

        wrapper_reports[wname] = {
            "neutral_kl_inst_vs_base_nats": kl_neutral,
            "chat_template_amplification": amp_ratio,
            "tok1_ridge_fve_user_content": {
                "h_target": tok1_fve_ht,
                "delta_raw": tok1_fve_delta_raw,
                "delta_role_centered_unit": tok1_fve_delta_cu,
                "reduction_frac_vs_h_target": tok1_reduction_frac,
            },
            "tasks": task_metrics,
            "auditing_summary": {
                "avg_fisher_snr_h_target": avg_snr_ht,
                "avg_fisher_snr_delta_centered_unit": avg_snr_cu,
                "avg_fisher_snr_ratio": avg_snr_ratio,
                "avg_auroc_h_target": avg_auroc_ht,
                "avg_auroc_delta_centered_unit": avg_auroc_cu,
            },
        }

    # Winner selection: prefer higher auditing Fisher SNR; break close ties (<5%) by lower neutral KL
    if args.force_base_wrapper is not None:
        winning_wrapper = args.force_base_wrapper
    else:
        s_c = wrapper_reports["chatml"]["auditing_summary"]["avg_fisher_snr_delta_centered_unit"]
        s_p = wrapper_reports["plain"]["auditing_summary"]["avg_fisher_snr_delta_centered_unit"]
        kl_c = wrapper_reports["chatml"]["neutral_kl_inst_vs_base_nats"]
        kl_p = wrapper_reports["plain"]["neutral_kl_inst_vs_base_nats"]
        if abs(s_p - s_c) <= 0.05 * max(s_p, s_c, 1e-9):
            winning_wrapper = "plain" if kl_p < kl_c else "chatml"
        else:
            winning_wrapper = "plain" if s_p > s_c else "chatml"

    role_means_path = save_role_means(
        out_dir / f"role_means_L{args.layer_index}.pt",
        wrapper_role_means[winning_wrapper],
        tau_web_median=align_res["tau_web_median"],
        tau_web_p75=align_res["tau_web_p75"],
        extra={"base_wrapper": winning_wrapper, "layer_index": args.layer_index},
    )

    # ------------------------------------------------------------------
    # 6. GO / NO-GO Gate 0 Verification
    # ------------------------------------------------------------------
    win_rep = wrapper_reports[winning_wrapper]
    gate_fix1 = bool(align_res["fix1_strict_passed"])
    gate_snr = bool(win_rep["auditing_summary"]["avg_fisher_snr_ratio"] >= 1.25)
    gate_tok1 = bool(win_rep["tok1_ridge_fve_user_content"]["reduction_frac_vs_h_target"] >= 0.30)
    if taboo_report["skipped"] or not taboo_report["loras"]:
        gate_taboo = True
    else:
        gate_taboo = all(v["passes_kl_lt_3"] for v in taboo_report["loras"].values())

    gate0_passed = gate_fix1 and gate_snr and gate_tok1 and gate_taboo

    # Recommended Taboo mode (non_thinking if majority / lower KL)
    if taboo_report["loras"]:
        nt_wins = sum(v["recommended_mode"] == "non_thinking" for v in taboo_report["loras"].values())
        taboo_mode = "non_thinking" if nt_wins >= len(taboo_report["loras"]) / 2 else "thinking"
    else:
        taboo_mode = "non_thinking"

    phase0_config = {
        "base_model": args.base_model,
        "target_model": args.target_model,
        "layer_index": int(args.layer_index),
        "d_model": int(art_idanc["d_model"]),
        "base_wrapper": winning_wrapper,
        "tau_web_median": float(align_res["tau_web_median"]),
        "tau_web_p75": float(align_res["tau_web_p75"]),
        "chosen_alpha_id_anchored": float(align_res["best_alpha_id_anchored"]),
        "chosen_alpha_std_ridge": float(align_res["best_alpha_std_ridge"]),
        "align_id_anchored_path": str(align_idanc_path),
        "align_std_ridge_path": str(align_std_path),
        "role_means_path": str(role_means_path),
        "taboo_recommended_mode": taboo_mode,
        "gate0_passed": gate0_passed,
    }
    (out_dir / "phase0_config.json").write_text(json.dumps(phase0_config, indent=2), encoding="utf-8")

    alignment_summary = {
        k: v
        for k, v in align_res.items()
        if k not in ("artifact_id_anchored", "artifact_std_ridge")
    }
    report = {
        "wall_seconds": round(time.time() - t_start, 2),
        "config": phase0_config,
        "alignment": alignment_summary,
        "wrappers": wrapper_reports,
        "taboo_mode_check": taboo_report,
        "gate0": {
            "passed": gate0_passed,
            "fix1_strict_passed": gate_fix1,
            "fix1_best_passed": bool(align_res["fix1_best_passed"]),
            "snr_boost_ge_1_25x": gate_snr,
            "tok1_cut_ge_30pct": gate_tok1,
            "taboo_kl_lt_3": gate_taboo,
            "winning_wrapper": winning_wrapper,
        },
    }
    (out_dir / "phase0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Crisp stdout summary
    print("\n" + "=" * 78)
    print(f"PHASE 0 SUMMARY — Layer {args.layer_index} ({args.base_model} -> {args.target_model})")
    print("=" * 78)
    print(
        f"Fix 1 Alignment (web_val FVE): "
        f"Id-Anc={align_res['fve_id_anchored']:.4f} (alpha={align_res['best_alpha_id_anchored']:g}) | "
        f"Identity={align_res['fve_identity']:.4f} | "
        f"Id+Bias={align_res['fve_identity_bias']:.4f} | "
        f"Std-Ridge={align_res['fve_std_ridge']:.4f}"
    )
    print(
        f"Fix 3 Noise Floor: tau_web_median={align_res['tau_web_median']:.4f}, "
        f"tau_web_p75={align_res['tau_web_p75']:.4f}"
    )
    for wname in ("chatml", "plain"):
        wr = wrapper_reports[wname]
        tag = " [WINNER]" if wname == winning_wrapper else ""
        print(
            f"Wrapper={wname:6s}{tag}: KL={wr['neutral_kl_inst_vs_base_nats']:.3f} nats | "
            f"Amp={wr['chat_template_amplification']:.3f} | "
            f"AvgSNR_ratio={wr['auditing_summary']['avg_fisher_snr_ratio']:.2f}x | "
            f"AvgAUROC={wr['auditing_summary']['avg_auroc_h_target']:.3f}->{wr['auditing_summary']['avg_auroc_delta_centered_unit']:.3f} | "
            f"tok1_cut={wr['tok1_ridge_fve_user_content']['reduction_frac_vs_h_target']*100:.1f}%"
        )
    if taboo_report["loras"]:
        for lid, lv in taboo_report["loras"].items():
            print(
                f"Taboo {lid}: KL(no-think)={lv['kl_enable_thinking_false']:.3f}, "
                f"KL(think)={lv['kl_enable_thinking_true']:.3f} -> {lv['recommended_mode']}"
            )
    print(f"GATE 0 VERDICT: {'PASS' if gate0_passed else 'FAIL'}  (saved -> {out_dir})")
    print("=" * 78 + "\n", flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--target-model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer-index", type=int, default=24)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--web-dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--web-dataset-config", default="sample-10BT")
    p.add_argument("--web-custom-path", default=None, help="Optional local JSONL/parquet for offline tests")
    p.add_argument("--n-web-train", type=int, default=1600)
    p.add_argument("--n-web-val", type=int, default=400)
    p.add_argument("--web-max-tokens", type=int, default=64)
    p.add_argument("--chat-max-tokens", type=int, default=512)
    p.add_argument("--n-diag-pairs", type=int, default=50)
    p.add_argument("--refusal-source", choices=["auto", "hf", "builtin"], default="auto")
    p.add_argument("--n-neutral", type=int, default=20)
    p.add_argument(
        "--taboo-loras",
        nargs="*",
        default=[o.repo_id for o in PHASE0_TABOO_ORGANISMS],
    )
    p.add_argument("--skip-taboo", action="store_true", default=False)
    p.add_argument("--force-base-wrapper", choices=["chatml", "plain"], default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fail-exit", action="store_true", default=False,
                   help="Always exit 0 even when Gate 0 fails (useful for diagnostics/tests)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_phase0(args)
    if not report["gate0"]["passed"] and not args.no_fail_exit:
        sys.exit(2)


if __name__ == "__main__":
    main()
