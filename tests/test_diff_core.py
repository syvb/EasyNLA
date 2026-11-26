"""CPU unit tests for nla/diff/ core modules: alignment (Fix 1), wrappers (Fix 2 & §2.4),
metrics, and benchmarks.

Tests include real `Qwen/Qwen3-8B` tokenizer verification of ChatML `enable_thinking=False`
vs `Base(Plain)` 1-to-1 token alignment!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from nla.diff.alignment import (
    apply_alignment,
    doc_grouped_folds,
    evaluate_alignment_suite,
    fit_alignment,
    load_alignment,
    predict_aligned_base,
    save_alignment,
)
from nla.diff.benchmarks import (
    ALL_TABOO_WORDS,
    NEUTRAL_QUERIES,
    TABOO_ELICITATION_PROMPTS,
    flatten_binary_pairs,
    get_taboo_organisms,
    load_eval_awareness_pairs,
    load_historical_period_items,
    load_math_items,
    load_refusal_pairs,
    load_sycophancy_pairs,
)
from nla.diff.metrics import (
    bootstrap_auroc_ci,
    bootstrap_ci,
    compute_chat_template_amplification,
    compute_fisher_snr,
    compute_next_token_kl,
    compute_probe_auroc_cv,
    compute_tok1_ridge_fve,
    longest_verbatim_ngram,
    unit_normalize,
    verbatim_ngram_precision,
)
from nla.diff.wrappers import (
    apply_role_centering,
    compute_role_means,
    encode_aligned_pair,
    encode_raw_user_only,
    load_role_means,
    map_inst_pos_to_base_pos,
    save_role_means,
)

# ---------------------------------------------------------------------------
# Fix 1: Identity-Anchored Ridge Alignment
# ---------------------------------------------------------------------------

def test_id_anchored_beats_identity_and_std_ridge_when_small_drift():
    """Synthetic high-d regime mirroring Base->Instruct: W_true = I + small low-rank rotation + noise.

    With N_train comparable to d, Standard Ridge shrinks W toward 0 and underperforms
    Identity, whereas Identity-Anchored Ridge shrinks dW toward 0 and beats BOTH.
    """
    torch.manual_seed(42)
    d = 64
    n_train_docs, tok_per_doc = 40, 4  # N_train = 160 (~2.5 * d)
    n_val = 200

    # True map: near-identity + small rank-4 drift + constant bias + mild noise
    U = torch.randn(d, 4) * 0.15
    V = torch.randn(4, d) * 0.15
    dW_true = U @ V
    b_true = torch.randn(d) * 0.5

    Hb_tr = torch.randn(n_train_docs * tok_per_doc, d) * 3.0
    Ht_tr = Hb_tr + (Hb_tr @ dW_true.T) + b_true + torch.randn_like(Hb_tr) * 0.35
    doc_ids = [f"doc_{i // tok_per_doc}" for i in range(n_train_docs * tok_per_doc)]

    Hb_va = torch.randn(n_val, d) * 3.0
    Ht_va = Hb_va + (Hb_va @ dW_true.T) + b_true + torch.randn_like(Hb_va) * 0.35

    # Check direct eigh solver agrees with torch.linalg.solve to < 1e-6
    art = fit_alignment(Hb_tr, Ht_tr, mode="id_anchored", alpha=1.0)
    pred = predict_aligned_base(Hb_va, art)
    X64 = Hb_tr.double()
    R64 = (Ht_tr - Hb_tr).double()
    xm, rm = X64.mean(0), R64.mean(0)
    C = ((X64 - xm).T @ (X64 - xm)) / len(X64) + 1.0 * torch.eye(d, dtype=torch.float64)
    M = ((X64 - xm).T @ (R64 - rm)) / len(X64)
    dW_ref = torch.linalg.solve(C, M).T
    b_ref = rm - (xm @ dW_ref.T)
    pred_ref = (Hb_va.double() + Hb_va.double() @ dW_ref.T + b_ref).float()
    assert torch.allclose(pred, pred_ref, atol=1e-5)

    # Evaluate full suite (with large alpha grid where standard ridge shrinks toward 0)
    res = evaluate_alignment_suite(
        Hb_tr, Ht_tr, doc_ids, Hb_va, Ht_va,
        alphas=(1e1, 1e2, 1e3, 1e4), n_folds=5, seed=0, layer_index=24,
    )
    assert res["fve_id_anchored"] >= res["fve_identity"] - 1e-4
    assert res["fve_identity"] > res["fve_std_ridge"], (
        f"Expected Identity ({res['fve_identity']:.4f}) > Std Ridge ({res['fve_std_ridge']:.4f})"
    )
    assert res["fix1_strict_passed"] is True
    assert res["tau_web_median"] > 0.0
    assert res["tau_web_p75"] >= res["tau_web_median"]

    # Save / load round-trip
    with tempfile.TemporaryDirectory() as td:
        p = save_alignment(Path(td) / "align_L24.pt", res["artifact_id_anchored"])
        loaded = load_alignment(p)
        d1 = apply_alignment(Hb_va, Ht_va, res["artifact_id_anchored"])
        d2 = apply_alignment(Hb_va, Ht_va, loaded)
        assert torch.allclose(d1, d2, atol=1e-7)


def test_doc_grouped_folds_no_leakage():
    doc_ids = [f"d{i // 3}" for i in range(60)]
    folds = doc_grouped_folds(doc_ids, n_folds=5, seed=7)
    by_doc: dict[str, set[int]] = {}
    for d, f in zip(doc_ids, folds):
        by_doc.setdefault(d, set()).add(int(f))
    assert all(len(s) == 1 for s in by_doc.values())
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Fix 2 & §2.4: ChatML vs Base(Plain) 1-to-1 Alignment & Role Centering
# ---------------------------------------------------------------------------

def test_qwen3_chatml_and_plain_exact_1to1_alignment():
    """Verify exact 1-to-1 token ID alignment on the REAL Qwen/Qwen3-8B tokenizer."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Edge cases: trailing punctuation, leading/trailing spaces, multi-turn history, rollout
    cases = [
        ("How do rainbows form?", None, "Light refracts inside water droplets."),
        ("Wait — why does 0.1 + 0.2 != 0.3?\nPlease explain!", None, "IEEE-754 binary floats cannot represent 0.1 exactly."),
        (
            "Give me a second example.",
            [
                {"role": "user", "content": "What is recursion?"},
                {"role": "assistant", "content": "A function calling itself."},
            ],
            "Sure, tree traversal is another classic example.",
        ),
    ]

    for user_text, history, asst_text in cases:
        inst_c, base_c = encode_aligned_pair(
            tok, user_text, base_wrapper="chatml", history=history,
            assistant_text=asst_text, enable_thinking=False,
        )
        assert inst_c.input_ids == base_c.input_ids
        # Verify enable_thinking=False injected <think>\n\n</think>\n\n before rollout
        header_txt = tok.decode(inst_c.input_ids[inst_c.header_span[0] : inst_c.header_span[1]])
        assert "<think>" in header_txt and "</think>" in header_txt

        inst_p, base_p = encode_aligned_pair(
            tok, user_text, base_wrapper="plain", history=history,
            assistant_text=asst_text, enable_thinking=False,
        )
        # Exact 1-to-1 token ID match on user_content and asst_rollout
        iu0, iu1 = inst_p.user_span
        bu0, bu1 = base_p.user_span
        assert inst_p.input_ids[iu0:iu1] == base_p.input_ids[bu0:bu1]
        assert len(inst_p.user_content_positions) == (iu1 - iu0)
        assert 0 not in inst_p.user_content_positions and 0 not in base_p.user_content_positions

        ir0, ir1 = inst_p.rollout_span
        br0, br1 = base_p.rollout_span
        assert ir1 > ir0
        assert inst_p.input_ids[ir0:ir1] == base_p.input_ids[br0:br1]

        # Position mapper maps every active position to identical token ID (except asst_last trigger)
        for p in inst_p.user_content_positions:
            bp = map_inst_pos_to_base_pos(p, inst_p, base_p)
            assert inst_p.input_ids[p] == base_p.input_ids[bp]
            assert inst_p.role_at(p) == "user_content"
            assert base_p.role_at(bp) == "user_content"
        for p in inst_p.asst_rollout_positions:
            bp = map_inst_pos_to_base_pos(p, inst_p, base_p)
            assert inst_p.input_ids[p] == base_p.input_ids[bp]
            assert inst_p.role_at(p) == "asst_rollout"
            assert base_p.role_at(bp) == "asst_rollout"
        assert map_inst_pos_to_base_pos(inst_p.asst_last_idx, inst_p, base_p) == base_p.asst_last_idx
        assert inst_p.role_at(inst_p.asst_last_idx) == "asst_last"
        assert base_p.role_at(base_p.asst_last_idx) == "asst_last"

        # Template positions must be labeled non-active and rejected for cross-wrapper mapping
        assert inst_p.role_at(0) == "chat_bos"
        assert len(inst_p.asst_header_positions) > 0
        for p in inst_p.asst_header_positions:
            assert inst_p.role_at(p) == "asst_header"
            with pytest.raises(ValueError):
                map_inst_pos_to_base_pos(p, inst_p, base_p)

        # Raw user encoding matches exact user_ids
        raw_enc = encode_raw_user_only(tok, user_text, user_ids=inst_p.input_ids[iu0:iu1])
        assert raw_enc.input_ids == inst_p.input_ids[iu0:iu1]


def test_role_centering_removes_constant_offset_and_boosts_unit_snr():
    """Verify Fix 2 role centering eliminates constant role bias, de-collapses the unit sphere,
    and boosts Fisher SNR when role offsets vary across positions."""
    torch.manual_seed(0)
    d = 64
    n = 100
    labels = np.array([0] * 50 + [1] * 50)
    signal_dir = torch.zeros(d); signal_dir[0] = 1.0
    offset_a = torch.zeros(d); offset_a[1] = 20.0
    offset_b = torch.zeros(d); offset_b[2] = 20.0

    # Role A (even indices) and Role B (odd indices) have huge distinct constant offsets
    roles = ["asst_last" if i % 2 == 0 else "user_content" for i in range(n)]
    deltas = torch.randn(n, d) * 0.5
    for i, r in enumerate(roles):
        deltas[i] += offset_a if r == "asst_last" else offset_b
    deltas[50:] += signal_dir * 3.0

    # Single-role cone collapse before centering (cosine between class means > 0.95)
    asst_raw = deltas[0::2]
    asst_raw_u = unit_normalize(asst_raw)
    cos_before = torch.nn.functional.cosine_similarity(
        asst_raw_u[:25].mean(0, keepdim=True), asst_raw_u[25:].mean(0, keepdim=True)
    ).item()
    assert cos_before > 0.95

    # Mixed-role SNR before centering is crushed by inter-role constant offsets
    snr_raw_unit = compute_fisher_snr(unit_normalize(deltas), labels)

    means = compute_role_means({
        "asst_last": deltas[0::2],
        "user_content": deltas[1::2],
    })
    centered = apply_role_centering(deltas, roles, means)
    assert centered[0::2].mean(dim=0).abs().max().item() < 1e-5
    assert centered[1::2].mean(dim=0).abs().max().item() < 1e-5

    # After centering, class means on the unit sphere become cleanly antipodal (< -0.9)
    asst_cen_u = unit_normalize(centered[0::2])
    cos_after = torch.nn.functional.cosine_similarity(
        asst_cen_u[:25].mean(0, keepdim=True), asst_cen_u[25:].mean(0, keepdim=True)
    ).item()
    assert cos_after < -0.8

    # Mixed-role SNR jumps dramatically once role offsets are removed
    snr_centered_unit = compute_fisher_snr(unit_normalize(centered), labels)
    assert snr_centered_unit > 10.0 * snr_raw_unit

    with tempfile.TemporaryDirectory() as td:
        p = save_role_means(Path(td) / "role_means_L24.pt", means, tau_web_median=12.3, tau_web_p75=16.7)
        loaded = load_role_means(p)
        assert torch.allclose(loaded["asst_last"], means["asst_last"])
        assert abs(loaded["tau_web_median"] - 12.3) < 1e-6
        assert abs(loaded["tau_web_p75"] - 16.7) < 1e-6


# ---------------------------------------------------------------------------
# Metrics: Probing, tok1 Ridge FVE, KL, Verbatim Overlap, Bootstrap CIs
# ---------------------------------------------------------------------------

def test_metrics_suite_sanity():
    torch.manual_seed(1)
    n, d = 120, 64
    y = np.array([0] * 60 + [1] * 60)
    X_sig = torch.randn(n, d)
    X_sig[60:, 0] += 3.0
    X_noise = torch.randn(n, d)

    assert compute_fisher_snr(X_sig, y) > 0.1
    assert compute_probe_auroc_cv(X_sig, y) > 0.95
    assert 0.3 <= compute_probe_auroc_cv(X_noise, y) <= 0.7

    # tok1 Ridge FVE: activations deterministically keyed by token ID -> high FVE; random -> ~0
    toks = np.tile(np.arange(10), 20)  # 200 rows across 10 token IDs
    codebook = torch.randn(10, 32) * 5.0
    H_tok = codebook[toks] + torch.randn(200, 32) * 0.2
    H_rand = torch.randn(200, 32)
    fve_tok = compute_tok1_ridge_fve(H_tok, toks)
    fve_rand = compute_tok1_ridge_fve(H_rand, toks)
    assert fve_tok > 0.95
    assert fve_rand < 0.05

    # Next-token KL
    logits = torch.randn(4, 50)
    assert torch.allclose(compute_next_token_kl(logits, logits), torch.zeros(4), atol=1e-6)
    kl_diff = compute_next_token_kl(logits, logits + torch.randn_like(logits))
    assert (kl_diff > 0).all()

    # Chat template amplification
    amp = compute_chat_template_amplification(torch.ones(5, 8), torch.ones(5, 8) * 2.5)
    assert abs(amp - 2.5) < 1e-5

    # Verbatim n-gram quoting
    src = "The quick brown fox jumps over the lazy dog near the river."
    expl_copy = "The model is reading about how the quick brown fox jumps over a fence."
    expl_abs = "The assistant prepares a cautious refusal due to policy boundaries."
    assert longest_verbatim_ngram(expl_copy, src) == 6
    assert longest_verbatim_ngram(expl_abs, src) <= 1
    assert verbatim_ngram_precision(expl_copy, src, n=4) > 0.2
    assert verbatim_ngram_precision(expl_abs, src, n=4) == 0.0

    # Bootstrap CIs
    ci = bootstrap_ci([0.8, 0.85, 0.9, 0.75, 0.95], n_boot=500, seed=0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    aci = bootstrap_auroc_ci(y, X_sig[:, 0].numpy(), n_boot=300, seed=0)
    assert aci["auroc"] > 0.95 and aci["ci_low"] <= aci["auroc"] <= aci["ci_high"]


# ---------------------------------------------------------------------------
# Benchmarks Registry Integrity
# ---------------------------------------------------------------------------

def test_bundled_benchmarks_integrity():
    assert len(NEUTRAL_QUERIES) == 30
    assert len(TABOO_ELICITATION_PROMPTS) == 50
    assert len(ALL_TABOO_WORDS) == 20

    ea = load_eval_awareness_pairs(100)
    assert len(ea) == 100
    assert len({p["benign"] for p in ea}) == 100
    assert len({p["positive"] for p in ea}) == 100
    prompts, labels, _ = flatten_binary_pairs(ea)
    assert len(prompts) == 200 and sum(labels) == 100

    syc = load_sycophancy_pairs(100)
    assert len(syc) == 100
    assert len({p["benign"] for p in syc}) == 100
    assert len({p["positive"] for p in syc}) == 100
    assert all("ground_truth" in p and p["ground_truth"] for p in syc)

    ref = load_refusal_pairs(25, source="builtin")
    assert len(ref) == 25
    # Verify no punctuation giveaway: benign and harmful prompts share identical terminal punctuation per pair
    for pair in ref:
        assert pair["benign"].endswith("?") == pair["positive"].endswith("?"), (
            f"Terminal punctuation mismatch in {pair['pair_id']}"
        )

    hist = load_historical_period_items(60)
    assert len(hist) == 60
    assert sum(h["binary_label"] == 0 for h in hist) == 30
    assert sum(h["binary_label"] == 1 for h in hist) == 30

    math_items = load_math_items(40, source="builtin")
    assert len(math_items) == 40
    assert all(m["prompt"].endswith("Answer:") for m in math_items)

    paired = get_taboo_organisms("paired")
    distinct = get_taboo_organisms("distinct")
    assert len(paired) == 10 and sum(o.regime == "narrow" for o in paired) == 5
    assert len(distinct) == 10 and len({o.secret_word for o in distinct}) == 10


def test_long_multiturn_history_truncates_and_separates_history_from_sys_prefix():
    """Verify multi-turn conversations exceeding max_total_tokens drop oldest turns cleanly
    and label earlier turns as 'history' instead of 'sys_prefix'."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    # Build >1000-token multi-turn history (previously raised ValueError at max_total_tokens=512)
    long_history = []
    for i in range(10):
        long_history.append({"role": "user", "content": f"Turn {i}: " + ("oceanographic data " * 30)})
        long_history.append({"role": "assistant", "content": f"Reply {i}: " + ("atmospheric profile " * 30)})
    user_text = "Summarize the key takeaway in one concise sentence."

    inst_p, base_p = encode_aligned_pair(
        tok,
        user_text,
        base_wrapper="plain",
        history=long_history,
        enable_thinking=False,
        max_total_tokens=512,
    )
    assert len(inst_p.input_ids) <= 512
    assert inst_p.kept_history_start_idx > 0
    assert len(inst_p.user_content_positions) > 0
    assert inst_p.input_ids[inst_p.user_span[0] : inst_p.user_span[1]] == (
        base_p.input_ids[base_p.user_span[0] : base_p.user_span[1]]
    )

    # Short multi-turn conversation: earlier-turn content must have role == 'history', NOT 'sys_prefix'
    short_history = [
        {"role": "user", "content": "Hello, let us discuss alpine ecology."},
        {"role": "assistant", "content": "Sure, alpine ecosystems adapt to low temperatures."},
    ]
    inst_s, _ = encode_aligned_pair(
        tok,
        "What animals live above the tree line?",
        base_wrapper="plain",
        history=short_history,
        enable_thinking=False,
        max_total_tokens=512,
    )
    assert len(inst_s.history_positions) > 10
    assert all(inst_s.role_at(p) == "history" for p in inst_s.history_positions)
    assert all(inst_s.role_at(p) == "sys_prefix" for p in inst_s.sys_prefix_positions)
    assert set(inst_s.history_positions).isdisjoint(set(inst_s.sys_prefix_positions))

