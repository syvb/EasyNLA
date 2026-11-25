"""Unit tests for Fix 3 (--soft-norm-tau) across all three EasyNLA patch sites.

Verifies:
  1. tau == 0.0 reproduces baseline EasyNLA outputs bitwise / within 1e-5 across:
       - HF injection (`karvonen_inject_in_residual` & `register_karvonen_hook`)
       - vLLM steering (`build_steering_vector` simulated against vllm-lens norm_match)
       - AR critic MSE loss (`weighted_mse_loss` vs `F.mse_loss`)
       - GRPO policy advantages (`apply_soft_norm_to_advantages`)
  2. tau > 0.0:
       - Tiny noise vector (||v|| = 0.05 * tau) is injected at ~0.05 * ||h_p||
         (NOT 1.0 * ||h_p||!) in BOTH HF and vLLM paths, and both paths agree
         within 1e-5.
       - Large signal vector (||v|| = 10 * tau) is injected near ~0.995 * ||h_p||.
       - Exact zero vector produces finite, unperturbed output (no 0/0 NaN).
       - Critic weight w_i ≈ 0.002494 for ||v|| = 0.05 * tau and is invariant to
         micro-batch size.
       - GRPO weighted_adv shrinks noise-prompt advantages by w_i / w_mean and
         is NOT canceled by per-group standardization std_j.
  3. End-to-end tiny-Llama CPU forward + backward test for:
       - `register_karvonen_hook(..., soft_norm_tau=tau)` + `grpo_update_microbatched`
       - `init_coldstart_ar` checkpoint round-trip via `NLACriticModel.from_pretrained`
       - Sidecar round-trip with `injection_scale="raw"`, `mse_scale="sqrt_d_model"`,
         `soft_norm_tau`.
"""

from __future__ import annotations

import math
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    deserialize_sidecar,
    serialize_sidecar,
)
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import NLATokenMeta, normalize_activation
from nla.train_rl_vllm import grpo_update_microbatched
from nla.utils.hooks import register_karvonen_hook
from nla.utils.soft_norm import (
    apply_soft_norm_to_advantages,
    soft_norm_scale_factor,
    soft_norm_weights,
    weighted_mse_loss,
)
from nla.utils.vllm_steer import build_steering_vector


def _tiny_llama(d: int = 32, layers: int = 3, vocab: int = 100) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=d,
            intermediate_size=d * 2,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=vocab,
        )
    ).float()


def _install_fake_vllm_lens(monkeypatch):
    """Register a minimal stand-in `vllm_lens.SteeringVector` for CPU tests."""
    @dataclass
    class FakeSteeringVector:
        activations: torch.Tensor
        layer_indices: list[int]
        scale: float
        norm_match: bool
        position_indices: list[int]

    mod = types.ModuleType("vllm_lens")
    mod.SteeringVector = FakeSteeringVector
    monkeypatch.setitem(sys.modules, "vllm_lens", mod)
    return FakeSteeringVector


def _simulate_vllm_lens_apply(h_p: torch.Tensor, sv) -> torch.Tensor:
    """Exact formula executed by patched `vllm_lens._worker_ext._apply_steering`."""
    v = sv.activations[0, 0].to(h_p.device, h_p.dtype)
    if sv.norm_match:
        # Mirrors vllm_lens norm_match(norm_ref[rel], v): scale v to ||h_p||
        v = v * (h_p.norm() / (v.norm() + 1e-8))
    return h_p + v * sv.scale


def test_tau_zero_matches_baseline_hf_injection():
    torch.manual_seed(7)
    INJ, L, R = 42, 41, 43
    ids = torch.tensor([[1, L, INJ, R, 5, 6], [2, 3, L, INJ, R, 7]])
    resid = torch.randn(2, 6, 64)
    vecs = torch.randn(2, 64) * 3.7

    out_default = karvonen_inject_in_residual(ids, resid, vecs, INJ, L, R)
    out_zero = karvonen_inject_in_residual(ids, resid, vecs, INJ, L, R, soft_norm_tau=0.0)
    assert torch.equal(out_default, out_zero)

    # Verify exact Karvonen formula at marker positions
    for b, p in [(0, 2), (1, 3)]:
        h_p = resid[b, p]
        expected = h_p + h_p.norm() * (vecs[b] / (vecs[b].norm() + 1e-9))
        assert torch.allclose(out_zero[b, p], expected, atol=1e-6)


def test_soft_norm_hf_and_vllm_agree_and_shrink_noise(monkeypatch):
    _install_fake_vllm_lens(monkeypatch)
    torch.manual_seed(11)
    tau = 20.0
    d = 64
    INJ, L, R = 42, 41, 43
    ids = torch.tensor([[1, L, INJ, R, 5]])
    resid = torch.randn(1, 5, d)
    h_p = resid[0, 2]
    h_norm = h_p.norm().item()

    # Case A: tiny noise vector ||v|| = 0.05 * tau
    direction = F.normalize(torch.randn(d), dim=-1)
    v_noise = direction * (0.05 * tau)
    out_hf_noise = karvonen_inject_in_residual(
        ids, resid, v_noise.unsqueeze(0), INJ, L, R, soft_norm_tau=tau
    )
    delta_hf = out_hf_noise[0, 2] - h_p
    expected_ratio = 0.05 / math.sqrt(0.05**2 + 1.0)  # ~0.049938
    assert abs(delta_hf.norm().item() / h_norm - expected_ratio) < 1e-4
    # Must NOT be unit-normalized to 1.0 * ||h_p||
    assert delta_hf.norm().item() < 0.06 * h_norm

    sv_noise = build_steering_vector(v_noise, marker_pos=2, injection_layer=1, soft_norm_tau=tau)
    out_vllm_noise = _simulate_vllm_lens_apply(h_p, sv_noise)
    assert torch.allclose(out_hf_noise[0, 2], out_vllm_noise, atol=1e-5)

    # Case B: strong signal vector ||v|| = 10 * tau
    v_sig = direction * (10.0 * tau)
    out_hf_sig = karvonen_inject_in_residual(
        ids, resid, v_sig.unsqueeze(0), INJ, L, R, soft_norm_tau=tau
    )
    delta_sig = out_hf_sig[0, 2] - h_p
    expected_sig_ratio = 10.0 / math.sqrt(10.0**2 + 1.0)  # ~0.995037
    assert abs(delta_sig.norm().item() / h_norm - expected_sig_ratio) < 1e-4
    sv_sig = build_steering_vector(v_sig, marker_pos=2, injection_layer=1, soft_norm_tau=tau)
    out_vllm_sig = _simulate_vllm_lens_apply(h_p, sv_sig)
    assert torch.allclose(out_hf_sig[0, 2], out_vllm_sig, atol=1e-5)

    # Case C: exact zero vector -> finite, zero perturbation (no 0/0 NaN)
    v_zero = torch.zeros(d)
    out_hf_zero = karvonen_inject_in_residual(
        ids, resid, v_zero.unsqueeze(0), INJ, L, R, soft_norm_tau=tau
    )
    assert torch.isfinite(out_hf_zero).all()
    assert torch.allclose(out_hf_zero[0, 2], h_p, atol=1e-7)
    sv_zero = build_steering_vector(v_zero, marker_pos=2, injection_layer=1, soft_norm_tau=tau)
    assert sv_zero.scale == 0.0 and sv_zero.norm_match is False
    out_vllm_zero = _simulate_vllm_lens_apply(h_p, sv_zero)
    assert torch.isfinite(out_vllm_zero).all()
    assert torch.allclose(out_vllm_zero, h_p, atol=1e-7)

    # Case D: tau == 0.0 vLLM matches baseline scale=1.0, norm_match=True
    sv_base = build_steering_vector(v_noise, marker_pos=2, injection_layer=1, soft_norm_tau=0.0)
    assert sv_base.scale == 1.0 and sv_base.norm_match is True
    out_hf_base = karvonen_inject_in_residual(
        ids, resid, v_noise.unsqueeze(0), INJ, L, R, soft_norm_tau=0.0
    )
    out_vllm_base = _simulate_vllm_lens_apply(h_p, sv_base)
    assert torch.allclose(out_hf_base[0, 2], out_vllm_base, atol=1e-5)


def test_soft_norm_critic_weights_and_microbatch_invariance():
    torch.manual_seed(3)
    tau = 10.0
    d = 32
    mse_scale = math.sqrt(d)

    # Build 8 vectors: 4 noise (0.05*tau) and 4 signal (5.0*tau)
    dirs = F.normalize(torch.randn(8, d), dim=-1)
    norms = torch.tensor([0.05, 0.05, 0.05, 0.05, 5.0, 5.0, 5.0, 5.0]) * tau
    golds = dirs * norms.unsqueeze(-1)
    preds = torch.randn(8, d, requires_grad=True)

    w = soft_norm_weights(golds, tau)
    expected_w_noise = (0.05**2) / (0.05**2 + 1.0)  # ~0.0024938
    expected_w_sig = (5.0**2) / (5.0**2 + 1.0)      # ~0.9615385
    assert torch.allclose(w[:4], torch.full((4,), expected_w_noise), atol=1e-5)
    assert torch.allclose(w[4:], torch.full((4,), expected_w_sig), atol=1e-5)

    # tau == 0.0 equals standard F.mse_loss
    pred_n = normalize_activation(preds, mse_scale)
    gold_n = normalize_activation(golds, mse_scale)
    w_zero = soft_norm_weights(golds, 0.0)
    assert torch.equal(w_zero, torch.ones(8))
    loss_w0 = weighted_mse_loss(pred_n, gold_n, w_zero, w_zero.sum())
    loss_ref = F.mse_loss(pred_n, gold_n)
    assert torch.allclose(loss_w0, loss_ref, atol=1e-6)

    # Micro-batch invariance across chunk sizes 1, 2, 4, 8
    w_sum = w.sum()
    full_loss = weighted_mse_loss(pred_n, gold_n, w, w_sum)
    (full_grad,) = torch.autograd.grad(full_loss, preds, retain_graph=True)

    for cmb in (1, 2, 4):
        preds_mb = preds.detach().clone().requires_grad_(True)
        total_loss_val = 0.0
        for cs in range(0, 8, cmb):
            sl = slice(cs, cs + cmb)
            pn = normalize_activation(preds_mb[sl], mse_scale)
            gn = normalize_activation(golds[sl], mse_scale)
            l_mb = weighted_mse_loss(pn, gn, w[sl], w_sum)
            l_mb.backward()
            total_loss_val += l_mb.item()
        assert abs(total_loss_val - full_loss.item()) < 1e-6
        assert torch.allclose(preds_mb.grad, full_grad, atol=1e-6)


def test_grpo_post_standardization_advantage_weighting():
    """Ensure w_i is applied AFTER group standardization and is NOT canceled by std_j."""
    torch.manual_seed(5)
    tau = 15.0
    d = 32
    group_size = 8

    # Prompt 0: tiny noise (0.05 * tau), Prompt 1: strong signal (5.0 * tau)
    p0 = F.normalize(torch.randn(d), dim=-1) * (0.05 * tau)
    p1 = F.normalize(torch.randn(d), dim=-1) * (5.0 * tau)
    prompt_acts = torch.stack([p0, p1], dim=0)

    # Give BOTH prompts identical standardized advantages
    group_t = torch.tensor([0] * group_size + [1] * group_size, dtype=torch.long)
    rewards = torch.tensor(
        [-0.5, -0.4, -0.6, -0.3, -0.7, -0.2, -0.8, -0.5] * 2, dtype=torch.float32
    )
    adv = torch.zeros_like(rewards)
    for gi in (0, 1):
        m = group_t == gi
        r_g = rewards[m]
        adv[m] = (r_g - r_g.mean()) / (r_g.std() + 1e-6)

    # Confirm pre-weighting advantages are identical across the two prompts
    assert torch.allclose(adv[:group_size], adv[group_size:], atol=1e-6)

    # tau == 0.0 is an exact no-op
    adv_0, m_0 = apply_soft_norm_to_advantages(adv, group_t, prompt_acts, tau=0.0)
    assert torch.equal(adv_0, adv) and m_0 == {}

    # tau > 0.0 scales prompt 0 down by w_0 / w_mean and prompt 1 up by w_1 / w_mean
    w_adv, metrics = apply_soft_norm_to_advantages(adv, group_t, prompt_acts, tau=tau)
    w = soft_norm_weights(prompt_acts, tau)
    w_mean = w.mean().item()
    expected_scale_0 = w[0].item() / (w_mean + 1e-8)
    expected_scale_1 = w[1].item() / (w_mean + 1e-8)
    assert torch.allclose(w_adv[:group_size], adv[:group_size] * expected_scale_0, atol=1e-5)
    assert torch.allclose(w_adv[group_size:], adv[group_size:] * expected_scale_1, atol=1e-5)
    # Ratio between prompt 0 and prompt 1 magnitudes must match w_0 / w_1 (~0.00259)
    ratio = w_adv[:group_size].abs().mean().item() / w_adv[group_size:].abs().mean().item()
    assert abs(ratio - (w[0].item() / w[1].item())) < 1e-5
    assert ratio < 0.005
    assert "av/soft_norm_w_mean" in metrics and "av/soft_norm_ess_frac" in metrics


def test_hook_and_grpo_end_to_end_tiny_llama():
    """Tiny-Llama CPU forward+backward through register_karvonen_hook(soft_norm_tau) + GRPO."""
    INJ, L, R = 42, 41, 43
    d = 32
    tau = 10.0

    def run_once(soft_norm_tau: float, vec_scale: float):
        base = _tiny_llama(d=d, layers=3, vocab=100)
        actor = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )
        torch.manual_seed(1)
        with torch.no_grad():
            for n, p in actor.named_parameters():
                if "lora_" in n:
                    p.copy_(torch.randn_like(p) * 0.05)
        vref = [None]
        register_karvonen_hook(actor, vref, INJ, L, R, layer_idx=1, soft_norm_tau=soft_norm_tau)
        optim = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.0)

        class Tok:
            eos_token_id = 0

        seq = torch.tensor([1, L, INJ, R, 5, 6, 7, 8, 9, 10], dtype=torch.long)
        torch.manual_seed(2)
        v = F.normalize(torch.randn(d), dim=-1) * vec_scale
        grpo_update_microbatched(
            actor,
            optim,
            Tok(),
            [seq],
            [5],
            [v],
            torch.tensor([1.0]),
            vref,
            "cpu",
            micro_batch=1,
            kl_beta=0.01,
            do_step=False,
            n_total=1,
        )
        return {
            n: p.grad.clone()
            for n, p in actor.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    # With tau=0, scaling vector magnitude does NOT change injection (unit-normed) -> identical grads
    g_t0_small = run_once(0.0, 0.5)
    g_t0_large = run_once(0.0, 50.0)
    for k in g_t0_small:
        assert torch.allclose(g_t0_small[k], g_t0_large[k], atol=1e-5)

    # With tau=10, scaling vector magnitude DOES change injection -> distinct grads
    g_t10_small = run_once(tau, 0.5)
    g_t10_large = run_once(tau, 50.0)
    max_diff = max((g_t10_small[k] - g_t10_large[k]).abs().max().item() for k in g_t10_small)
    assert max_diff > 1e-4


def test_init_coldstart_ar_roundtrip():
    from scripts.init_coldstart_ar import init_coldstart_ar

    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "base"
        out_dir = Path(td) / "coldstart_ar"
        _tiny_llama(d=32, layers=4, vocab=80).save_pretrained(base_dir)
        init_coldstart_ar(str(base_dir), ar_num_layers=3, output_dir=out_dir, dtype=torch.float32)
        reloaded = NLACriticModel.from_pretrained(str(out_dir), torch_dtype=torch.float32)
        assert reloaded.config.num_hidden_layers == 3
        assert torch.allclose(reloaded.value_head.weight, torch.eye(32), atol=1e-6)


def test_sidecar_optional_diff_fields_roundtrip():
    # Absent optional fields -> omitted in YAML -> mse_scale still defaults to sqrt_d_model
    meta_plain = NLADatasetMeta(
        dataset_id="test_plain",
        stage="rl",
        row_count=10,
        extraction=NLAExtractionMeta(
            base_model="dummy",
            d_model=64,
            layer_index=2,
            norm="none",
            corpus="c",
            corpus_slice={"start": 0, "length": 10},
            positions_per_doc=1,
        ),
        tokens=NLATokenMeta(
            injection_char="X",
            injection_token_id=10,
            injection_left_neighbor_id=9,
            injection_right_neighbor_id=11,
        ),
        prompt_templates={"actor": "a {injection_char} b", "critic": "<text>{explanation}</text>"},
    )
    yml_plain = serialize_sidecar(meta_plain)
    assert "injection_scale" not in yml_plain and "mse_scale" not in yml_plain
    rt_plain = deserialize_sidecar(yml_plain)
    assert rt_plain.extraction.injection_scale is None

    # Explicit Fix 3 diff fields round-trip cleanly
    meta_diff = NLADatasetMeta(
        dataset_id="test_diff",
        stage="rl",
        row_count=10,
        extraction=NLAExtractionMeta(
            base_model="dummy",
            d_model=64,
            layer_index=2,
            norm="none",
            corpus="c",
            corpus_slice={"start": 0, "length": 10},
            positions_per_doc=3,
            injection_scale="raw",
            mse_scale="sqrt_d_model",
            soft_norm_tau=12.5,
        ),
        tokens=meta_plain.tokens,
        prompt_templates=meta_plain.prompt_templates,
    )
    yml_diff = serialize_sidecar(meta_diff)
    assert "injection_scale: raw" in yml_diff
    assert "mse_scale: sqrt_d_model" in yml_diff
    assert "soft_norm_tau: 12.5" in yml_diff
    rt_diff = deserialize_sidecar(yml_diff)
    assert rt_diff.extraction.injection_scale == "raw"
    assert rt_diff.extraction.mse_scale == "sqrt_d_model"
    assert rt_diff.extraction.soft_norm_tau == 12.5


def test_negative_tau_rejected():
    with pytest.raises(ValueError):
        soft_norm_weights(torch.randn(2, 8), -1.0)
    with pytest.raises(ValueError):
        soft_norm_scale_factor(1.0, -0.5)
    with pytest.raises(ValueError):
        karvonen_inject_in_residual(
            torch.tensor([[1, 2, 3, 4]]),
            torch.randn(1, 4, 8),
            torch.randn(1, 8),
            2,
            1,
            3,
            soft_norm_tau=-0.1,
        )
