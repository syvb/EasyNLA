"""Future-lens unit tests. CPU-only. Run: .venv/bin/python -m pytest tests/test_future_lens.py -q

Model-backed tests use the locally cached Qwen/Qwen3-0.6B (skipped if absent).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SMALL = "Qwen/Qwen3-0.6B"


def _have_model():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(SMALL)
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(not _have_model(), reason=f"{SMALL} not cached")


# ---------------------------------------------------------------- pure -----

def test_scale_for_injection_per_row():
    from nla.future_lens.inject import scale_for_injection
    v = torch.randn(3, 16)
    a = torch.tensor([1.0, 5.0, 10.0])
    out = scale_for_injection(v, a)
    assert torch.allclose(out.norm(dim=-1), a, atol=1e-5)
    assert torch.allclose(scale_for_injection(v, 7.0).norm(dim=-1), torch.full((3,), 7.0), atol=1e-5)


def test_affine_identity_init():
    from nla.future_lens.inject import AffineInjector
    aff = AffineInjector(8)
    x = torch.randn(4, 8)
    assert torch.allclose(aff(x), x, atol=1e-6)


def test_exact_match_reward():
    from nla.future_lens.rewards import exact_match_reward
    tgt = np.array([5, 6, 7, 8, 9])
    # full match on K=3, one wrong, overrun truncated, short (len violation)
    r, viol = exact_match_reward([5, 6, 7], tgt, 3, length_penalty=0.1)
    assert r == pytest.approx(1.0) and not viol
    r, viol = exact_match_reward([5, 0, 7], tgt, 3, length_penalty=0.1)
    assert r == pytest.approx(2 / 3) and not viol
    r, viol = exact_match_reward([5, 6, 7, 8], tgt, 3, length_penalty=0.1)
    assert r == pytest.approx(1.0 - 0.1) and viol
    r, viol = exact_match_reward([5, 6], tgt, 3, length_penalty=0.1)
    assert r == pytest.approx(2 / 3 - 0.1) and viol
    r, viol = exact_match_reward([], tgt, 3, length_penalty=0.1)
    assert r == pytest.approx(-0.1) and viol


def test_per_offset_hits_and_logp_sum_reward():
    from nla.future_lens.rewards import per_offset_hits, target_logp_sum_reward
    tgt = np.array([5, 6, 7, 8, 9])
    assert per_offset_hits([5, 0], tgt, 4) == [1, 0, 0, 0]          # short readout = misses, no None
    assert per_offset_hits([5, 6, 7, 8, 9, 9], tgt, 9) == [1, 1, 1, 1, 1, 0, 0, 0, 0]  # k > len(target) safe
    # summed reward: an honest K=9 readout at -3.5/token beats "1 good token + EOS"
    honest = target_logp_sum_reward([-3.5] * 9, 9)
    lazy = target_logp_sum_reward([-1.5], 9, length_violation=True)
    assert honest == pytest.approx(-3.5) and lazy < honest
    assert target_logp_sum_reward(None, 3) == pytest.approx(-4.0)
    assert target_logp_sum_reward([-1.0, -1.0, -1.0], 3) == pytest.approx(-1.0)


def test_group_advantages():
    from nla.future_lens.rewards import group_advantages
    r = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ok = torch.ones(8, dtype=torch.bool)
    adv = group_advantages(r, g, ok, n_groups=2, norm="none")
    assert torch.allclose(adv[:4], torch.tensor([0.5, -0.5, 0.5, -0.5]))
    assert torch.all(adv[4:] == 0)                    # zero-variance group -> no gradient
    adv_std = group_advantages(r, g, ok, n_groups=2, norm="std")
    assert adv_std[0] > adv[0]                         # std-normalised is larger (std<1)
    ok2 = ok.clone(); ok2[0] = False
    adv2 = group_advantages(r, g, ok2, n_groups=2, norm="none")
    assert adv2[0] == 0 and torch.allclose(adv2[1:4], torch.tensor([-1 / 3, 2 / 3, -1 / 3]))


def test_ngram_model_greedy():
    from nla.future_lens.baselines import NGramModel
    seqs = [np.array([1, 2, 3, 1, 2, 3, 1, 2, 4]), np.array([1, 2, 3, 1, 2, 3])]
    m = NGramModel(order=2)
    for s in seqs:
        m.add(s)
    assert m.predict([1]) == 2 and m.predict([2]) == 3
    assert m.greedy([1], 3) == [2, 3, 1]
    m4 = NGramModel(order=4)
    for s in seqs:
        m4.add(s)
    assert m4.predict([3, 1, 2]) == 3        # trigram context (3,1,2) seen -> 3
    assert m4.predict([9, 9, 1]) == 2        # unseen higher-order contexts -> backoff to bigram
    assert NGramModel(order=2).predict([7]) == 0   # empty model -> 0, no crash


def test_shuffle_activations_changes_rows():
    from nla.future_lens.data import shuffle_activations
    rows = [{"activation_layer": l, "activation_vector": np.full(4, i, dtype=np.float16)}
            for i, l in enumerate([4, 4, 4, 8, 8, 8])]
    before = [r["activation_vector"].copy() for r in rows]
    shuffle_activations(rows, seed=0)
    for r, b in zip(rows, before):
        assert not np.array_equal(r["activation_vector"], b)
    # layer 4 rows still hold layer-4 vectors (values 0,1,2)
    assert {int(r["activation_vector"][0]) for r in rows[:3]} == {0, 1, 2}


# ---------------------------------------------------------- model-backed ----

@needs_model
def test_prompt_marker_neighbours_stable_across_layer_and_k():
    from transformers import AutoTokenizer
    from nla.datagen.injection_tokens import build_token_meta
    from nla.future_lens.data import DEFAULT_TEMPLATE, build_prompt_messages, encode_prompt, fill_template
    from nla.injection import marker_well_formed
    tok = AutoTokenizer.from_pretrained(SMALL)
    meta = build_token_meta(tok, fill_template(DEFAULT_TEMPLATE, 0, 1, "{injection_char}"))
    for layer in (4, 12, 24):
        for k in (1, 5, 9):
            ids = encode_prompt(tok, build_prompt_messages(DEFAULT_TEMPLATE, layer, k), meta.injection_char)
            assert marker_well_formed(ids, meta.injection_token_id,
                                      meta.injection_left_neighbor_id, meta.injection_right_neighbor_id)
    txt = tok.apply_chat_template(build_prompt_messages(DEFAULT_TEMPLATE, 4, 1), tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert "<think>" in txt and "</think>" in txt      # thinking explicitly closed off


@needs_model
def test_replace_embed_hook_writes_exactly_the_marker():
    """Spec: confirm the marker's layer-0 input equals alpha*h/|h| and nothing else changes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.datagen.injection_tokens import build_token_meta
    from nla.future_lens.data import DEFAULT_TEMPLATE, build_prompt_messages, encode_prompt, fill_template
    from nla.future_lens.inject import prepare_vectors, register_replace_embed_hook
    tok = AutoTokenizer.from_pretrained(SMALL)
    model = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=torch.float32).eval()
    meta = build_token_meta(tok, fill_template(DEFAULT_TEMPLATE, 0, 1, "{injection_char}"))
    inj, lft, rgt = meta.injection_token_id, meta.injection_left_neighbor_id, meta.injection_right_neighbor_id
    d = model.config.hidden_size
    ids = torch.tensor([encode_prompt(tok, build_prompt_messages(DEFAULT_TEMPLATE, 8, 3), meta.injection_char),
                        encode_prompt(tok, build_prompt_messages(DEFAULT_TEMPLATE, 8, 9), meta.injection_char)])
    captured = {}
    model.model.layers[0].register_forward_pre_hook(
        lambda m, a, kw: captured.__setitem__("x", (a[0] if a else kw["hidden_states"]).detach().clone()),
        with_kwargs=True)
    vref = [None]
    register_replace_embed_hook(model, vref, inj, lft, rgt)
    with torch.no_grad():
        model(input_ids=ids)
        clean = captured["x"]
        raw = torch.randn(2, d) * 30
        alphas = torch.tensor([50.0, 120.0])
        vref[0] = prepare_vectors(raw, alphas, "replace_embed")
        model(input_ids=ids)
        injected = captured["x"]
        vref[0] = None
        model(input_ids=ids)
        again = captured["x"]
    pos = [(ids[b] == inj).nonzero().item() for b in range(2)]
    for b in range(2):
        want = raw[b] / raw[b].norm() * alphas[b]
        assert torch.allclose(injected[b, pos[b]], want, atol=1e-4)
        mask = torch.ones(ids.shape[1], dtype=torch.bool); mask[pos[b]] = False
        assert torch.equal(injected[b, mask], clean[b, mask])
    assert torch.equal(again, clean)                  # hook is inert when vectors_ref is None
    # count mismatch fails loud
    vref[0] = prepare_vectors(raw[:1], alphas[:1], "replace_embed")
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(input_ids=ids)


@needs_model
def test_replace_embed_hook_under_generate_and_peft():
    """Decode steps (S=1) must not trip the count assert; PEFT wrapper still injects."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.datagen.injection_tokens import build_token_meta
    from nla.future_lens.data import DEFAULT_TEMPLATE, build_prompt_messages, encode_prompt, fill_template
    from nla.future_lens.inject import prepare_vectors, register_replace_embed_hook
    tok = AutoTokenizer.from_pretrained(SMALL)
    model = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=torch.float32).eval()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    meta = build_token_meta(tok, fill_template(DEFAULT_TEMPLATE, 0, 1, "{injection_char}"))
    vref = [None]
    register_replace_embed_hook(model, vref, meta.injection_token_id,
                                meta.injection_left_neighbor_id, meta.injection_right_neighbor_id)
    ids = torch.tensor([encode_prompt(tok, build_prompt_messages(DEFAULT_TEMPLATE, 8, 2), meta.injection_char)])
    vref[0] = prepare_vectors(torch.randn(1, model.config.hidden_size), 60.0, "replace_embed")
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    assert out.shape[1] == ids.shape[1] + 3


@needs_model
def test_target_logp_reward_prefers_true_continuation():
    """Surprisal machinery: the true continuation must score far above random tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.future_lens.rewards import target_logp_reward
    tok = AutoTokenizer.from_pretrained(SMALL)
    model = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=torch.float32).eval()
    text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy"
    ids = tok.encode(text, add_special_tokens=False)
    prefix, true_next = np.array(ids[:-3]), ids[-3:]
    rng = np.random.default_rng(0)
    rand = [int(x) for x in rng.integers(1000, 50000, size=3)]
    lp = target_logp_reward(model, [prefix, prefix, prefix], [true_next, rand, []], "cpu",
                            pad_id=tok.eos_token_id, micro_batch=2, disable_adapter=False)
    assert lp[2] is None
    assert lp[0] > -2.0 and lp[1] < -8.0 and lp[0] > lp[1] + 5
    # left-padding must not change the answer: score alone vs in a batch with a longer prefix
    long_prefix = np.array(tok.encode("Once upon a time, " * 5 + text, add_special_tokens=False)[:-3])
    lp2 = target_logp_reward(model, [prefix, long_prefix], [true_next, true_next], "cpu",
                             pad_id=tok.eos_token_id, micro_batch=2, disable_adapter=False)
    assert abs(lp2[0] - lp[0]) < 1e-3
