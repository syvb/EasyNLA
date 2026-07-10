"""Verification of the embed_replace injection hook (classic-NLA mechanism)
added alongside the Karvonen hook for the injection-method comparison.

Concrete executable checks on a tiny Llama, CPU-only.
Run: python tests/test_injection_methods.py
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS = []

INJ, LEFT, RIGHT = 42, 41, 43
D = 64


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, None)); print(f"PASS {name}")
    except Exception as e:
        RESULTS.append((name, e)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=D, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=100)).float()


def marked_ids():
    # [B=2, S=8], one valid marker per row at different positions
    return torch.tensor([
        [1, LEFT, INJ, RIGHT, 5, 6, 7, 8],
        [1, 2, 3, LEFT, INJ, RIGHT, 7, 8],
    ])


def t_replaces_marker_row_with_scaled_vector():
    from nla.utils.hooks import register_embed_replace_hook
    from nla.schema import normalize_activation
    model = tiny_model()
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=8.0)
    captured = {}
    # registered AFTER the replace hook -> sees its modified output
    model.get_input_embeddings().register_forward_hook(
        lambda m, a, o: captured.__setitem__("emb", o.detach().clone()))
    ids = marked_ids()
    v = torch.randn(2, D) * 37.0  # far off scale on purpose
    baseline = model.get_input_embeddings()(ids).detach()

    vref[0] = v
    model(input_ids=ids)
    vref[0] = None

    emb = captured["emb"]
    want = normalize_activation(v, 8.0)
    assert torch.allclose(emb[0, 2], want[0], atol=1e-5), "row 0 marker not replaced"
    assert torch.allclose(emb[1, 4], want[1], atol=1e-5), "row 1 marker not replaced"
    assert abs(emb[0, 2].norm().item() - 8.0) < 1e-3, "injection_scale not applied"
    # every non-marker row untouched
    mask = torch.ones(2, 8, dtype=torch.bool); mask[0, 2] = False; mask[1, 4] = False
    assert torch.allclose(emb[mask], baseline[mask]), "non-marker rows modified"


def t_raw_scale_passes_vector_through():
    from nla.utils.hooks import register_embed_replace_hook
    model = tiny_model()
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=None)
    captured = {}
    model.get_input_embeddings().register_forward_hook(
        lambda m, a, o: captured.__setitem__("emb", o.detach().clone()))
    v = torch.randn(2, D) * 37.0
    vref[0] = v
    model(input_ids=marked_ids())
    vref[0] = None
    assert torch.allclose(captured["emb"][0, 2], v[0], atol=1e-5), (
        "scale=None must inject the raw vector")


def t_noop_when_vectors_none_or_decode_step():
    from nla.utils.hooks import register_embed_replace_hook
    model = tiny_model()
    ref_logits = model(input_ids=marked_ids()).logits
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=8.0)
    # vectors None -> bit-identical forward
    out = model(input_ids=marked_ids()).logits
    assert torch.equal(out, ref_logits), "hook changed output with vectors=None"
    # seq_len==1 (autoregressive decode step) -> no-op, and crucially no
    # count-mismatch RuntimeError even though vectors are staged
    vref[0] = torch.randn(2, D)
    model(input_ids=torch.tensor([[5], [7]]))
    vref[0] = None


def t_gradient_flow_and_marker_row_isolation():
    from nla.utils.hooks import register_embed_replace_hook
    import torch.nn.functional as F
    model = tiny_model()
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=8.0)
    ids = marked_ids()
    vref[0] = torch.randn(2, D)
    logits = model(input_ids=ids).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 100), ids[:, 1:].reshape(-1))
    loss.backward()
    vref[0] = None
    g = model.get_input_embeddings().weight.grad
    assert g is not None and torch.isfinite(g).all()
    # the marker id (42) only occurs at replaced positions -> its embedding row
    # must receive ZERO gradient; a used, non-replaced id must receive some
    assert torch.all(g[INJ] == 0), "replaced marker row got gradient"
    assert g[LEFT].abs().sum() > 0, "left-neighbor row got no gradient"


def t_generate_prefill_injects_decode_noops():
    from nla.utils.hooks import register_embed_replace_hook
    model = tiny_model()
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=8.0)
    ids = marked_ids()[:1]
    vref[0] = torch.randn(1, D)
    try:
        seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=3, do_sample=False, pad_token_id=0)
    finally:
        vref[0] = None
    assert seq.shape[1] == ids.shape[1] + 3
    # injection must change what gets generated vs a fresh un-hooked model
    # (same seed weights) ONLY via the marker row; just assert it ran — the
    # decode steps would have raised count-mismatch if the seq_len<2 guard broke


def t_surplus_marker_still_fails_loud():
    from nla.utils.hooks import register_embed_replace_hook
    model = tiny_model()
    vref = [None]
    register_embed_replace_hook(model, vref, INJ, LEFT, RIGHT, injection_scale=8.0)
    two_markers = torch.tensor([[1, LEFT, INJ, RIGHT, LEFT, INJ, RIGHT, 8]])
    vref[0] = torch.randn(1, D)
    try:
        model(input_ids=two_markers)
    except RuntimeError:
        pass
    else:
        raise AssertionError("2 valid markers with 1 vector must raise")
    finally:
        vref[0] = None


def t_methods_actually_differ():
    from nla.utils.hooks import register_embed_replace_hook, register_karvonen_hook
    ids = marked_ids()
    v = torch.randn(2, D)
    m1, m2 = tiny_model(), tiny_model()  # identical weights (same seed)
    r1, r2 = [None], [None]
    register_embed_replace_hook(m1, r1, INJ, LEFT, RIGHT, injection_scale=8.0)
    register_karvonen_hook(m2, r2, INJ, LEFT, RIGHT)
    r1[0] = v; r2[0] = v
    l1 = m1(input_ids=ids).logits
    l2 = m2(input_ids=ids).logits
    r1[0] = None; r2[0] = None
    assert not torch.allclose(l1, l2), (
        "embed_replace and karvonen produced identical logits — dispatch broken?")


if __name__ == "__main__":
    check("replaces_marker_row_with_scaled_vector", t_replaces_marker_row_with_scaled_vector)
    check("raw_scale_passes_vector_through", t_raw_scale_passes_vector_through)
    check("noop_when_vectors_none_or_decode_step", t_noop_when_vectors_none_or_decode_step)
    check("gradient_flow_and_marker_row_isolation", t_gradient_flow_and_marker_row_isolation)
    check("generate_prefill_injects_decode_noops", t_generate_prefill_injects_decode_noops)
    check("surplus_marker_still_fails_loud", t_surplus_marker_still_fails_loud)
    check("methods_actually_differ", t_methods_actually_differ)
    n_fail = sum(1 for _, e in RESULTS if e)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)
