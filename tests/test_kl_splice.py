"""CPU checks for nla/utils/kl_splice.py against an independent reference.

The reference is a plain unpadded forward over the whole prefix, one row at a
time, with the layer-K output at the last position replaced directly — no
cache, no left padding, no 4D mask. SpliceKL must match it.

Run: python -m pytest tests/test_kl_splice.py   (or python tests/test_kl_splice.py)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.utils.arch_adapters import resolve_decoder_layers  # noqa: E402
from nla.utils.kl_splice import SpliceKL, orthogonal_fail_vectors  # noqa: E402

K = 2


def _tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def _reference(model, ids, vec):
    """(natural layer-K h at the last position, KL with vec spliced there)."""
    layer = resolve_decoder_layers(model)[K]
    state = {}

    def hook(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        state["h"] = h[0, -1].detach().clone()
        if vec is not None:
            v = vec / vec.norm() * h[0, -1].norm()
            h = torch.cat([h[:, :-1], v.to(h.dtype)[None, None]], dim=1)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return out

    x = torch.tensor([ids])
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            lp_s = F.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
    finally:
        handle.remove()
    with torch.no_grad():
        lp_o = F.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
    return state["h"], (lp_o.exp() * (lp_o - lp_s)).sum()


def _prefixes():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (n,), generator=g).tolist() for n in (5, 12, 9)]


def test_matches_unpadded_reference():
    model = _tiny_model()
    prefs = _prefixes()
    # several vectors per prefix, interleaved order, micro_batch < #prefixes
    order = [0, 1, 2, 1, 0, 1]
    vecs = torch.randn(len(order), 64, generator=torch.Generator().manual_seed(2))
    splice = SpliceKL(model, K, micro_batch=2)
    got = splice.kl([prefs[i] for i in order], vecs)
    want = torch.stack([_reference(model, prefs[i], vecs[j])[1] for j, i in enumerate(order)])
    assert torch.all(want > 1e-4), want
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def test_natural_activation_gives_zero_kl():
    model = _tiny_model()
    prefs = _prefixes()
    nat = torch.stack([_reference(model, p, None)[0] for p in prefs])
    got = SpliceKL(model, K).kl(prefs, nat * 3.0)   # scale must not matter
    assert got.abs().max() < 1e-5, got


def test_gradient_reaches_vectors_only():
    model = _tiny_model()
    prefs = _prefixes()
    vecs = torch.randn(3, 64, requires_grad=True)
    splice = SpliceKL(model, K)
    splice.kl(prefs, vecs).sum().backward()
    assert vecs.grad is not None and vecs.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    # hook is inert outside kl(): plain forwards are unchanged
    x = torch.tensor([prefs[1]])
    with torch.no_grad():
        a = model(input_ids=x).logits
    assert splice._splice is None
    with torch.no_grad():
        b = model(input_ids=x).logits
    torch.testing.assert_close(a, b)


def test_rl_score_with_critic_kl():
    """RL scoring in kl mode: MSE rewards unchanged, -KL per valid rollout,
    orthogonal floor for every rollout (incl. failed ones)."""
    import nla.train_rl_vllm as rl
    from transformers import AutoTokenizer
    model = _tiny_model()
    splice = SpliceKL(model, K)
    prefs = _prefixes()
    G = 3
    expl = ["a cat", None, "a dog", "x", "y", "z", "p", "q", None]   # 3 prompts x G
    prefixes = [prefs[i // G] for i in range(len(expl))]
    golds = [torch.randn(64, generator=torch.Generator().manual_seed(10 + i // G))
             for i in range(len(expl))]
    preds = torch.randn(len(expl), 64, generator=torch.Generator().manual_seed(3))
    calls = {"n": 0}

    def fake_predict(critic, bx, attn, scale):          # one row per valid rollout, in order
        out = preds[calls["n"]:calls["n"] + bx.shape[0]]
        calls["n"] += bx.shape[0]
        return out

    real = rl.critic_predict
    rl.critic_predict = fake_predict
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        args = (None, tok, expl, golds, "Summary: <text>{explanation}</text> <summary>", 8.0, "cpu")
        plain = rl.score_with_critic(*args, batch_size=4)
        calls["n"] = 0
        fail_dir = torch.randn(64)
        rewards, neg_kls, floors = rl.score_with_critic(
            *args, batch_size=4, splice=splice, prefixes=prefixes, fail_dir=fail_dir)
    finally:
        rl.critic_predict = real
    assert rewards == plain
    valid = [i for i, e in enumerate(expl) if e is not None]
    assert [i for i, k in enumerate(neg_kls) if k is not None] == valid
    want = splice.kl([prefixes[i] for i in valid], preds[:len(valid)])
    torch.testing.assert_close(torch.tensor([-neg_kls[i] for i in valid]), want, rtol=1e-4, atol=1e-6)
    want_f = splice.kl(prefixes, orthogonal_fail_vectors(torch.stack(golds), fail_dir))
    torch.testing.assert_close(torch.tensor(floors), -want_f, rtol=1e-4, atol=1e-6)


def test_orthogonal_fail_vectors():
    g = torch.randn(4, 64)
    u = orthogonal_fail_vectors(g, torch.randn(64))
    cos = F.cosine_similarity(u, g, dim=-1)
    assert cos.abs().max() < 1e-5, cos


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
