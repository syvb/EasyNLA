"""CPU tests for the bottleneck experiment harness (no GPU, no vLLM).

Uses a tiny random Qwen3 to validate the custom generation loop against HF
generate(), the identity tap (C0' == C0), the replace_all tap, and the
teacher-forced metric plumbing. Run:

    python -m pytest tests/test_bottleneck_cpu.py -x -q

Falls back to constructing a tiny Qwen3 from config if the hub model is
unavailable offline.
"""

import pytest
import torch

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=1024, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception:
        pytest.skip("no tokenizer available offline")
    return model, tok, cfg


@pytest.fixture()
def bm(tiny, monkeypatch):
    from experiments.bottleneck import patched
    model, tok, cfg = tiny

    def fake_init(self, m_ckpt="x", layer_index=1, device="cpu",
                  attn_implementation="sdpa"):
        from nla.utils.arch_adapters import resolve_decoder_layers
        self.device = device
        self.layer_index = layer_index
        self.model = model
        self.tokenizer = tok
        self.eos_ids = {5}          # arbitrary small-vocab EOS for the tiny model
        self.pad_id = 0
        self._mode = None
        self._captured = None
        self._replacement = None
        self._replace_mask = None
        self._hook_fired = 0
        self._handle = resolve_decoder_layers(self.model)[layer_index] \
            .register_forward_hook(self._tap)

    monkeypatch.setattr(patched.BottleneckModel, "__init__", fake_init)
    m = patched.BottleneckModel(layer_index=1)
    yield m
    m._handle.remove()


def _prompts():
    torch.manual_seed(1)
    return [torch.randint(6, 1000, (n,)).tolist() for n in (5, 9, 7)]


def test_clean_matches_hf_generate(bm):
    prompts = _prompts()
    ours = bm.generate(prompts, 12, condition="clean")
    for i, p in enumerate(prompts):
        t = torch.tensor([p])
        out = bm.model.generate(
            input_ids=t, attention_mask=torch.ones_like(t), max_new_tokens=12,
            do_sample=False, eos_token_id=list(bm.eos_ids), pad_token_id=bm.pad_id)
        ref = out[0, t.shape[1]:].tolist()
        for j, tokid in enumerate(ref):
            if tokid in bm.eos_ids:
                ref = ref[:j + 1]
                break
        assert ours[i] == ref, f"prompt {i}: {ours[i]} != {ref}"


def test_identity_matches_clean(bm):
    prompts = _prompts()
    assert bm.generate(prompts, 12, condition="identity") == \
        bm.generate(prompts, 12, condition="clean")


def test_codec_condition_calls_and_perturbs(bm):
    from experiments.bottleneck.patched import StepLog

    class FakeCodec:
        def __init__(self):
            self.calls = 0

        def roundtrip(self, h):
            self.calls += 1
            fake_recs = []
            for _ in range(h.shape[0]):
                class V:  # minimal VerbalizeRecord stand-in
                    n_tokens, text, truncated, extract_failed, steer_verified = \
                        3, "z", False, False, True
                class R:
                    verb, cosine, h_norm, pred_norm = V(), 0.9, 1.0, 1.0
                fake_recs.append(R())
            return -h, fake_recs  # sign-flip: a maximally visible perturbation

    codec = FakeCodec()
    logs: list[StepLog] = []
    prompts = _prompts()
    clean = bm.generate(prompts, 8, condition="clean")
    pert = bm.generate(prompts, 8, codec=codec, condition="nla", step_logs=logs)
    assert codec.calls > 0 and logs
    assert pert != clean, "sign-flipped stream should change greedy tokens"
    # first generated token comes from CLEAN prefill in both conditions
    for i in range(len(prompts)):
        assert pert[i][0] == clean[i][0], "prefill must be clean (first token equal)"


def test_replace_all_identity_bitwise(bm):
    prompts = _prompts()
    S = max(len(p) for p in prompts)
    ids = torch.zeros((len(prompts), S), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, p in enumerate(prompts):
        ids[r, :len(p)] = torch.tensor(p)
        mask[r, :len(p)] = 1
    logits_clean, h = bm.forward_teacher_forced(ids, mask, capture=True)
    assert h is not None and h.shape == (*ids.shape, bm.model.config.hidden_size)
    logits_ident, _ = bm.forward_teacher_forced(
        ids, mask, replacement=h.clone(), replace_mask=mask.bool())
    assert torch.equal(logits_clean, logits_ident)
    # and a real perturbation must move logits
    logits_pert, _ = bm.forward_teacher_forced(
        ids, mask, replacement=-h, replace_mask=mask.bool())
    assert not torch.equal(logits_clean, logits_pert)


def test_score_continuation_nll(bm):
    p, c = [7, 8, 9, 10], [11, 12, 13]
    nll = bm.score_continuation_nll(p, c)
    full = torch.tensor([p + c])
    logits = bm.model(input_ids=full).logits.float()
    lp = torch.log_softmax(logits[0, len(p) - 1:-1], dim=-1)
    ref = float(-lp.gather(-1, torch.tensor(c).unsqueeze(-1)).mean())
    assert abs(nll - ref) < 1e-5
