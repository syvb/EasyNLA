"""Head-contribution RL: RL-split selection + export, and the trainer's head-specific paths.

CPU-only, tiny random Qwen3 + the Qwen3 tokenizer from the local HF cache (skipped if
absent); nothing is downloaded or sent anywhere. Run: python -m pytest tests/test_heads_rl.py -q
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import nla.train_rl_self_contained as rl  # noqa: E402
from nla.utils import register_karvonen_hook  # noqa: E402
from test_heads_datagen import _tiny, _tokenizer_or_skip  # noqa: E402


def _tiny_dataset(tmp_path, tok):
    model_dir = tmp_path / "tiny"
    _tiny(vocab=len(tok), layers=3).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    words = ["the", "old", "harbor", "town", "kept", "its", "lighthouse", "burning", "every", "night", "ships"]
    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(words, size=int(n))) for n in rng.integers(70, 160, size=24)]
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"doc_id": [f"d{i:02d}" for i in range(24)], "text": texts,
                             "domain": ["news"] * 24, "url": [""] * 24}), corpus)
    out = tmp_path / "heads"
    overrides = [
        f"base_model={model_dir}", f"output_dir={out}", f"corpus.path={corpus}", "corpus.length=24",
        "extract.chunk_size=8", "extract.chunks_per_part=1", "extract.positions_per_doc=3",
        "extract.layers_per_position=3",
        "extract.extractor_kwargs={batch_size: 4, max_length: 256, device_map: cpu, dtype: float32}",
    ]
    cfg = str(REPO / "configs/datagen/heads_qwen3_8b_finefineweb_100k.yaml")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "nla.datagen.heads.run_pipeline", "--config", cfg,
                    "--stages", "corpus,extract,stats", "--override", *overrides], check=True, cwd=REPO, env=env)
    return out, env


def _select(out, env, rl_per_head):
    subprocess.run([sys.executable, "-m", "nla.datagen.heads.split_select", "--output-dir", str(out),
                    "--av-sft-frac", "0.3", "--ar-sft-frac", "0.3", "--rl-frac", "0.25", "--eval-frac", "0.15",
                    "--per-head-av", "2", "--per-head-ar", "2", "--per-head-eval", "1",
                    "--per-head-rl", str(rl_per_head)], check=True, cwd=REPO, env=env)
    return pq.read_table(out / "warmstart" / "selection.parquet").to_pylist()


def test_rl_selection_appends_and_export_loads(tmp_path):
    tok = _tokenizer_or_skip()
    out, env = _tiny_dataset(tmp_path, tok)
    base = _select(out, env, 0)
    with_rl = _select(out, env, 2)
    # the warm-start rows (and their example_idx) are untouched; RL rows come after all of them
    assert with_rl[:len(base)] == base
    tail = with_rl[len(base):]
    assert tail and all(r["split"] == "rl" for r in tail)
    splits = dict(zip(*pq.read_table(out / "splits.parquet").to_pydict().values(), strict=True))
    assert all(splits[r["doc_id"]] == "rl" for r in tail)

    sft = tmp_path / "sft"
    subprocess.run([sys.executable, "-m", "nla.datagen.heads.export_sft", "--output-dir", str(out),
                    "--sft-dir", str(sft), "--rl-only"], check=True, cwd=REPO, env=env)
    t = pq.read_table(sft / "rl.parquet").to_pylist()
    assert len(t) == len(tail)
    by_ex = {r["example_idx"]: r for r in tail}
    for r in t:
        src = by_ex[r["example_idx"]]
        assert r["critic_prefix"] == f"Attention head L{src['layer']}H{src['head']}: "
        assert "Attention head" not in r["prompt"][0]["content"]        # the AV prompt names no head
        assert 0 < np.linalg.norm(r["activation_vector"]) <= 4.0 + 1e-4
    rows = rl.load_rl_dataset(str(sft / "rl.parquet"), extra_cols=["critic_prefix"])
    assert [r["critic_prefix"] for r in rows] == [r["critic_prefix"] for r in t]
    assert rows[0]["activation"].shape == (64,)


def test_score_with_critic_raw_and_prefix(monkeypatch):
    tok = _tokenizer_or_skip()
    d = 8
    preds = torch.arange(3 * d, dtype=torch.float32).view(3, d) / 10
    seen = []

    def fake_predict(critic, ids, attn, mse_scale_f):
        seen.extend(tok.decode(row[attn[i].bool()]) for i, row in enumerate(ids))
        return preds[: ids.shape[0]]
    monkeypatch.setattr(rl, "critic_predict", fake_predict)
    acts = [torch.randn(d) for _ in range(3)]
    expl = ["alpha", None, "gamma"]
    pre = ["Attention head L24H1: ", "Attention head L24H2: ", "Attention head L24H3: "]
    tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
    r = rl.score_with_critic(None, tok, expl, acts, tmpl, 4.0, "cpu", prefixes=pre, raw=True, target_scale=2.0)
    assert r[1] is None                                       # failed extraction stays None (floor set later)
    for k, i in enumerate((0, 2)):
        want = -float(((preds[k] - 2.0 * acts[i]) ** 2).mean())
        assert math.isclose(r[i], want, rel_tol=1e-5)
    assert seen == [tmpl.format(explanation=pre[0] + "alpha"), tmpl.format(explanation=pre[2] + "gamma")]
    # default (normalized) mode is unchanged: scale-free in the target
    r_n = rl.score_with_critic(None, tok, ["alpha"], [acts[0] * 5], tmpl, 4.0, "cpu")
    r_n2 = rl.score_with_critic(None, tok, ["alpha"], [acts[0]], tmpl, 4.0, "cpu")
    assert math.isclose(r_n[0], r_n2[0], rel_tol=1e-5)


def test_rollout_batched_rows_and_vectors():
    tok = _tokenizer_or_skip()
    model = _tiny(vocab=len(tok), layers=3)
    marker = tok.encode("㈎", add_special_tokens=False)
    assert len(marker) == 1
    inj = marker[0]
    pieces = [tok.encode(t, add_special_tokens=False) for t in ("Describe <", "> please")]
    prompt_ids = pieces[0] + [inj] + pieces[1]
    left, right = pieces[0][-1], pieces[1][0]
    prompt_text = tok.decode(prompt_ids)
    assert tok.encode(prompt_text, add_special_tokens=False) == prompt_ids
    vref = [None]
    register_karvonen_hook(model, vref, inj, left, right, layer_idx=1, preserve_magnitude=True)
    applied = []
    model.model.layers[0].register_forward_pre_hook(
        lambda m, a: applied.append(vref[0].clone()) if a[0].shape[1] > 1 else None)
    acts = [torch.full((64,), float(i + 1)) / 8 for i in range(3)]
    torch.manual_seed(0)
    res = rl.rollout_batched(model, tok, [prompt_text] * 3, acts, vref, 2, 6, 1.0, "cpu",
                             {tok.eos_token_id}, chunk_rows=4)
    assert vref[0] is None
    assert [r["prompt_idx"] for r in res] == [0, 0, 1, 1, 2, 2]
    for r in res:
        assert r["prompt_len"] == len(prompt_ids)
        assert r["full_ids"][: len(prompt_ids)].tolist() == prompt_ids
        assert r["full_ids"].numel() == len(prompt_ids) + r["n_resp"] and 1 <= r["n_resp"] <= 6
    # each prefill got the vectors of exactly its rows, in order (chunks of 4 then 2)
    assert [a.shape[0] for a in applied] == [4, 2]
    flat = torch.cat(applied)
    for k, pi in enumerate([0, 0, 1, 1, 2, 2]):
        torch.testing.assert_close(flat[k], acts[pi])


def test_rollout_batched_injection_matches_unbatched():
    """The batched path injects exactly what the per-prompt path does: same first-token
    logits for a row whether it is generated alone or in a mixed batch."""
    tok = _tokenizer_or_skip()
    model = _tiny(vocab=len(tok), layers=3)
    inj = tok.encode("㈎", add_special_tokens=False)[0]
    pieces = [tok.encode(t, add_special_tokens=False) for t in ("Describe <", "> please")]
    ids = torch.tensor([pieces[0] + [inj] + pieces[1]])
    vref = [None]
    register_karvonen_hook(model, vref, inj, pieces[0][-1], pieces[1][0], layer_idx=1, preserve_magnitude=True)
    acts = torch.randn(3, 64)
    with torch.no_grad():
        vref[0] = acts
        batched = model(ids.expand(3, -1)).logits[:, -1]
        singles = []
        for i in range(3):
            vref[0] = acts[i:i + 1]
            singles.append(model(ids).logits[0, -1])
    vref[0] = None
    torch.testing.assert_close(batched, torch.stack(singles), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(singles[0], singles[1])        # the injection matters


def test_matryoshka_cap_and_open_explanation():
    opening = [10, 11, 12, 13]
    resp = opening + list(range(100, 130)) + [2]            # 30 content tokens + a stop id
    ids, capped = rl.cap_response(resp, opening, 5)
    assert capped and ids == opening + [100, 101, 102, 103, 104]
    assert rl.cap_response(resp, opening, 30, {2}) == (resp, False)   # stopped within its cap
    assert rl.cap_response(resp, opening, 29, {2}) == (resp[:33], True)
    assert rl.cap_response([7, 8, 9], opening, 2) == ([7, 8], True)   # no opening: count from 0
    assert rl.open_explanation("<explanation>\nfoo bar</explanation>") == "foo bar"
    assert rl.open_explanation("<explanation>\nfoo bar baz") == "foo bar baz"
    assert rl.open_explanation("<explanation>\nfoo bar.\n</expl") == "foo bar."
    assert rl.open_explanation("<explanation>\n") == ""
    assert rl.open_explanation("no tag at all") is None


def test_kl_taper_weights():
    w = rl.kl_taper_weights(4 + 130, offset=4, start=16, end=120, floor=0.1)
    assert torch.allclose(w[: 4 + 17], torch.ones(21))         # opening + content tokens 0..16
    assert math.isclose(float(w[4 + 68]), 1 - 0.9 * 52 / 104, rel_tol=1e-6)
    assert torch.allclose(w[4 + 120:], torch.full((10,), 0.1))
    assert (w[1:] <= w[:-1]).all()


def test_rollout_batched_caps():
    tok = _tokenizer_or_skip()
    model = _tiny(vocab=len(tok), layers=3)
    inj = tok.encode("㈎", add_special_tokens=False)[0]
    pieces = [tok.encode(t, add_special_tokens=False) for t in ("Describe <", "> please")]
    prompt_text = tok.decode(pieces[0] + [inj] + pieces[1])
    vref = [None]
    register_karvonen_hook(model, vref, inj, pieces[0][-1], pieces[1][0], layer_idx=1, preserve_magnitude=True)
    opening = tok.encode("<explanation>\n", add_special_tokens=False)
    caps = [9, 2, 5]
    acts = [torch.randn(64) for _ in caps]
    res = rl.rollout_batched(model, tok, [prompt_text] * 3, acts, vref, 2, 20, 1.0, "cpu",
                             {tok.eos_token_id}, chunk_rows=2, caps=caps, opening_ids=opening)
    assert sorted(r["prompt_idx"] for r in res) == [0, 0, 1, 1, 2, 2]
    assert [r["prompt_idx"] for r in res] == [1, 1, 2, 2, 0, 0]   # generated in cap order
    for r in res:
        resp = r["full_ids"][r["prompt_len"]:].tolist()
        off = rl.content_offset(resp, opening)
        n_content = len(resp) - off - (resp[-1] == tok.eos_token_id)
        assert n_content <= caps[r["prompt_idx"]] and r["n_resp"] == len(resp)
        if r["capped"]:
            assert len(resp) - off == caps[r["prompt_idx"]] and tok.eos_token_id not in resp


def test_kl_weight_fn_scales_kl_term():
    new_lp = torch.zeros(6, requires_grad=True)
    ref_lp = torch.full((6,), -1.0)
    w = torch.tensor([1, 1, 1, 0.5, 0.1, 0.1])
    loss_w, _ = rl.grpo_token_loss(new_lp, ref_lp, torch.tensor(0.0), kl_beta=0.03 * w)
    loss_1, _ = rl.grpo_token_loss(new_lp, ref_lp, torch.tensor(0.0), kl_beta=0.03)
    assert math.isclose(float(loss_w) / float(loss_1), float(w.mean()), rel_tol=1e-5)
