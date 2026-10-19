"""Head-contribution datagen: capture numerics, stats, selection, prompts, batch flow.

CPU-only. Capture tests use a tiny random Qwen3 (no download). The end-to-end
test also needs the Qwen3 tokenizer from the local HF cache (skipped if
absent); the Message Batches API is replaced by a fake client — nothing is
sent anywhere. Run: python -m pytest tests/test_heads_datagen.py -q
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.datagen.heads import batch_api, store  # noqa: E402
from nla.datagen.heads.capture import HeadCaptureExtractor  # noqa: E402
from nla.datagen.heads.extract import sample_layers  # noqa: E402
from nla.datagen.heads.prompts import LITTLE, PromptConfig, parse_response, render_user_prompt  # noqa: E402

TOKENIZER = "Qwen/Qwen3-0.6B-Base"


class _Tok:
    pad_token_id = 0
    eos_token_id = 0
    all_special_ids = (0,)
    truncation_side = "right"


def _tiny(vocab: int = 500, layers: int = 3, seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    return Qwen3ForCausalLM(cfg).float().eval()


def _docs(lengths, vocab=500, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, vocab, (n,), generator=g).tolist() for n in lengths]


def _row(caps, layer, doc, pos):
    c = caps[layer]
    hits = np.nonzero((c["doc"] == doc) & (c["pos"] == pos))[0]
    assert len(hits) == 1
    return int(hits[0])


# ── capture ──────────────────────────────────────────────────────────────

def test_capture_matches_eager_reference():
    """Attention summaries, resid_pre and head_out agree with an eager forward,
    and resid_pre + o_proj(head_out) is the post-attention residual."""
    model = _tiny()
    ref = _tiny()
    ref.load_state_dict(model.state_dict())
    ref.set_attn_implementation("eager")
    ex = HeadCaptureExtractor.from_model(model, _Tok(), batch_size=2)
    docs = _docs([90, 60, 120])
    plan = [[(55, (0, 2)), (89, (1,))], [(59, (0, 1, 2))], [(70, (2,)), (119, (0,))]]
    caps = ex.capture(docs, plan, top_k=8)

    for i, d in enumerate(docs):
        got = {}
        hooks = []
        for li, layer in enumerate(ref.model.layers):
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, li=li, got=got: got.__setitem__(("z", li), a[0].detach())))
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, li=li, got=got: got.__setitem__(("mid", li), a[0].detach())))
        out = ref(input_ids=torch.tensor([d]), output_attentions=True, output_hidden_states=True)
        for h in hooks:
            h.remove()
        for pos, lays in plan[i]:
            for layer in lays:
                r = _row(caps, layer, i, pos)
                c = caps[layer]
                probs = out.attentions[layer][0, :, pos]                       # [H, S]
                w, idx = probs.topk(8, dim=-1)
                torch.testing.assert_close(torch.from_numpy(c["attn_top_w"][r]), w, atol=1e-5, rtol=0)
                assert (torch.from_numpy(c["attn_top_idx"][r]).long() == idx).all()
                torch.testing.assert_close(torch.from_numpy(c["attn_first_w"][r]), probs[:, 0], atol=1e-5, rtol=0)
                torch.testing.assert_close(torch.from_numpy(c["attn_self_w"][r]), probs[:, pos], atol=1e-5, rtol=0)
                ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
                torch.testing.assert_close(torch.from_numpy(c["attn_entropy"][r]), ent, atol=1e-4, rtol=0)

                resid = store.from_bf16_bits(c["resid_pre"][r])
                z = store.from_bf16_bits(c["head_out"][r])
                torch.testing.assert_close(resid, out.hidden_states[layer][0, pos].to(torch.bfloat16).float())
                torch.testing.assert_close(z, got[("z", layer)][0, pos].to(torch.bfloat16).float(),
                                           atol=1e-2, rtol=1e-2)
                wo = model.model.layers[layer].self_attn.o_proj.weight
                mid = resid + wo @ z
                torch.testing.assert_close(mid, got[("mid", layer)][0, pos], atol=3e-2, rtol=3e-2)
                # per-head writes sum to the full attention write
                per_head = sum(wo[:, h * 16:(h + 1) * 16] @ z[h * 16:(h + 1) * 16] for h in range(4))
                torch.testing.assert_close(per_head, wo @ z, atol=1e-5, rtol=1e-5)


def test_capture_padding_and_batching_invariant():
    """A doc's captures don't depend on what it's batched (and right-padded) with."""
    model = _tiny()
    docs = _docs([200, 55, 140, 75])
    plan = [[(150, (0, 2))], [(54, (1,))], [(60, (0,)), (139, (2,))], [(74, (1, 2))]]
    batched = HeadCaptureExtractor.from_model(model, _Tok(), batch_size=4).capture(docs, plan, top_k=8)
    alone = HeadCaptureExtractor.from_model(model, _Tok(), batch_size=1).capture(docs, plan, top_k=8)
    for layer in batched:
        for r, (i, pos) in enumerate(zip(batched[layer]["doc"], batched[layer]["pos"], strict=True)):
            ra = _row(alone, layer, i, pos)
            for k in ("attn_top_w", "attn_first_w", "attn_entropy"):
                np.testing.assert_allclose(batched[layer][k][r], alone[layer][k][ra], atol=1e-5)
            for k in ("resid_pre", "head_out"):
                np.testing.assert_allclose(store.from_bf16_bits(batched[layer][k][r]).numpy(),
                                           store.from_bf16_bits(alone[layer][k][ra]).numpy(), atol=1e-2)


def test_capture_stops_before_lm_head():
    model = _tiny(layers=4)
    called = []
    model.lm_head.register_forward_pre_hook(lambda m, a: called.append(1))
    later = []
    model.model.layers[3].register_forward_pre_hook(lambda m, a: later.append(1))
    ex = HeadCaptureExtractor.from_model(model, _Tok())
    caps = ex.capture(_docs([80]), [[(60, (0, 2))]], top_k=4)
    assert sorted(caps) == [0, 2] and not called and not later


def test_bf16_codec_roundtrip():
    x = torch.randn(5, 33) * 1e3
    bits = store.to_bf16_bits(x)
    assert bits.dtype == np.uint16
    assert torch.equal(store.from_bf16_bits(bits), x.to(torch.bfloat16).float())
    col = store.fixed_array(bits)
    assert col.type.byte_width == 66
    assert np.array_equal(store.fixed_numpy(col.slice(2, 2), np.uint16, 33), bits[2:4])
    chunked = pa.chunked_array([col.slice(0, 2), col.slice(2, 3)])
    assert np.array_equal(store.fixed_numpy(chunked, np.uint16, 33), bits)


def test_sample_layers_deterministic():
    a = sample_layers(list(range(36)), 4, "doc-1", 77, 42)
    assert a == sample_layers(list(range(36)), 4, "doc-1", 77, 42)
    assert len(set(a)) == 4 and list(a) == sorted(a)
    assert a != sample_layers(list(range(36)), 4, "doc-1", 78, 42)


# ── prompts ──────────────────────────────────────────────────────────────

def _decode(ids):
    return "".join(f" w{i}" for i in ids)


def test_render_user_prompt_marks_sources():
    ids = list(range(100))
    text = render_user_prompt(_decode, ids, pos=80, layer=12, head=7, n_layers=36, n_heads=32,
                              top_idx=[42, 0, 80, 79, 5], top_w=[0.6, 0.2, 0.1, 0.05, 0.01],
                              first_w=0.2, self_w=0.1, cfg=PromptConfig(context_tokens=50, window_tokens=3))
    assert "Head: L12H7 (layer 12 of 36, head 7 of 32)" in text
    assert "…" + _decode(range(31, 80)) + "⟦ w80⟧" in text            # context truncated to 50 tokens
    assert '1. 0.60 on ⟦ w42⟧ (position 42) in "… w39 w40 w41⟦ w42⟧ w43 w44 w45…"' in text
    assert "2. 0.20 on the first token ⟦ w0⟧ (position 0)" in text
    assert "3. 0.10 on the current token itself ⟦ w80⟧" in text
    assert "4. 0.05 on the previous token ⟦ w79⟧" in text
    assert "w5⟧" not in text                                          # below min_source_weight
    assert "Weight on the sources listed: 0.95." in text


def test_parse_response():
    prompt = "… the ancient ⟦ Colosseum⟧ in Rome …"
    p = parse_response("<description>location attributes of 'Colosseum': Rome, Italy</description>", prompt)
    assert p.description == "location attributes of 'Colosseum': Rome, Italy" and p.quote_ok and not p.is_little
    assert p.quotes == ("Colosseum",)
    bad = parse_response("<description>the head's view of 'Rome, Italy'</description>", prompt)
    assert bad.quotes == ("Rome, Italy",) and not bad.quote_ok
    assert parse_response("<description>Little or nothing.</description>", prompt).is_little
    assert parse_response("no tags", prompt).fail_reason == "no_tags"
    assert parse_response("<description>this document is about Rome</description>", prompt).mentions_document
    assert parse_response("<description>" + "a " * 50 + "</description>", prompt).fail_reason == "too_long"


# ── end to end (tiny model + real tokenizer + fake Batch API) ────────────

def _tokenizer_or_skip():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip(f"{TOKENIZER} tokenizer not in the local HF cache")


class _FakeBatches:
    """Message Batches stand-in: every batch ends immediately."""

    def __init__(self, respond):
        self.store: dict[str, list] = {}
        self.respond = respond

    def create(self, requests):
        bid = f"msgbatch_fake{len(self.store):03d}"
        self.store[bid] = requests
        return NS(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        n = len(self.store[bid])
        return NS(id=bid, processing_status="ended",
                  request_counts=NS(processing=0, succeeded=n, errored=0, canceled=0, expired=0))

    def results(self, bid):
        for req in self.store[bid]:
            yield self.respond(req)


def _ok(req, text):
    usage = NS(input_tokens=1000, output_tokens=40, cache_read_input_tokens=600, cache_creation_input_tokens=0)
    msg = NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
             usage=usage)
    return NS(custom_id=req["custom_id"], result=NS(type="succeeded", message=msg))


def test_pipeline_end_to_end(tmp_path):
    tok = _tokenizer_or_skip()
    model_dir = tmp_path / "tiny"
    model = _tiny(vocab=len(tok), layers=3)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    words = ["the", "old", "harbor", "town", "kept", "its", "lighthouse", "burning", "every", "night", "while", "ships", "from", "distant", "ports", "carried", "salt", "grain", "and", "letters", "across", "the", "cold", "northern", "sea"]
    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(words, size=int(n))) for n in rng.integers(70, 160, size=12)]
    texts[3] = "too short"
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"doc_id": [f"d{i:02d}" for i in range(12)], "text": texts,
                             "domain": ["news", "hobby"] * 6,
                             "url": [f"https://example.com/{i}" for i in range(12)]}), corpus)
    out = tmp_path / "heads"
    overrides = [
        f"base_model={model_dir}", f"output_dir={out}", f"corpus.path={corpus}", "corpus.length=12",
        "extract.chunk_size=5", "extract.chunks_per_part=1", "extract.positions_per_doc=3",
        "extract.layers_per_position=2",
        "extract.extractor_kwargs={batch_size: 4, max_length: 256, device_map: cpu, dtype: float32}",
        "select.per_head_av=3", "select.per_head_ar=3", "split={av_sft: 0.4, ar_sft: 0.4, rl: 0.1, eval: 0.1, seed: 1}",
        "warmstart.little_threshold=0.99", "warmstart.max_requests_per_batch=10",
    ]
    cfg = str(REPO / "configs/datagen/heads_qwen3_8b_finefineweb_100k.yaml")
    subprocess.run([sys.executable, "-m", "nla.datagen.heads.run_pipeline", "--config", cfg,
                    "--stages", "corpus,extract,stats,select,prepare", "--override", *overrides],
                   check=True, cwd=REPO, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})

    meta = store.read_meta(out)
    assert meta.n_layers == 3 and meta.n_heads == 4 and meta.layers == [0, 1, 2]
    docs = store.read_docs(out)
    assert "d03" not in docs.column("doc_id").to_pylist()            # < 51 tokens → skipped
    assert docs.column("domain").to_pylist()[0] in ("news", "hobby")
    n_records = sum(store.read_records(out, layer, columns=["pos"]).num_rows for layer in meta.layers)
    assert n_records == docs.num_rows * 3 * 2

    # stats: dev_norm is ‖W_h (z − z̄)‖ in residual space, recomputed directly
    stats = store.load_stats(out)
    rec = store.read_records(out, 1, columns=["head_out"])
    z = store.from_bf16_bits(store.fixed_numpy(rec.column("head_out"), np.uint16, 64)).view(-1, 4, 16)
    wo = store.load_wo(out, 1)
    z_bar = z.mean(0)
    torch.testing.assert_close(stats.mean_head_out[1], z_bar, atol=1e-5, rtol=1e-5)
    dev = torch.stack([(wo[:, h * 16:(h + 1) * 16] @ (z[:, h] - z_bar[h]).T).norm(dim=0) for h in range(4)], 1)
    dn = pa.concat_tables([pq.read_table(f) for f in store.part_files(store.devnorms_dir(out, 1))])
    torch.testing.assert_close(torch.from_numpy(store.fixed_numpy(dn.column("dev_norm"), np.float32, 4)), dev,
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(stats.dev_rms[1], dev.pow(2).mean(0).sqrt(), atol=1e-4, rtol=1e-4)

    # selection: per head per split, unique docs, docs from the right split
    sel = pq.read_table(out / "warmstart" / "selection.parquet").to_pylist()
    splits = dict(zip(*pq.read_table(out / store.SPLITS_NAME).to_pydict().values(), strict=True))
    for (layer, head, split) in {(r["layer"], r["head"], r["split"]) for r in sel}:
        picks = [r for r in sel if (r["layer"], r["head"], r["split"]) == (layer, head, split)]
        assert len({r["doc_id"] for r in picks}) == len(picks)
        assert all(splits[r["doc_id"]] == split for r in picks)
    assert {r["split"] for r in sel} == {"av_sft", "ar_sft"}

    # prepare: every selected example is a request or a rule row; prompts are well-formed
    req_dir = out / "warmstart" / "requests"
    bodies = [json.loads(line) for f in sorted(req_dir.glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    rules = pq.read_table(out / "warmstart" / "rule_descriptions.parquet").column("example_idx").to_pylist()
    assert sorted([int(b["custom_id"][2:]) for b in bodies] + rules) == list(range(len(sel)))
    assert all(len(f.read_text().splitlines()) <= 10 for f in req_dir.glob("shard_*.jsonl"))
    p = bodies[0]["params"]
    assert p["model"] == "claude-haiku-5-5" and "temperature" not in p
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"} and p["output_config"] == {"effort": "low"}
    assert "<text>" in p["messages"][0]["content"] and "Where this head's attention went" in p["messages"][0]["content"]

    # submit: dry run sends nothing; then a real (fake-client) run, collect, retry
    ns = NS(output_dir=str(out), confirm_spend=False, max_shards=None)
    batch_api.cmd_submit(ns, client_factory=lambda: pytest.fail("dry run must not build a client"))

    def respond(req):
        cid = req["custom_id"]
        if int(cid[2:]) % 5 == 0:
            return NS(custom_id=cid, result=NS(type="errored", error=NS(type="api_error")))
        return _ok(req, "<description>the identity of an earlier token</description>")

    fake = NS(messages=NS(batches=_FakeBatches(respond)))
    fake.with_options = lambda **kw: fake
    class _Rejected(Exception):
        status_code = 400

    def _fail(exc):
        def create(requests):
            raise exc
        broken = NS(messages=NS(batches=NS(create=create)))
        broken.with_options = lambda **kw: broken
        return broken

    with pytest.raises(_Rejected):
        batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=1),
                             client_factory=lambda: _fail(_Rejected()))
    assert all(s["status"] == "prepared" for s in batch_api.load_state(out)["shards"].values())
    with pytest.raises(TimeoutError):
        batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=1),
                             client_factory=lambda: _fail(TimeoutError()))
    with pytest.raises(SystemExit, match="mid-submission"):
        batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=1), client_factory=lambda: fake)
    state = batch_api.load_state(out)
    stuck = [n for n, s in state["shards"].items() if s["status"] == "submitting"]
    state["shards"][stuck[0]]["status"] = "prepared"        # the manual reconciliation: no batch existed
    batch_api.save_state(out, state)

    batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=1), client_factory=lambda: fake)
    state = batch_api.load_state(out)
    assert sum(1 for s in state["shards"].values() if s.get("batch_id")) == 1
    batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=None), client_factory=lambda: fake)
    copts = NS(output_dir=str(out), max_words=40, price_input=0.05, price_output=0.25, poll_seconds=0.0)
    batch_api.cmd_wait(copts, client_factory=lambda: fake)
    desc = pq.read_table(out / "warmstart" / "descriptions.parquet").to_pylist()
    assert len(desc) == len(sel)
    errored = {r["example_idx"] for r in desc if r["fail_reason"] == "errored"}
    assert errored and all(ex % 5 == 0 for ex in errored)

    batch_api.cmd_retry(NS(output_dir=str(out), max_tokens=None, max_words=40))
    fake.messages.batches.respond = lambda req: _ok(req, "<description>" + LITTLE + "</description>")
    batch_api.cmd_submit(NS(output_dir=str(out), confirm_spend=True, max_shards=None), client_factory=lambda: fake)
    batch_api.cmd_collect(copts, client_factory=lambda: fake)
    desc = {r["example_idx"]: r for r in pq.read_table(out / "warmstart" / "descriptions.parquet").to_pylist()}
    assert len(desc) == len(sel) and all(r["ok"] for r in desc.values())
    assert all(desc[ex]["is_little"] for ex in errored)               # retried rows replaced the failures
    assert all(desc[ex]["source"] == "rule" for ex in rules)

    # prepare refuses to rebuild once anything was submitted
    with pytest.raises(SystemExit):
        batch_api.cmd_prepare(NS(output_dir=str(out)))


def _extract_cmd(model_dir, corpus, out, *extra):
    return [sys.executable, "-m", "nla.datagen.heads.extract", "--base-model", str(model_dir),
            "--corpus", str(corpus), "--corpus-length", "10", "--id-column", "doc_id",
            "--positions-per-doc", "3", "--layers-per-position", "2", "--chunk-size", "3",
            "--chunks-per-part", "1", "--extractor-kwargs",
            '{"batch_size": 3, "max_length": 256, "device_map": "cpu", "dtype": "float32"}',
            "--output-dir", str(out), *extra]


def _all_records(out, layers):
    return {lay: store.read_records(out, lay).sort_by([("doc_id", "ascending"), ("pos", "ascending")])
            for lay in layers}


def test_extract_resume_and_shards(tmp_path):
    tok = _tokenizer_or_skip()
    model_dir = tmp_path / "tiny"
    model = _tiny(vocab=len(tok), layers=3, seed=3)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    rng = np.random.default_rng(1)
    words = ["a", "quiet", "river", "ran", "past", "the", "mill", "where", "farmers", "brought", "wheat", "each", "autumn"]
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"doc_id": [f"d{i}" for i in range(10)],
                             "text": [" ".join(rng.choice(words, size=90)) for _ in range(10)]}), corpus)
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1"}
    run = lambda *a: subprocess.run(_extract_cmd(model_dir, corpus, *a), check=True, cwd=REPO, env=env)

    one = tmp_path / "one"
    run(one)
    base = _all_records(one, [0, 1, 2])

    # crash simulation: a part lost its marker and one of its files → recomputed identically
    (one / "_parts" / "part_00000003.parquet.done").unlink()
    (store.records_dir(one, 1) / "part_00000003.parquet").unlink(missing_ok=True)
    run(one)
    for lay, t in _all_records(one, [0, 1, 2]).items():
        assert t.equals(base[lay])

    # two shards into one dir == one process
    two = tmp_path / "two"
    run(two, "--num-shards", "2", "--shard-index", "1")
    run(two, "--num-shards", "2", "--shard-index", "0")
    for lay, t in _all_records(two, [0, 1, 2]).items():
        assert t.equals(base[lay])

    # an incompatible re-run into the same dir is refused, not interleaved
    bad = subprocess.run(_extract_cmd(model_dir, corpus, one, "--top-k", "8"), check=False, cwd=REPO,
                         env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "incompatible run" in bad.stderr


def test_pipeline_multi_gpu_flags(tmp_path, monkeypatch):
    """num_gpus > 1 launches one extract process per GPU with its own shard + CUDA_VISIBLE_DEVICES."""
    from nla.datagen.heads import run_pipeline
    launched = []

    class _P:
        def __init__(self, cmd, env):
            launched.append((cmd, env["CUDA_VISIBLE_DEVICES"]))

        def wait(self):
            return 0

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", _P)
    cfg = {"base_model": "m", "output_dir": str(tmp_path),
           "corpus": {"path": "c.parquet", "length": 10, "id_column": "doc_id"},
           "extract": {"layers": "all", "layers_per_position": 4, "positions_per_doc": 10, "num_gpus": 4,
                       "extractor_kwargs": {"batch_size": 12}}}
    run_pipeline._extract(cfg, NS())
    assert [gpu for _, gpu in launched] == ["0", "1", "2", "3"]
    for i, (cmd, _) in enumerate(launched):
        assert cmd[-4:] == ["--num-shards", "4", "--shard-index", str(i)]
        assert cmd[cmd.index("--layers-per-position") + 1] == "4" and "--num-gpus" not in cmd


# ── self-descriptions + probe ────────────────────────────────────────────

def test_probe_recovers_planted_signal():
    from nla.datagen.heads.probe import crossfit
    rng = np.random.default_rng(0)
    words = [f"tok{i}" for i in range(6)]
    labels = rng.integers(0, 6, size=300)
    centers = rng.normal(size=(6, 16))
    y = centers[labels] + 0.1 * rng.normal(size=(300, 16))
    good = [f"carries {words[k]} from earlier" for k in labels]
    fve, err = crossfit(good, y)
    assert fve > 0.9 and err.shape == (300,)
    noise = [f"carries {words[k]} from earlier" for k in rng.permutation(labels)]
    assert crossfit(noise, y)[0] < 0.05
    assert abs(crossfit([""] * 300, y)[0]) < 0.05
    assert crossfit(centers[labels] + rng.normal(size=(300, 16)), y)[0] > 0.5   # dense features


def test_write_coords_isometry():
    from nla.datagen.heads.probe import write_coords
    w = torch.randn(64, 16)
    z = torch.randn(5, 16)
    c = torch.from_numpy(write_coords(z, w))
    torch.testing.assert_close(torch.cdist(c, c), torch.cdist(z @ w.T, z @ w.T), atol=1e-4, rtol=1e-4)


def test_self_describe_injection(tmp_path):
    from nla.datagen.heads import self_describe
    tok = _tokenizer_or_skip()
    model_dir = tmp_path / "tiny"
    _tiny(vocab=len(tok), layers=3).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    sd = self_describe.SelfDescriber(str(model_dir), device_map="cpu", dtype="float32", style="completion",
                                     max_new_tokens=4, batch_size=2)
    assert sd.prompt_text.endswith("<description>") and sd.style == "completion"
    ids = torch.tensor([sd.prompt_ids] * 2)
    p = sd.marker_pos
    v = torch.randn(2, 64)

    def run(capture_layer, pre):
        # Own hooks, registered after the injection hook: output_hidden_states records
        # layer outputs before later forward hooks run, so it would not show the edit.
        got = {}
        if pre:
            h = sd.layers[capture_layer].register_forward_pre_hook(lambda m, a: got.setdefault("x", a[0].clone()))
        else:
            h = sd.layers[capture_layer].register_forward_hook(lambda m, a, o: got.setdefault("x", o.clone()))
        sd.model(input_ids=ids)
        h.remove()
        return got["x"]

    base1 = run(1, pre=False)
    with sd._inject("l1_add", v, 0):
        added1 = run(1, pre=False)
    expect = base1[:, p] + base1[:, p].norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(added1[:, p], expect)
    mask = torch.ones(ids.shape[1], dtype=torch.bool)
    mask[p] = False
    torch.testing.assert_close(added1[:, mask], base1[:, mask])        # only the marker moves

    base2 = run(2, pre=True)
    with sd._inject("same_layer", v, 2):
        x = run(2, pre=True)
    torch.testing.assert_close(x[:, p] / x[:, p].norm(dim=-1, keepdim=True), v / v.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(x[:, p].norm(dim=-1), base2[:, p].norm(dim=-1))
    torch.testing.assert_close(x[:, mask], base2[:, mask])
    # decode steps (one position) pass through untouched; generation runs end to end
    out = sd.describe(v, [2, 2], "same_layer")
    assert len(out) == 2 and all(isinstance(t, str) for t in out)



def test_soft_token_injection(tmp_path):
    from nla.datagen.heads import soft_token
    tok = _tokenizer_or_skip()
    model_dir = tmp_path / "tiny"
    _tiny(vocab=len(tok), layers=3).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    sd = soft_token.SoftTokenDescriber(str(model_dir), device_map="cpu", dtype="float32", batch_size=2)
    rec = soft_token.Recipe("t", "a {m} b {m} →", beta=2.0, max_new_tokens=3)
    ids, marks = sd.encode(rec)
    assert len(marks) == 2 and ids[marks[0]] == ids[marks[1]] == tok(" X", add_special_tokens=False)["input_ids"][-1]
    x = torch.tensor([ids] * 2)
    v = torch.randn(2, 64)

    def attn_out(layer):
        got = {}
        h = sd.layers[layer].self_attn.o_proj.register_forward_hook(lambda m, a, o: got.setdefault("x", o.clone()))
        sd.model(input_ids=x)
        h.remove()
        return got["x"]

    base = attn_out(1)
    with sd._inject(1, marks, len(ids), v):
        got = attn_out(1)
    torch.testing.assert_close(got[:, marks], base[:, marks] + v[:, None])   # the write lands at the marks
    keep = torch.ones(len(ids), dtype=torch.bool)
    keep[marks] = False
    torch.testing.assert_close(got[:, keep], base[:, keep])
    # a zero write reads exactly like the clean prompt; "delta" lists top_k tokens
    zero = torch.zeros(2, 64)
    clean = sd.model.generate(input_ids=x[:1], attention_mask=torch.ones_like(x[:1]), max_new_tokens=3,
                              do_sample=False, pad_token_id=0)
    assert sd.describe(rec, zero, [1, 1])[0] == soft_token.clean(tok.decode(clean[0, len(ids):]))
    out = sd.describe(soft_token.Recipe("d", rec.prompt, beta=2.0, readout="delta", top_k=5), v, [1, 2])
    assert all(len(t.split("', '")) == 5 for t in out)

# ── quick-SFT plumbing ───────────────────────────────────────────────────

def test_karvonen_preserve_magnitude():
    from nla.injection import karvonen_inject_in_residual
    ids = torch.tensor([[5, 41, 42, 43, 6]])
    h = torch.randn(1, 5, 8)
    v = torch.randn(1, 8) * 0.3
    unit = karvonen_inject_in_residual(ids, h, v, 42, 41, 43)
    raw = karvonen_inject_in_residual(ids, h, v, 42, 41, 43, preserve_magnitude=True)
    torch.testing.assert_close(unit[0, 2], h[0, 2] + h[0, 2].norm() * v[0] / v[0].norm(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(raw[0, 2], h[0, 2] + h[0, 2].norm() * v[0])
    torch.testing.assert_close(raw[0, [0, 1, 3, 4]], h[0, [0, 1, 3, 4]])
    tiny = karvonen_inject_in_residual(ids, h, v * 1e-6, 42, 41, 43, preserve_magnitude=True)
    torch.testing.assert_close(tiny, h, atol=1e-4, rtol=0)      # an idle head's ~0 write injects ~nothing


def test_run_sync_writes_batch_format_results(tmp_path):
    out = tmp_path / "heads"
    ws = out / "warmstart"
    (ws / "requests").mkdir(parents=True)
    sel = pa.table({"example_idx": [0, 1, 2], "split": ["av_sft"] * 3, "layer": pa.array([1, 1, 1], pa.int16()),
                    "head": pa.array([0, 1, 2], pa.int16()), "rank": [0, 0, 0], "doc_id": ["a", "b", "c"],
                    "pos": [60, 61, 62], "dev_norm": [1.0, 1.0, 1.0], "dev_rel": [1.0, 1.0, 1.0]})
    pq.write_table(sel, ws / "selection.parquet")
    pq.write_table(pa.table({"example_idx": pa.array([2], pa.int64()), "first_w": pa.array([0.99], pa.float32())}),
                   ws / "rule_descriptions.parquet")
    lines = [json.dumps({"custom_id": batch_api.custom_id(i), "params": batch_api.request_params(
        f"prompt with 'tok{i}'", {"model": "m", "max_tokens": 8, "effort": "low", "thinking": "adaptive",
                                  "cache_system": True})}) for i in (0, 1)]
    (ws / "requests" / "shard_0000.jsonl").write_text("\n".join(lines) + "\n")
    (ws / "requests" / "prepare.json").write_text(json.dumps({"estimate": {"usd": 1.0}}))
    batch_api.save_state(out, {"shards": {"shard_0000": {"status": "prepared", "n_requests": 2, "batch_id": None}}})

    class _Msgs:
        async def create(self, **params):
            i = params["messages"][0]["content"][-2]
            usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            return NS(stop_reason="end_turn", usage=usage,
                      content=[NS(type="text", text=f"<description>carries 'tok{i}'</description>")])

    ns = NS(output_dir=str(out), confirm_spend=False, concurrency=2, max_words=40, price_input=0.1, price_output=0.5,
            deadline_minutes=None)
    batch_api.cmd_run_sync(ns, client_factory=lambda: pytest.fail("dry run must not build a client"))
    ns.confirm_spend = True
    batch_api.cmd_run_sync(ns, client_factory=lambda: NS(messages=_Msgs()))
    st = batch_api.load_state(out)["shards"]["shard_0000"]
    assert st["status"] == "ended" and st["results"] == "results/shard_0000.jsonl"
    desc = {r["example_idx"]: r for r in pq.read_table(ws / "descriptions.parquet").to_pylist()}
    assert desc[0]["description"] == "carries 'tok0'" and desc[0]["quote_ok"] and desc[1]["ok"]
    assert desc[2]["source"] == "rule"


def test_run_sync_deadline_keeps_what_came_back(tmp_path):
    """--deadline-minutes ends the stage with partial results kept, and retry re-sends too-long answers."""
    import asyncio
    out = tmp_path / "heads"
    ws = out / "warmstart"
    (ws / "requests").mkdir(parents=True)
    n = 6
    pq.write_table(pa.table({"example_idx": list(range(n)), "split": ["av_sft"] * n,
                             "layer": pa.array([1] * n, pa.int16()), "head": pa.array(list(range(n)), pa.int16()),
                             "rank": [0] * n, "doc_id": [f"d{i}" for i in range(n)], "pos": [60] * n,
                             "dev_norm": [1.0] * n, "dev_rel": [1.0] * n}), ws / "selection.parquet")
    pq.write_table(pa.table({"example_idx": pa.array([], pa.int64()), "first_w": pa.array([], pa.float32())}),
                   ws / "rule_descriptions.parquet")
    cfg = {"model": "m", "max_tokens": 8, "effort": "low", "thinking": "adaptive", "cache_system": True}
    lines = [json.dumps({"custom_id": batch_api.custom_id(i), "params": batch_api.request_params(f"p{i}", cfg)})
             for i in range(n)]
    (ws / "requests" / "shard_0000.jsonl").write_text("\n".join(lines) + "\n")
    (ws / "requests" / "prepare.json").write_text(json.dumps({"estimate": {"usd": 1.0}}))
    pq.write_table(pa.table({"custom_id": [batch_api.custom_id(i) for i in range(n)], "example_idx": list(range(n)),
                             "shard": ["shard_0000"] * n}), ws / "requests" / "index.parquet")
    batch_api.save_state(out, {"shards": {"shard_0000": {"status": "prepared", "n_requests": n, "batch_id": None}}})

    class _Msgs:
        async def create(self, **params):
            i = int(params["messages"][0]["content"][1:])
            if i >= 4:
                await asyncio.sleep(30)                    # never answers before the deadline
            text = "<description>" + ("word " * 50 if i == 0 else f"thing {i}") + "</description>"
            usage = NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text=text)])

    ns = NS(output_dir=str(out), confirm_spend=True, concurrency=8, max_words=40, price_input=0.1,
            price_output=0.5, deadline_minutes=0.02)
    batch_api.cmd_run_sync(ns, client_factory=lambda: NS(messages=_Msgs()))
    st = batch_api.load_state(out)["shards"]["shard_0000"]
    assert st["status"] == "ended" and st["truncated"]
    desc = {r["example_idx"]: r for r in pq.read_table(ws / "descriptions.parquet").to_pylist()}
    assert set(desc) == {0, 1, 2, 3} and desc[0]["fail_reason"] == "too_long" and desc[3]["ok"]
    batch_api.cmd_retry(NS(output_dir=str(out), max_tokens=None, max_words=40))
    retried = [json.loads(line)["custom_id"] for f in (ws / "requests").glob("retry*.jsonl")
               for line in f.read_text().splitlines()]
    assert retried == [batch_api.custom_id(0)]          # the too-long answer is re-sent
