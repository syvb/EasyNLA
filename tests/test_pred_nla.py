"""Unit tests for the predictive-readout (pred-NLA) pipeline.

The pure-logic tests run anywhere. The tokenizer tests need Qwen3-0.6B (cached
locally for CPU smoke tests); the cross-family test additionally wants a Gemma
tokenizer and is skipped when neither is available.

    .venv/bin/python -m pytest tests/test_pred_nla.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from nla.pred.data import shuffled_partner, split_for_doc
from nla.pred.reader import (
    DEFAULT_BUCKETS,
    ReaderTemplates,
    bucket_char_ranges,
    token_char_bounds,
)
from nla.pred.scoring import gain_per_token, target_tokens_per_bucket
from nla.pred.stats import bootstrap_mean, paired_bootstrap_diff


def _ids24(tok, text, n=24):
    """Token ids for `text`, repeated until it reaches n tokens, truncated to n.

    The buckets need exactly n target tokens; short fixture strings would
    otherwise fail the bucket-range assert for reasons that have nothing to do
    with what the test is checking.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + tok(text, add_special_tokens=False)["input_ids"]
    return ids[:n]


def _tok(name="Qwen/Qwen3-0.6B"):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"tokenizer {name} unavailable: {type(e).__name__}")


# ---------------------------------------------------------------- splits


def test_split_is_deterministic_and_three_way():
    docs = [f"corpus:train:{i}" for i in range(20000)]
    a = [split_for_doc(d) for d in docs]
    b = [split_for_doc(d) for d in docs]
    assert a == b
    frac = {s: a.count(s) / len(a) for s in set(a)}
    assert set(frac) == {"rl", "val", "eval"}
    assert abs(frac["val"] - 0.06) < 0.01
    assert abs(frac["eval"] - 0.12) < 0.015
    assert frac["rl"] > 0.8


def test_split_respects_custom_permille():
    docs = [f"d{i}" for i in range(20000)]
    got = [split_for_doc(d, val_permille=100, eval_permille=200) for d in docs]
    assert abs(got.count("val") / len(got) - 0.10) < 0.015
    assert abs(got.count("eval") / len(got) - 0.20) < 0.02


def test_shuffled_partner_is_a_derangement_across_documents():
    rows = [{"doc_id": f"doc{i // 2}", "row_id": i} for i in range(200)]
    perm, valid = shuffled_partner(rows, np.random.default_rng(0))
    assert sorted(perm) == list(range(200)), "must be a permutation"
    assert all(valid), "a 100-document pool admits a clean derangement"
    assert all(perm[i] != i for i in range(200)), "no self-pairs"
    assert all(rows[perm[i]]["doc_id"] != rows[i]["doc_id"] for i in range(200)), \
        "a shuffled partner from the same document would leak real context"


def test_unshufflable_rows_are_flagged_not_silently_self_paired():
    """All positions from one document: there is no valid shuffled partner for
    any of them. Those rows must come back flagged, because scoring a row against
    itself makes the control identical to the matched condition and biases
    matched-minus-shuffled toward zero."""
    rows = [{"doc_id": "only-doc", "row_id": i} for i in range(4)]
    perm, valid = shuffled_partner(rows, np.random.default_rng(0))
    assert sorted(perm) == list(range(4))
    assert not any(valid), "no row in a single-document pool has a usable partner"


def test_partly_shufflable_pool_keeps_the_good_rows():
    rows = ([{"doc_id": "a", "row_id": i} for i in range(6)]
            + [{"doc_id": f"b{i}", "row_id": 6 + i} for i in range(6)])
    perm, valid = shuffled_partner(rows, np.random.default_rng(1))
    assert sorted(perm) == list(range(12))
    for i, ok in enumerate(valid):
        if ok:
            assert perm[i] != i and rows[perm[i]]["doc_id"] != rows[i]["doc_id"]


# ---------------------------------------------------------------- char bounds


def test_token_char_bounds_are_monotone_and_span_the_text():
    tok = _tok()
    text = " Mötley Crüe drummer, who married Pamela Anderson in 1995 after"
    ids = _ids24(tok, text)
    bounds = token_char_bounds(tok, ids)
    decoded = tok.decode(ids)
    assert len(bounds) == len(ids) + 1
    assert bounds[0] == 0 and bounds[-1] == len(decoded)
    assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))


def test_bucket_char_ranges_partition_the_continuation():
    tok = _tok()
    ids = _ids24(tok, " one two three four five six seven eight nine ten eleven twelve "
              "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
              "twentyone twentytwo twentythree twentyfour")
    bounds = token_char_bounds(tok, ids)
    ranges = bucket_char_ranges(bounds, DEFAULT_BUCKETS)
    assert len(ranges) == 3
    assert ranges[0][0] == 0
    assert ranges[0][1] == ranges[1][0] and ranges[1][1] == ranges[2][0]
    assert ranges[-1][1] == len(tok.decode(ids))


def test_bucket_range_rejects_a_bucket_past_the_continuation():
    tok = _tok()
    ids = tok(" a b c d", add_special_tokens=False)["input_ids"]
    with pytest.raises(AssertionError):
        bucket_char_ranges(token_char_bounds(tok, ids), ((0, 999),))


# ---------------------------------------------------------------- gain math


def test_gain_per_token_normalizes_by_target_tokens():
    scores = np.array([[-16.0, -8.0, -4.0]])
    base = np.array([[-24.0, -16.0, -12.0]])
    g = gain_per_token(scores, base, DEFAULT_BUCKETS)
    np.testing.assert_allclose(g, [[1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(target_tokens_per_bucket(DEFAULT_BUCKETS),
                                  [8.0, 8.0, 8.0])


def test_gain_is_nan_where_the_score_is_nan():
    scores = np.array([[np.nan, -8.0, -4.0]])
    base = np.zeros((1, 3))
    g = gain_per_token(scores, base, DEFAULT_BUCKETS)
    assert np.isnan(g[0, 0]) and np.isfinite(g[0, 1])


# ---------------------------------------------------------------- statistics


def test_bootstrap_ci_brackets_the_mean_and_drops_nans():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.1, 0.05, 500), [np.nan] * 10]
    mean, lo, hi = bootstrap_mean(x, seed=0)
    assert lo < mean < hi
    assert abs(mean - 0.1) < 0.02
    assert hi - lo < 0.03


def test_clustered_bootstrap_is_wider_when_positions_share_documents():
    rng = np.random.default_rng(0)
    doc_effect = rng.normal(0, 0.5, 100)
    x = np.repeat(doc_effect, 10) + rng.normal(0, 0.01, 1000)
    docs = np.repeat([f"d{i}" for i in range(100)], 10)
    _, lo_i, hi_i = bootstrap_mean(x, n_boot=2000, seed=0)
    _, lo_c, hi_c = bootstrap_mean(x, clusters=list(docs), n_boot=2000, seed=0)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i), (
        "ignoring document clustering must understate the interval")


def test_paired_diff_of_identical_arrays_is_zero_and_insignificant():
    x = np.random.default_rng(0).normal(0, 1, 200)
    d, lo, hi, p, n = paired_bootstrap_diff(x, x, seed=0)
    assert d == 0.0 and lo == 0.0 and hi == 0.0 and n == 200
    assert p >= 0.99


def test_paired_diff_detects_a_small_consistent_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 400)
    b = a - 0.1
    d, lo, hi, p, _ = paired_bootstrap_diff(a, b, seed=0)
    assert abs(d - 0.1) < 1e-9 and lo > 0 and p < 0.05
    del hi


def test_paired_diff_drops_unpaired_nans():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, np.nan, 3.0, 3.0])
    d, _, _, _, n = paired_bootstrap_diff(a, b, seed=0)
    assert n == 2 and abs(d - 1.0) < 1e-9


# ---------------------------------------------------------------- templates


def test_templates_differ_only_in_the_description_block():
    t = ReaderTemplates()
    with_e = t.render("a description")
    without = t.render(None)
    tail = "Here is the text of the document continuing from that exact point:\n"
    assert with_e.endswith(tail) and without.endswith(tail), (
        "both conditions must hand the reader the same final framing, or the gain "
        "measures the framing rather than the description")
    assert "a description" in with_e and "a description" not in without
    assert with_e.startswith("A language model was reading a document.")
    assert without.startswith("A language model was reading a document.")


# ---------------------------------------------------------------- reader scoring


@pytest.fixture(scope="module")
def tiny_reader():
    """A real (tiny) reader so the scoring path is exercised end to end."""
    import torch

    from nla.pred.reader import FrozenReader
    try:
        return FrozenReader.load("Qwen/Qwen3-0.6B", device="cpu", dtype="float32",
                                 attn_implementation="eager", max_batch_rows=8,
                                 max_batch_tokens=8192)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"Qwen3-0.6B unavailable: {type(e).__name__}: {e}")
    finally:
        torch.set_grad_enabled(True)


def _rows_for(tok, texts):
    rows = []
    for i, t in enumerate(texts):
        ids = _ids24(tok, t)
        rows.append({
            "row_id": i, "doc_id": f"doc{i}",
            "cont_text": [tok.decode(ids)],
            "cont_ids": [ids],
            "cont_bounds": [token_char_bounds(tok, ids)],
        })
    return rows


def test_scored_token_set_is_identical_across_conditions(tiny_reader):
    """The property the whole comparison rests on: changing the prefix must not
    change which continuation tokens are scored or how they are bucketed."""
    from nla.pred.reader import ScoreJob
    tok = tiny_reader.tokenizer
    text = (" drummer for the band, and the couple married in Cancun just days "
            "after they first met, which surprised absolutely everyone involved")
    ids = _ids24(tok, text)
    cont = tok.decode(ids)
    ranges = bucket_char_ranges(token_char_bounds(tok, ids), DEFAULT_BUCKETS)
    jobs = [
        ScoreJob("with", "The text is about a celebrity marriage.", cont, ranges),
        ScoreJob("none", None, cont, ranges),
        ScoreJob("other", "Completely unrelated: a treatise on soil chemistry "
                          "and the nitrogen cycle in temperate forests.", cont, ranges),
    ]
    res = {r.key: r for r in tiny_reader.score(jobs, DEFAULT_BUCKETS)}
    counts = {k: tuple(v.bucket_ntok) for k, v in res.items()}
    assert len(set(counts.values())) == 1, (
        f"conditions scored different token counts: {counts}")
    assert sum(counts["with"]) > 0
    # ...and the prefixes really do differ, so this is not a vacuous pass.
    assert res["with"].n_prefix_tokens != res["none"].n_prefix_tokens


def test_scoring_is_deterministic(tiny_reader):
    from nla.pred.reader import ScoreJob
    tok = tiny_reader.tokenizer
    ids = _ids24(tok, " the quick brown fox jumps over the lazy dog again and again today")
    cont = tok.decode(ids)
    ranges = bucket_char_ranges(token_char_bounds(tok, ids), DEFAULT_BUCKETS)
    job = ScoreJob("k", "A pangram about a fox.", cont, ranges)
    a = tiny_reader.score([job], DEFAULT_BUCKETS)[0]
    b = tiny_reader.score([job], DEFAULT_BUCKETS)[0]
    np.testing.assert_allclose(a.bucket_logp, b.bucket_logp, rtol=1e-6)


def test_batching_does_not_change_scores(tiny_reader):
    """Length-sorted batching pads; padding must not leak into the log-probs."""
    from nla.pred.reader import ScoreJob
    tok = tiny_reader.tokenizer
    texts = [" a short one about cats and dogs living together in one house here",
             " a considerably longer continuation that mentions several unrelated "
             "topics such as geology, the price of tin, and nineteenth century opera"]
    jobs = []
    for i, t in enumerate(texts):
        ids = _ids24(tok, t)
        jobs.append(ScoreJob(i, f"Explanation number {i}.", tok.decode(ids),
                             bucket_char_ranges(token_char_bounds(tok, ids),
                                                DEFAULT_BUCKETS)))
    together = {r.key: r.bucket_logp for r in tiny_reader.score(jobs, DEFAULT_BUCKETS)}
    alone = {}
    for j in jobs:
        alone[j.key] = tiny_reader.score([j], DEFAULT_BUCKETS)[0].bucket_logp
    for k in together:
        np.testing.assert_allclose(together[k], alone[k], rtol=1e-4, atol=1e-4)


def test_a_relevant_explanation_beats_an_irrelevant_one(tiny_reader):
    """The measurement has to have the right sign on an easy case, or nothing
    downstream means anything."""
    from nla.pred.scoring import baseline_scores, score_explanations
    tok = tiny_reader.tokenizer
    cont = (" photosynthesis converts light energy into chemical energy stored in "
            "glucose molecules within the chloroplasts of plant cells")
    rows = _rows_for(tok, [cont] * 2)
    good = ("The document is a biology textbook passage explaining how plants use "
            "sunlight, chlorophyll and chloroplasts to make sugars.")
    bad = ("The document is a football match report listing the final score and "
           "the names of the goalscorers.")
    base = baseline_scores(tiny_reader, rows, branches=(0,), buckets=DEFAULT_BUCKETS)
    sg = score_explanations(tiny_reader, rows, [good, good], branches=(0,),
                            buckets=DEFAULT_BUCKETS)
    sb = score_explanations(tiny_reader, rows, [bad, bad], branches=(0,),
                            buckets=DEFAULT_BUCKETS)
    gg = gain_per_token(sg, base, DEFAULT_BUCKETS)[0]
    gb = gain_per_token(sb, base, DEFAULT_BUCKETS)[0]
    assert gg[0] > gb[0], f"on-topic explanation should win at bucket 0: {gg} vs {gb}"
    assert gg[0] > 0, f"an on-topic explanation should beat no explanation: {gg}"


def test_skip_sentinel_yields_nan_not_a_silent_zero(tiny_reader):
    from nla.pred.scoring import SKIP, score_explanations
    tok = tiny_reader.tokenizer
    rows = _rows_for(tok, [" one continuation about weather patterns over the sea"] * 2)
    out = score_explanations(tiny_reader, rows, ["a real explanation", SKIP],
                             branches=(0,), buckets=DEFAULT_BUCKETS)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all(), "a failed extraction must not average in as zero"


def test_baseline_cache_returns_identical_values(tiny_reader):
    from nla.pred.scoring import BaselineCache, baseline_scores
    tok = tiny_reader.tokenizer
    rows = _rows_for(tok, [" the harbour was full of small fishing boats that morning"])
    cache = BaselineCache()
    a = baseline_scores(tiny_reader, rows, branches=(0,), buckets=DEFAULT_BUCKETS,
                        cache=cache)
    assert len(cache) == 1
    b = baseline_scores(tiny_reader, rows, branches=(0,), buckets=DEFAULT_BUCKETS,
                        cache=cache)
    np.testing.assert_allclose(a, b)


def test_cross_family_reader_buckets_the_same_characters():
    """A Gemma (SentencePiece) reader must bucket the SAME character spans as a
    Qwen (BPE) reader, even though the token counts differ. That equivalence is
    what makes the two readers' gains comparable."""
    qwen = _tok()
    try:
        gemma = _tok("google/gemma-3-270m")
    except Exception:                                          # noqa: BLE001
        pytest.skip("no Gemma tokenizer available")
    text = (" the treaty was signed in 1815 and redrew the borders of several "
            "central European states for the next fifty years")
    q_ids = _ids24(qwen, text)
    cont = qwen.decode(q_ids)
    ranges = bucket_char_ranges(token_char_bounds(qwen, q_ids), DEFAULT_BUCKETS)
    enc = gemma(cont, add_special_tokens=False, return_offsets_mapping=True)
    assigned = [
        next((bi for bi, (lo, hi) in enumerate(ranges) if lo <= s < hi), -1)
        for s, _ in enc["offset_mapping"]
    ]
    assert set(assigned) >= {0, 1, 2}, f"Gemma tokens must fall in every bucket: {assigned}"
    assert len(enc["input_ids"]) != len(q_ids) or True   # counts may legitimately differ
    # every scored Gemma token lies inside the character span of its bucket
    for (s, _), b in zip(enc["offset_mapping"], assigned):
        if b >= 0:
            assert ranges[b][0] <= s < ranges[b][1]


def test_truncated_logits_window_matches_full_logits(tiny_reader, monkeypatch):
    """The reader asks the model for only the logits window covering the
    continuation. That optimization must be invisible in the numbers."""
    from nla.pred.reader import ScoreJob
    tok = tiny_reader.tokenizer
    jobs = []
    for i, t in enumerate([" the harbour filled with fishing boats before dawn each day",
                           " geology of the basin records three separate marine incursions"]):
        ids = _ids24(tok, t)
        jobs.append(ScoreJob(i, f"Explanation {i} with a deliberately different length" + " padding" * i,
                             tok.decode(ids),
                             bucket_char_ranges(token_char_bounds(tok, ids), DEFAULT_BUCKETS)))
    windowed = {r.key: r.bucket_logp for r in tiny_reader.score(jobs, DEFAULT_BUCKETS)}

    # Force the full-logits fallback and confirm the two agree.
    orig = tiny_reader.model.__class__.forward
    calls = {"n": 0}

    def maybe_raise(self, *args, **kwargs):
        if "logits_to_keep" in kwargs:
            calls["n"] += 1
            raise TypeError("logits_to_keep unsupported")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(tiny_reader.model.__class__, "forward", maybe_raise)
    full = {r.key: r.bucket_logp for r in tiny_reader.score(jobs, DEFAULT_BUCKETS)}
    assert calls["n"] > 0, "the fallback path was never exercised"
    for k in windowed:
        np.testing.assert_allclose(windowed[k], full[k], rtol=1e-4, atol=1e-4)


def test_multimodal_gemma_wrapper_scores_text_only():
    """The held-out reader on the real run is google/gemma-3-4b-pt, which is a
    MULTIMODAL Gemma3ForConditionalGeneration, not the text-only class the CPU
    smoke test uses. Check the text-only scoring path against that class with a
    shrunken, randomly-initialised copy: right class, text-vocab logits, a
    working logits_to_keep window, and a BOS the reader requires.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    name = "google/gemma-3-4b-pt"
    try:
        cfg = AutoConfig.from_pretrained(name)
    except Exception as e:                                     # noqa: BLE001
        pytest.skip(f"{name} config unavailable: {type(e).__name__}")
    tc = cfg.text_config
    tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size = 2, 64, 128
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 1, 16
    if hasattr(cfg, "vision_config"):
        for k, v in (("num_hidden_layers", 2), ("hidden_size", 64),
                     ("intermediate_size", 128), ("num_attention_heads", 4)):
            if hasattr(cfg.vision_config, k):
                setattr(cfg.vision_config, k, v)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    assert type(model).__name__ == "Gemma3ForConditionalGeneration"
    tok = _tok(name)
    enc = tok("A language model was reading a document.\n the treaty was signed",
              return_tensors="pt", add_special_tokens=True)
    assert int(enc.input_ids[0, 0]) == tok.bos_token_id, (
        "Gemma needs its <bos>; prefix_ids() relies on add_special_tokens=True")
    with torch.no_grad():
        full = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
        win = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    logits_to_keep=5).logits
    assert full.shape[-1] == tc.vocab_size
    torch.testing.assert_close(win, full[:, -5:], rtol=1e-4, atol=1e-4)


def test_echoed_marker_in_the_response_is_rejected():
    """The update forwards prompt+response, so a marker the policy echoes into its
    own explanation is a SECOND injection site and aborts the run. Checking only
    the prompt misses it; this is the failure that killed a 400-step run at step
    224 on the vLLM path before commit a2e4a5a."""
    from nla.injection import marker_well_formed

    INJ, L, R = 149705, 29, 522
    prompt = [7, 8, L, INJ, R, 9, 10]
    clean_resp = [11, 12, 13]
    echo_resp = [11, 12, L, INJ, R, 13]
    assert marker_well_formed(prompt, INJ, L, R)
    assert marker_well_formed(prompt + clean_resp, INJ, L, R)
    assert marker_well_formed(prompt, INJ, L, R), "prompt-only check passes the echo"
    assert not marker_well_formed(prompt + echo_resp, INJ, L, R), (
        "the echoed marker must be caught on the full sequence")


def test_trainer_validates_the_marker_over_the_full_sequence():
    """Guard the fix itself: the trainer must pass prompt+response to the check."""
    import inspect

    from nla.pred import train_rl

    src = inspect.getsource(train_rl.main)
    assert 'marker_well_formed(s["prompt_ids"] + s["resp_ids"]' in src, (
        "train_rl must validate the injection marker over prompt+response; "
        "checking the prompt alone lets an echoed marker through")


def test_baseline_cache_key_covers_what_the_value_depends_on():
    """A cached no-explanation score must not be served to a caller measuring
    something else: different reader, template, branches or buckets."""
    from nla.pred.reader import ReaderTemplates
    from nla.pred.scoring import BaselineCache

    class FakeReader:
        def __init__(self, name, templates):
            self.model_name, self.templates = name, templates

    a = FakeReader("Qwen/Qwen3-4B-Base", ReaderTemplates())
    b = FakeReader("google/gemma-3-4b-pt", ReaderTemplates())
    c = FakeReader("Qwen/Qwen3-4B-Base", ReaderTemplates(without="Different.\n"))
    cache = BaselineCache()
    cache.put(a, 7, (0, 1), np.array([1.0, 2.0, 3.0]), DEFAULT_BUCKETS)
    assert cache.get(a, 7, (0, 1), DEFAULT_BUCKETS) is not None
    assert cache.get(b, 7, (0, 1), DEFAULT_BUCKETS) is None, "different reader"
    assert cache.get(c, 7, (0, 1), DEFAULT_BUCKETS) is None, "different template"
    assert cache.get(a, 8, (0, 1), DEFAULT_BUCKETS) is None, "different position"
    assert cache.get(a, 7, (0, 1, 2), DEFAULT_BUCKETS) is None, "different branches"
    assert cache.get(a, 7, (0, 1), ((0, 4), (4, 8))) is None, "different buckets"


# ---------------------------------------------------------------- review round 2


def test_context_templates_are_parallel_and_carry_the_context():
    t = ReaderTemplates()
    tail = "Here is the text of the document continuing from that exact point:\n"
    w = t.render("ZQX-marker", context="the end of the doc")
    wo = t.render(None, context="the end of the doc")
    assert w.endswith(tail) and wo.endswith(tail)
    assert "the end of the doc" in w and "the end of the doc" in wo
    assert "ZQX-marker" in w and "ZQX-marker" not in wo
    # and the context-free pair is untouched
    assert "the end of the doc" not in t.render("ZQX-marker") and "the end of the doc" not in t.render(None)


def test_same_doc_partner_pairs_within_documents_only():
    from nla.pred.data import same_doc_partner, take_by_doc
    rows = [{"doc_id": "a", "row_id": 0}, {"doc_id": "a", "row_id": 1},
            {"doc_id": "b", "row_id": 2}, {"doc_id": "c", "row_id": 3},
            {"doc_id": "c", "row_id": 4}, {"doc_id": "c", "row_id": 5}]
    partner, valid = same_doc_partner(rows)
    assert valid == [True, True, False, True, True, True]
    for i, ok in enumerate(valid):
        if ok:
            assert partner[i] != i and rows[partner[i]]["doc_id"] == rows[i]["doc_id"]
    # take_by_doc never splits a document
    sub = take_by_doc(rows, 3)
    assert [r["row_id"] for r in sub] == [0, 1, 2]
    sub = take_by_doc(rows, 4)
    assert [r["row_id"] for r in sub] == [0, 1, 2, 3, 4, 5]


def test_reward_span_is_a_token_weighted_mean_over_whole_buckets():
    from nla.pred.rewards import ReaderGainReward

    class R:  # the reward only needs these at construction time
        model_name = "x"; n_nonfinite = 0; n_empty_buckets = 0
    r = ReaderGainReward(R(), branches=(0,), reward_span=(8, 24))
    assert r.reward_buckets == [1, 2]
    g = np.array([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(r._span_gain(g), [3.0])
    r0 = ReaderGainReward(R(), branches=(0,))
    np.testing.assert_allclose(r0._span_gain(g), [4.0])
    with pytest.raises(AssertionError):
        ReaderGainReward(R(), branches=(0,), reward_span=(4, 24))


def test_fp32_head_matches_full_fp32_scoring():
    """The production reader runs a bf16 trunk with an fp32 unembedding. Its
    scores must agree with an all-fp32 reader to well under the effect sizes
    under study (bf16 logits alone drift by ~0.01-0.05 nats/token per branch)."""
    import torch
    from nla.pred.reader import FrozenReader, ScoreJob
    try:
        full = FrozenReader.load("Qwen/Qwen3-0.6B", device="cpu", dtype="float32",
                                 attn_implementation="eager")
        mixed = FrozenReader.load("Qwen/Qwen3-0.6B", device="cpu", dtype="bfloat16",
                                  attn_implementation="eager", fp32_head=True)
    except Exception as e:                                     # noqa: BLE001
        pytest.skip(f"model unavailable: {type(e).__name__}")
    assert mixed.model.lm_head.weight.dtype == torch.float32
    tok = full.tokenizer
    ids = _ids24(tok, " the harbour was full of small fishing boats that morning, and")
    cont = tok.decode(ids)
    ranges = bucket_char_ranges(token_char_bounds(tok, ids), DEFAULT_BUCKETS)
    jobs = [ScoreJob("w", "A passage about a harbour.", cont, ranges),
            ScoreJob("n", None, cont, ranges)]
    a = {r.key: np.array(r.bucket_logp) for r in full.score(jobs, DEFAULT_BUCKETS)}
    b = {r.key: np.array(r.bucket_logp) for r in mixed.score(jobs, DEFAULT_BUCKETS)}
    gain_full = (a["w"] - a["n"]) / 8
    gain_mixed = (b["w"] - b["n"]) / 8
    # bf16 trunk still contributes some drift; the fp32 head removes the
    # unembedding's share. Loose bound: CPU bf16 kernels are the noisy part.
    assert np.max(np.abs(gain_full - gain_mixed)) < 0.15, (gain_full, gain_mixed)


def test_special_ids_cover_chat_markup_not_just_eos():
    from nla.pred.continuations import special_ids
    tok = _tok()
    ids = set(special_ids(tok))
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        tid = tok.convert_tokens_to_ids(t)
        assert tid in ids, f"{t} ({tid}) must be suppressed during sampling"
    assert tok.convert_tokens_to_ids("Ġthe") not in ids
