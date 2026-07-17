"""Verification sweep of the activation-whitening pipeline (nla/whitening.py +
scripts/compute_whitening_stats.py + scripts/whiten_dataset.py).

Every test is a concrete executable check (algebraic properties, round-trips,
an end-to-end run of both CLI scripts on a synthetic parquet) — not a lint.
CPU-only. Run: python tests/test_whitening.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS = []

# No regularization — for tests asserting exact whiteness/round-trips.
EXACT = dict(shrinkage=0.0, floor_quantile=0.0)


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, None)); print(f"PASS {name}")
    except Exception as e:
        RESULTS.append((name, e)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def _anisotropic_sample(n=6000, d=24, seed=0):
    """Gaussian with a random, well-anisotropic covariance (cond ~ 1e4)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    scales = np.logspace(-2, 2, d)
    a = q * scales
    mean = rng.standard_normal(d) * 5
    return rng.standard_normal((n, d)) @ a.T + mean


# ------------------------------------------------------------- numerics ----
def t_whitened_cov_is_identity():
    from nla.whitening import compute_stats_from_array, whiten
    x = _anisotropic_sample()
    stats = compute_stats_from_array(x, **EXACT)
    xw = whiten(x, stats)
    cov = np.cov(xw, rowvar=False)
    assert np.abs(np.diag(cov) - 1).max() < 1e-6, f"diag off by {np.abs(np.diag(cov) - 1).max()}"
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 1e-6, f"offdiag {np.abs(off).max()}"
    assert np.abs(xw.mean(axis=0)).max() < 1e-8


def t_roundtrip_inverse():
    from nla.whitening import compute_stats_from_array, unwhiten, whiten
    x = _anisotropic_sample(seed=1)
    stats = compute_stats_from_array(x)  # default regularization — inverse must hold regardless
    back = unwhiten(whiten(x, stats), stats)
    rel = np.abs(back - x).max() / np.abs(x).max()
    assert rel < 1e-9, f"round-trip rel err {rel}"


def t_zca_symmetry_and_inverse_pair():
    from nla.whitening import compute_stats_from_array
    stats = compute_stats_from_array(_anisotropic_sample(seed=2))
    assert np.abs(stats.w - stats.w.T).max() < 1e-10, "W not symmetric (not ZCA)"
    assert np.abs(stats.w_inv - stats.w_inv.T).max() < 1e-10
    prod = stats.w @ stats.w_inv
    assert np.abs(prod - np.eye(stats.d_model)).max() < 1e-8, "W·W⁻¹ ≠ I"


def t_rank_deficiency_fails_loudly():
    from nla.whitening import compute_stats_from_array
    rng = np.random.default_rng(3)
    d = 30
    # rank-deficient: data lives in a 5-dim subspace
    x = rng.standard_normal((500, 5)) @ rng.standard_normal((5, d))
    for kwargs in ({"shrinkage": 0.0, "floor_quantile": 0.0},
                   {"shrinkage": 0.0, "floor_quantile": 0.05}):  # floor of a ~0 tail is ~0 — must NOT rescue
        try:
            compute_stats_from_array(x, **kwargs)
            raise RuntimeError(f"rank-deficient data should assert ({kwargs})")
        except AssertionError:
            pass
    stats = compute_stats_from_array(x, shrinkage=0.01, floor_quantile=0.0)  # shrinkage does rescue
    assert stats.eigenvalues[0] > 0


def t_floor_whitens_bulk_exactly():
    # flooring at q leaves the top (1−q) of directions whitened EXACTLY
    # (variance spectrum == 1) and only caps the bottom tail (< 1).
    from nla.whitening import compute_stats_from_array
    x = _anisotropic_sample(seed=4)
    q = 0.25
    stats = compute_stats_from_array(x, shrinkage=0.0, floor_quantile=q)
    v = stats.whitened_variance_spectrum()  # ascending in eigenvalue
    d = stats.d_model
    n_floored = int(np.ceil(q * d)) - 1  # quantile interpolates; at least the strict-below bucket
    assert np.abs(v[n_floored + 1:] - 1.0).max() < 1e-12, "bulk not exactly whitened"
    assert (v[:n_floored] < 1.0).all(), "tail not attenuated"
    assert stats.expected_whitened_variance() < 1.0


def t_shrinkage_bounds_amplification():
    # eigenvalue floor is ≈ λ·(trΣ/d) ⇒ max whitening gain ≈ sqrt(mean/floor)
    from nla.whitening import compute_stats_from_array
    x = _anisotropic_sample(seed=4)
    lam = 0.05
    stats = compute_stats_from_array(x, shrinkage=lam, floor_quantile=0.0)
    iso = stats.eigenvalues.mean()  # trace preserved by shrinkage
    assert stats.eigenvalues[0] >= lam * iso * 0.5, "floor not applied"


def t_streaming_matches_in_memory():
    from nla.whitening import compute_stats_from_array, compute_stats_from_moments
    x = _anisotropic_sample(seed=5)
    ref = compute_stats_from_array(x)
    # accumulate in 7 uneven chunks, as the parquet path does
    n, s, outer = 0, np.zeros(x.shape[1]), np.zeros((x.shape[1], x.shape[1]))
    for chunk in np.array_split(x, 7):
        n += len(chunk); s += chunk.sum(axis=0); outer += chunk.T @ chunk
    streamed = compute_stats_from_moments(n, s, outer)
    # relative to ‖W‖: chunk-order fp noise in Σ (~1e-12 in eigenvalues) is
    # amplified 1/√λ by whitening, so absolute tolerance depends on conditioning
    assert np.abs(streamed.w - ref.w).max() < 1e-8 * np.abs(ref.w).max()
    assert np.abs(streamed.mean - ref.mean).max() < 1e-10


def t_npz_roundtrip_and_hash():
    from nla.whitening import compute_stats_from_array, load_stats, save_stats
    stats = compute_stats_from_array(
        _anisotropic_sample(seed=6), source="synthetic", base_model="test/m", layer_index=7)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "stats.npz")
        save_stats(stats, p)
        loaded = load_stats(p)  # asserts stored sha == recomputed sha
        assert loaded.sha256() == stats.sha256()
        assert np.array_equal(loaded.w, stats.w)
        assert np.array_equal(loaded.mean, stats.mean)
        assert np.array_equal(loaded.eigenvalues_raw, stats.eigenvalues_raw)
        assert (loaded.n_samples, loaded.source) == (stats.n_samples, "synthetic")
        assert (loaded.base_model, loaded.layer_index) == ("test/m", 7)
        assert (loaded.shrinkage, loaded.floor_quantile) == (stats.shrinkage, stats.floor_quantile)


def t_norm_tag_resolution():
    from nla.schema import NORM_RAW, NORM_WHITENED_ZCA, resolve_activation_norm
    assert resolve_activation_norm(None) == NORM_RAW  # pre-field sidecars
    assert resolve_activation_norm("none") == NORM_RAW
    assert resolve_activation_norm(NORM_WHITENED_ZCA) == NORM_WHITENED_ZCA
    try:
        resolve_activation_norm("mystery_norm_v9")
        raise RuntimeError("unknown norm should assert")
    except AssertionError:
        pass


# ------------------------------------------------- end-to-end (scripts) ----
def _write_synthetic_avsft_parquet(path: Path, n: int, d: int, seed: int = 7):
    """Tiny av_sft-shaped parquet + valid sidecar, mirroring stage3's schema —
    including the real HF files' list child field name 'element' (from_arrays
    produces 'item'; the rewriter must reconcile)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar_local
    from nla.schema import NLATokenMeta

    x = _anisotropic_sample(n, d, seed).astype(np.float32)
    av_type = pa.list_(pa.field("element", pa.float32(), nullable=False), d)
    prompt_t = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
    msg = [{"role": "user", "content": "explain <concept>㈎</concept>"}]
    av_col = pa.FixedSizeListArray.from_arrays(
        pa.array(x.reshape(-1), type=pa.float32()), d).cast(av_type)
    table = pa.table({
        "prompt": pa.array([msg] * n, type=prompt_t),
        "response": pa.array([f"<explanation>\nrow {i}\n</explanation>" for i in range(n)]),
        "activation_vector": av_col,
        "doc_id": pa.array([f"doc{i // 3}" for i in range(n)]),
    })
    # 4 row groups so the rewriter's batch loop is actually exercised
    pq.write_table(table, str(path), row_group_size=(n + 3) // 4)
    meta = NLADatasetMeta(
        dataset_id="synthetic-test",
        stage="av_sft",
        row_count=n,
        extraction=NLAExtractionMeta(
            base_model="test/model", d_model=d, layer_index=2, norm="none",
            corpus="synthetic", corpus_slice={"start": 0, "length": n},
            positions_per_doc=3,
        ),
        tokens=NLATokenMeta(
            injection_char="㈎", injection_token_id=149705,
            injection_left_neighbor_id=1, injection_right_neighbor_id=2,
        ),
        prompt_templates={"actor": "explain <concept>{injection_char}</concept>"},
    )
    write_sidecar_local(path, meta)
    return x


def t_end_to_end_scripts():
    import pyarrow.parquet as pq

    from nla.datagen.sidecar import read_sidecar_local
    from nla.schema import NORM_WHITENED_ZCA
    from nla.whitening import load_stats

    n, d = 800, 16
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_pq = td / "av_sft.parquet"
        stats_npz = td / "stats.npz"
        white_pq = td / "white" / "av_sft.parquet"
        x = _write_synthetic_avsft_parquet(raw_pq, n, d)

        # no regularization so whitened cov is exactly identity (floor/shrink
        # behavior has its own unit tests); --out WITHOUT .npz suffix must
        # still produce + report the real file.
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/compute_whitening_stats.py"),
             "--parquet", str(raw_pq), "--out", str(td / "stats"),
             "--shrinkage", "0.0", "--floor-quantile", "0.0"],
            capture_output=True, text=True)
        assert r.returncode == 0, f"stats script failed:\n{r.stdout}\n{r.stderr}"
        assert stats_npz.exists(), "suffix-less --out did not land at stats.npz"

        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/whiten_dataset.py"),
             "--parquet", str(raw_pq), "--stats", str(stats_npz), "--out", str(white_pq)],
            capture_output=True, text=True)
        assert r.returncode == 0, f"whiten script failed:\n{r.stdout}\n{r.stderr}"
        assert "distribution gate OK" in r.stdout

        # vectors: near-identity covariance, near-zero mean
        out = pq.read_table(str(white_pq))
        xw = out.column("activation_vector").combine_chunks().flatten().to_numpy(
            zero_copy_only=False).reshape(n, d)
        cov = np.cov(xw.astype(np.float64), rowvar=False)
        assert np.abs(np.diag(cov) - 1).max() < 0.05, "whitened diag far from 1"
        assert np.abs(xw.mean(axis=0)).max() < 0.05, "whitened mean far from 0"

        # schema (incl. 'element' child name) and non-activation columns identical
        inp = pq.read_table(str(raw_pq))
        assert out.schema.equals(inp.schema), f"schema changed:\n{out.schema}\nvs\n{inp.schema}"
        for c in ("prompt", "response", "doc_id"):
            assert out.column(c).equals(inp.column(c)), f"column {c} changed"

        # sidecar contract: tag + provenance block + lineage, tokens preserved
        meta = read_sidecar_local(white_pq)
        assert meta.extraction.norm == NORM_WHITENED_ZCA
        stats = load_stats(str(stats_npz))
        wb = meta.extraction.whitening
        assert wb["stats_sha256"] == stats.sha256()
        assert wb["n_samples"] == n
        assert wb["base_model"] == "test/model" and wb["layer_index"] == 2
        assert meta.parent_datasets == ["synthetic-test"]
        assert meta.dataset_id.startswith("synthetic-test-wzca-")
        assert meta.row_count == n
        assert meta.tokens.injection_char == "㈎"
        assert meta.prompt_templates["actor"].startswith("explain")

        # exactness: parquet content == whiten(x) in float32
        from nla.whitening import whiten
        expect = whiten(x.astype(np.float64), stats).astype(np.float32)
        assert np.array_equal(xw, expect), "stored vectors != whiten(raw)"

        # refuses to double-whiten
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/whiten_dataset.py"),
             "--parquet", str(white_pq), "--stats", str(stats_npz),
             "--out", str(td / "double.parquet")],
            capture_output=True, text=True)
        assert r.returncode != 0, "double-whitening should be refused"
        assert "already transformed" in (r.stdout + r.stderr)


def t_wrong_stats_rejected():
    """unwhiten(whiten(x)) == x for ANY stats, so only a distribution/provenance
    check can catch a wrong stats file — verify both layers actually fire."""
    from nla.whitening import compute_stats_from_array, save_stats

    n, d = 800, 16
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_pq = td / "av_sft.parquet"
        _write_synthetic_avsft_parquet(raw_pq, n, d)

        # 1. stats from a DIFFERENT distribution (shifted + scaled), no
        #    provenance (base_model unset) → must die at the distribution gate
        rng = np.random.default_rng(99)
        other = rng.standard_normal((4000, d)) * 3.7 + 42.0
        wrong = compute_stats_from_array(other, source="other-distribution")
        save_stats(wrong, str(td / "wrong.npz"))
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/whiten_dataset.py"),
             "--parquet", str(raw_pq), "--stats", str(td / "wrong.npz"),
             "--out", str(td / "w1.parquet")],
            capture_output=True, text=True)
        assert r.returncode != 0, "wrong-distribution stats were accepted"
        assert "does not match the stats' distribution" in (r.stdout + r.stderr)

        # 2. right distribution, wrong provenance → must die at the pin check
        x = _anisotropic_sample(n, d, seed=7)
        pinned = compute_stats_from_array(x, base_model="other/model", layer_index=2)
        save_stats(pinned, str(td / "pinned.npz"))
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/whiten_dataset.py"),
             "--parquet", str(raw_pq), "--stats", str(td / "pinned.npz"),
             "--out", str(td / "w2.parquet")],
            capture_output=True, text=True)
        assert r.returncode != 0, "wrong-model stats were accepted"
        assert "stats were computed on" in (r.stdout + r.stderr)


def t_config_asserts_norm_consistency():
    # load_nla_config needs a live tokenizer (not available CPU-side), so test
    # the norm/whitening consistency logic the same way it runs there.
    from nla.schema import resolve_activation_norm
    for norm_raw, whitening, ok in [
        (None, None, True),
        ("none", None, True),
        ("whitened_zca_v1", {"stats_sha256": "ab"}, True),
        ("whitened_zca_v1", None, False),
        ("none", {"stats_sha256": "ab"}, False),
    ]:
        try:
            norm = resolve_activation_norm(norm_raw)
            assert (norm == "none") == (whitening is None)
            passed = True
        except AssertionError:
            passed = False
        assert passed == ok, f"norm={norm_raw!r} whitening={whitening!r}: expected ok={ok}"


def t_fve_baselines_work_on_whitened_vectors():
    # whitened-space FVE reuses compute_predict_mean_baselines unchanged;
    # with isotropic unit variance + mse_scale=None the raw-variance baseline
    # is per-element variance ≈ 1. (NB the OTHER return, the meannorm
    # baseline, is degenerate on whitened data — μ≈0 normalizes to noise —
    # see docs/whitening_experiment.md.)
    import torch

    from nla.schema import compute_predict_mean_baselines
    from nla.whitening import compute_stats_from_array, whiten
    x = _anisotropic_sample(seed=8)
    stats = compute_stats_from_array(x, **EXACT)
    xw = torch.from_numpy(whiten(x, stats))
    _, raw_var = compute_predict_mean_baselines(xw, mse_scale=None)
    assert abs(raw_var - 1.0) < 0.02, f"whitened per-element variance {raw_var} ≉ 1"


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name, fn)
    failed = [n for n, e in RESULTS if e is not None]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)
