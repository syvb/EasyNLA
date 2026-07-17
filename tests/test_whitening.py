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
    stats = compute_stats_from_array(x, shrinkage=0.0)
    xw = whiten(x, stats)
    cov = np.cov(xw, rowvar=False)
    assert np.abs(np.diag(cov) - 1).max() < 1e-6, f"diag off by {np.abs(np.diag(cov) - 1).max()}"
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 1e-6, f"offdiag {np.abs(off).max()}"
    assert np.abs(xw.mean(axis=0)).max() < 1e-8


def t_roundtrip_inverse():
    from nla.whitening import compute_stats_from_array, unwhiten, whiten
    x = _anisotropic_sample(seed=1)
    stats = compute_stats_from_array(x)
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


def t_shrinkage_fixes_rank_deficiency():
    from nla.whitening import compute_stats_from_array
    rng = np.random.default_rng(3)
    d = 30
    # rank-deficient: data lives in a 5-dim subspace
    x = rng.standard_normal((500, 5)) @ rng.standard_normal((5, d))
    try:
        compute_stats_from_array(x, shrinkage=0.0)
        raise RuntimeError("shrinkage=0 on rank-deficient data should assert")
    except AssertionError:
        pass
    stats = compute_stats_from_array(x, shrinkage=0.01)  # must not raise
    assert stats.eigenvalues[0] > 0


def t_shrinkage_bounds_amplification():
    # eigenvalue floor is ≈ λ·(trΣ/d) ⇒ max whitening gain ≈ sqrt(mean/floor)
    from nla.whitening import compute_stats_from_array
    x = _anisotropic_sample(seed=4)
    lam = 0.05
    stats = compute_stats_from_array(x, shrinkage=lam)
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
    assert np.abs(streamed.w - ref.w).max() < 1e-8
    assert np.abs(streamed.mean - ref.mean).max() < 1e-10


def t_npz_roundtrip_and_hash():
    from nla.whitening import compute_stats_from_array, load_stats, save_stats
    stats = compute_stats_from_array(_anisotropic_sample(seed=6), source="synthetic")
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "stats.npz")
        save_stats(stats, p)
        loaded = load_stats(p)  # asserts stored sha == recomputed sha
        assert loaded.sha256() == stats.sha256()
        assert np.array_equal(loaded.w, stats.w)
        assert np.array_equal(loaded.mean, stats.mean)
        assert loaded.n_samples == stats.n_samples
        assert loaded.source == "synthetic"


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
    """Tiny av_sft-shaped parquet + valid sidecar, mirroring stage3's schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar_local
    from nla.schema import NLATokenMeta

    x = _anisotropic_sample(n, d, seed).astype(np.float32)
    av_type = pa.list_(pa.float32(), d)
    prompt_t = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
    msg = [{"role": "user", "content": "explain <concept>㈎</concept>"}]
    table = pa.table({
        "prompt": pa.array([msg] * n, type=prompt_t),
        "response": pa.array([f"<explanation>\nrow {i}\n</explanation>" for i in range(n)]),
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(x.reshape(-1), type=pa.float32()), d),
        "doc_id": pa.array([f"doc{i // 3}" for i in range(n)]),
    })
    # 4 row groups so the rewriter's row-group loop is actually exercised
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

        # shrinkage 0 so whitened cov is exactly identity — shrinkage's
        # deliberate under-whitening of weak directions (diag < 1) is covered
        # by t_shrinkage_* unit tests; this test validates the plumbing.
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/compute_whitening_stats.py"),
             "--parquet", str(raw_pq), "--out", str(stats_npz), "--shrinkage", "0.0"],
            capture_output=True, text=True)
        assert r.returncode == 0, f"stats script failed:\n{r.stdout}\n{r.stderr}"

        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/whiten_dataset.py"),
             "--parquet", str(raw_pq), "--stats", str(stats_npz), "--out", str(white_pq)],
            capture_output=True, text=True)
        assert r.returncode == 0, f"whiten script failed:\n{r.stdout}\n{r.stderr}"

        # vectors: near-identity covariance, near-zero mean
        out = pq.read_table(str(white_pq))
        xw = out.column("activation_vector").combine_chunks().flatten().to_numpy(
            zero_copy_only=False).reshape(n, d)
        cov = np.cov(xw.astype(np.float64), rowvar=False)
        assert np.abs(np.diag(cov) - 1).max() < 0.05, "whitened diag far from 1"
        assert np.abs(xw.mean(axis=0)).max() < 0.05, "whitened mean far from 0"

        # non-activation columns byte-identical; row-group structure preserved
        inp = pq.read_table(str(raw_pq))
        for c in ("prompt", "response", "doc_id"):
            assert out.column(c).equals(inp.column(c)), f"column {c} changed"
        assert (pq.ParquetFile(str(white_pq)).num_row_groups
                == pq.ParquetFile(str(raw_pq)).num_row_groups)

        # sidecar contract: tag + provenance block + lineage, tokens preserved
        meta = read_sidecar_local(white_pq)
        assert meta.extraction.norm == NORM_WHITENED_ZCA
        stats = load_stats(str(stats_npz))
        wb = meta.extraction.whitening
        assert wb["stats_sha256"] == stats.sha256()
        assert wb["n_samples"] == n
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
    # is per-element variance ≈ 1.
    import torch

    from nla.schema import compute_predict_mean_baselines
    from nla.whitening import compute_stats_from_array, whiten
    x = _anisotropic_sample(seed=8)
    stats = compute_stats_from_array(x, shrinkage=0.0)
    xw = torch.from_numpy(whiten(x, stats))
    _, raw_var = compute_predict_mean_baselines(xw, mse_scale=None)
    assert abs(raw_var - 1.0) < 0.02, f"whitened per-element variance {raw_var} ≉ 1"


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in list(globals().items()) if k.startswith("t_")):
        check(name, fn)
    failed = [n for n, e in RESULTS if e is not None]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)
