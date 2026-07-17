# Experiment: NLA on whitened activations (Qwen3-8B, layer 24)

**Question.** What does an NLA learn when reconstruction error weights every
direction of the residual stream equally — instead of being dominated by the
few highest-variance directions?

**Method.** Train the standard Qwen3-8B L24 NLA recipe unchanged, but on
ZCA-whitened activations x̃ = W(x − μ), W = Σ^{-1/2} of the train activation
distribution. MSE in whitened space is Mahalanobis distance in raw space, so
the GRPO reward (−MSE) and FVE stop being gameable by explaining only the
dominant variance directions.

The transform is **offline**: `scripts/whiten_dataset.py` rewrites the
parquets and restamps the sidecars (`extraction.norm: whitened_zca_v1` + a
provenance block); the trainers run with **no code changes**. Gold
explanations are reused as-is — whitening is a fixed invertible linear map,
so each explanation still describes the same underlying activation.

## Headline expectations

- **Whitened held-out FVE will read much lower than the ~70% raw benchmark.
  This is the metric being harder, not the run failing.** The raw-space
  number is inflated by anisotropy; in whitened space the predict-the-mean
  baseline has per-element variance ≈ 1 in *every* direction.
- Success criterion is therefore **relative**: whitened-trained NLA vs the
  existing raw-trained NLA, both evaluated in *both* spaces (see Evals).

## Pipeline

Everything below assumes the HF warmstart data
(`asher577/easynla-warmstart-data`) is downloaded to `<data>` — no datagen,
no Anthropic API cost. Known gotcha: some sidecar `row_count`s there are
stale; `whiten_dataset.py` recounts and corrects on rewrite.

### 0 · Whitening stats (CPU, minutes)

Compute from the **train** split only; whiten every split (including val)
with the SAME stats — computing stats on val would leak val statistics into
the transform and quietly corrupt held-out FVE.

```bash
python scripts/compute_whitening_stats.py \
    --parquet <data>/av_sft_train.parquet \
    --out <data>/whitening_stats.npz
```

Notes:
- 364k rows at d=4096 is n/d ≈ 89 — the sample covariance's weak directions
  are noisy, which is what the default `--shrinkage 0.01` (toward
  (trΣ/d)·I) is for. It caps the amplification of a near-null direction at
  ~10× the mean direction, at the cost of deliberately **under-whitening
  weak directions** (their whitened variance lands below 1 — measured
  directly in `tests/test_whitening.py::t_shrinkage_*`). If the printed
  condition number and self-check look tame, don't touch it.
- If the RL parquet (~2× larger) is present, computing stats from it instead
  is a fine variation; both are train-side. Record which one in the run notes.

### 1 · Whiten every split

```bash
for f in av_sft_train av_sft_val ar_sft_train ar_sft_val rl_shuf; do
    python scripts/whiten_dataset.py \
        --parquet <data>/$f.parquet \
        --stats   <data>/whitening_stats.npz \
        --out     <data_w>/$f.parquet
done
```

Each rewrite round-trips (whiten → unwhiten) the first batch against the raw
input and aborts on mismatch, so a wrong stats file dies here, not after SFT.
Keep `whitening_stats.npz` with the run artifacts — `W⁻¹` is required to map
anything back to the raw residual stream later.

### 2 · AV / AR warm-start SFT (unchanged recipe, whitened parquets)

Standard `docs/train_new_model.md` §2–3 commands, pointing `--parquet` /
`--sidecar` / `--heldout-parquet` at `<data_w>/…`. One epoch each, same lrs.

Scale sanity: whitened vector norms concentrate tightly around √d = 64
(the stats script prints the measured mean). Karvonen injection norm-matches
to `injection_scale` regardless, and `mse_scale`'s per-vector normalization
becomes a near-no-op — both interact *more* benignly with whitened data than
with raw (raw norms: mean ≈ 277 with real spread).

### 3 · RL (unchanged recipe)

Standard §4 vLLM command on `<data_w>/rl_shuf.parquet`. Watch, in order:

1. `av/steer_apply_rate`, `av/inject_fail_count` — injection health (should
   be unaffected; whitening only changes vector *values*).
2. `eval/extraction_rate` + reward trend — if advantages collapse early,
   suspect the critic warm-start quality (see Risks), not the RL config.
3. Held-out whitened FVE vs its own predict-mean baseline (logged as usual —
   `compute_predict_mean_baselines` is space-agnostic).

## Evals — the 2×2 that makes results interpretable

Evaluate BOTH models in BOTH spaces (linear maps make this cheap post-hoc):

|  | raw-space FVE | whitened-space FVE |
|---|---|---|
| raw-trained NLA (existing ckpt) | ~70% (known) | transform its AR preds: x̂̃ = W(x̂ − μ) |
| whitened-trained NLA (this run) | unwhiten its AR preds: x̂ = W⁻¹x̂̃ + μ | primary metric |

- The interesting cells are the off-diagonal: does whitened training beat raw
  training *on the whitened metric*, and how much raw-space FVE does it give
  up? Both cells are computable from saved generations + the stats npz; no
  retraining.
- `text_judges` (+ `source_match`) run unchanged — a second, space-independent
  probe of whether whitened training changes *what the explanations say*
  (the hypothesis: more mention of low-variance/rare features).
- Single-vector baseline: rerun `scripts/compute_fve_baseline.py` on the
  whitened parquets — the ~50% raw-space figure does not transfer.

## Risks / expected failure modes

- **Warm-start mismatch (main scientific risk).** Gold explanations were
  written (by the API model) about raw-salient content. Whitened
  reconstruction rewards directions those texts may under-describe, so the
  AR warm start could be weaker; if RL rewards are flat at step 0, this is
  the first suspect. Mitigation is a judgment call: accept it (it's part of
  the question) — do NOT regenerate explanations before seeing the 2×2.
- **Recipe was tuned on raw.** lrs, group size, injection_scale carried over
  untested. Whitened SFT val CE will not be comparable to the 1.51-ish raw
  numbers; only compare within-run trends.
- **Shrinkage bias.** Whitened "unit variance" is only approximate in weak
  directions (see §0). FVE baselines are computed from the data itself, so
  metrics stay internally consistent.
- **Sidecar contract.** Whitened sidecars fail loudly on pre-whitening code
  (`extraction.whitening` is an unknown field there) — deliberate. Anything
  that consumes AR outputs as residual-stream vectors (steering, patching
  demos) MUST unwhiten first; `NLAConfig.activation_norm` /
  `cfg.whitening` is the flag to check.

## Cost (4×H200 box, ~$3/GPU-hr)

| Stage | Wall clock | Cost |
|---|---|---|
| stats + whitening rewrites (CPU-bound) | <1 h | ~$5 |
| AV SFT (1 epoch, 1 GPU) + AR SFT (1 epoch, 1 GPU, concurrent) | ~8–14 h | $40–85 |
| merge LoRAs | minutes | — |
| GRPO RL (4 GPU) | ~3–5 h | $40–60 |
| datagen / API | skipped | $0 |
| **Total** | **~1 day** | **~$90–150** (≈$220–280 if the box idles through SFT) |
