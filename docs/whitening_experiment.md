# Experiment: NLA on whitened activations (Qwen3-8B, layer 24)

**Question.** What does an NLA learn when reconstruction error weights every
direction of the residual stream equally — instead of being dominated by the
few highest-variance directions?

**Method.** Train the standard Qwen3-8B L24 NLA recipe unchanged, but on
ZCA-whitened activations x̃ = W(x − μ), W = Σ^{-1/2} of the train activation
distribution. Precisely: since the reward/FVE pipeline L2-normalizes both
prediction and gold to ‖·‖=√d before MSE, the optimized quantity is the
**angle in the (regularized) Mahalanobis geometry** of the centered
distribution — raw-space training optimizes the same angle in Euclidean
geometry, which is dominated by the high-variance directions. (Do not
describe this as "Mahalanobis distance": all norm information is discarded
by the normalization in both arms.)

The transform is **offline**: `scripts/whiten_dataset.py` rewrites the
parquets and restamps the sidecars (`extraction.norm: whitened_zca_v1` + a
provenance block); the trainers on this branch run with **no code changes**.
Gold explanations are reused as-is — whitening is a fixed invertible linear
map, so each explanation still describes the same underlying activation
(and regenerating them would confound text distribution with training
space).

**What the treatment bundles.** Offline whitening changes three things at
once: (a) the AV's input geometry (raw injections lie in a narrow cone
around the dominant directions; whitened ones are near-orthogonal and
centered), (b) the AR head's target geometry (isotropic, full-rank demand
on the Linear(d,d) head), and (c) the reward metric. The research question
is (c); a result difference could in principle come from (a)/(b). If the
2×2 (below) shows an interesting effect, the sharper follow-up is whitening
**inside the critic loss only** (map pred and gold by W before MSE — small
trainer change) to isolate (c).

## Headline expectations

- **Whitened held-out FVE will read much lower than the ~70% raw benchmark.
  This is the metric being harder, not the run failing.**
- Success criterion is therefore **relative**: whitened-trained NLA vs the
  existing raw-trained NLA, both evaluated in *both* spaces (see Evals),
  with the whitened-space column anchored by step 0b below.

## Pipeline

Assumes the HF warmstart data (`asher577/easynla-warmstart-data`) is at
`<data>`. It contains **only** `{av,ar}_sft_{train,val}.parquet` + sidecars
— **no RL parquet**, so the RL split must be regenerated (step 0c). Known
gotcha: some sidecar `row_count`s there are stale; `whiten_dataset.py`
recounts and corrects on rewrite.

> ⚠️ **Never point a pre-whitening checkout at whitened parquets.** Old
> `load_nla_config` ignores `extraction.norm` and trains silently as if the
> data were raw. Only datagen-side readers fail loudly. Same for model
> checkpoints: if a whitened-trained checkpoint gets a hand-written
> `kind: nla_model` sidecar, it must carry `extraction.norm` +
> `extraction.whitening`, or this branch too will load it as raw.

### 0a · Whitening stats (CPU, minutes)

Compute from `av_sft_train.parquet` — **exactly that file, no substitutes**
(the 2× larger regenerated RL split would shave bulk eigenvalue noise from
~10.6% to ~7.5% — statistically immaterial, and a fixed choice keeps runs
reproducible). Never compute stats on a val split: whiten val with the
TRAIN stats or held-out FVE silently leaks.

```bash
python scripts/compute_whitening_stats.py \
    --parquet <data>/av_sft_train.parquet \
    --out <data>/whitening_stats.npz
```

- Regularization default is **eigenvalue flooring at the 5th percentile**:
  95% of directions are whitened *exactly*; only the noisy tail is capped.
  (Shrinkage toward (trΣ/d)·I is available but its target is inflated by
  the outlier top of the spectrum, under-whitening a broad band of exactly
  the low-variance directions this experiment is about.)
- **Go/no-go:** the script prints the achieved whitened-variance spectrum
  (`whitened var: mean/min/frac dirs < 0.9`). If more than ~10% of
  directions sit below 0.9, the treatment is materially attenuated — stop
  and revisit the floor before spending GPU time. n/d ≈ 89 is otherwise
  fine (Marchenko–Pastur bulk noise ≤ ~12% in the 1/√λ scaling).
- Robustness check (cheap, recommended): compute stats on two disjoint
  halves (`--max-rows` + a reversed file) and check each half's whitened
  variance under the other's W — heavy-tailed "massive activation" rows
  hurt Σ estimation more than Gaussian intuition suggests.

### 0b · The free cell first ($0, do before ANY training)

Evaluate the **existing raw-trained NLA in whitened space** before training
anything: regenerate explanations for the held-out split with the raw
checkpoint (or reuse saved ones if complete), run its critic, map
predictions to whitened space (convention below), compute whitened FVE.
This (i) debugs the cross-space mapping code on real data, (ii) sizes the
headroom whitened training could claim, and (iii) is the baseline the
headline comparison needs. If this cell is already near the whitened-space
ceiling, the experiment's interesting outcome disappears — find out for $0.

### 0c · Regenerate the RL split (GPU extraction, no API cost)

The RL split never touches the paid stage-2 explanations (stage-1 `rl_raw`
goes straight to stage-3 build). Re-run datagen with the **original**
`configs/datagen/qwen3_8b_finefineweb_100k.yaml` (same corpus slice +
seeds), stages 0, 1, 3, shuffle only:

```bash
python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_8b_finefineweb_100k.yaml \
    --stages 0,1,3,shuffle
```

Then **verify doc-disjointness** of the regenerated `rl_shuf.parquet`
against the HF val splits (`doc_id` intersection must be empty — stage-1
seeding should reproduce the original split; trust but verify). Extraction
is the dominant cost: ~45 min on 8×H100 for the full 100k docs (from the
stage-0 war story in `nla/storage.py`), so ~6 GPU-hours.

### 1 · Whiten every split

```bash
for f in av_sft_train av_sft_val ar_sft_train ar_sft_val rl_shuf; do
    python scripts/whiten_dataset.py \
        --parquet <data>/$f.parquet \
        --stats   <data>/whitening_stats.npz \
        --out     <data_w>/$f.parquet
done
```

Each rewrite checks provenance (stats pinned to base_model/layer) and a
distribution gate (whitened first batch must match the stats' expected
variance/mean) — a wrong stats file dies here, not after SFT. Keep
`whitening_stats.npz` with the run artifacts — `W⁻¹` is required to map
anything back to the raw residual stream.

### 2 · AV / AR warm-start SFT (unchanged recipe, whitened parquets)

Standard `docs/train_new_model.md` §2–3 commands with `--seed 0`, pointing
`--parquet` / `--sidecar` / `--heldout-parquet` at `<data_w>/…`. One epoch
each, same lrs (tuned on raw — a caveat, not a blocker; whitened val CE is
not comparable to the raw-run 1.51-ish numbers, only within-run trends).

**Abort criterion:** AR SFT logs held-out FVE on gold explanations. Compare
it (in its own space) against the raw AR warm-start's ≈50%-of-baseline
mark: if the whitened AR's gold-explanation FVE is near zero at epoch end,
the warm start failed (likely the raw-salience mismatch of the reused
explanations) and RL on top will be noise — stop and diagnose before §3.

Scale sanity: karvonen injection norm-matches to `injection_scale`
regardless of input scale, and `mse_scale`'s per-vector normalization is
close to a no-op on whitened **gold** vectors (norms concentrate near
√d = 64 — the stats script prints the measured spread; heavy tails widen
it somewhat). Note the norm-matching also means whitened injections carry
no norm cue — raw injections didn't either (already norm-matched), so this
is not a regression, but whitened injected *directions* are near-orthogonal
across samples where raw ones clustered in a cone.

### 3 · RL (unchanged recipe)

Standard §4 vLLM command on `<data_w>/rl_shuf.parquet`, `--seed 0`. Watch,
in order:

1. `av/steer_apply_rate`, `av/inject_fail_count` — injection health
   (should be unaffected; whitening only changes vector values).
2. **Within-group reward std and the critic's held-out FVE** — GRPO
   advantages are std-normalized per group, so they *never* visibly
   collapse: a near-constant-reward critic makes RL take full-size steps
   on pure noise while advantage magnitudes look healthy. Flat within-group
   reward spread early = warm-start problem (see §2 abort criterion).
3. `eval/extraction_rate` + reward trend.
4. Held-out whitened FVE vs its own predict-mean baseline. **Baseline
   caveat:** of the two logged predict-mean baselines, only the
   raw-variance one (`fve_nrm` denominator, ≈1 in whitened space) is
   meaningful; the *meannorm* baseline degenerates on whitened data
   (μ ≈ 0 normalizes to an arbitrary direction, baseline ≈ 2, coincides
   with the failed-extraction floor) — ignore `…meannorm…` in whitened
   runs.

**Required pre-run change (small):** the eval loop currently persists only
truncated explanations (500 chars) and scalar rewards to wandb — the 2×2
below needs a final-eval artifact dump per checkpoint: row idx, full
explanation, and the critic's d-dim prediction vector (JSONL/npz). Without
it the promised post-hoc analysis cannot be executed; alternative is
re-generating from saved checkpoints, which costs GPU time and must be
budgeted.

## Evals — the 2×2 that makes results interpretable

Evaluate BOTH models in BOTH spaces:

|  | raw-space FVE | whitened-space FVE |
|---|---|---|
| raw-trained NLA (existing ckpt) | ~70% (known) | step 0b (free cell) |
| whitened-trained NLA (this run) | map preds raw-ward | primary metric |

**Pinned cross-space convention** (the naive "just apply W/W⁻¹" is
ill-defined: the normalize-to-√d loss is scale-invariant, so each critic's
output norm is an arbitrary calibration constant, and the affine map makes
that constant matter through the large μ term):

1. In the critic's native space, rescale each prediction to the **train-set
   mean gold norm of that space** (raw: ≈277; whitened: measured by the
   stats script, ≈√d). Report the per-model calibration factor.
2. Apply the affine map (x̂ = W⁻¹x̂̃ + μ, or x̂̃ = W(x̂ − μ)).
3. Evaluate in the target space exactly as native evals do
   (normalize-to-√d, MSE vs the raw-variance predict-mean baseline).

Normalize-then-map vs map-then-normalize give different numbers — this
order (calibrate, map, target-normalize) is the convention; state it with
the results. Step 0b exercises it end-to-end before the expensive cell
exists.

- `text_judges` (+ `source_match`) run unchanged — a space-independent
  probe of whether whitened training changes *what the explanations say*
  (hypothesis: more mention of low-variance/rare features). Needs
  `ANTHROPIC_API_KEY` — the only API cost in the plan, skippable.
- Single-vector baseline: rerun `scripts/compute_fve_baseline.py` on the
  whitened parquets — the ~50% raw-space figure does not transfer.

**Decision policy.** One seed per arm (`--seed 0`), identical data order.
The repo's own raw runs show ~3pp FVE path-to-path spread, so treat
off-diagonal differences **< 5pp as inconclusive** rather than over-reading
the 2×2; larger effects or consistent direction across both off-diagonal
cells are reportable.

## Risks / expected failure modes

- **Warm-start mismatch (main scientific risk).** Gold explanations were
  written about raw-salient content; whitened reconstruction rewards
  directions those texts may under-describe. Detected by the §2 abort
  criterion and the §3 reward-spread check. Do NOT regenerate explanations
  before seeing the 2×2 — that would confound text distribution with
  training space (and cost API money).
- **Recipe tuned on raw.** lrs, group size, injection_scale carried over
  untested; the AR head now faces an isotropic full-rank target.
- **Regularization bias.** Whitened "unit variance" holds exactly only
  above the eigenvalue floor; the stats script reports the achieved
  spectrum (go/no-go in §0a). FVE baselines are computed from the data
  itself, so metrics stay internally consistent.
- **Contract gaps.** Old checkouts ignore the norm tag (warning above);
  model-checkpoint sidecars need manual norm propagation. Anything that
  consumes AR outputs as residual-stream vectors (steering, patching)
  MUST unwhiten first — `NLAConfig.activation_norm` / `cfg.whitening` is
  the flag to check.

## Cost (4×H200 box, ~$3/GPU-hr; H100 rates similar)

| Stage | Wall clock | Cost |
|---|---|---|
| stats + whitening rewrites (CPU) + free cell (0b) | ~1–2 h | ~$10 |
| RL-split regeneration: datagen stages 0,1,3 (GPU extraction) | ~1–6 h depending on GPUs | $15–25 |
| AV SFT (1 epoch, 1 GPU) + AR SFT (1 epoch, 1 GPU, concurrent) | ~8–14 h | $40–85 |
| merge LoRAs | minutes | — |
| GRPO RL (4 GPU) | ~3–5 h | $40–60 |
| 2×2 re-scoring / artifact-dump evals | ~1 h | ~$5 |
| gold-explanation datagen / API | skipped (text_judges optional extra) | $0 |
| **Total** | **~1–1.5 days** | **~$110–185** (≈$250–320 if the box idles through SFT) |
