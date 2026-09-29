# KL-NLA: training and judging NLAs with a splice-KL objective

Writeup of the `sv/kl-nla` thread, 2026-09-25 to 2026-09-28. Model: Qwen3-8B, layer 24. Detailed logs and the
pre-registrations are in [`kl_nla.md`](kl_nla.md) (Phases 0, 1, hedging control, 1b) and
[`kl_nla_phase2.md`](kl_nla_phase2.md) (RL pilot). Every number below is from those runs. Brackets are 95%
bootstrap intervals over documents.

## TL;DR

- **As a metric, splice-KL is useful.** It ranks explanations the same way FVE does, but per row it agrees
  with MSE only loosely (ρ 0.3–0.6). It shows what FVE hides: the AR's reconstruction error falls on the
  directions the model's output depends on. Gold explanations reach FVE 0.64 but cost 0.86 nats of KL; a
  random error of the same size costs 0.12.
- **As an AR training loss it helps, about half through calibration.** One epoch of KL training takes KL
  recovered on the SFT AV's explanations from 0.72 to 0.86 (+0.139), at an FVE cost of 0.50 → 0.34. Shrinking
  an MSE-trained AR toward a good prior gets 55% of that for free; the rest (+0.063) is extra output-relevant
  content read from the same text. MSE-trained ARs are overconfident.
- **As an RL reward for the verbalizer it did not help.** In a matched 100-step pilot, −KL and −MSE rewards
  raised frozen-judge KL recovered about equally. −KL was no better on KL (−0.007 [−0.017, +0.002]), even
  under the judge that favours it, and cost ~0.035 FVE.
- **Scoring more than the next token doesn't change this.** The splice's effect is almost entirely on the
  next token: the 16 tokens after carry only ~5% of the KL, and rank explanations identically.
- **Total cost ≈ $32** of Secure GPU time. ~$13 of it went on aborted or slow pods during the RL setup,
  mostly fixed infrastructure bugs.

## 1. Question

An NLA has a verbalizer (AV) that describes a residual activation in text, and a reconstructor (AR) that
maps the text back to a vector. It is scored and RL-trained by reconstruction MSE (FVE), which weights every
direction of the vector equally. SAE work uses a different yardstick: splice the reconstruction back into the
model and measure how much the model's output changes, KL(p_orig ‖ p_splice). The questions:

1. Is splice-KL a sensible way to judge NLA explanations, and how does it differ from FVE?
2. Does training the AR on KL make it extract more functionally relevant content?
3. Does a −KL RL reward give the AV better explanations?

## 2. Method

**The objective.** For an activation h taken from the output of layer K at position t of a prefix, the AR's
prediction replaces h in a frozen copy of the target model. The prediction is rescaled to ‖h‖, because the AR
is direction-only. The loss is KL between the original and spliced next-token distributions at t, and
optionally at later positions (§5.5). "KL recovered" = 1 − KL / KL(mean activation direction) is the KL
counterpart of FVE.

**Source text.** The datasets store the decoded prefix up to t (`detokenized_text_truncated`), not token ids.
Prefixes are re-tokenized and kept only if they give back exactly `n_raw_tokens` tokens. That held for 100% of
the rows used, and splicing the stored activation back in gives KL ≈ 0 (median 5×10⁻⁴ nats at 8B in bf16),
which confirms layer, position and tokenization.

**Implementation** (`nla/utils/kl_splice.py`, `SpliceKL`):
- Each unique prefix runs once into a KV cache.
- All candidate reconstructions of a prefix run together as extra query tokens at the same position, with a
  4D mask so each sees only the cached prefix and itself: copy 0 unspliced (p_orig), the rest spliced. One
  cached prefix serves every rollout of a prompt, and gradients flow through the single spliced position.
- The multi-position version makes each candidate a block [x_t, y₁…y_m] with a block-diagonal causal mask,
  where y is the model's own greedy continuation.

**Tests** (`tests/test_kl_splice.py`):
- a match against a plain unpadded forward with the replacement applied by hand;
- ~0 KL for the natural activation;
- gradient flow to the vectors only;
- the RL scorer's KL path.

Mask and position mutants (copies seeing each other, no in-block causal mask, wrong positions) all fail the
reference test.

**Wiring.** Behind `--recon-loss {mse,kl}`, default `mse`, unchanged:
- `nla/train_sft.py` (AR): KL loss, with held-out KL and KL recovered;
- `nla/train_rl_vllm.py`: reward −KL, failures scored as the KL of an orthogonal vector (the analogue of the
  −2.0 MSE floor), the co-trained AR minimizing KL, and eval KL.

## 3. Setup

- **Models:** the SFT (no RL) NLA `syvb/nanonla-qwen3-8b-L24-av` / `-ar`. The AR loads through EasyNLA's
  `NLACriticModel` and reproduces the earlier fvecmp FVE exactly. Target: `Qwen/Qwen3-8B`.
- **Eval rows:** 2,844 rows / 287 documents of `asher577/easynla-warmstart-data:av_sft_val` that neither the
  AV nor the AR trained on, with the explanations fvecmp saved. Those are gold (Claude, written from the text
  only), SFT AV greedy/sampled, RL p0.0 greedy/sampled, the last 40 words quoted, and a wrong document's
  explanation. No generation was needed for Phases 0–1.
- **Vector references:** the mean activation direction (the KL-recovered baseline), a vector orthogonal to the
  true one, and the true vector rotated to cosine c (to map FVE onto KL).
- **Frozen judges** for RL: `A_kl` (Phase 1 KL-trained AR), `A_mse` (Phase 1 MSE-trained AR), and `A_cal`
  (A_mse shrunk toward A_kl's prior at α = 0.6, the calibrated MSE AR from §5.3).

## 4. Code

All on branch `sv/kl-nla` (off `origin/main`):

| file | what |
|---|---|
| `nla/utils/kl_splice.py` | `SpliceKL` (single- and multi-position), prefix re-tokenization, loader |
| `nla/train_sft.py`, `nla/train_rl_vllm.py` | `--recon-loss kl` |
| `scripts/kl_audit.py` | Phase 0/1/1b audit: FVE + KL for text and vector conditions, gates, next-token diagnostic, `--future M` |
| `scripts/kl_hedge_control.py` | no-info baselines + cross-fit shrinkage |
| `scripts/kl_phase1_contrasts.py`, `scripts/kl_rl_compare.py` | paired contrasts |
| `scripts/merge_prepared_ar.py` | merge a LoRA fit on a prepared AR into a full critic (checked vs LoRA load) |
| `scripts/kl_rl_prep.py`, `scripts/kl_curve_eval.py` | RL data (minus audit docs); frozen-judge checkpoint curves |
| `scripts/pod_kl_nla.sh`, `scripts/runpod_kl_nla.py`, `scripts/hf_sync.py` | RunPod job and launcher (Secure only, HF as the store, watchdog, heartbeat) |
| `utils/patch_vllm_lens.py` | fix: patches fresh vllm-lens installs and is idempotent (§7) |

## 5. Results

### 5.1 Phase 0: splice-KL as a metric (SFT AR)

| condition | KL (nats) | KL recovered | FVE |
|---|---|---|---|
| gold | 0.859 | 0.842 [0.826, 0.857] | 0.641 |
| RL p0.0 greedy | 1.284 | 0.764 | 0.533 |
| RL p0.0 sampled | 1.401 | 0.743 | 0.484 |
| SFT AV greedy | 1.527 | 0.720 [0.697, 0.741] | 0.491 |
| SFT AV sampled | 2.029 | 0.627 | 0.366 |
| quote (last 40 words) | 2.150 | 0.605 | 0.383 |
| wrong document | 11.286 | −1.075 | −0.841 |
| mean direction | 5.446 | 0 | −0.203 |
| orthogonal vector | 11.161 | −1.049 | −2.568 |

- **Same ordering as FVE.** A wrong document's explanation is as damaging as an orthogonal vector, and worse
  than predicting the mean.
- **Loose agreement per row:** Spearman(KL, MSE) is 0.29 (gold) to 0.60 (quote).
- **The error lands on output-relevant directions.** Rotating the true vector at random gives FVE 0.643 and
  KL recovered 0.979 at cos 0.9, and FVE −0.07 but still 0.902 at cos 0.7. Gold explanations match cos 0.9 on
  FVE but lose 0.86 nats of KL, not 0.115: about 7× the damage of an isotropic error of the same size.
- **KL is heavy-tailed:** SFT AV greedy has median 0.45 nats, mean 1.53.
- **Next-token guessing is not the driver.** The SFT AV names the target's greedy next token in 60% of eligible
  rows, and those rows carry 56% of its KL gain: proportional, not concentrated.

### 5.2 Phase 1: AR trained on KL vs MSE (matched)

Both arms continue the SFT AR for one epoch (782 steps, effective batch 64, LoRA r128, lr 5e-5) on the AR's
own training data. They differ only in the loss.

| AR | KL rec., SFT AV greedy | FVE, SFT AV greedy | KL rec., gold | FVE, gold |
|---|---|---|---|---|
| sft (start) | 0.720 | 0.491 | 0.842 | 0.641 |
| + 1 epoch MSE | 0.721 | 0.499 | 0.845 | 0.651 |
| + 1 epoch KL | **0.860** | 0.343 | **0.915** | 0.464 |

- **KL minus MSE AR, SFT AV greedy: +0.139 [+0.126, +0.153].** It's positive for every explanation type:
  gold +0.071, RL greedy +0.110, quote +0.189. The MSE control is flat (+0.001), so the gain comes from the
  loss, not the extra epoch. Top-1 agreement is +0.109.
- **The gain is in the tail.** On the quarter of rows where the MSE AR is worst, KL goes 4.97 → 2.31 nats. On
  its best quarter it's slightly worse (0.047 → 0.073).
- **Not a next-token effect:** the gain on rows whose explanation doesn't name the next token (+0.128) is at
  least as large as on rows that do (+0.090).
- **The warning sign:** with the KL AR, a *wrong* document's explanation beats the mean baseline on 30% of
  rows (was 5%). Wrong-document KL falls from 11.3 to 6.9 nats.

### 5.3 Hedging control: calibration or content?

An explanation-free input gives each AR one constant vector, its learned prior. The MSE AR's predictions were
shrunk toward a prior, with the blend weight cross-fit over documents. The decision rule was pre-registered.

| shrink the MSE AR toward | gap closed (of +0.139) |
|---|---|
| the mean direction (α = 0.7) | 7.4% |
| the KL AR's prior (α = 0.6) | **54.7% [50.6, 58.7]** |

- **Verdict (pre-registered): mostly calibration, narrowly.** About half the Phase 1 gain comes free by
  shrinking an overconfident MSE AR toward a good prior. The mean direction is a poor prior for KL.
- **The other half is real content:** raw KL AR − best-shrunk MSE AR = **+0.063 [+0.056, +0.069]** (gold +0.052).
- **The KL AR is already calibrated:** its best α is 1.0. Shrinkage also reproduces its gentler wrong-document
  damage (−1.08 → −0.32, vs −0.28).

### 5.4 RL pilot: −KL vs −MSE reward

Two arms on `nla.train_rl_vllm`: merged SFT AV + fresh LoRA, 100 steps × 32 prompts × 8 rollouts, AR
co-trained (`--ar-lora`) on each arm's own loss, one H200 each at ~28–30 s/step. Every 20 steps, greedy
explanations on 500 audit rows were scored by the frozen judges.

| step | KL rec. A_cal (KL / MSE arm) | KL rec. A_kl (KL / MSE) | FVE A_mse (KL / MSE) | names next token (KL / MSE) |
|---|---|---|---|---|
| 0 | 0.799 / 0.799 | 0.850 / 0.850 | 0.504 / 0.504 | 0.61 / 0.61 |
| 20 | 0.832 / 0.851 | 0.882 / 0.900 | 0.551 / 0.585 | 0.67 / 0.66 |
| 40 | 0.840 / 0.837 | 0.886 / 0.881 | 0.564 / 0.599 | 0.68 / 0.74 |
| 60 | 0.843 / 0.857 | 0.888 / 0.901 | 0.556 / 0.595 | 0.69 / 0.72 |
| 80 | 0.858 / 0.855 | 0.902 / 0.904 | 0.576 / 0.605 | 0.68 / 0.71 |
| 100 | 0.845 / 0.856 | 0.884 / 0.902 | 0.572 / 0.608 | 0.77 / 0.63 |

- **Shape:** both arms rise in the first ~20 steps, then plateau, on every judge. Quoting stays ≈ 0, vagueness
  is flat, and specificity rises alike in both.
- **Both rewards improve the functional metric:** gain vs step 0 pooled over steps 20–100, under A_cal, is
  +0.046 [+0.027, +0.065] for the KL arm and +0.054 [+0.034, +0.075] for the MSE arm.
- **−KL gives no KL advantage:** paired, pooled, KL arm − MSE arm = −0.007 [−0.017, +0.002] under A_cal and
  −0.008 [−0.016, +0.000] under A_kl, the judge that favours the KL arm.
- **It costs FVE:** ~0.035 lower at every checkpoint.
- **Next-token drift is not established:** pooled +0.004 [−0.021, +0.029]. One checkpoint (step 100) shows
  +0.138, but step 40 went the other way.

### 5.5 Phase 1b: does the activation matter beyond the next token?

Same audit with the splice kept at t and KL scored on the target's own 16 greedy continuation tokens (teacher-forced).

| | next token | 16 later tokens |
|---|---|---|
| share of KL, gold (SFT AR) | 94.6% | **5.4% [4.7, 6.2]** |
| share of KL, SFT AV greedy | 95.3% | **4.7% [4.0, 5.4]** |
| mean-direction splice, KL at t, t+1, t+2… | 5.446 | 0.060, 0.018, ≤ 0.015 |

- **Verdict (pre-registered): stop.** A single-position change to the layer-24 residual barely reaches later
  tokens; its only route is attention in layers 25–35. The rankings by next, future and total KL are identical
  for all three ARs.
- **Low row-level ρ (≈ 0.2) is noise, not a second signal:** future KL is at the bf16 floor. Splicing the true
  activation back in already gives 0.012 nats of future KL.
- **So next-token KL wasn't too narrow a measure; it's where the effect is.** Multi-position KL would not
  change §5.4.

## 6. Conclusions

1. **Keep splice-KL as an evaluation metric beside FVE.** It is cheap (~2 min for 2,844 rows × 15
   conditions on one H100), valid (natural activation → 0, sensible ordering), and it measures something FVE
   does not: whether the reconstruction preserves what the model does with the activation. FVE 0.64 hides
   errors that cost 7× more than their size suggests.
2. **MSE-trained ARs are miscalibrated.** Shrinking their output toward a good prior (e.g. a KL-trained AR's
   no-info prediction) is a free improvement in KL recovered: 0.72 → 0.80 on SFT AV explanations.
3. **A KL-trained AR reads somewhat more output-relevant content from the same explanations** (+0.063 over a
   calibrated MSE AR), at a large FVE cost.
4. **That doesn't become better explanations under RL here.** The −MSE reward already moves explanations the
   way KL rewards them, and −KL doesn't move them further.
5. **The single-position splice's effect is a next-token effect.** Any future attempt to reward "what the
   model uses later" needs a different intervention than replacing one position.

## 7. Limitations

- **One model and layer** (Qwen3-8B L24), one seed per arm, and the SFT NLA from nanoNLA as the starting point.
- **The RL pilot is short:** 100 steps at a quarter of the tuned batch. It shows direction and early shape,
  not endpoints. A null over 100 steps is weak evidence against a slow effect.
- **Home advantage:** ARs are scored on the metric they were trained for. The calibrated judge (built from
  the MSE AR) and cross-scoring limit this but don't remove it.
- **Teacher forcing:** the multi-position audit uses the original model's continuation. In free generation,
  effects on later tokens run through the next-token distribution, which next-token KL already measures.
- **Single intervention:** the splice replaces one position at one layer. Explanations of multi-position or
  multi-layer state were not tested.

## 8. Engineering notes

These are worth knowing before the next RunPod RL run. All are fixed on this branch.

- **The upstream vllm-lens patcher refused every fresh install.** `utils/patch_vllm_lens.py` checked all
  hunks against the pristine file, but later hunks anchor on text earlier hunks write, so on a fresh
  vllm-lens 1.1.0 it printed "version drift" and patched nothing. The trainer then (rightly) refused to run.
  Checking in order alone isn't idempotent, because hunk 7's replacement contains its own anchor. The fix
  rebuilds from the pristine backup every run. A regression test was added; the old baseline test missed it
  because its synthetic baseline concatenates every hunk's anchor. **This is an upstream bug worth a PR to
  `main`.**
- **The vLLM v1 weight sync needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.** The engine runs in a subprocess, and
  `apply_model` ships a `functools.partial`.
- **Launcher safety:**
  - Secure Cloud only. `plan` now prices with `securePrice`; the old `lowestPrice` is the Community price,
    about 30% lower.
  - RL stages only on ≥ 141 GB GPUs (H200 / H200 NVL). The first pod silently fell back to an 80 GB H100.
  - A 5-min heartbeat pushes the pod's log to HF, since RunPod has no remote log access.
  - `hf_transfer` for the ~50 GB of downloads per pod.
- **Machine variance is large.** One H200 ran at ~36 MB/s downloads, a 16-min AR merge and 2.3 min/step
  (generation 28–80 s against 6 s elsewhere). Check `time/gen_s` in the first steps (> 20 s at 32 × 8 means a
  bad machine) and relaunch.
- **`train_sft` treats `--base-ckpt` as a prepared critic only if it's a local directory** containing
  `value_head.safetensors`. A repo id silently re-truncates the model with an identity head. Also, the
  existing `merge_lora_to_hf.py` assumes a LoRA fit on a freshly truncated raw base; use
  `merge_prepared_ar.py` for LoRA on a prepared AR.

## 9. Cost

These are estimates from pod uptime × Secure price (H100 $3.49/h, H200 $4.59/h).

| step | GPU | ≈ cost |
|---|---|---|
| Phase 0 audit | H100, ~20 min | $1.1 |
| Phase 1 (two AR arms + audit) | H100, 1.4 h | $4.9 |
| Hedging control | H100, ~25 min | $1.5 |
| RL pilot, useful arms (MSE 1.05 h, KL 1.2 h) | 2 × H200 | $10.3 |
| RL pilot, aborted/slow pods (80 GB fallback, patcher bug, serialization bug, stalled setup, slow machine) | H100/H200 | $12.8 |
| Phase 1b multi-position audit | H100, ~17 min | $1.0 |
| **total** | | **≈ $32** |

## 10. Artifacts and reproduction

- **HF dataset `syvb/kl-nla-qwen3-8b` (private):**
  - evals: `evals/audit0`, `evals/audit1`, `evals/hedge`, `evals/audit_future`, `evals/curves_rl_kl`,
    `evals/curves_rl_mse`, `evals/rl_compare`;
  - checkpoints: `ckpts/ar_kl`, `ckpts/ar_mse` (Phase 1 LoRA ARs), `ckpts/ar_kl_merged`, `ckpts/ar_mse_merged`,
    and `ckpts/rl_kl`, `ckpts/rl_mse` (RL LoRA every 20 steps + co-trained ARs).
- **W&B `kl-nla`:** group `kl-nla-phase1` (AR arms), group `kl-nla-rl-pilot` (runs `rl_kl`, `rl_mse`,
  `curves_rl_kl`, `curves_rl_mse`).
- **Re-running a stage:** `python scripts/runpod_kl_nla.py launch --stages <stage> --no-keep`. Stages are
  `audit0`, `ar_kl,ar_mse,audit1`, `hedge`, `audit_future`, and `rl_kl` / `rl_mse` (add
  `--gpu "NVIDIA H200"`). `plan` prices a stage without spending. Contrasts:
  `scripts/kl_phase1_contrasts.py <audit dir>` and `scripts/kl_rl_compare.py <kl curves> <mse curves> <out>`.
- **CPU checks:** `python tests/test_kl_splice.py`, `python tests/test_patcher_baseline.py`.
