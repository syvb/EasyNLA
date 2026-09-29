# KL-NLA Phase 2 plan: does a KL reward make better explanations?

> **Summary of the whole thread:** [`kl_nla_writeup.md`](kl_nla_writeup.md).

Status 2026-09-28: the **slim pilot** below was adopted, replacing Steps A and B further down (kept for the
record). Follows `docs/kl_nla.md` (Phases 0, 1, and the hedging control).

## Adopted: slim RL pilot to see the shape of the curves (2026-09-28)

The user wants the shape of the curves more than a final verdict, at lower cost. Two standing rules apply: the
AR is **always co-trained, never frozen**, and every run **pushes to HF and tracks on W&B**.

- **No best-of-16 probe and no separate smoke run.** The drift curves below answer the probe's question more
  directly. The KL arm launches first and acts as the smoke test; the MSE arm launches once it steps cleanly.
- **Both arms:** `nla.train_rl_vllm --config configs/rl_vllm.yaml`, merged SFT AV + fresh LoRA (r128/α16
  rsLoRA). Overrides:
  - `--num-steps 100 --batch-prompts 32 --group-size 8`;
  - `--train-critic --ar-lora` (the AR co-trained as LoRA);
  - `--eval-every 10 --eval-n-prompts 128 --save-every 20 --val-rows 2000`;
  - `--vllm-gpu-mem 0.30`, and `--evals base_fve` (no judge key).
  - The KL arm starts from Phase 1 `ar_kl` with `--recon-loss kl`; the MSE arm from Phase 1 `ar_mse` with
    `--recon-loss mse`. Both Phase 1 ARs are merged into full critic dirs first (`scripts/merge_prepared_ar.py`,
    checked against the LoRA-loaded AR).
- **Data:** `rl_full` minus the 287 audit docs (`scripts/kl_rl_prep.py`).
- **Curves**, two kinds:
  1. each arm's own reward/eval curve from the trainer, every 10 steps, in W&B;
  2. **frozen-judge curves** from `scripts/kl_curve_eval.py`. At steps 0 (SFT AV), 20, 40, 60, 80, 100:
     greedy explanations on 500 audit rows, scored by `A_cal` (calibrated MSE AR), `A_kl` and `A_mse`.
     Reported: KL recovered, FVE, extraction rate, and the drift measures (length, next-token naming, 4-gram
     quoting, vagueness, specificity). These go to W&B as `curves_rl_kl` / `curves_rl_mse` with x = RL step,
     and to HF `evals/curves_rl_*`. `A_cal` is built from the MSE arm's starting AR, so a KL-arm lead under it
     is conservative. `A_kl` favours the KL arm. Both are shown.
- **Hardware:** one Secure H200 per arm ($4.59/h), ~1.5–2 h each including the vllm-lens build →
  **~$15–20 total**. Pod stages `rl_kl` / `rl_mse` in `scripts/pod_kl_nla.sh`. Adapters are pushed to HF every
  20 min during training, then everything (including the co-trained AR) at the end. W&B group `kl-nla-rl-pilot`.
- **What this can and can't show:** direction and early shape (does −KL move the frozen judges, and does
  drift start?), not endpoints. 100 steps at 32 prompts per step is short and noisy.

### Result (2026-09-28)

Pods: MSE arm `cchut9g12r3i5s`, KL arm `21pjb9nfafe9va` (both H200 Secure, ~28–30 s/step). HF
`syvb/kl-nla-qwen3-8b`: `ckpts/rl_kl`, `ckpts/rl_mse` (LoRA every 20 steps + co-trained AR),
`evals/curves_rl_kl`, `evals/curves_rl_mse`, `evals/rl_compare` (`scripts/kl_rl_compare.py`). W&B `kl-nla` /
`kl-nla-rl-pilot`: training runs `rl_kl` / `rl_mse`, frozen-judge runs `curves_rl_kl` / `curves_rl_mse`.

Frozen-judge curves, greedy explanations on the same 500 audit rows (240 docs):

| step | KL rec. A_cal (kl / mse arm) | KL rec. A_kl (kl / mse) | FVE A_mse (kl / mse) | names next token (kl / mse) |
|---|---|---|---|---|
| 0 | 0.799 / 0.799 | 0.850 / 0.850 | 0.504 / 0.504 | 0.61 / 0.61 |
| 20 | 0.832 / 0.851 | 0.882 / 0.900 | 0.551 / 0.585 | 0.67 / 0.66 |
| 40 | 0.840 / 0.837 | 0.886 / 0.881 | 0.564 / 0.599 | 0.68 / 0.74 |
| 60 | 0.843 / 0.857 | 0.888 / 0.901 | 0.556 / 0.595 | 0.69 / 0.72 |
| 80 | 0.858 / 0.855 | 0.902 / 0.904 | 0.576 / 0.605 | 0.68 / 0.71 |
| 100 | 0.845 / 0.856 | 0.884 / 0.902 | 0.572 / 0.608 | 0.77 / 0.63 |

- **Shape:** both arms rise fast in the first 20 steps and then plateau, on every judge. Quoting stays at ~0,
  vagueness is flat, specificity rises slightly and alike, length is similar.
- **Both rewards improve the functional metric about equally.** Gain vs step 0, pooled over steps 20–100,
  under A_cal: KL arm +0.046 [+0.027, +0.065], MSE arm +0.054 [+0.034, +0.075]. Under A_kl: +0.040 vs +0.049.
- **−KL gives no KL advantage.** KL arm − MSE arm, paired and pooled over steps 20–100:
  A_cal −0.007 [−0.017, +0.002], and A_kl −0.008 [−0.016, +0.000]. That holds even under the judge that favours the KL arm.
- **It costs FVE:** the KL arm is ~0.035 lower at every checkpoint (step 100: 0.572 vs 0.608).
- **Next-token drift: not established.** Pooled +0.004 [−0.021, +0.029]. Step 100 alone is +0.138
  [+0.090, +0.188], but step 40 goes the other way (−0.066), so it's one checkpoint, not a trend.
- **Reading:** at this scale the −MSE reward already moves explanations in the direction KL rewards, and a
  −KL reward doesn't move them further. Phase 1's AR-side result (a KL-trained AR reads +0.063 more
  output-relevant content from the same explanations) does not become better *explanations* under RL here.
  Limits: one seed, 100 steps, 32 prompts per step, next-token KL only.
- **Cost:** ~$23 of Secure H200/H100 time. About $10 of that was the two useful arms. The rest went on startup
  bugs found and fixed on the way (the upstream vllm-lens patcher refused fresh installs; the vLLM v1
  weight sync needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`), an 80 GB fallback, and one very slow machine.

## Where we are

- **KL as a metric works.** It ranks explanations like FVE (gold > RL > SFT AV > quote ≫ wrong). Per row it
  agrees with MSE only loosely (ρ 0.3–0.6). It shows the AR's error sits on output-relevant directions: gold
  explanations reach FVE 0.64 but leave 0.86 nats of KL, where a random error of the same size leaves 0.12.
- **KL as an AR loss helps, about half through calibration.** One epoch of KL training takes KL recovered on
  the SFT AV's explanations from 0.72 to 0.86. Shrinking the MSE AR toward a good prior gets to 0.80 for free. The
  remaining +0.063 [+0.056, +0.069] is extra content read from the same text. FVE falls from 0.50 to 0.34.
- **Open question:** does optimizing the *verbalizer* against −KL give explanations that carry more of what
  the model uses, or does optimization find an exploit? Two known candidates: next-token guessing (KL is
  measured at one position), and quoting the recent text. The pred-NLA best-of-16 check is the cautionary tale:
  selection under that reward moved toward window-derivable content even though greedy outputs looked fine.

**Why calibration does not bias the RL signal.** GRPO normalizes rewards within each prompt's group of G
rollouts: A = (r − mean_group) / std_group. All G rollouts share the prompt, the activation and the prefix, so any
per-prompt offset or scale cancels. That includes "gain over the AR's own prior", since the prior's KL is
constant within a group. So raw −KL from the KL AR is a fine training signal. Calibration matters only when
*comparing arms* afterwards, and step B handles that by scoring both arms with scorers neither was trained against.

## Step A: best-of-16 probe (1×H100, ~1 h, ≈ $3.5)

Cheap preview of the direction RL would push, using the SFT AV's own samples and no training.

**Rows.** 1,000 of the 2,844 audit rows (seeded, spread over all 287 docs; doc-disjoint from every AV/AR
training set).

**Samples.** 16 explanations per row from the merged SFT AV (`syvb/nanonla-qwen3-8b-L24-av`), temperature 1.0,
max 192 new tokens, Karvonen injection at layer 1. This is the same setup as fvecmp's `av_sample`; the batched
generator is ported from `sv/pred-nla`'s `nla/pred/av.py`.

**Frozen scorers.** No scorer changes during step A.
- `A_kl`: the Phase 1 KL AR. Its −KL is the KL arm's reward.
- `A_mse`: the Phase 1 MSE AR. Its −MSE is the MSE arm's reward.
- `A_cal`: `A_mse` shrunk toward `A_kl`'s long-generic prior at α = 0.6, the calibrated MSE AR from the hedging
  control. This is the **neutral judge**: it was never trained on KL, and its calibration is fixed and known.

**Selection.** For each row: the best of 16 under −KL(`A_kl`) and the best under −MSE(`A_mse`). Plus two
references: one random sample (= fvecmp's `av_sample` level) and the SFT AV's greedy explanation
(from `explanations.json`). All 16 are scored, so best-of-{1, 2, 4, 8, 16} curves come free by subsampling.

**Measures for each pick,** with doc-bootstrap CIs:
- KL recovered under `A_kl` and `A_cal`; FVE under `A_mse`; the full cross-scoring matrix.
- **Drift:**
  - token length;
  - names-the-greedy-next-token rate (the Phase 0 diagnostic);
  - verbatim overlap: the share of the explanation's word 4-grams that appear in the prefix;
  - vagueness: cosine between `A_kl`'s prediction and `A_kl`'s no-info prior;
  - specificity: KL of the pick's `A_kl` prediction spliced into a *different* row's prefix, minus KL on its
    own row (bigger = more row-specific).

**Pre-registered go rule** (RL-KL goes ahead only if all three hold, comparing the best-of-16 KL pick with the
random sample):
- **A1 (neutral gain):** KL recovered under `A_cal` rises, CI > 0. The KL-selected explanations are better for
  an AR that was not trained on KL.
- **A2 (no FVE collapse):** FVE under `A_mse` is no more than 0.02 below the random sample's.
- **A3 (no drift):** next-token-naming rate +≤ 10 pp, 4-gram overlap +≤ 5 pp, specificity not lower (CI),
  length within ±20%.

If A1 holds but A3 fails, the KL reward prefers an exploit. Then don't run RL-KL as is; the options are the
multi-position KL of `docs/kl_nla.md` Phase 1b, or a mixed reward, to be decided then. If A1 fails, KL
selection doesn't generalize past its own AR; stop.

## Step B: RL pilot, −KL vs −MSE (2×H200 in parallel, ≈ 10–14 GPU-h)

### Arms (identical except the reward and the AR it starts from)

| | KL arm | MSE arm |
|---|---|---|
| reward | −KL(p_orig ‖ p_splice), failures get the orthogonal-vector floor | −MSE (paper), failures −2.0 |
| AR start | Phase 1 `ar_kl`, merged | Phase 1 `ar_mse`, merged |
| AR co-training | on KL (`--ar-lora`, r64) | on MSE (`--ar-lora`, r64) |

The shared recipe is `nla.train_rl_vllm --config configs/rl_vllm.yaml` with these overrides:
- `--av-ckpt syvb/nanonla-qwen3-8b-L24-av` (merged SFT AV; fresh LoRA r128/α16 rsLoRA per the config);
- `--batch-prompts 64 --group-size 8` (= one rank's share of the tuned 256 × 8 recipe) `--num-steps 250`;
- `--lr 1e-4 --critic-lr 8e-5 --ar-lora` (LoRA AR co-training frees ~20 GB vs full FT, which pays for the
  second 8B target);
- `--vllm-gpu-mem 0.30 --max-new-tokens 256 --length-penalty 0.01`;
- `--evals base_fve --eval-every 25 --eval-n-prompts 128 --save-every 50 --val-rows 2000 --seed 0`;
- KL arm only: `--recon-loss kl --kl-micro-batch 8`.

`text_judges` is off, since it needs `ANTHROPIC_API_KEY` and we don't have one.

**Data:** `syvb/nanonla-qwen3-8b-L24-data-full:rl_full.parquet` (30k rows) with the 287 audit docs removed.
The trainer reserves the last 2,000 rows (doc-disjoint) for its own running eval. 250 steps × 64 prompts =
16k prompts, about 60% of the rest.

**Memory (H200, 141 GB), estimate to be measured in the smoke run:**
- vLLM at 0.30: ~42 GB, of which ~26 GB is KV cache for 512 rollouts;
- HF actor + LoRA: ~17 GB;
- AR (bf16 backbone + LoRA + Adam): ~12 GB;
- activations: ~30 GB;
- KL target: 16 GB.

That's ~117 GB for the KL arm and ~101 GB for the MSE arm. If the KL arm doesn't fit, the fallback is 2×H100
per arm with the target on the second GPU, which needs a `--kl-device` flag (~10 lines).

**Stop rules** (manual, checked on W&B every eval; either arm):
- extraction rate < 70% at two consecutive evals;
- `av/kl_to_ref` > 5;
- more than 10% of rollouts hit the length cap for two evals;
- the arm's own eval metric (`eval/kl` or `eval/fve_pct`) gets worse for three consecutive evals.

A stopped arm keeps its last saved checkpoint and is still evaluated.

### Final evaluation (1×H100, ~1.5 h, ≈ $5)

For each arm's step-250 and step-100 checkpoints: greedy and one temperature-1 sample per row on all 2,844
audit rows (merged SFT AV + the arm's LoRA; HF generate with the same injection). Then score them with the
frozen scorers `A_cal`, `A_kl` and `A_mse` in the Phase 0/1 audit format, alongside the existing references
(SFT AV greedy/sample, RL p0.0 greedy/sample, gold). Also report the drift measures from step A, plus a blinded
read of 40 random row pairs (KL arm vs MSE arm explanation, order shuffled). I'll write that up qualitatively,
since there's no LLM judge.

**Pre-registered outcome:**
- **Primary:** KL recovered of greedy explanations, KL arm − MSE arm, under the neutral judge `A_cal`,
  doc-bootstrap CI > 0.
- **Safety:** the KL arm's FVE under `A_mse` is not below the SFT AV greedy's (0.499), and its drift measures
  vs the MSE arm are within the A3 thresholds.
- **KL reward wins** if the primary holds and safety holds. It's a **trade-off** if the primary holds and
  safety fails, reported as such. **No evidence** if the primary fails.
- Secondary: the same contrasts under `A_kl`, and under each arm's own co-trained final AR (home-advantaged,
  reported for completeness only).

## Engineering before anything launches

Each piece is CPU-smoked on Qwen3-0.6B stand-ins first, like Phases 0/1.

1. **`scripts/merge_prepared_ar.py`.** The Phase 1 ARs are LoRA on the *prepared* SFT AR, but the existing
   `merge_lora_to_hf.py` rebuilds from a freshly truncated raw base, so it doesn't apply. Load the prepared AR,
   inject and load the LoRA (as `kl_audit.load_ar` does), merge, and save a full critic dir. Check: re-score
   500 audit rows with merged vs LoRA-loaded ARs; KL and MSE must match to bf16 noise. Push the merged ARs to
   HF (`ckpts/ar_kl_merged`, `ckpts/ar_mse_merged`, ~11 GB each). Runs on the step A pod.
2. **`scripts/kl_bon.py`** (step A): generation, scoring by the three scorers, selection, curves, drift
   measures, the go rule.
3. **RL data prep:** filter `rl_full` and write the sidecar.
4. **Pod stages:**
   - `bon` (step A);
   - `vllm_env`: build the vllm-lens venv on the pod with `scripts/install_vllm_lens.sh` (needs `uv`), about
     10–15 min per RL pod;
   - `rl_smoke`: 10 steps of the KL arm at 16 × 4, logging peak memory and s/step;
   - `rl_kl` and `rl_mse`: one per pod, pushing checkpoints and W&B as they go;
   - `rl_gen` + `audit2`: the final evaluation.
5. **Launcher:** `--gpu "NVIDIA H200"` for the RL stages. Watchdog: 8 h per RL pod. Secure Cloud
   only (the launcher no longer falls back to Community Cloud).

## Risks and limits

- **Untested code:** the vLLM loop's KL path has never run. The smoke run is the gate, and a failure costs about $3.
- **Memory:** only estimated so far; the fallback is above.
- **Scale:** one seed, 250 steps, 64 prompts per step (a quarter of the tuned batch). This is a pilot.
  A null result is weak evidence; a clear win or a clear exploit is informative.
- **Shared handicaps:** a fresh LoRA on the merged SFT AV costs ~12 pp FVE at the start (README), for both arms alike.
- **Single-position KL:** it can pull the AV toward next-token content. That's measured, not prevented.
- **Moving rewards:** the co-trained ARs change the reward during training. Final judging uses frozen scorers,
  so it isn't affected.

## Budget and order

| step | GPU | est. cost | gate to the next step |
|---|---|---|---|
| engineering + CPU smoke | local | $0 | smokes pass |
| A: merge ARs + best-of-16 probe | 1×H100 | ~$3.5 (Secure $3.49/h) | A1 ∧ A2 ∧ A3 |
| B0: RL smoke (KL arm, 10 steps) | 1×H200 | ~$3.5 (Secure $4.59/h) | runs; memory and s/step measured |
| B: two RL arms, in parallel | 2×H200 | ~$46–64 (Secure $4.59/h) | stop rules |
| B-eval: generate + cross-score | 1×H100 | ~$5 | — |

Total if every gate passes: about $60–80. Wall-clock: about a day, most of it the two RL arms running side by side.
