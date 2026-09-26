# KL-NLA Phase 2 plan: does a KL reward make better explanations?

Status 2026-09-26: plan only; nothing built or launched. Follows `docs/kl_nla.md` (Phases 0, 1, and the hedging
control). Every paid step below needs a go-ahead.

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
