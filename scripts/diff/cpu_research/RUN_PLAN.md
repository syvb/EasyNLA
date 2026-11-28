Run the experiment as a two-arm pilot (Arm 2 against Arm 0) in four stages with a stop check after each, and only after five small code changes. The first two stages need one GPU; rent the 4×H200 node only once they pass. Do not run `docs/diff_nla_gpu_handoff.md` as written: its first command exits with an error on the real 8B pair, and two later steps would measure the wrong thing without any error.

What I need from you: say whether I should make the five changes on a new branch off `diff-stuff`. Nothing in the pipeline has been changed so far, and `diff-stuff` is untouched.

The plan's own figure for this pilot is about $235 and 44 H200 GPU-hours. The first stop check costs about one GPU-hour.

## The runbook as written stops at its first command and mismeasures two later steps

These three are verified by running the repo's own scripts on `Qwen/Qwen3-8B` and `Qwen/Qwen3-8B-Base` at layer 24 on CPU. Numbers are in the [CPU evidence report](https://ampcode.com/artifacts/diff-nla-cpu-evidence-37aee2c3-0986-433d-a620-021500c013d8).

- **Phase 0 exits with code 2.** Gate 0 needs a mean audit SNR ratio of 1.25 at `asst_last`. The measured ratio is 0.68 under the Plain wrapper and 0.57 under ChatML. `phase0_config.json` is written before the exit and nothing downstream reads `gate0_passed`, so later phases run anyway if you ignore the exit code.
- **Benchmark 5 would run in thinking mode.** `phase4_eval_suite.py` line 499 picks the mode with the lower KL. In thinking mode both models put probability above 0.9999 on `<think>`, so that KL is near zero for a reason that has nothing to do with the organism. Every training row is no-think. In thinking mode `asst_last` is the `\n` after `assistant`; in training it is the `\n\n` after `</think>`, and Arm 2 subtracts the no-think role mean from it.
- **Benchmarks 2, 3, 4 and 6 read `asst_last` only** (lines 389, 423, 431). That is the position where the diff carries less audit signal than the raw activation. At user-content tokens the ratio is 1.34 to 1.97 and content controls fall to 0.54 to 0.68.

## Make five code changes before renting GPUs

| # | Change | Where | Why |
| --- | --- | --- | --- |
| 1 | Drop massive-activation rows (norm above five times the median) before the alignment fit, its CV, the FVE check and tau; before user-token pooling in Phase 4; resample when Phase 2 lands on one | `phase0_geometry_gate.py` lines 584–592, `phase4_eval_suite.py` lines 457–461, 515–519, 607–624, `phase2_build_corpus.py` `_sample_positions_for_conv` | 116 of 100,797 web tokens carry 95.6% of the alignment objective. Dropping them moves held-out FVE from 0.851 to 0.926 and tau from 63.8 to 49.5 |
| 2 | Add `--taboo-thinking {off,on,auto}` with default `off`, and report both KLs | `phase0_geometry_gate.py` line 476, `phase4_eval_suite.py` line 499 | The mode check is vacuous, see above |
| 3 | Compute the Gate 0 table at user-content positions and gate on it; keep `asst_last` in the report | `phase0_geometry_gate.py` lines 635–639. The user-content activations are already extracted there and used only for the `tok1` metric | The gate currently tests the position where the diff is weakest |
| 4 | Cache, generate and judge Benchmarks 2–4 and 6 at user-content tokens as well, and refusal at rollout tokens | `phase4_eval_suite.py` `run_cache_benchmarks`, `run_generate`; `phase4_judge_and_report.py` | Same reason. On rollout tokens the diff separates refusals from compliance at 1.11 to 1.42 times the raw activation |
| 5 | Report a logit-lens baseline for Benchmark 5 next to every arm | `phase4_judge_and_report.py` | In no-think mode an untrained logit-lens readout of the user-token diff picks the secret word first of five on 91.5% of prompts. An arm that does not beat that has not shown anything |

All five are small. The tiny-model end-to-end tests in `tests/test_phase0_phase2_e2e.py` and `tests/test_phase4_e2e.py` exercise the scripts they touch, and the suite passes today (`pytest tests`: 37 passed, 1 skipped).

## Run four stages and stop at the first failed check

| Stage | Hardware | Wall time | Stop check |
| --- | --- | --- | --- |
| 1. Geometry and benchmark cache | 1 GPU | about 1 h | Diff beats raw at user-content tokens; Taboo no-think KL recorded |
| 2. Corpus and warm start, two arms | 1 GPU for the corpus, H200s for SFT | about 4 h plus the explanation batch | Arm 2's reconstructor has learned a direction |
| 3. RL, two arms | 4×H200 | 3 to 4 h per arm | Tripwires hold; step time measured in a 5-step smoke first |
| 4. Evaluation and decision | 1 GPU plus API | about 1 h plus the judge batch | Pre-registered comparison |

Phase 0, the benchmark cache and all of Phase 2 run on a single GPU, so a one-GPU node with `$WORK_ROOT` on a persistent volume is enough until SFT. The plan's budget bills those hours at the 4-GPU rate. Wall times for stages 1, 2 and 4 are the plan's and handoff's estimates. The RL time is the README's figure for the default config on 4×H200; the handoff says 1.5 to 2 h and the plan says 3.5 h. None of the three is a measurement of these configs, which is why stage 3 starts with a smoke run.

### Stage 0: preflight

- [ ] One node type for every arm. `--hardware h100` switches the reconstructor from full fine-tuning to LoRA, so arms trained on different hardware are not comparable
- [ ] At least 500 GB of fast disk (handoff section 2.1)
- [ ] `HF_TOKEN` only matters for XSTest. I checked the Hugging Face API: WildChat-1M, UltraChat, PKU-SafeRLHF, FineWeb-Edu, JailbreakBench, MATH-500, both Qwen models and the one Taboo LoRA I checked (`taboo-snow`) are not gated; `walledai/XSTest` is. Without access the refusal pairs come from JailbreakBench, which is what my CPU numbers used
- [ ] `ANTHROPIC_API_KEY` exported in the shell that launches RL. Every RL config lists `text_judges` under `evals`, and `nla/utils/text_judges.py` line 264 aborts without the key
- [ ] `claude-sonnet-4-6` and `claude-opus-4-8` are still listed on Anthropic's models page. `claude-sonnet-5-5` ($2/$10 per MTok against $3/$15) and `claude-opus-5-5` ($4/$20 against $5/$25) are cheaper; switching is optional and changes the teacher and the judge
- [ ] Both venvs built, `utils/patch_vllm_lens.py` re-run, tests pass (handoff sections 2.3 and 2.4)

### Stage 1: geometry and benchmark cache

Run Phase 0 exactly as in handoff section 3, adding `--no-fail-exit` only if change 3 has not landed. Then run Phase 4a-1 `cache-benchmarks` straight away. It reads only `phase0_config.json` and the alignment files, not the corpus, so it does not need to wait for Phase 2.

```bash
CUDA_VISIBLE_DEVICES=0 "$CORE_PYTHON" scripts/diff/phase0_geometry_gate.py \
  --output-dir "$P0_DIR" --layer-index 24 \
  --n-web-train 1600 --n-web-val 400 --web-max-tokens 64 \
  --n-diag-pairs 50 --n-neutral 20 --batch-size 16 --dtype bf16 --device cuda

CUDA_VISIBLE_DEVICES=0 "$CORE_PYTHON" scripts/diff/phase4_eval_suite.py --step cache-benchmarks \
  --phase0-dir "$P0_DIR" --corpus-dir "$CORPUS_DIR" --eval-dir "$EVAL_DIR" \
  --taboo-preset paired --batch-size 16 --dtype bf16 --device cuda
```

Stop if, with massive rows dropped, the mean audit SNR ratio at user-content tokens is below 1.25 or the content-control ratio is above 1. On CPU these were 1.34 to 1.97 and 0.54 to 0.68. Also check in `benchmark_meta.json` that the target model refuses harmful prompts more often than benign ones.

Two things in Phase 0 to know. It caches the target pass in `_phase0_pass1_target.pt`, and the cache key leaves out `--web-max-tokens`, `--chat-max-tokens`, the web dataset and the dtype, so pass `--overwrite` when you change any of those. And tau comes from this run: note the value, since the launcher and Phase 4 read it from `phase0_config.json`.

### Stage 2: corpus and warm start for Arm 2 and Arm 0

Run handoff Phase 2a-1, 2a-2, 2b for `arm2` then `arm0` (skip the `arm1` batch), and 2c. Then run only the warm start. The launcher prints its commands with `--dry-run`; the two SFT jobs per arm each use one GPU, so on a 4-GPU node you can run all four at once with `CUDA_VISIBLE_DEVICES` and then start the launcher with `--resume`, which skips finished SFT stages. I read that path in the launcher and did not run it.

- [ ] After 2a-2, read `phase2_extract_diagnostics.json`. On CPU the Fix 3 weights were nearly uniform (ESS 96 to 100%), so expect Fix 3 to change little
- [ ] After 2b, read 30 explanations per role from `arm2/sft_explained.parquet`. The teacher never sees an activation, only the two models' next-token behaviour. If explanations for a role are interchangeable across rows, the warm start teaches format and nothing else
- [ ] After AR SFT, read the last `[heldout@...] mse` line for each arm. Targets and predictions are unit-normalised, so cosine is `1 - mse/2`

Stop if Arm 2's held-out reconstructor MSE is still at or above 1.8 (cosine 0.1 or less). The threshold is my judgement. The reason for the check is the measurement in the next section: Arm 0's reconstructor starts on its target and Arm 2's does not, and the verbalizer's RL reward is only as good as the reconstructor.

### Stage 3: RL for both arms

- [ ] 5-step smoke first: `--extra-rl-args "--num-steps 5 --max-rows 1024 --no-wandb --save-every 0 --save-steps"`. Confirm `av/steer_apply_rate` is at least 0.99 and read the seconds per step. Multiply by 400 before committing to both arms
- [ ] Arm 2, then Arm 0, with `scripts/diff/run_phase3_arm.sh ... --hardware h200 --seed 0` (handoff section 3). I ran `--dry-run` for all five arms on both hardware settings: every command line builds, tau reaches only Arm 2 and Arm 2b, and resume arguments are wired
- [ ] Watch `eval/cos_mean`, `eval/extraction_rate`, `av/steer_apply_rate` and the two judge means, as in the handoff's tripwire table. Do not compare `eval/fve_pct` across arms
- [ ] On a crash, rerun the same command with `--resume`. At most 25 RL steps are lost

400 steps at 256 prompts is 102,400 prompts from a 155k-row corpus. Each arm sees about two thirds of its rows once.

### Stage 4: evaluation and the decision to continue

- [ ] `generate` for steps 0, 100 and 400 of both arms first; add 50, 200 and 300 only if the trajectory matters
- [ ] A second `generate` pass for `arm2` at step 400 with `--soft-norm-tau 0.0` into `"${EVAL_DIR}_tau0"`. Link `benchmark_cache.pt` into that directory first: `generate` loads the cache from its own `--eval-dir` (line 1066) and the handoff's instruction omits this. Taboo `org_minus_inst` diffs at user tokens have a median norm of 0.49 tau, which the trained injection scales to 0.44 of full strength; training rows sat between 0.71 and 0.91
- [ ] Phase 4b judge and report

Fix the comparison before looking at results. My proposal: the primary result is the paired Arm 2 minus Arm 0 difference in judged AUROC, averaged over Benchmarks 2 to 4, read at user-content tokens at step 400. Secondary results are Taboo Hit@3 on the 50%-mix organisms against the logit-lens baseline, and the drop on the Historical and MATH controls. Run Arm 1, Arm 0c or a second seed only if the primary result favours Arm 2; they explain a win and add nothing to a loss.

## Two new CPU measurements: the reconstructor gap is real, and bf16 extraction is safe

### Arm 2's reconstructor starts far from its target

The reconstructor is the first 25 blocks of the instruct model with an identity value head (`init_critic_from_base` in `nla/train_sft.py`). For Arm 0 that is already the right space. For Arm 2 it is not. On 840 CPU-generated conversations (`ar_init_gap.py`, 6,000 rows per role, split by conversation):

| Role | Cosine of the target with the instruct activation, Arm 0 / Arm 2 | Held-out FVE of the best linear map from the instruct activation, Arm 0 / Arm 2 | Cosine with the role's mean direction, Arm 0 / Arm 2 |
| --- | --- | --- | --- |
| `user_content` | 1.00 / 0.08 | 0.98 / 0.37 | 0.77 / 0.02 |
| `asst_last` (840 rows) | 1.00 / 0.23 | 0.93 / 0.69 | 0.88 / 0.01 |
| `asst_rollout` | 1.00 / 0.24 | 0.98 / 0.37 | 0.70 / 0.02 |

So even a reconstructor that recovered the instruct activation perfectly from the text could reach only 0.37 FVE on Arm 2's target with a linear head, and it starts at cosine 0.08 to 0.24. Arm 0 gets cosine 0.70 to 0.88 for predicting a constant. Both arms get the same 19k-row, 297-step warm start, which is about one twentieth of the 371k rows the README's 70% FVE figure rests on. Reconstruction scores will therefore favour Arm 0 whatever the diff is worth, which is why the stage 2 check and the judged benchmarks carry the comparison.

A linear map from the base activation alone reaches 0.39 to 0.72 FVE on Arm 2's target. A clean residual would be unpredictable from the base activation, so part of what Arm 2 is asked to verbalise is still a function of prompt content.

### bf16 extraction does not disturb the diff

Everything in the CPU report was computed in fp32 and the runbook extracts in bf16, which the report listed as a risk. `bf16_fidelity.py` runs the repo's own `extract_at_positions` in bf16 on 50 chat prompts and 24 web passages and compares with the fp32 reference:

| Rows | Median relative error on the activation | Median relative error on the aligned diff | Median cosine of bf16 diff with fp32 diff |
| --- | --- | --- | --- |
| Chat, user-content (688) | 0.7% | 2.6% | 0.9997 |
| Chat, `asst_last` (50) | 0.9% | 1.9% | 0.9998 |
| Web tokens (1,512) | 0.9% | 5.4% | 0.9986 |

The noise on the diff is about 0.04 to 0.05 tau. The refusal Fisher SNR ratio on those prompts is 0.548 in fp32 and 0.547 in bf16 at `asst_last`, and 0.719 and 0.723 on the user-token mean. Those four numbers are the plain refusal statistic on 20 pairs with the stock alignment, there only to compare dtypes; they are not the gate table. Keep `--dtype bf16`.

## What else the review found

Severity is my judgement. "Run" means I executed it; "read" means I read the code and did not execute that path.

| Finding | Where | Basis | Effect |
| --- | --- | --- | --- |
| One third of training rows are `asst_last`, a single fixed token | `phase2_build_corpus.py` `_sample_positions_for_conv` | read | A third of each arm's budget goes to the position where the diff is weakest. It matches what the benchmarks read today |
| `--hardware h100` with `arm2b` runs 400 steps, not the 200 in the arm's own config | `run_phase3_arm.sh`, `rl_vllm_h100_80gb.yaml` | run (`--dry-run`) | Only matters if you run the cold-start arm on H100 |
| Handoff names the Phase 0 Taboo pilots as `snow` and `snow_50_mix`; the code uses `snow` and `blue_50_mix`. It expects KL under 0.5 nats, which only the vacuous thinking-mode KL meets | `docs/diff_nla_gpu_handoff.md`, `nla/diff/benchmarks.py` line 74 | run | Documentation only. Measured no-think KL is 2.83 and 4.66 |
| The AR SFT training log's FVE for Arm 2 divides a weighted loss by an unweighted baseline | `nla/train_sft.py` lines 964–967, 1007–1009 | read | Read the held-out line instead |
| RL time per arm is given as 1.5–2 h, 3–4 h and 3.5 h in three documents | handoff, README, plan | read | Budget on the smoke run |
| The SFT stages have no mid-run resume | `nla/train_sft.py` | read | A crash costs at most one SFT stage, about half an hour |
| The handoff's `--soft-norm-tau 0.0` diagnostic pass into a separate eval directory fails with a missing `benchmark_cache.pt` | `phase4_eval_suite.py` line 1066 | read | Symlink the cache into the new directory first |
| `gate0_passed` and `taboo_recommended_mode` are written and never read | `phase0_geometry_gate.py` lines 749–750 | run (grep) | A failed gate does not stop later phases |

I found no defect in the parts I read line by line: the alignment solver, the wrapper and position bookkeeping, Phase 2's chunked extraction and split handling, the soft-norm weighting in SFT and RL, the vLLM steering and its per-request verification, the checkpoint commit order, and the resume logic.

Not read in full: `nla/train_rl_vllm.py` outside rollout, reward, advantage, critic update, eval and checkpointing; `utils/patch_vllm_lens.py`; `nla/train_rl_self_contained.py`; the report-writing half of `phase4_judge_and_report.py`. Nothing GPU-side was executed: vLLM is not installed here.

## Alternatives

- **Run the runbook as written with `--no-fail-exit`.** Lost because Arm 2 would train on a target whose largest direction is an alignment artifact, Benchmark 5 would read a position that never occurs in training, and a null result on `asst_last` is what the CPU evidence already predicts.
- **Run all three arms in one go.** Lost because Arm 1 only explains an Arm 2 win. It costs another 14 GPU-hours and can be added after stage 4 without redoing anything.
- **4×H100 node.** Fine if every arm uses it; the plan's figure is about 50% more wall time for roughly the same money. Lost only because it is not the plan's configuration: the reconstructor becomes a LoRA.
- **Initialise Arm 2's value head from the fitted linear map instead of identity.** It would close part of the gap in the table above. Lost as a default because it changes the experiment; it is the first thing to try if the stage 2 check fails.
- **Drop the experiment.** Still defensible if the `asst_last` benchmarks are the result you care about.

## Risks

- No NLA has been trained in this work. Every prediction about which arm wins rests on linear statistics of activations. The earliest signal is the stage 2 reconstructor check, then step-0 evaluation.
- My corpus numbers come from 840 conversations sampled with HF generation, not 60k from vLLM. The earliest signal is `phase2_extract_diagnostics.json`.
- The refusal pairs were JailbreakBench, not XSTest. The earliest signal is the per-task ratios in `phase0_report.json`.
- The stage 2 threshold (MSE 1.8) is a judgement, not a derived number. If Arm 2 sits just under it, treat RL as a gamble and watch `eval/cos_mean` over the first 50 steps.
- The bf16 check used PyTorch's CPU bf16 kernels, not a GPU's. The earliest signal is Phase 0's printed table against the CPU report's.

## Out of scope

Layers other than 24, the 0.6B pair, multi-layer or multi-position inputs, and the population-conditioned designs the plan defers.

## Decision

| Option | What happens next |
| --- | --- |
| A. Make the five changes on a new branch, then you run the staged pilot (my recommendation) | I implement and test them on CPU with the tiny-model suite, push the branch, and update this plan's commands to match |
| B. Run the staged pilot without the changes | Use `--no-fail-exit`, expect no Arm 2 advantage on the `asst_last` benchmarks, and treat Benchmark 5 as uninterpretable |
| C. Drop the experiment | Nothing |

Without an answer I will make no further changes.
