# Plan: L24 head-NLA RL from the SFT warm-starts

**Status:** implemented and CPU-tested. **The real run waits for approval.**

## Goal

The L24 SFT warm-starts pass both pre-RL checks (`docs/head_nla_l24_plan.md`):
- The AR reconstructs 0.063 of the normalized write from gold explanations.
- It reconstructs 0.049 from the AV's own explanations.

A blinded judge found that the AV's faithful explanations reconstruct as well as gold, and that the
generic templates it falls back on (50% of outputs) earn about 0.

**Question:** does RL on the reconstruction reward lift the AV toward gold, push it off the templates,
and keep it faithful?

## What runs: the repo's NLA RL, with only the head-data contract changed

Trainer: `nla.train_rl_self_contained`, with **`configs/rl_sgpu.yaml`**, the repo's single-GPU RL
recipe. Hyperparameters are unchanged from that file except where this table says:

| setting | recipe (`rl_sgpu.yaml` / shared with `rl_vllm.yaml`) | this run | why |
|---|---|---|---|
| AV lr / AR lr | 1e-4 / 8e-5 | same | |
| batch | 64 prompts × group 8 | same | |
| steps | 400 | same, plus a 5.0 h wall-clock stop | budget guard |
| max new tokens | 256 | same | |
| length penalty | 0.01/token past 192 (hinged) | same | |
| KL | k3, β 0.01, frozen SFT adapter as reference | same | |
| AR | co-trained (LoRA + value head), 8-bit Adam | same | |
| AV policy | continues the SFT LoRA adapter | same (our SFT adapter) | |
| precision | 4-bit base | **bf16** | the SFT checkpoints are bf16 (`ar_meta.quant: none`, which the trainer asserts) |
| injection | `h + ‖h‖·v/‖v‖` | **`h + ‖h‖·v`** (`--inject-preserve-magnitude`) | how the AV was SFT-trained |
| reward / AR loss | MSE on mse_scale-normalized vectors | **raw MSE**, targets √d·v (`--ar-raw-loss`) | how the AR was SFT-trained |
| AR input text | `<summary template>(explanation)` | `<template>("Attention head L24Hh: " + explanation)` (`--critic-prefix-column`) | how the AR was SFT-trained; the AV prompt still names no head |
| failed-extraction reward | −2 | **−2 × the row's predict-zero MSE** | the raw counterpart of the normalized floor |
| rollouts | one `generate()` per prompt | batched, 256 rows per call (`--rollout-batch`) | same sampler; speed only |
| micro-batches | logp 2, AR 4 | logp 8, AR 16 | an 80 GB GPU in bf16 |
| evals | base_fve + text_judges | base_fve | text_judges scores against document text this data doesn't carry; the faithfulness judge runs afterwards instead |
| checkpoints | every 100 steps | every 50 | so the best checkpoint by held-out FVE can be checked, not only the last |

**Not used: the faster `train_rl_vllm` path.** It needs merged checkpoints, the patched vllm-lens venv and
several GPUs (README: 4×H200). That is over the RunPod budget ($43), and it can't be tested on CPU. Its
injection would need `scale=‖v‖` on the SteeringVector for magnitude preservation.

Every change is behind a flag with the old default, so the residual-stream recipe is untouched. The
full test suite passes: 35 passed, 1 skipped.

## Data

- **Prompts:** the seeded L24 datagen, regenerated on the pod, with `select.per_head_rl: 1000`. That
  gives 32k RL prompts from the rl split's 1,500 documents, at most one per document per head, weighted
  toward large writes like the other splits.
- **Row order:** RL rows are appended after all warm-start rows. The pod asserts that the first 48,000
  rows equal the saved selection, so the SFT/eval rows, and therefore their explanations, are the same.
- **Export:** `export_sft --rl-only` writes `rl.parquet`. Each row has the AV prompt, v (the SFT
  normalization) and `critic_prefix`.
- **Held-out eval:** `--val-rows 2000` holds out about 6% of the rl documents by doc hash. The live
  held-out FVE is measured on 128 of them.
- **Final comparison:** on the SFT eval split (6,360 rows, documents no model trained on). The RL run
  never sees it.

## Pipeline (`scripts/runpod/heads_l24_rl_remote.sh`, one secure H100)

| stage | est. time |
|---|---|
| install, download the SFT checkpoints and SFT/eval data from the private HF repos | 10 min |
| datagen (seeded) + selection check + `rl.parquet` export | 12 min |
| RL: 400 steps × 64 prompts × 8 (estimated 45–60 s/step; stops at 5.0 h) | ≤ 5 h |
| keep checkpoints: copy the last and the best (by held-out FVE) into results/ right away | 2 min |
| checks: RL AV + RL AR on all 6,360 eval rows, for the last and the best checkpoint | 30–40 min |
| checks: RL AV + SFT AR, and SFT AV + RL AR (3,200 rows each) | 25 min |
| collect | 2 min |

**Time and cost:**
- About 6.5–7 h end to end, at most about 7.3 h (the RL stage is capped).
- At $3.99/hr that is about $27, at most about $29.
- The launcher's limit is `--max-hours 8`, and the watchdog self-deletes at 8.5 h (≤ $34). The RunPod
  balance is $43.
- No `--hold`: the pod is fetched and deleted as soon as DONE or FAILED appears.
- **Checkpoint safety:**
  - Checkpoints live outside `results/`. The last and the best are copied into `results/` right after
    RL, before the checks.
  - On any failure, the newest checkpoint is copied into `results/rl_ckpt_partial/` before FAILED is
    written. So a late failure or timeout never loses the RL training.

**Afterwards, locally:**
- The blinded faithfulness judge (Sonnet 5.5) on the same 823 eval rows as the SFT judgment, for a
  paired comparison. About $8.
- A wandb summary, and HF uploads of the RL adapter + co-trained AR (private).

**Logging:** wandb, live, in project `head-nla-qwen3-8b`, group `l24-rl`. The plots cover reward, FVE,
extraction rate, KL, entropy, length, AR loss, and the held-out FVE every 10 steps with a sample table.

## What would count as success

**Method:**
- Every comparison is **paired on the same eval rows** against the SFT checks run that saved per-row
  errors (`/home/syvb/l24_regen/results/checks`).
- The tool is `scripts/heads_compare_checks.py`: FVE in both runs on the shared rows, and the difference
  with a 95% paired bootstrap CI.

**Criteria:**
1. **Held-out reconstruction rises:** AV-generated explanations through the RL AR (`checks_rl`, and
   `checks_rl_best`) beat the SFT pairing's `av/true` of 0.049, with a CI that excludes 0, on all
   6,360 rows.
2. **The AV improves on its own,** not only the AR adapting to it: the RL AV scored by the **SFT AR**
   (`checks_rlav_sftar`) beats the SFT AV on the same 3,200 rows.
3. **Co-adaptation diagnostic, not a pass/fail:** `ar/gold` under the RL AR (`checks_sftav_rlar`) against
   0.063.
   - The recipe co-trains the AR only on AV outputs, so some drift on gold text is expected.
   - If 2 passes and 3 drops, the run still succeeded.
   - If 1 passes but 2 doesn't, the gain is a private AV↔AR code.
4. **Faithfulness:** the blinded judge on the same 823 rows as the SFT judgment (paired).
   - Faithfulness rises above 2.69/5.
   - The template rate falls below 50%.
   - The unsupported-specifics rate does not rise above 68%.
5. **No token-listing drift (set in advance).** The SFT AV's measured baselines on all 6,360 eval rows
   are a quote rate of 0.0% and copying of 0.039 (gold: 0%, 0.113). Copying is the share of content
   words that are one of the write's own readout tokens.
   - **The run counts as violating the "explanations don't list tokens" constraint, whatever its FVE,
     if** the RL AV's eval quote rate exceeds **5%** or its copying exceeds **0.23** (twice gold's).
   - `av/quote_rate` and `eval/quote_rate` are logged live in wandb, so this is visible during the run.
6. **No degeneration:**
   - Extraction stays near 100%.
   - KL stays bounded.
   - Length doesn't run to the cap.

## Risks and guards

- **Private code between AV and AR:** the AR co-trains on the AV's outputs, so the two could drift into a
  code that reconstructs well but means little. The cross pairings (criteria 2 and 3) and the
  faithfulness judge (criterion 4) catch this.
- **Token copying:** the AR would reward explanations that list the write's tokens. The KL to the SFT
  adapter slows that. The post-run quote/copying measures and the judge's flags show whether it happened.
- **Running slower than estimated:** `--max-hours 5.0` stops the trainer at a step boundary with a saved
  checkpoint. It is checked at the top of every step too, so failing steps can't skip it. The checks
  still run.
- **Memory:** estimated about 58 GB peak of 80 (bf16 AV 16.4 GB and two LoRA adapters about 3 GB; AR
  11 GB; KV cache for 256 rollouts about 14 GB; update activations about 20 GB).
  - If it runs out of memory before the first checkpoint, the stage retries once at half batch.
  - Any other failure fails the stage, with the newest checkpoint kept.
- **Greedy vs sampled:** the checks and the judge use greedy decoding. Templates may persist in the
  greedy mode even if RL reshapes the T=1 distribution. The trainer's live eval samples at T=1, and
  its wandb table can be judged too if the greedy numbers look flat.
- **Grammatical heads (H2, H3 and similar):** the reward signal is weak there, since even gold barely
  helps the AR. Don't expect much change on those heads.

## Review (3 Fable reviewers: code correctness, method and recipe match, ops and cost)

- **No blockers in the code or the method.** Verified by running things:
  - scale consistency (√d applied once; raw MSE identical to `train_sft --ar-raw-loss` and the checks)
  - batched vs unbatched rollouts, including left-padded mixed lengths (logits differ by 2e-7)
  - prefix threading
  - default-path invariance
  - stop/save/resume
  - the raw failure floor: a failure would beat the SFT AV's real explanation on only 0.7% of rows, and
    never for ‖v‖ ≥ 1
  - constraint compliance
- **Ops blockers fixed:**
  - the RL checkpoint was unprotected until the last stage (now copied out right after RL, plus partial
    saves in the ERR trap)
  - `rsync` was likely missing from the image (replaced by `cp`)
- **Adopted:**
  - best-checkpoint checks
  - paired same-row comparisons
  - quote-rate logging with thresholds set in advance
  - an OOM retry
  - criterion 3 reframed as a diagnostic
  - a `dev_norm` check in the selection check
  - the loop-top time check
  - 5.0 h RL cap and `--max-hours 8`
- **Checked and not needed:** "rl.parquet is ordered by head". The export's shuffle runs before the RL
  branch writes, and the rows are mixed.
- **Considered, not adopted (recipe fidelity):**
  - dropping low-norm rows (27% of rows have ‖v‖ < 0.5 and carry mostly noise)
  - a lower AR lr (8e-5 was tuned for the full-FT AR)

  Both are the first follow-ups if the AR's gold reading collapses or the gain is weak.

## CPU tests (done)

- **`tests/test_heads_rl.py`:**
  - RL selection appends without moving warm-start rows, and `rl.parquet` loads in the trainer's
    format with prefixes.
  - Raw-loss scoring with the head prefix (and the default normalized mode unchanged).
  - Batched rollouts give the right rows and the right per-row injected vectors.
  - Batched and unbatched injection give the same logits.
- **End to end on Qwen3-0.6B:**
  - 3 steps of `train_rl_self_contained` with every head flag, then the checks on the RL checkpoints.
  - The exact `--config configs/rl_sgpu.yaml` command line with every override.
  - The `--max-hours` stop and save.
