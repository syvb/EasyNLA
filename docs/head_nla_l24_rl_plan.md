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
| steps | 400 | same, plus a 5.5 h wall-clock stop | budget guard |
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
| RL: 400 steps × 64 prompts × 8 (estimated ~45 s/step; stops early at 5.5 h) | ~5 h |
| checks: RL AV + RL AR on all 6,360 eval rows | 20 min |
| checks: RL AV + SFT AR, and SFT AV + RL AR (3,200 rows each) | 25 min |
| collect | 2 min |

**Time and cost:**
- About 6.2 h expected, 7 h at most (the RL stage is capped).
- At $3.99/hr that is about $25, at most about $28.
- The launcher's limit is `--max-hours 7.5`, and the watchdog self-deletes at 8 h (≤ $32).
- No `--hold`: the pod is fetched and deleted as soon as DONE or FAILED appears.

**Afterwards, locally:**
- The blinded faithfulness judge (Sonnet 5.5) on the same 823 eval rows as the SFT judgment, for a
  paired comparison. About $8.
- A wandb summary, and HF uploads of the RL adapter + co-trained AR (private).

**Logging:** wandb, live, in project `head-nla-qwen3-8b`, group `l24-rl`. The plots cover reward, FVE,
extraction rate, KL, entropy, length, AR loss, and the held-out FVE every 10 steps with a sample table.

## What would count as success

1. **Held-out reconstruction rises:** AV-generated explanations through the RL AR reach eval FVE above
   0.049 (SFT), on all 6,360 rows with a bootstrap CI.
2. **The AV improves on its own,** not only the AR adapting to it: the RL AV scored by the **SFT AR**
   beats the SFT AV (0.049).
3. **The AR doesn't lose the ability to read gold text:** `ar/gold` under the RL AR stays near or
   above 0.063.
4. **Faithfulness:**
   - Judged faithfulness of the RL AV rises above 2.69/5.
   - The template rate falls below 50%.
   - The unsupported-specifics rate does not rise above 68%.
5. **No degeneration:**
   - Extraction stays near 100%.
   - KL stays bounded.
   - Length doesn't run to the cap.
   - Outputs don't drift toward token lists or quotes. Checked after the run: the quote rate and the
     copying measure on the eval generations.

## Risks and guards

- **Private code between AV and AR:** the AR co-trains on the AV's outputs, so the two could drift into a
  code that reconstructs well but means little. The cross pairings (criteria 2 and 3) and the
  faithfulness judge (criterion 4) catch this.
- **Token copying:** the AR would reward explanations that list the write's tokens. The KL to the SFT
  adapter slows that. The post-run quote/copying measures and the judge's flags show whether it happened.
- **Running slower than estimated:** `--max-hours 5.5` stops the trainer at a step boundary with a saved
  checkpoint, and the checks still run.
- **Memory:** estimated about 50 GB of 80 (bf16 AV 16 GB, AR 11.5 GB, KV cache for 256 rollouts about
  14 GB, activations). If it OOMs, the stage fails fast in the first step, and the fix is a smaller
  `--rollout-batch` or micro-batch.
- **Grammatical heads (H2, H3 and similar):** the reward signal is weak there, since even gold barely
  helps the AR. Don't expect much change on those heads.

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
