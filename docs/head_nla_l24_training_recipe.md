# Head NLA, Qwen3-8B layer 24: training recipe of the final runs

This document lists everything needed to retrain the two final models. Infrastructure (pods, launchers,
uploads, retries) is left out. Memory-only settings are marked as such where they appear.

| Model | Checkpoint | Built from |
|---|---|---|
| **Std RL** | `syvb/head-nla-qwen3-8b-l24-std-rl-ckpts/iter_000400` | SFT pair → 400 GRPO steps |
| **Matryoshka** | `syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts/iter_000640` (exported as `-l24-mtry-av` / `-l24-mtry-ar`) | std RL step 400 → 40 AR-only steps → 200 matryoshka steps |

Lineage: data (§1) → warm-start explanations (§2) → AR and AV SFT (§3) → std RL (§4) → matryoshka (§5).

**Exact resolved configs** are stored next to the checkpoints:
- `iter_000400/run_config.yaml` and `iter_000640/run_config.yaml` in the checkpoint repos;
- the SFT configs, in dataset `syvb/head-nla-qwen3-8b` under `l24/std_rl/results/{av,ar}_sft_run_config.yaml`.

**Code:** branch `sv/head-nla`.
- The std SFT and RL started on the training code of `440bd5c`.
- Later commits to the trainers up to `5c08cd8` add only opt-in flags, so the current code reproduces the std run:
  - `--heldout-head-prefix`;
  - `--stop-file`, which the std run used for its stop and resume at step 260;
  - the matryoshka flags.
- The matryoshka run used `5c08cd8`.

**To replicate the training exactly,** start from the published data in `syvb/head-nla-qwen3-8b` rather than regenerating it.
- The Haiku explanations are sampled, so a regenerated set would differ.
- Files:
  - `l24/sft_data/{av_sft,ar_sft,eval}.parquet`;
  - `l24/rl_data/rl.parquet`;
  - `l24/meta/` (selection, splits, head stats, `W_O`).
- Each parquet has its `.nla_meta.yaml` sidecar. Sections 1–2 say how these files were made.

## 1. Data

**Base model:** `Qwen/Qwen3-8B` in bf16.
- Layer 24 of 36.
- All 32 query heads (8 KV heads, GQA groups of 4).
- `head_dim` 128, `d_model` 4096.

Config: `configs/datagen/heads_qwen3_8b_l24.yaml`, run with:
```bash
python -m nla.datagen.heads.run_pipeline --config configs/datagen/heads_qwen3_8b_l24.yaml --stages corpus,extract,stats,select
```
`output_dir` and `corpus.path` in that config point at `/workspace`; change them to suit.

| Stage | Setting |
|---|---|
| Corpus | `m-a-p/FineFineWeb`: 200 random shard files × 50 docs ≥ 1,000 chars, seed 42. The first 10,000 docs are used. A copy is on HF at `l24/meta/corpus.parquet`. |
| Extract | Per doc, 10 token positions drawn uniformly from positions ≥ 50 that are not special tokens. The RNG is keyed on (seed 42, doc id). Truncated to 2,048 tokens, no BOS, giving 100k records. |
| What a record stores | The input to layer 24's `self_attn.o_proj`: all heads' outputs z, head h at `[h·128:(h+1)·128]`. Hook: `nla/datagen/heads/capture.py`. It also stores attention statistics, which are used for evaluation only and never for training data. |
| Stats | Per head, over all 100k records: the mean output z̄_h, and the RMS (`dev_rms`) and quantiles (0.5, 0.9, 0.99, 0.999) of ‖W_O,h (z_h − z̄_h)‖. |
| Document split | av_sft 0.35, ar_sft 0.35, rl 0.15, eval 0.15, by document, seed 42. |
| Selection | Per head: 650 av_sft, 650 ar_sft and 200 eval examples (48,000 rows in all), at most one per document per head. The sampling weight is 0.25 + 0.75·min(dev/rms, 4), so larger deviations are picked more often. Seed 42. |
| RL prompts | Rerun `select` with `--override select.per_head_rl=1000`. This appends 1,000 rl-split rows per head (32,000 in all); the first 48,000 rows are unchanged. |

**The vector each model is trained on** comes from `export_sft.normalized_writes`, defaults `--cap-quantile 0.999 --max-norm 4`:

v = cap(W_O,h (z_h − z̄_h) / dev_rms_h)

- The norm of v is capped at the smaller of the head's q0.999 / rms and 4.
- The AV injection and the AR loss below both use only v's direction, so neither the scale nor the cap affects the final runs.

## 2. Warm-start explanations (SFT targets)

Every explanation is a function of v alone: never the document, the attention pattern, the residual or the head's name.

**Readouts.** Command: `python scripts/heads_soft_token_readouts.py --output-dir <data> --work-dir <work>`.
Code: `nla/datagen/heads/soft_token.py` and `scripts/heads_soft_token_diag.py`.
- **avg16:**
  1. Add v (β = 1, i.e. ‖v‖ × the norm of the placeholder's residual) at layer 24's attention output, at a placeholder token.
  2. Do this in 2 fixed prompts × 2 carriers:
     - prompts: `assoc` ("Words that come to mind:…") and `continue` ("Here is some text:\n\n{m}");
     - carriers: `" something"` and `" it"`.
  3. Average the next-token log-prob gain over the uninjected prompt.
  4. Keep the top 16 tokens by that gain, among the 128 most likely tokens.
- **lens:** the top 8 tokens of (v ⊙ final-norm weight) · W_U.

**Explanations.** Command: `python scripts/heads_soft_token_explain.py --texts <work>/texts.parquet --out <work>/explain.parquet`.

| Setting | Value |
|---|---|
| Model | `claude-haiku-5-5`, adaptive thinking, effort `medium`, `max_tokens` 4000 |
| Input | avg16 split into distinctive tokens and common filler (a token that appears in ≥ 8% of all avg16 lists), plus lens |
| Output rules | One or two prose sentences about concepts. No quoted or listed tokens, no mention of readouts, no stock openings. |
| Rule breaks | Sent back for a rewrite, up to 4 times. Still failing → `ok=False`, dropped at export. |
| Prompt | The `SYSTEM` prompt in the script: v26 from the prompt lab, `scripts/heads_explain_lab/`. |

Over 95% of the selection was explained.
- The published explanations were made in a first data pass.
- They were then mapped onto a seeded regeneration of the same selection (48,000 of 48,000 sites identical) by (doc, position, layer, head). The mapping is `l24/meta/explain_remap.parquet`.

**Export.** Command:
```bash
python -m nla.datagen.heads.export_sft --output-dir <data> --sft-dir <sft> --descriptions <explain.parquet> --description-column explain
```
`--rl-only` writes `rl.parquet` after the RL selection. Row-order shuffle seed 0.

| File | Rows | Content |
|---|---|---|
| `av_sft.parquet` | 20,690 | Fixed actor prompt + v; response `<explanation>\n{explanation}\n</explanation>` |
| `ar_sft.parquet` | 20,702 | Critic prompt around `Attention head L24H{h}: {explanation}`; target √4096 · v (the scale is irrelevant under the direction-only loss) |
| `eval.parquet` | 6,360 | Held-out pre-RL checks |
| `rl.parquet` | 32,000 | Actor prompt + v + `critic_prefix` = `Attention head L24H{h}: ` |

**Prompts** (verbatim, from the sidecar `.nla_meta.yaml`). The actor prompt names no head; the AR's input does.
- Injection marker `㈎` (token 149705), between `<concept>` tags (neighbour ids 29 and 522).
- Actor:
  ```
  You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the information that one attention head added to one token.

  We will pass the head's contribution, as a vector enclosed in <concept> tags, into your context. You must then describe it in one or two sentences, enclosed within <explanation> tags.

  <concept>㈎</concept>

  Please provide an explanation.
  ```
  The prompt goes through Qwen3's chat template.
- Critic: `Summary of the following text: <text>{critic_prefix}{explanation}</text> <summary>`.

## 3. SFT (standard NLA recipe, `docs/train_new_model.md`)

**AV injection** (`nla/injection.py`, hooked by `nla/utils/hooks.py`) is a norm-matched add at the output of decoder layer 1, at the marker token:

h ← h + ‖h‖ · v/‖v‖

That is `inject_preserve_magnitude: false`, so the AV sees only v's direction.

**AR.** Qwen3-8B truncated to 25 blocks (layer index + 1), with the final norm stripped and a value head on the last token's state. The loss is the MSE between prediction and target after both are normalized, so it is direction-only (`ar_raw_loss: false`).

| | AR SFT | AV SFT |
|---|---|---|
| Command | `python -m nla.train_sft --mode ar …` | `python -m nla.train_sft --mode av …` |
| Data | `ar_sft.parquet` | `av_sft.parquet` (held-out: `eval.parquet`, 2,000 rows, every 100 steps) |
| LoRA | r 128, α 16, rsLoRA, dropout 0, all linear modules | same |
| Batch | 64 | 64 (run as 32 × 2 gradient accumulation; memory only) |
| Steps | 324 (one epoch, the default) | 324 (one epoch) |
| LR | 2e-5 → 2e-6 | 1e-4 → 1e-5 |
| Schedule | linear warm-up 50 steps, then cosine to the minimum | same |
| Optimizer | AdamW 8-bit, β (0.9, 0.95), weight decay 0, grad clip 1.0 | same |
| Other | `--ar-num-layers 25`, value-head LR multiplier 1, `max_len` 1024 | `max_len` 1024 |
| Seed | 0 | 0 |

Published as `syvb/head-nla-qwen3-8b-l24-std-ar` and `-std-av`.

## 4. Std RL: 400 steps

Command: `python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml` plus the flags below. The full resolved config is `iter_000400/run_config.yaml`.

```bash
--base-ckpt Qwen/Qwen3-8B --quant none --av-ckpt <AV SFT> --ar-ckpt <AR SFT> \
--rl-parquet rl.parquet --sidecar rl.parquet --critic-prefix-column critic_prefix \
--rollout-batch 256 --num-steps 400 --save-every 50 --val-rows 2000 --evals base_fve --seed 0
```

| Setting | Value |
|---|---|
| Prompts per step × group size | 64 × 8 = 512 rollouts. A group is 8 samples on the same write (same head, same token). |
| Sampling | temperature 1.0, `max_new_tokens` 256 |
| Reward | −MSE of the normalized vectors = 2·cos(AR prediction, v) − 2. If no closed `<explanation>` appears within 256 tokens, the reward is −2. |
| Length penalty | 0.01 per response token past 192 (= 256 − 64), subtracted from the reward. This is `rl_sgpu.yaml`'s default. |
| Advantage | (r − group mean) / (group std + 1e-6). Rollouts that failed the injection checks are excluded. |
| AV loss | On-policy policy gradient (one update per rollout batch, so it equals GRPO at ratio 1) + β·KL to the reference |
| KL | k3 estimator, log-ratio clamped at 12, β = 0.01. The reference is the SFT AV (adapter disabled). |
| AV optimizer | LoRA r 128 / α 16 / rsLoRA, LR 1e-4 constant, AdamW 8-bit β (0.9, 0.95), weight decay 0, grad clip 1.0 |
| AR co-training | One step per RL step, on that step's explanations with their targets (the same normalized MSE as SFT). It trains the AR's LoRA + value head, LR 8e-5 constant, AdamW 8-bit β (0.9, 0.95). |
| Data | `rl.parquet`, 32,000 rows, shuffled with seed 0. 400 × 64 = 25,600 draws ≈ 0.8 epoch. |
| Validation | `val_rows 2000` holds out 6.2% of the RL documents by document hash, about 2,000 rows, never trained on. `base_fve` is computed on 128 of them every 10 steps. |
| Precision | bf16 base, no quantization; one GPU |
| Memory only | micro-batches: log-prob 8, AR 16; eval generation 64 |

The run stopped at step 260 and resumed from that checkpoint with identical settings (the optimizer state was restored).

## 5. Matryoshka: 240 more steps, from std RL step 400

The flags are the same as §4 plus the ones below. The resolved config is `iter_000640/run_config.yaml`, and the design doc is `docs/head_nla_l24_matryoshka_plan.md`.

```bash
--av-ckpt syvb/head-nla-qwen3-8b-l24-std-av \
--resume-from-lora <std iter_000400> --ar-ckpt <std iter_000400>/critic --start-step 400 \
--num-steps 640 --trunc-min-tokens 1 --trunc-max-tokens 120 --kl-beta 0.03 \
--kl-taper-start 16 --kl-taper-end 120 --kl-taper-floor 0.1 --av-freeze-steps 40 --eval-trunc-lengths 4 16 64
```

| Setting | Value |
|---|---|
| Start | Policy LoRA and co-trained AR from std step 400. The optimizer state is restored from that checkpoint. |
| KL reference | Still the std SFT AV (`-std-av`). |
| Truncation | Each prompt group gets a cap L ~ U{1, …, 120} content tokens, from an RNG keyed on (seed, step). Generation stops after L tokens of explanation (counted after the `<explanation>\n` opening). The AR scores, and co-trains on, that prefix. An explanation that closes before L is scored whole. Nearly all rollouts are cut. |
| KL | β 0.03 per token × a taper: 1 for content tokens < 16, falling linearly to 0.1 at 120. |
| AR-only warm-up | Steps 400–439 do rollouts and AR updates only, with the AV frozen, so the AR learns to read prefixes first. |
| Matryoshka steps | 440–639 (200 steps) update the AV and AR together. |
| Unchanged from §4 | everything else: reward, length penalty, learning rates, batch, group size, temperature, data, validation |
| Extra eval | `eval/fve_pct_t{4,16,64}`: held-out explanations cut to their first 4, 16 or 64 tokens |

## 6. Scores used during training

- **`base_fve`:** 1 − Σ err / Σ err of predicting the mean of the training targets, on the held-out validation documents. The baseline err was 0.9992 on the training set and 0.9920 on the eval set.
- **err:** the squared error between the normalized prediction and the normalized target, so it is direction-only.

The viewer's per-head FVE uses each head's own mean training write as the baseline, so it differs from `base_fve` (`docs/head_nla_results.md`).
