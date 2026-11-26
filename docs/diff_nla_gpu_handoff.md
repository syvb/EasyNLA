# Direct Diff-Target NLA (`Qwen3-8B`, Layer 24) — GPU Execution Handoff Guide

This guide gives a GPU-equipped agent (e.g., on RunPod or Vast.ai) everything needed to execute the full **Minimal Direct Diff-Target NLA** experiment described verbatim in [`docs/diff_nla_experiment_plan.md`](diff_nla_experiment_plan.md), incorporating all engineering, statistical, and methodological hardening from the pre-GPU code review.

All code on branch `diff-stuff` has been implemented and CPU-tested end-to-end (unit tests + synthetic tiny-model E2E tests covering Phases 0, 2, 3 dry-run/config validation, and 4 including mocked vLLM steering and survivorship-bias checks). **No 8B model training or GPU benchmark extraction has been run yet** — that is your job on GPU.

---

## 1. Implementation Map

| Component | Files | Purpose |
| :--- | :--- | :--- |
| **Experiment plan (verbatim)** | `docs/diff_nla_experiment_plan.md` | Original experiment specification |
| **Fix 3 soft-norm math & trainer patches** | `nla/utils/soft_norm.py`, `nla/injection.py`, `nla/utils/hooks.py`, `nla/utils/vllm_steer.py`, `nla/utils/prompts.py`, `nla/datagen/sidecar.py`, `nla/train_sft.py`, `nla/train_rl_vllm.py`, `nla/train_rl_self_contained.py`, `scripts/show_nla_generations.py`, `scripts/init_coldstart_ar.py` | `--soft-norm-tau` (HF + vLLM parity, globally weight-normalized DP critic MSE, weighted-variance `eval/weighted_fve_pct`, post-standardization GRPO advantages), `--eval-parquet`, `--save-steps` (incl. step 0), `--save-critic-at-ckpts` / `--no-save-critic-at-ckpts`, `--fail-reward` (`-4.0` in all diff configs), `--disable-av-thinking`, `eval/cos_mean` & `eval/opt_scale_fve_pct` |
| **Core diff library** | `nla/diff/alignment.py`, `nla/diff/wrappers.py`, `nla/diff/metrics.py`, `nla/diff/model_io.py`, `nla/diff/benchmarks.py`, `nla/diff/benchmarks_data/{eval_awareness,sycophancy}_pairs.json` | Fix 1 float64 `eigh` Identity-Anchored Ridge + 5-fold doc-grouped CV; Fix 2 exact 1-to-1 ChatML vs Plain alignment (with clean left-truncation of long multi-turn history), role masking & centering; Fisher SNR, dual-space probe AUROC, closed-form `tok1` Ridge FVE, KL, bootstrap CIs; bundled 100-pair benchmarks, matched-punctuation fallback refusal pairs, & verified `adamkarvonen/Qwen3-8B-taboo-*` registry |
| **Phase 0 (Gate 0)** | `scripts/diff/phase0_geometry_gate.py` | Web alignment fit + 5-fold CV (`fix1_best_passed`), neutral-KL-gated Base-wrapper A/B check, per-task auditing SNR ratio check, `tok1` / `tau_web_median`, Taboo `<think>` Mode Check |
| **Phase 2 (Corpus & SFT)** | `scripts/diff/phase2_build_corpus.py`, `scripts/diff/phase2_contrastive_sft.py` | Resumable rollouts (exact prompt token IDs persisted for bitwise-identical prefix reuse) -> sequential Target/Base extraction (training-split-only Fix 2 role means + per-arm unit role means + Fix 3 ESS diagnostics) -> Anthropic Batch explanations (`--mode contrastive|standard`, `--cache-from` cross-arm reuse) -> EasyNLA parquets + sidecars for `arm0`, `arm1`, `arm2`, and optional `arm0c` |
| **Phase 3 (Training)** | `configs/diff/rl_vllm_{arm0,arm0c,arm1,arm2,arm2b_coldstart}_h200.yaml`, `configs/diff/rl_vllm_h100_80gb.yaml`, `scripts/diff/run_phase3_arm.sh` | One-command SFT -> merge -> 4-GPU DP vLLM GRPO RL launcher per arm (`--seed` supported for multi-seed runs) |
| **Phase 4 (Eval & Report)** | `scripts/diff/phase4_eval_suite.py`, `scripts/diff/phase4_judge_and_report.py` | Offline 6-benchmark activation cache (`cache-benchmarks`, chat-wrapped B6 + Target behavioral refusal audit), single-pass vLLM/HF generation + cross-arm comparable `WithinRole FVE%` / `WithinRole Cos` + AR `tok1` (`generate`), zero-survivorship-bias Anthropic Batch LLM-judge + paired item-level bootstrap CIs + plots + qualitative examples |
| **Automated CPU test suite** | `tests/test_soft_norm.py`, `tests/test_diff_core.py`, `tests/test_phase0_phase2_e2e.py`, `tests/test_phase4_e2e.py` | Full unit + E2E verification on tiny Llama models with real `Qwen/Qwen3-8B` tokenizer |

---

## 2. GPU Provisioning & Environment Setup

### 2.1 Recommended Hardware & Disk Budget
- **Preferred**: **4×H200 141GB** (exact plan config: full-FT AR SFT + co-trained full-FT AR in RL, `batch_prompts=256`, `group_size=8`, ~1.5–2 hrs per 400-step RL arm).
- **Budget alternative**: **4×H100 80GB** — pass `--hardware h100` to `scripts/diff/run_phase3_arm.sh`. This automatically switches SFT to micro-batch `16 × grad_accum 4` (same effective batch 64) + LoRA AR SFT (merged to bf16 HF before RL), and RL to `configs/diff/rl_vllm_h100_80gb.yaml` (`ar_lora: true`, `gradient_checkpointing: true`, `vllm_gpu_mem: 0.25`, `logp_micro_batch: 4`, `critic_micro_batch: 4`).
- **Disk budget**: Provision **>= 500 GB** fast NVMe (e.g., `/workspace` on RunPod):
  - HF cache (`Qwen/Qwen3-8B` + `Qwen/Qwen3-8B-Base` + 10 Taboo LoRAs): ~35 GB.
  - Phase 2 corpus + `_extract_chunks/`: ~40–60 GB (delete `_extract_chunks/` after Phase 2c finishes to reclaim ~25 GB).
  - Phase 3 checkpoints with `save_critic_at_ckpts: true` at 6 steps (`0, 50, 100, 200, 300, 400`): ~66 GB/arm (~200–330 GB for 3–5 arms). If disk is tight (< 300 GB), pass `--extra-rl-args "--save-steps 0 100 200 400"` or `--extra-rl-args "--no-save-critic-at-ckpts"` (keeping only `critic_latest` at step 400).

### 2.2 Environment Variables
Put these in your shell (and verify before every phase):
```bash
export WORK_ROOT=/workspace/diff_nla           # fast NVMe mount
export HF_HOME=/workspace/hf_cache
export HF_TOKEN="hf_..."                      # needed for Qwen3 / WildChat / gated XSTest
export WANDB_API_KEY="..."                    # or pass --no-wandb for offline runs
export ANTHROPIC_API_KEY="sk-ant-..."         # Phase 2b SFT explanations + Phase 4b judges + RL text_judges
# Optional: dedicated key for Batch API (providers.py falls back to ANTHROPIC_API_KEY if unset)
# export ANTHROPIC_API_KEY_BATCH="sk-ant-..."

# CRITICAL for vLLM IPC weight sync in Phase 3:
unset PYTORCH_CUDA_ALLOC_CONF
```

### 2.3 Install Both Virtualenvs
EasyNLA uses two Python environments (see `docs/vllm-lens-setup.md`):
1. **Core venv (`.venv`)** — Phase 0, Phase 2 extract/SFT/materialize, Phase 3 SFT + merge, Phase 4 `cache-benchmarks` & `phase4_judge_and_report.py`:
```bash
cd /path/to/EasyNLA
git checkout diff-stuff
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .        # includes scikit-learn, scipy, matplotlib in pyproject.toml
```
2. **Patched `vllm-lens` venv** — Phase 2a vLLM rollouts, Phase 3 `train_rl_vllm`, Phase 4a `generate --backend vllm`:
```bash
bash scripts/install_vllm_lens.sh
# Verify where install_vllm_lens.sh created the venv (check script output; e.g. /opt/vllm-lens-venv or .venv-vllm):
export VLLM_PYTHON=/path/to/vllm-lens-venv/bin/python
# Ensure our diff package + scientific deps are visible inside the vllm-lens venv too:
"$VLLM_PYTHON" -m pip install --no-deps -e .
"$VLLM_PYTHON" -m pip install scikit-learn scipy matplotlib pyyaml pyarrow datasets peft anthropic
# Re-verify the mandatory vllm-lens patch (idempotent):
"$VLLM_PYTHON" utils/patch_vllm_lens.py
```

### 2.4 Pre-Flight Automated Test Verification (~45s)
Run the test suite before burning GPU hours:
```bash
.venv/bin/pytest -v tests/test_soft_norm.py tests/test_diff_core.py tests/test_phase0_phase2_e2e.py tests/test_phase4_e2e.py
.venv/bin/python tests/test_core_functions.py
```

---

## 3. Phase-by-Phase Execution Runbook

Set shared paths once:
```bash
export CORE_PYTHON="$(pwd)/.venv/bin/python"
export VLLM_PYTHON="/path/to/vllm-lens-venv/bin/python"
export P0_DIR="${WORK_ROOT}/phase0"
export CORPUS_DIR="${WORK_ROOT}/corpus"
export CKPT_ROOT="${WORK_ROOT}/ckpts"
export EVAL_DIR="${WORK_ROOT}/eval"
mkdir -p "$P0_DIR" "$CORPUS_DIR" "$CKPT_ROOT" "$EVAL_DIR"
```

### Phase 0 — Geometry, Fix 1 Alignment, Base-Wrapper & Taboo Mode Check (Gate 0)
Runs on **1 GPU** (~15–25 min). Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), fits Fix 1 Identity-Anchored Ridge vs Standard Ridge vs Raw Identity (`W=I`) via 5-fold doc-grouped CV, A/B tests `chatml` vs `plain` Base wrappers (enforcing neutral `KL <= 3.0 nats` before choosing by SNR), computes `tau_web_median`, and runs the 20-prompt `<think>` Mode Check on 2 pilot Taboo LoRAs (`snow_narrow`, `snow_50_mix` by default).

> **Gated dataset note**: `Paul/XSTest` is gated on HuggingFace. Accept its license with your `HF_TOKEN` beforehand; if access is unavailable, `load_refusal_pairs` logs a warning and falls back cleanly to `JailbreakBench/JBB-Behaviors` or the matched 25-pair builtin set (use `--refusal-source xstest|jbb|builtin|auto`).

```bash
CUDA_VISIBLE_DEVICES=0 "$CORE_PYTHON" scripts/diff/phase0_geometry_gate.py \
  --base-model Qwen/Qwen3-8B-Base \
  --target-model Qwen/Qwen3-8B \
  --layer-index 24 \
  --output-dir "$P0_DIR" \
  --n-web-train 1600 \
  --n-web-val 400 \
  --web-max-tokens 64 \
  --n-diag-pairs 50 \
  --n-neutral 20 \
  --batch-size 16 \
  --dtype bf16 \
  --device cuda
```

**Inspect `$P0_DIR/phase0_report.json` and `$P0_DIR/phase0_config.json` before proceeding:**
- **Gate 0 checks** (script exits non-zero if any fails; pass `--no-fail-exit` only to inspect diagnostics):
  1. `fix1_best_passed`: `id_anchored` held-out web FVE is strictly positive, beats `identity` (`W=I`), and achieves `>= 99%` of the best ridge FVE (note: on the expanded `[1e-3..1e6]` grid at `N/d ≈ 25`, `std_ridge` may land near OLS and closely tie `id_anchored` — see **Section 4, Issue #2 & #13**). Check that the selected `alpha` did not hit a grid endpoint (`1e-3` or `1e6`).
  2. Winning `base_wrapper` (`chatml` vs `plain`) satisfies neutral `KL <= 3.0 nats` (if any candidate does) and achieves `chat_template_amplification <= 2.5` (ideally `< 1.5`).
  3. Cleaned `Delta_tilde` satisfies per-task auditing Fisher SNR gating (`mean_task_snr_ratio >= 1.25` AND `min_task_snr_ratio >= 0.85` across Refusal / Eval-Awareness / Sycophancy — preventing one task from masking regression on another), while collapsing `tok1` Ridge FVE on `user_content` and Historical Period Fisher SNR!
  4. Record `tau_web_median` from `$P0_DIR/phase0_config.json` (auto-read by `run_phase3_arm.sh` and `phase4_eval_suite.py`).
  5. Check `taboo_mode_check`: confirm matched `enable_thinking` KL is small (`< 0.5 nats` on neutral queries) for both Narrow and 50%-Mix LoRAs.

---

### Phase 1 — 5-Step GPU Smoke Test (Recommended Before Full Training)
Validate `vllm-lens` IPC sync, `av/steer_apply_rate`, and `--soft-norm-tau` on GPU once Phase 2c produces `arm2/rl_shuf.parquet` (or run a 5-step smoke test at the start of Phase 3 with `--extra-rl-args "--num-steps 5 --max-rows 1024 --no-wandb --save-every 0 --save-steps"`). Confirm:
- `av/steer_apply_rate == 1.0` (must be `>= 0.99`)
- `eval/cjk_fraction < 0.02`
- No CUDA OOM / IPC errors when `PYTORCH_CUDA_ALLOC_CONF` is unset.

---

### Phase 2 — Multi-Arm Chat Corpus + Contrastive SFT Warm-Start Data

#### Phase 2a-1: Target Rollouts (1 GPU, vLLM, ~15–25 min)
Samples ~60k deduplicated prompts (30k WildChat + 20k UltraChat + 10k PKU-SafeRLHF, automatically blocklisting all Phase 4 evaluation prompts from HF + builtins), left-truncates long multi-turn histories cleanly (skipping degenerate prompts that cannot fit), generates 1 `enable_thinking=False` rollout per prompt (`max_new_tokens=96, temp=0.7`), persists the exact prompt token IDs & user spans in `rollouts.parquet` for bitwise-identical prefix reuse in extraction, and assigns doc-disjoint `sft` / `val` / `rl` splits (~20k / ~5k / ~155k activation rows across 3 positions per conversation).
```bash
CUDA_VISIBLE_DEVICES=0 "$VLLM_PYTHON" scripts/diff/phase2_build_corpus.py \
  --step rollouts \
  --phase0-dir "$P0_DIR" \
  --output-dir "$CORPUS_DIR" \
  --n-wildchat 30000 \
  --n-ultrachat 20000 \
  --n-safety 10000 \
  --backend vllm \
  --tp-size 1 \
  --vllm-gpu-mem 0.85 \
  --rollout-max-new-tokens 96 \
  --rollout-temperature 0.7 \
  --dtype bf16 \
  --device cuda
```
*(If you have 4 GPUs idle during Phase 2a-1, `--tp-size 2` works too, though `tp=1` on H200/H100 already blasts through 60k × 96 tokens in ~15 min.)*

#### Phase 2a-2: Sequential Target & Base Activation Extraction (1 GPU, ~1.5–2.5 hrs, crash-resumable)
Loads Target first, extracts all chunks (`_extract_chunks/target_*.pt`, asserting bitwise prefix match against `rollouts.parquet` and recording top-5 logits + 20-token greedy continuations on SFT rows only), unloads Target, loads Base (`base_*.pt`), computes **training-split-only** Fix 2 per-role means (`role_means_L24_train.pt`, strictly excluding `val`) AND per-arm training-split role means (`role_means_L24_train_by_arm.pt`, for cross-arm comparable within-role reconstruction metrics), and streams `base_{sft,rl,val}.parquet` for `arm0`, `arm1`, and `arm2` (with identical held-out `val.parquet` rows across all arms!).
```bash
CUDA_VISIBLE_DEVICES=0 "$CORE_PYTHON" scripts/diff/phase2_build_corpus.py \
  --step extract \
  --phase0-dir "$P0_DIR" \
  --output-dir "$CORPUS_DIR" \
  --chunk-convs 5000 \
  --sft-cont-tokens 20 \
  --sft-cont-batch-size 64 \
  --batch-size 16 \
  --dtype bf16 \
  --device cuda
```
*(If interrupted, simply re-run the exact command — completed chunks in `_extract_chunks/` are skipped automatically!)*

**MANDATORY CHECK after Phase 2a-2 — Inspect `$CORPUS_DIR/phase2_extract_diagnostics.json`!**
Review `arm2_fix3_train_diagnostics`:
- Check per-role `median_norm_over_tau` (`asst_last`, `asst_rollout`, `user_content`), `w_max_over_w_min`, and `ess_frac` (effective sample size fraction).
- **Why this matters (see Section 4, Issue #14)**: If `ess_frac > 0.90` on chat-only RL rows (e.g., `median_norm / tau` sits around `1.0–1.7` across roles, as observed on `Qwen3-0.6B`), Fix 3's soft-norm weights are nearly uniform on chat-only data, AND any role with `median_norm / tau ≈ 1.0` sits near the web noise floor! Record these 8B numbers before interpreting Fix 3's effect.

#### Phase 2b: SFT Explanations via Anthropic Batch API (CPU, ~$15–25 total)
Generates ~20k SFT explanations per arm with chunk-level crash resume (`<output>.chunks/chunk_*.parquet`) and cross-arm `(doc_id, position)` reuse via `--cache-from`. **Run `arm2` first**:
- `arm0c` (optional teacher-prompt control: raw `h_target` + Arm 2's exact contrastive explanations) reuses ~100% of `arm2`'s explanations at **~\$0 extra API cost** via `--cache-from`!
- `arm1` reuses ~85% of `arm2`'s active-position contrastive explanations via `--cache-from` and only calls the API for its extra template-token positions!
- `arm0` **MUST** pass `--mode standard` (the script defaults to `--mode contrastive`!).

```bash
# 1. Arm 2 (Fixed Diff-NLA — contrastive explanations)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --input "$CORPUS_DIR/arm2/base_sft.parquet" \
  --output "$CORPUS_DIR/arm2/sft_explained.parquet" \
  --mode contrastive \
  --provider batch \
  --model claude-sonnet-4-6

# 2. Arm 0 (Raw Activation NLA — matched 20k chat STANDARD single-model explanations!)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --input "$CORPUS_DIR/arm0/base_sft.parquet" \
  --output "$CORPUS_DIR/arm0/sft_explained.parquet" \
  --mode standard \
  --provider batch \
  --model claude-sonnet-4-6

# 3. Optional Arm 0c (Teacher-Prompt Control: raw h_target + Arm 2's contrastive SFT, ~$0 extra API!)
#    Strongly recommended to separate target-vector effect from contrastive teacher-prompt effect!
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --input "$CORPUS_DIR/arm0/base_sft.parquet" \
  --output "$CORPUS_DIR/arm0c/sft_explained.parquet" \
  --mode contrastive \
  --cache-from "$CORPUS_DIR/arm2/sft_explained.parquet" \
  --provider batch \
  --model claude-sonnet-4-6

# 4. Arm 1 (Naive Diff — reuses ~85% of Arm 2's explanations via --cache-from; defer if running Lean Pilot first)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --input "$CORPUS_DIR/arm1/base_sft.parquet" \
  --output "$CORPUS_DIR/arm1/sft_explained.parquet" \
  --mode contrastive \
  --cache-from "$CORPUS_DIR/arm2/sft_explained.parquet" \
  --provider batch \
  --model claude-sonnet-4-6
```
> **Note on Arm 0 warm-start**: Generating matched 20k chat SFT explanations for `arm0` (above, ~$8–10 via Batch API) avoids the massive domain- and 18× scale-confounder of `asher577/easynla-warmstart-data`. Adding **Arm 0c** (~$0 extra API) further decouples the contrastive teacher prompt from the diff target vector (see **Section 4, Issue #10 & #15**).

#### Phase 2c: Materialize EasyNLA Parquets + Sidecars (CPU, ~2 min)
Converts all completed arms (`arm0`, `arm1`, `arm2`, plus `arm0c` automatically whenever `$CORPUS_DIR/arm0c/sft_explained.parquet` exists) into exact EasyNLA schemas (`av_sft_train.parquet`, `ar_sft_train.parquet`, `av_sft_val.parquet`, `rl_shuf.parquet`, `val.parquet` + `.nla_meta.yaml` sidecars with verified `critic_suffix_ids` and `Qwen3-8B` cached marker token `㈎`, id `149705`):
```bash
"$CORE_PYTHON" scripts/diff/phase2_build_corpus.py \
  --step materialize \
  --phase0-dir "$P0_DIR" \
  --output-dir "$CORPUS_DIR" \
  --av-ar-split shared \
  --device cpu
```
Once verified, you may delete `$CORPUS_DIR/_extract_chunks/` to free ~25 GB of disk.

---

### Phase 3 — SFT + On-Policy 4-GPU DP vLLM GRPO RL Training

**Recommended Execution Order (Lean Pilot First)**:
1. Train **Arm 2** (Fixed Diff-NLA) + **Arm 0** (Raw Activation NLA) first — and strongly consider **Arm 0c** (Raw `h_target` + Contrastive SFT, zero extra API cost) BEFORE Arm 1, since Arm 0c cleanly isolates the diff target vector from the contrastive teacher prompt (while Arm 1 bundles Fix 2 + Fix 3 + Base wrapper with almost the same ridge alignment as Arm 2 on the expanded grid).
2. Run Phase 4 evaluation on Arm 2 vs Arm 0 (and Arm 0c) (`step 0` and `step 400`).
3. Once primary arms show clear gains (or to diagnose why), run **Arm 1** (Naive Diff ablation) and optionally **Arm 2b** (Cold-Start RL ablation, zero SFT).
4. **Multi-seed recommendation**: If GPU budget allows, run a second seed (`--seed 1` into `--ckpt-root "${CKPT_ROOT}_seed1"`) for **Arm 2** and **Arm 0** to quantify RL training-seed variance alongside item-level paired bootstrap CIs.

Use `scripts/diff/run_phase3_arm.sh` (preview any command first by appending `--dry-run`; pass `--hardware h100` on 4×H100 80GB; pass `--seed 0|1` for multi-seed runs):

```bash
# Arm 2 (Main Treatment: Fixes 1 + 2 + 3; auto-reads tau_web_median from phase0_config.json)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm2 \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200 \
  --seed 0

# Arm 0 (Raw Activation Baseline: standard chat SFT, tau=0.0)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm0 \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200 \
  --seed 0

# Optional Arm 0c (Teacher-Prompt Control: raw h_target + Arm 2's contrastive SFT, tau=0.0)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm0c \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200 \
  --seed 0

# Arm 1 (Naive Diff Ablation: Std Ridge, unmasked templates in train, uncentered, tau=0.0)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm1 \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200 \
  --seed 0

# Optional Arm 2b (Cold-Start Diff-NLA: skips SFT, runs init_coldstart_ar.py + --disable-av-thinking)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm2b \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200 \
  --seed 0
```

#### Mandatory RL Monitoring Tripwires (Check Every 10–50 Steps in W&B / Logs)
| Metric | Healthy Range | Action if Violated |
| :--- | :--- | :--- |
| `av/steer_apply_rate` | `>= 0.99` (ideally `1.00`) | **STOP immediately** — `vllm-lens` patch broken or marker token mismatch |
| `eval/extraction_rate` | `>= 0.95` | All `configs/diff/*.yaml` now set `fail_reward: -4.0` up front (lower bound of $-\text{MSE}_{\text{norm}} \in [-4, 0]$). If still dipping, inspect sample rollouts for truncation or tag drift |
| `eval/cjk_fraction` | `< 0.02` | If spiking on step 0, check `injection_token_id` sidecar match |
| `eval/cos_mean` & `eval/opt_scale_fve_pct` | Rising monotonically (target `cos >= 0.50` / `opt_scale_fve >= 25%`) | **Do NOT judge Arm 2 by raw `eval/fve_pct` alone** (see **Issue #1** — centered diffs have `mse_rawvar ≈ 1.0`, so `cos = 0.50` yields `eval/fve_pct = 0.0%`!). Use Phase 4's `WithinRole FVE%` / `WithinRole Cos` for cross-arm reconstruction comparison |
| `eval/weighted_fve_pct` (Arm 2/2b) | Higher than unweighted `eval/fve_pct` (uses exact weighted variance baseline) | Confirms Fix 3 concentrates signal on large-diff tokens |
| `eval_judge/coherence_mean`, `eval_judge/repetitiveness_mean` | Coherence `>= 7`, Repetitiveness `<= 3` | Watch for RL mode collapse / boilerplate templates |

**Crash / Preemption Resume**:
Re-run `run_phase3_arm.sh` with `--skip-sft --skip-merge --extra-rl-args "--resume-from-lora ${CKPT_ROOT}/arm2/rl_vllm/iter_000200 --start-step 200"` (`train_rl_vllm.py` automatically bypasses the overwrite guard and restores `critic_latest` + `optim_latest.pt` whenever `--resume-from-lora` is set!).

---

### Phase 4 — Offline 6-Benchmark Evaluation, Batch Judging & Final Report

#### Phase 4a-1: Cache All Benchmark Activations ONCE (1 GPU, ~15–25 min)
Extracts Layer-24 Base, Instruct, and all 10 Taboo LoRA activations (`--taboo-preset paired` = 5 words `snow, smile, blue, gold, ship` × `{narrow, 50_mix}`; or `--taboo-preset distinct` for 10 distinct words), automatically runs the 10-prompt `<think>` Mode Check per Taboo LoRA, chat-wraps Benchmark 6 (Historical Period & MATH) at `asst_last` so Fix 2 centering stays in-distribution, and records 48-token greedy Target completions (`--behavior-audit-max-tokens 48`) on B2–B4 to audit actual Target refusal behavior alongside prompt labels.
```bash
CUDA_VISIBLE_DEVICES=0 "$CORE_PYTHON" scripts/diff/phase4_eval_suite.py \
  --step cache-benchmarks \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --eval-dir "$EVAL_DIR" \
  --taboo-preset paired \
  --n-refusal-pairs 100 \
  --n-eval-awareness-pairs 100 \
  --n-sycophancy-pairs 100 \
  --n-taboo-prompts 50 \
  --n-historical 300 \
  --n-math 200 \
  --behavior-audit-max-tokens 48 \
  --batch-size 16 \
  --dtype bf16 \
  --device cuda
```
Inspect `$EVAL_DIR/benchmark_meta.json` to confirm:
1. All 10 Taboo organisms had clean matched KL (`< 0.5 nats`), AND
2. `behavior_audit.refusal` shows the Target model actually refuses unsafe prompts significantly more often than safe prompts (if not, check `behavior_auroc_ci` alongside `auroc_ci` in Phase 4b!).

#### Phase 4a-2: Generate Explanations & Score Benchmark 1 AR Reconstruction Across Checkpoints (1 GPU per checkpoint, ~3–5 min each)
Run across every arm (`arm2`, `arm0`, optional `arm0c`, `arm1`, optional `arm2b`) and saved RL checkpoint (`0, 50, 100, 200, 300, 400`). Using `--backend vllm` runs all ~4,500 prompts per checkpoint in one batched pass (~2–3 min on H200/H100) and verifies per-request `vllm-lens` steering coverage via `llm.apply_model(read_reset_steer_log)`. Note that `phase4_eval_suite.py` requires `$CORPUS_DIR/role_means_L24_train.pt` (refusing silent Phase 0 fallback unless `--allow-phase0-role-means` is explicitly passed) and computes cross-arm comparable `within_role_fve_pct` & `within_role_cos_mean` from `$CORPUS_DIR/role_means_L24_train_by_arm.pt`.
```bash
for ARM in arm2 arm0 arm0c arm1; do
  for STEP in 0 50 100 200 300 400; do
    ITER=$(printf "iter_%06d" "$STEP")
    AV_LORA="${CKPT_ROOT}/${ARM}/rl_vllm/${ITER}"
    if [[ ! -d "$AV_LORA" ]]; then
      continue
    fi
    AR_CKPT="${AV_LORA}/critic"
    AR_FLAG=()
    if [[ -d "$AR_CKPT" ]]; then
      AR_FLAG=(--ar-ckpt "$AR_CKPT")
    fi
    CUDA_VISIBLE_DEVICES=0 "$VLLM_PYTHON" scripts/diff/phase4_eval_suite.py \
      --step generate \
      --phase0-dir "$P0_DIR" \
      --corpus-dir "$CORPUS_DIR" \
      --eval-dir "$EVAL_DIR" \
      --arm "$ARM" \
      --step-num "$STEP" \
      --av-lora "$AV_LORA" \
      "${AR_FLAG[@]}" \
      --backend vllm \
      --tp-size 1 \
      --vllm-gpu-mem 0.80 \
      --n-val-rows 1000 \
      --dtype bf16 \
      --device cuda
  done
done
```
> **Taboo `Org - Inst` Fix 3 Diagnostic Ablation (Recommended)**: Inspect `taboo_diff_norm_diagnostics` in `$EVAL_DIR/arm2_step0400_local_metrics.json`. If `ratio_to_tau` (`median_norm / tau_web_median`) is `< 0.5` for `org_minus_inst`, run one diagnostic pass of `arm2@400` into a separate `--eval-dir "${EVAL_DIR}_tau0"` with `--soft-norm-tau 0.0` to test whether Base-vs-Instruct `tau` over-attenuates narrow Taboo LoRA diffs (see **Section 4, Issue #3**).

#### Phase 4b: Batch LLM-Judging, Paired Bootstrap CIs, Trajectory Plots & Final Report (CPU)
Grades all discovered `*_step*_generations.jsonl` files in one deduplicated Anthropic Batch job per judge model, persisting every judge output into `$EVAL_DIR/report/judge_cache.jsonl` (crash-safe & resumable!). **Zero survivorship bias**: failed extractions or unparseable judge replies are NEVER dropped — they count as misses (`0`) in Taboo Hit@3, MATH, Historical Period, Benchmark 1 `source_match_acc`, and binary accuracy/balanced accuracy, and as neutral `5/10` (`0.5`) in binary AUROC, with per-benchmark and overall `Ext%` reported in every table.
```bash
"$CORE_PYTHON" scripts/diff/phase4_judge_and_report.py \
  --eval-dir "$EVAL_DIR" \
  --provider batch \
  --judge-model claude-sonnet-4-6 \
  --taboo-judge-model claude-opus-4-8 \
  --n-text-judges 200 \
  --n-boot 2000
```
*(Verify the exact latest Anthropic model IDs on your account before submitting the batch — see **Section 4, Issue #12**.)*

**Deliverables produced in `$EVAL_DIR/report/`:**
- `final_summary.json` — full nested JSON with 95% bootstrap CIs, extraction rates, Refusal `behavior_auroc_ci`, and item-level `paired_diffs` (`arm2_minus_arm0`, `arm2_minus_arm1`, `arm2_minus_arm0c`, `arm0c_minus_arm0`)
- `final_report.md` — publication-ready Markdown with 4 tables:
  1. Headline Trajectory Table (`Overall Ext%`, `FVE%`, `WithinRole FVE%`, `WithinRole Cos`, `VerbLen`, plus per-benchmark score + `(Ext%)`)
  2. Benchmark 1 Reconstruction Quality, all 5 EasyNLA `text_judges` dimensions (`Unique Info`, `Coherence`, `Writing Qual`, `Specificity`, `Repetitiveness`), `Source Match`, and AR `tok1` FVE
  3. Full Benchmark 5 Taboo Breakdown (`Regime × Diff Mode × Position` + `Ext%` + Off-Diagonal FP)
  4. Paired Arm Comparisons (item-level paired bootstrap Δ [95% CI] at every matched RL step)
- `plots/trajectory_grid.png` — 3×2 trajectory plot grid across RL steps with shaded 95% CI bands
- `qualitative_examples.md` — 30 side-by-side Arm 0 / Arm 0c / Arm 1 / Arm 2 examples across all benchmarks

---

## 4. Flagged Potential Experiment Issues & Built-In Mitigations

The following 16 mathematical, geometric, statistical, and engineering issues were identified during implementation and pre-GPU review. Every one of them has a built-in diagnostic, fix, or explicit reporting safeguard already wired into the codebase:

1. **Zero-Mean Unit-Sphere FVE Compression & `-2.0` Failed-Extraction Floor Hazard (FIXED UP FRONT)**
   - **Why**: EasyNLA normalizes *both* predicted and gold vectors to radius $\sqrt{d}$, so $\text{MSE}_{\text{norm}} = 2 - 2\cos(\hat{u}, u)$. On raw activations (`arm0`), vectors share a large common mean direction so `mse_rawvar` is small, whereas on Fix-2 centered diffs ($\mathbb{E}[\tilde{\Delta}] = 0$), `mse_rawvar ≈ 1.0`. Consequently:
     - Raw `eval/fve_pct` is NOT comparable across centered (`arm2`) vs uncentered (`arm0`, `arm1`) arms! Even strong directional recovery $\cos = 0.50$ gives $\text{MSE}_{\text{norm}} = 1.0 \implies \text{FVE} = 0.0\%$, while the plan's `FVE >= 0.15` tripwire requires $\cos \ge 0.575$ on centered diffs.
     - Worse, antipodal predictions ($\cos < 0$) yield $r = -\text{MSE}_{\text{norm}} \in (-4.0, -2.0)$, strictly worse than EasyNLA's default `-2.0` failure penalty — creating an immediate reward-hacking incentive for AV to break `<explanation>` tags early in RL!
   - **Mitigation already implemented**:
     - All 6 `configs/diff/*.yaml` configs set `fail_reward: -4.0` up front (exact lower bound of $-\text{MSE}_{\text{norm}}$).
     - `train_rl_vllm.py` and `phase4_eval_suite.py` log `eval/cos_mean`, `eval/opt_scale_fve_pct`, and exact weighted-variance `eval/weighted_fve_pct` (`weighted_fve_baseline`).
     - `phase4_eval_suite.py` computes cross-arm comparable **`within_role_fve_pct`** and **`within_role_cos_mean`** by subtracting each arm's own training-split unit-sphere role mean (`role_means_L24_train_by_arm.pt`) from both gold and predicted vectors on the identical held-out `val.parquet` rows!

2. **Fix 1 Ridge $\alpha$ Scaling (`1/N` vs Sum-of-Squares)**
   - **Why**: Eq. 2.1 in the plan writes $\frac{1}{N}\sum_i \|\cdot\|^2 + \alpha \|\Delta W\|_F^2$ with $\alpha \in \{10^1, \dots, 10^6\}$. Under $1/N$-normalized MSE, eigenvalues of $\frac{1}{N}X^TX$ are $O(1\text{–}10^3)$, so the CV optimum typically lies in $[10^{-2}, 10^1]$ (below $10^1$), whereas under unnormalized sum-of-squares Ridge at $N = 80\text{k}$, $\alpha_{\text{sum}} \in [10^1, 10^6]$ is right in range.
   - **Mitigation already implemented**: `nla/diff/alignment.py` sweeps an expanded 10-point logarithmic grid `(1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)` via a single `float64` `eigh` decomposition and records both `alpha` and `alpha_sum_loss = alpha * N` in `phase0_report.json`. Check that the winner isn't pegged at `1e-3` or `1e6`.

3. **Fix 3 `tau_web_median` Possibly Over-Attenuating Narrow Taboo `Org - Inst` Diffs**
   - **Why**: `tau_web_median` is calibrated to Base-vs-Instruct post-training drift on FineWeb, whereas a single-secret-word Taboo LoRA's `||h_org - h_inst||` may be much smaller than `tau_web_median`, shrinking the injected vector via $\frac{\|v\|}{\sqrt{\|v\|^2 + \tau^2}}$.
   - **Mitigation already implemented**: `phase4_eval_suite.py` logs `median_norm` and `ratio_to_tau` for every `(organism, position_kind, diff_mode)` slice in `*_local_metrics.json`, and supports `--soft-norm-tau 0.0` for an instant diagnostic ablation on `Org - Inst`.

4. **Taboo `Org - Inst` Must NOT Subtract Base-vs-Instruct Role Means**
   - **Why**: `h_org - h_inst` already cancels the Instruct-vs-Base constant offset; subtracting `role_means_L24` would inject $-\mu_{\text{role}}$ as a spurious bias, while subtracting a per-organism mean would erase the constant secret-word signal itself.
   - **Mitigation already implemented**: `phase4_eval_suite.py` computes `org_minus_inst = h_org - h_inst` directly (no role-mean subtraction) and applies `role_means_L24` ONLY to `org_minus_base`.

5. **Cold-Start Arm 2b Requires Two Extra Fixes**
   - **Why**: (a) `NLACriticModel.from_pretrained` requires `value_head.safetensors`, so passing `Qwen/Qwen3-8B` directly as `--ar-ckpt` crashes. (b) Without AV SFT, `Qwen/Qwen3-8B`'s default chat template emits `<think>...</think>` blocks that consume `max_new_tokens=256` before reaching `<explanation>`.
   - **Mitigation already implemented**: `scripts/init_coldstart_ar.py` builds an identity-initialized 25-layer `NLACriticModel` HF checkpoint, and `run_phase3_arm.sh` / `rl_vllm_arm2b_coldstart_h200.yaml` pass `--disable-av-thinking` (`enable_thinking=False` on AV prompts).

6. **EasyNLA `--val-rows 50000` Default vs 155k `rl_shuf.parquet`**
   - **Why**: Stock `train_rl_vllm.py` carves out `--val-rows 50000` from `--rl-parquet` if no separate eval file is provided, silently cutting the 155k RL corpus to 105k!
   - **Mitigation already implemented**: Added `--eval-parquet` to `train_rl_vllm.py`. `run_phase3_arm.sh` passes `--rl-parquet rl_shuf.parquet --eval-parquet val.parquet` so all arms train on 100% of `rl_shuf.parquet` and evaluate inline on the identical doc-disjoint `val.parquet`.

7. **Checkpoint Critic Snapshots & Disk Usage**
   - **Why**: Stock `train_rl_vllm.py` only saved `critic_latest` (overwriting earlier critics) and skipped step 0, which prevents computing Benchmark 1 AR reconstruction trajectories across steps `0, 50, 100, 200, 300, 400`.
   - **Mitigation already implemented**: Added `--save-steps 0 50 100 200 300 400` and `--save-critic-at-ckpts` / `--no-save-critic-at-ckpts` (`BooleanOptionalAction`). Monitor disk (~66 GB/arm for full-FT critics); pass `--extra-rl-args "--no-save-critic-at-ckpts"` if disk is constrained.

8. **FineWeb-Edu Full-Dataset Download Trap**
   - **Why**: Calling `datasets.load_dataset("HuggingFaceFW/fineweb-edu")` without `name="sample-10BT", streaming=True` attempts to download multi-TB shards.
   - **Mitigation already implemented**: `nla/diff/benchmarks.py::load_fineweb_edu_docs` hard-defaults to `name="sample-10BT", streaming=True`.

9. **Taboo `adamkarvonen/Qwen3-8B-taboo-*` Naming & `adversarial_50_mix` Ambiguity**
   - **Why**: All 42 repos on HuggingFace (`21 words × {narrow, 50_mix}`) were verified via the HF API. One word (`adversarial`) has ambiguous secret-word vs regime semantics (`adversarial_50_mix`), and pairing `snow_narrow` + `snow_50_mix` means off-diagonal FP must exclude same-secret-word organisms.
   - **Mitigation already implemented**: Default `--taboo-preset paired` uses 5 unambiguous concrete nouns (`snow, smile, blue, gold, ship` × `{narrow, 50_mix}`); `--taboo-preset distinct` uses 10 distinct words; `phase4_judge_and_report.py` excludes any same-secret-word organism when computing off-diagonal FP.

10. **Arm 0 Warm-Start Domain Confounder & Shared vs Disjoint 20k SFT Rows**
    - **Why**: Using `asher577/easynla-warmstart-data` (371k FineFineWeb prose rows) for Arm 0 while Arm 1/2 use 20k chat SFT rows confounds target representation with warm-start domain and 18× SFT data volume. Meanwhile, splitting 20k SFT explanations into 10k AV + 10k AR halves SFT data per module.
    - **Mitigation already implemented**: Phase 2b supports matched 20k chat SFT for `arm0` (`--mode standard`), and `phase2_build_corpus.py` defaults to `--av-ar-split shared` (~19k train + 1k doc-disjoint val for both AV and AR, 100% doc-disjoint from `rl_shuf` and `val`), with `--av-ar-split disjoint` available if strict AV/AR SFT separation is desired.

11. **`tok1` Leakage Metric Role**
    - **Why**: At `asst_last`, every chat prompt ends on the identical `<|im_start|>assistant\n<think>\n\n</think>\n\n` token, making `tok1` Ridge FVE trivially 0 at `asst_last`.
    - **Mitigation already implemented**: Both Phase 0 and Phase 4 compute `tok1` Ridge FVE specifically on `user_content` tokens (where token identity varies richly and raw `h_target` leaks heavily).

12. **Anthropic Judge Model IDs**
    - **Why**: Default CLI flags use `--judge-model claude-sonnet-4-6` and `--taboo-judge-model claude-opus-4-8` (matching `nla/datagen/providers.py` and `nla/utils/text_judges.py`).
    - **Mitigation**: Verify active Anthropic API model IDs on your key before launching large Phase 2b / Phase 4b batches; override with `--model` / `--judge-model` / `--taboo-judge-model` as needed.

13. **Arm 1 vs Arm 2 Alignment Convergence on Expanded Ridge Grid (Interpretation Caveat)**
    - **Why**: On the expanded `[1e-3..1e6]` grid at `N/d ≈ 25`, cross-validated `std_ridge` selects a small `alpha` near OLS and closely matches `id_anchored` FVE (e.g., `0.8673` vs `0.8683` on `Qwen3-0.6B`; the pilot's "standard ridge worse than identity" effect only appears when `alpha >= 10` under the `1/N` objective). We intentionally do NOT artificially handicap Arm 1's alpha grid.
    - **Mitigation**: Gate 0 checks `fix1_best_passed` (not `FVE_identity > FVE_std_ridge`), and Arm 1 vs Arm 2 should be interpreted transparently as testing **Fix 2 (template masking + role centering) + Fix 3 + Base wrapper** rather than Fix 1 alone.

14. **Fix 3 Weight Dynamic Range on Chat-Only RL Corpora (Check `phase2_extract_diagnostics.json`)**
    - **Why**: On `Qwen3-0.6B`, Arm 2's chat-only training norms sat at `1.0–1.7 × tau_web_median` across roles (`ESS = 96%`, `w_max / w_min = 2.9`), making Fix 3 weights nearly uniform because the RL corpus contains no web rows where noise diffs collapse.
    - **Mitigation**: `phase2_build_corpus.py --step extract` writes `phase2_extract_diagnostics.json` with exact per-role `norm/tau` ratios, `w` quantiles, and `ess_frac`. Inspect the 8B table immediately after Phase 2a-2 before drawing conclusions about Fix 3!

15. **Teacher-Prompt Confound Between Arm 0 and Arm 2 (`arm0c` Control)**
    - **Why**: Arm 0 uses `--mode standard` (semantic/behavioral/next-token prompt), while Arms 1 and 2 use `--mode contrastive` (explicitly naming safety, persona, cognitive, and behavioral shifts and showing both Base and Instruct continuations). Without a crossed control, an Arm 2 advantage (especially at step 0) could partially reflect the teacher prompt rather than the diff target vector.
    - **Mitigation**: Optional **Arm 0c** (`raw h_target` + Arm 2's exact contrastive explanations via `--cache-from`, **~\$0 extra API cost**) is wired end-to-end across Phase 2b, Phase 2c, `configs/diff/rl_vllm_arm0c_h200.yaml`, `run_phase3_arm.sh --arm arm0c`, and Phase 4 (`arm2_minus_arm0c` and `arm0c_minus_arm0` paired bootstrap diffs).

16. **Benchmark 6 Chat-Wrapping & Refusal Target Behavior Audit**
    - **Why**: (a) Feeding raw un-wrapped prose to an Instruct chat-trained AV caused Benchmark 6 (Historical Period / MATH) to be out-of-distribution for Arm 2's role-centered chat representation. (b) Prompt-based refusal pairs only test intent-to-refuse if Target actually refuses the unsafe prompts and answers the safe ones.
    - **Mitigation**: `phase4_eval_suite.py` now chat-wraps Historical Period & MATH at `asst_last` with Fix 2 `asst_last` role centering for Arm 2, and records 48-token greedy Target completions (`--behavior-audit-max-tokens 48`) so `phase4_judge_and_report.py` reports Refusal `behavior_auroc_ci` alongside prompt-label `auroc_ci`.

---

## 5. Troubleshooting Cheat Sheet

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `train_rl_vllm` crashes on IPC weight sync | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in RunPod base image | `unset PYTORCH_CUDA_ALLOC_CONF` (`run_phase3_arm.sh` already does this automatically) |
| `[fatal] installed vllm-lens is missing required patch hunks` | `vllm-lens` venv rebuilt without running patcher | `"$VLLM_PYTHON" utils/patch_vllm_lens.py` |
| OOM during AR SFT or RL on 80GB H100 | Full-FT fp32 AR needs ~88 GB | Pass `--hardware h100` to `run_phase3_arm.sh` (switches AR SFT to LoRA 16×4 and RL to `rl_vllm_h100_80gb.yaml` with `ar_lora: true`) |
| `av/steer_apply_rate < 0.99` | Stale `vllm-lens` or sidecar/tokenizer mismatch | Check sidecar `injection_token_id` (`149705` / `㈎` for `Qwen3-8B`) and re-run `utils/patch_vllm_lens.py` |
| `FileNotFoundError: Missing training-split role means` in Phase 4 | Phase 2 extraction not run or wrong `--corpus-dir` | Point `--corpus-dir` to Phase 2 output containing `role_means_L24_train.pt` (do NOT silently fall back to Phase 0 diagnostic means) |
| `phase2_contrastive_sft.py` interrupted mid-batch | Network disconnect / pod preemption | Re-run the exact same command — completed chunks in `<output>.chunks/chunk_*.parquet` are skipped automatically |
| `phase4_judge_and_report.py` interrupted mid-batch | Network disconnect / pod preemption | Re-run the exact same command — append-only `$EVAL_DIR/report/judge_cache.jsonl` skips all already-judged prompts |
