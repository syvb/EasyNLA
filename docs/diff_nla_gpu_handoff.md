# Direct Diff-Target NLA (`Qwen3-8B`, Layer 24) — GPU Execution Handoff Guide

This guide gives a GPU-equipped agent (e.g., on RunPod or Vast.ai) everything needed to execute the full **Minimal Direct Diff-Target NLA** experiment described verbatim in [`docs/diff_nla_experiment_plan.md`](diff_nla_experiment_plan.md).

All code on branch `diff-stuff` has been implemented and CPU-tested end-to-end (unit tests + synthetic tiny-model E2E tests covering Phases 0, 2, 3 dry-run/config validation, and 4). **No 8B model training or GPU benchmark extraction has been run yet** — that is your job on GPU.

---

## 1. Implementation Map

| Component | Files | Purpose |
| :--- | :--- | :--- |
| **Experiment plan (verbatim)** | `docs/diff_nla_experiment_plan.md` | Original experiment specification |
| **Fix 3 soft-norm math & trainer patches** | `nla/utils/soft_norm.py`, `nla/injection.py`, `nla/utils/hooks.py`, `nla/utils/vllm_steer.py`, `nla/utils/prompts.py`, `nla/datagen/sidecar.py`, `nla/train_sft.py`, `nla/train_rl_vllm.py`, `nla/train_rl_self_contained.py`, `scripts/show_nla_generations.py`, `scripts/init_coldstart_ar.py` | `--soft-norm-tau` (HF + vLLM parity, weighted critic MSE, post-standardization GRPO advantages), `--eval-parquet`, `--save-steps` (incl. step 0), `--save-critic-at-ckpts`, `--fail-reward`, `--disable-av-thinking`, `eval/cos_mean` & `eval/opt_scale_fve_pct` |
| **Core diff library** | `nla/diff/alignment.py`, `nla/diff/wrappers.py`, `nla/diff/metrics.py`, `nla/diff/model_io.py`, `nla/diff/benchmarks.py`, `nla/diff/data/{eval_awareness,sycophancy}_pairs.json` | Fix 1 float64 `eigh` Identity-Anchored Ridge + 5-fold doc-grouped CV; Fix 2 exact 1-to-1 ChatML vs Plain alignment, role masking & centering; Fisher SNR, dual-space probe AUROC, closed-form `tok1` Ridge FVE, KL, bootstrap CIs; bundled 100-pair benchmarks & verified `adamkarvonen/Qwen3-8B-taboo-*` registry |
| **Phase 0 (Gate 0)** | `scripts/diff/phase0_geometry_gate.py` | Web alignment fit + 5-fold CV, Base-wrapper A/B gate, Fisher SNR / AUROC / `tok1` / `tau_web_median`, Taboo `<think>` Mode Check |
| **Phase 2 (Corpus & SFT)** | `scripts/diff/phase2_build_corpus.py`, `scripts/diff/phase2_contrastive_sft.py` | Resumable rollouts -> sequential Target/Base extraction (training-split-only Fix 2 role means) -> Anthropic Batch contrastive explanations (cross-arm cache) -> EasyNLA parquets + sidecars for `arm0`, `arm1`, `arm2` |
| **Phase 3 (Training)** | `configs/diff/rl_vllm_{arm0,arm1,arm2,arm2b_coldstart}_h200.yaml`, `configs/diff/rl_vllm_h100_80gb.yaml`, `scripts/diff/run_phase3_arm.sh` | One-command SFT -> merge -> 4-GPU DP vLLM GRPO RL launcher per arm |
| **Phase 4 (Eval & Report)** | `scripts/diff/phase4_eval_suite.py`, `scripts/diff/phase4_judge_and_report.py` | Offline 6-benchmark activation cache (`cache-benchmarks`), single-pass vLLM/HF generation + AR FVE/cosine/`tok1` (`generate`), Anthropic Batch LLM-judge + bootstrap CIs + plots + qualitative examples |
| **Automated CPU test suite** | `tests/test_soft_norm.py`, `tests/test_diff_core.py`, `tests/test_phase0_phase2_e2e.py`, `tests/test_phase4_e2e.py` | Full unit + E2E verification on tiny Llama models with real `Qwen/Qwen3-8B` tokenizer |

---

## 2. GPU Provisioning & Environment Setup

### 2.1 Recommended Hardware & Disk Budget
- **Preferred**: **4×H200 141GB** (exact plan config: full-FT AR SFT + co-trained full-FT AR in RL, `batch_prompts=256`, `group_size=8`, ~1.5–2 hrs per 400-step RL arm).
- **Budget alternative**: **4×H100 80GB** — pass `--hardware h100` to `scripts/diff/run_phase3_arm.sh`. This automatically switches SFT to micro-batch `16 × grad_accum 4` (same effective batch 64) + LoRA AR SFT (merged to bf16 HF before RL), and RL to `configs/diff/rl_vllm_h100_80gb.yaml` (`ar_lora: true`, `gradient_checkpointing: true`, `vllm_gpu_mem: 0.25`, `logp_micro_batch: 4`, `critic_micro_batch: 4`).
- **Disk budget**: Provision **>= 500 GB** fast NVMe (e.g., `/workspace` on RunPod):
  - HF cache (`Qwen/Qwen3-8B` + `Qwen/Qwen3-8B-Base` + 10 Taboo LoRAs): ~35 GB.
  - Phase 2 corpus + `_extract_chunks/`: ~40–60 GB (delete `_extract_chunks/` after Phase 2c finishes to reclaim ~25 GB).
  - Phase 3 checkpoints with `save_critic_at_ckpts: true` at 6 steps (`0, 50, 100, 200, 300, 400`): ~66 GB/arm (~200–265 GB for 3–4 arms). If disk is tight (< 300 GB), pass `--extra-rl-args "--save-steps 0 100 200 400"` or `--extra-rl-args "--no-save-critic-at-ckpts"` (keeping only `critic_latest` at step 400).

### 2.2 Environment Variables
Put these in your shell (and verify before every phase):
```bash
export WORK_ROOT=/workspace/diff_nla           # fast NVMe mount
export HF_HOME=/workspace/hf_cache
export HF_TOKEN="hf_..."                      # needed for Qwen3 / WildChat / gated datasets
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
pip install -e .
pip install scikit-learn scipy matplotlib
```
2. **Patched `vllm-lens` venv** — Phase 2a vLLM rollouts, Phase 3 `train_rl_vllm`, Phase 4a `generate --backend vllm`:
```bash
bash scripts/install_vllm_lens.sh
# Verify where install_vllm_lens.sh created the venv (check script output; e.g. /opt/vllm-lens-venv or .venv-vllm):
export VLLM_PYTHON=/path/to/vllm-lens-venv/bin/python
# Ensure our diff package + scientific deps are visible inside the vllm-lens venv too:
"$VLLM_PYTHON" -m pip install --no-deps -e .
"$VLLM_PYTHON" -m pip install scikit-learn scipy pyyaml pyarrow datasets peft anthropic
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
Runs on **1 GPU** (~15–25 min). Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), fits Fix 1 Identity-Anchored Ridge vs Standard Ridge vs Raw Identity (`W=I`) via 5-fold doc-grouped CV, A/B tests `chatml` vs `plain` Base wrappers, computes `tau_web_median`, and runs the 20-prompt `<think>` Mode Check on 2 pilot Taboo LoRAs (`snow_narrow`, `snow_50_mix` by default).

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
  1. `id_anchored` held-out web FVE is strictly positive and beats or matches `identity` (`W=I`) without overfitting (`std_ridge` shrinkage vs `id_anchored`). Check whether the selected `alpha` hit an endpoint of the grid (`1e-3` or `1e6`) — see **Section 4, Issue #2**.
  2. Winning `base_wrapper` (`chatml` vs `plain`) achieves `chat_template_amplification <= 2.5` (and ideally `< 1.5`).
  3. Cleaned `Delta_tilde` improves Fisher SNR on Refusal / Eval-Awareness / Sycophancy relative to raw `h_target`, while collapsing `tok1` Ridge FVE on `user_content` and Historical Period Fisher SNR!
  4. Record `tau_web_median` from `$P0_DIR/phase0_config.json` (auto-read by `run_phase3_arm.sh` and `phase4_eval_suite.py`).
  5. Check `taboo_mode_check`: confirm matched `enable_thinking` KL is small (`< 0.5 nats` on neutral queries) for both Narrow and 50%-Mix LoRAs.

---

### Phase 1 — 5-Step GPU Smoke Test (Recommended Before Full Corpus)
Validate `vllm-lens` IPC sync, `av/steer_apply_rate`, and `--soft-norm-tau` on GPU once Phase 2c produces `arm2/rl_shuf.parquet` (or run a 5-step smoke test at the start of Phase 3 with `--extra-rl-args "--num-steps 5 --max-rows 1024 --no-wandb --save-every 0 --save-steps"`). Confirm:
- `av/steer_apply_rate == 1.0` (must be `>= 0.99`)
- `eval/cjk_fraction < 0.02`
- No CUDA OOM / IPC errors when `PYTORCH_CUDA_ALLOC_CONF` is unset.

---

### Phase 2 — Multi-Arm Chat Corpus + Contrastive SFT Warm-Start Data

#### Phase 2a-1: Target Rollouts (1 GPU, vLLM, ~15–25 min)
Samples ~60k deduplicated prompts (30k WildChat + 20k UltraChat + 10k PKU-SafeRLHF, automatically blocklisting all Phase 4 evaluation prompts), generates 1 `enable_thinking=False` rollout per prompt (`max_new_tokens=96, temp=0.7`), and assigns doc-disjoint `sft` / `val` / `rl` splits (~20k / ~5k / ~155k activation rows across 3 positions per conversation).
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
Loads Target first, extracts all chunks (`_extract_chunks/target_*.pt`, with top-5 logits & 20-token greedy continuations on SFT rows only), unloads Target, loads Base (`base_*.pt`), computes **training-split-only** Fix 2 per-role means (`role_means_L24_train.pt`, strictly excluding `val`), and streams `base_{sft,rl,val}.parquet` for `arm0`, `arm1`, and `arm2`.
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

#### Phase 2b: Contrastive SFT Explanations via Anthropic Batch API (CPU, ~$15–25 total)
Generates ~20k SFT explanations per arm with a shared `(doc_id, position)` cache (`sft_explanation_cache.jsonl`). **Run `arm2` first**: `arm1` reuses ~85% of `arm2`'s cached active-position contrastive explanations automatically and only calls the API for its extra template-token positions!
```bash
# 1. Arm 2 (Fixed Diff-NLA — contrastive explanations)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --corpus-dir "$CORPUS_DIR" \
  --arm arm2 \
  --provider batch \
  --model claude-sonnet-4-6

# 2. Arm 0 (Raw Activation NLA — matched 20k chat single-model explanations for apples-to-apples control!)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --corpus-dir "$CORPUS_DIR" \
  --arm arm0 \
  --provider batch \
  --model claude-sonnet-4-6

# 3. Arm 1 (Naive Diff — run now or defer until after Lean Pilot Arm 2 vs Arm 0 passes!)
"$CORE_PYTHON" scripts/diff/phase2_contrastive_sft.py \
  --corpus-dir "$CORPUS_DIR" \
  --arm arm1 \
  --provider batch \
  --model claude-sonnet-4-6
```
> **Note on Arm 0 warm-start**: We strongly recommend generating matched 20k chat SFT explanations for `arm0` (above, ~$8–10 via Batch API) rather than using `asher577/easynla-warmstart-data` (371k FineFineWeb prose rows), which introduces a massive domain- and scale-confounder between Arm 0 and Arm 2. See **Section 4, Issue #10**.

#### Phase 2c: Materialize EasyNLA Parquets + Sidecars (CPU, ~2 min)
Converts all completed arms into exact EasyNLA schemas (`av_sft_train.parquet`, `ar_sft_train.parquet`, `av_sft_val.parquet`, `rl_shuf.parquet`, `val.parquet` + `.nla_meta.yaml` sidecars with verified `critic_suffix_ids` and `Qwen3-8B` cached marker token `㈎`, id `149705`):
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
1. Train **Arm 2** (Fixed Diff-NLA) + **Arm 0** (Raw Activation NLA) first.
2. Run Phase 4 evaluation on Arm 2 vs Arm 0 (`step 0` and `step 400`).
3. Once Arm 2 shows clear auditing gains over Arm 0 (or to diagnose why), run **Arm 1** (Naive Diff ablation) and optionally **Arm 2b** (Cold-Start RL ablation, zero SFT).

Use `scripts/diff/run_phase3_arm.sh` (preview any command first by appending `--dry-run`; pass `--hardware h100` on 4×H100 80GB):

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
  --hardware h200

# Arm 0 (Raw Activation Baseline: tau=0.0)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm0 \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200

# Arm 1 (Naive Diff Ablation: Std Ridge, unmasked templates, uncentered, tau=0.0)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm1 \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200

# Optional Arm 2b (Cold-Start Diff-NLA: skips SFT, runs init_coldstart_ar.py + --disable-av-thinking)
bash scripts/diff/run_phase3_arm.sh \
  --arm arm2b \
  --phase0-dir "$P0_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --ckpt-root "$CKPT_ROOT" \
  --core-python "$CORE_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --nproc 4 \
  --hardware h200
```

#### Mandatory RL Monitoring Tripwires (Check Every 10–50 Steps in W&B / Logs)
| Metric | Healthy Range | Action if Violated |
| :--- | :--- | :--- |
| `av/steer_apply_rate` | `>= 0.99` (ideally `1.00`) | **STOP immediately** — `vllm-lens` patch broken or marker token mismatch |
| `eval/extraction_rate` | `>= 0.95` | If dipping AND `reward/min < -2.0`, restart with `--extra-rl-args "--fail-reward -4.0"` (see **Issue #1**!) |
| `eval/cjk_fraction` | `< 0.02` | If spiking on step 0, check `injection_token_id` sidecar match |
| `eval/cos_mean` & `eval/opt_scale_fve_pct` | Rising monotonically (target `cos >= 0.50` / `opt_scale_fve >= 25%`) | **Watch these alongside `eval/fve_pct`!** On centered diffs, `cos = 0.50` gives `eval/fve_pct = 0.0%` (see **Issue #1**) |
| `eval/weighted_fve_pct` (Arm 2/2b) | Higher than unweighted `eval/fve_pct` | Confirms Fix 3 concentrates signal on large-diff tokens |
| `eval_judge/coherence_mean`, `eval_judge/repetitiveness_mean` | Coherence `>= 7`, Repetitiveness `<= 3` | Watch for RL mode collapse / boilerplate templates |

**Crash / Preemption Resume**:
Re-run `run_phase3_arm.sh` with `--skip-sft --skip-merge --extra-rl-args "--resume-from-lora ${CKPT_ROOT}/arm2/rl_vllm/iter_000200 --start-step 200 --allow-overwrite"` (`train_rl_vllm.py` automatically restores `critic_latest` and `optim_latest.pt` when `--resume-from-lora` is set!).

---

### Phase 4 — Offline 6-Benchmark Evaluation, Batch Judging & Final Report

#### Phase 4a-1: Cache All Benchmark Activations ONCE (1 GPU, ~15–20 min)
Extracts Layer-24 Base, Instruct, and all 10 Taboo LoRA activations (`--taboo-preset paired` = 5 words `snow, smile, blue, gold, ship` × `{narrow, 50_mix}`; or `--taboo-preset distinct` for 10 distinct words), automatically running the 10-prompt `<think>` Mode Check per Taboo LoRA and logging matched KL.
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
  --batch-size 16 \
  --dtype bf16 \
  --device cuda
```
Inspect `$EVAL_DIR/benchmark_meta.json` to confirm all 10 Taboo organisms had clean matched KL (`< 0.5 nats`).

#### Phase 4a-2: Generate Explanations & Score Benchmark 1 AR FVE Across Checkpoints (1 GPU per checkpoint, ~3–5 min each)
Run across every arm and saved RL checkpoint (`0, 50, 100, 200, 300, 400`). Using `--backend vllm` runs all ~4,500 prompts per checkpoint in one batched pass (~2–3 min on H200/H100) and automatically verifies `vllm-lens` patch hunks and steer coverage!
```bash
for ARM in arm2 arm0 arm1; do
  for STEP in 0 50 100 200 300 400; do
    ITER=$(printf "iter_%06d" "$STEP")
    AV_LORA="${CKPT_ROOT}/${ARM}/rl_vllm/${ITER}"
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

#### Phase 4b: Batch LLM-Judging, Bootstrap CIs, Trajectory Plots & Final Report (CPU)
Grades all discovered `*_step*_generations.jsonl` files in one deduplicated Anthropic Batch job per judge model, persisting every judge output into `$EVAL_DIR/report/judge_cache.jsonl` (crash-safe & resumable!).
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
- `final_summary.json` — full nested JSON with 95% bootstrap CIs for every arm, step, and benchmark slice
- `final_report.md` — publication-ready Markdown tables (Headline Trajectory incl. `FVE%`, `Cos`, `OptFVE%`, `VerbLen`; Benchmark 1 rubrics + AR `tok1` FVE; full Taboo Regime × Diff Mode × Position + Off-Diagonal FP table)
- `plots/trajectory_grid.png` — 3×2 trajectory plot grid across RL steps with shaded 95% CI bands
- `qualitative_examples.md` — 30 side-by-side Arm 0 vs Arm 1 vs Arm 2 examples across all benchmarks

---

## 4. Flagged Potential Experiment Issues & Built-In Mitigations

The following 12 subtle mathematical, geometric, and engineering issues were identified during implementation. Every one of them has a built-in diagnostic or mitigation switch already wired into the codebase:

1. **Zero-Mean Unit-Sphere FVE Compression & `-2.0` Failed-Extraction Floor Hazard (CRITICAL)**
   - **Why**: EasyNLA normalizes *both* predicted and gold vectors to radius $\sqrt{d}$, so $\text{MSE}_{\text{norm}} = 2 - 2\cos(\hat{u}, u)$. On raw activations (`arm0`), vectors share a large common mean direction so `mse_rawvar` is small, whereas on Fix-2 centered diffs ($\mathbb{E}[\tilde{\Delta}] = 0$), `mse_rawvar ≈ 1.0`. Consequently:
     - Even strong directional recovery $\cos = 0.50$ gives $\text{MSE}_{\text{norm}} = 1.0 \implies \text{FVE} = 0.0\%$, and $\cos = 0.40$ gives $\text{FVE} = -20\%$, even though optimal scalar-rescaled FVE is $\cos^2 = +16\%\text{–}25\%$. Reaching the plan's `FVE >= 0.15` tripwire requires $\cos \ge 0.575$!
     - Worse, antipodal predictions ($\cos < 0$) yield $r = -\text{MSE}_{\text{norm}} \in (-4.0, -2.0)$, which is **strictly worse** than EasyNLA's default `-2.0` failed-extraction/truncation penalty — creating a reward-hacking incentive for AV to break `<explanation>` tags early in RL!
   - **Mitigation already implemented**:
     - `train_rl_vllm.py` and `phase4_eval_suite.py` log `eval/cos_mean`, `eval/opt_scale_fve_pct` ($\max(0, \bar{\cos})^2 \times 100\%$), and `eval/weighted_fve_pct` alongside `eval/fve_pct`. Judge Arm 2 directional learning primarily by `eval/cos_mean` and `eval/opt_scale_fve_pct`!
     - `train_rl_vllm.py` supports `--fail-reward` (default `-2.0` for baseline parity). If `reward/min < -2.0` and `eval/extraction_rate` dips below 95%, immediately re-run with `--extra-rl-args "--fail-reward -4.0"`.
     - Fixed `valid_e` classification in `train_rl_vllm.py` so valid rollouts with antipodal MSE > 2.0 are not miscounted as failed extractions.

2. **Fix 1 Ridge $\alpha$ Scaling (`1/N` vs Sum-of-Squares)**
   - **Why**: Eq. 2.1 in the plan writes $\frac{1}{N}\sum_i \|\cdot\|^2 + \alpha \|\Delta W\|_F^2$ with $\alpha \in \{10^1, \dots, 10^6\}$. Under $1/N$-normalized MSE, eigenvalues of $\frac{1}{N}X^TX$ are $O(1\text{–}10^3)$, so the CV optimum may lie in $[10^{-2}, 10^1]$ (below $10^1$), whereas under unnormalized sum-of-squares Ridge ($\sum \|\cdot\|^2 + \alpha_{\text{sum}}\|\Delta W\|_F^2$) at $N = 80\text{k}$, $\alpha_{\text{sum}} \in [10^1, 10^6]$ is right in range.
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
   - **Why**: Stock `train_rl_vllm.py` only saved `critic_latest` (overwriting earlier critics) and skipped step 0, which prevents computing Benchmark 1 AR FVE trajectories across steps `0, 50, 100, 200, 300, 400`.
   - **Mitigation already implemented**: Added `--save-steps 0 50 100 200 300 400` and `--save-critic-at-ckpts` (saves `iter_NNNNNN/critic`). Monitor disk (~66 GB/arm for full-FT critics); drop intermediate critic snapshots if disk is constrained.

8. **FineWeb-Edu Full-Dataset Download Trap**
   - **Why**: Calling `datasets.load_dataset("HuggingFaceFW/fineweb-edu")` without `name="sample-10BT", streaming=True` attempts to download multi-TB shards.
   - **Mitigation already implemented**: `nla/diff/benchmarks.py::load_fineweb_edu_docs` hard-defaults to `name="sample-10BT", streaming=True`.

9. **Taboo `adamkarvonen/Qwen3-8B-taboo-*` Naming & `adversarial_50_mix` Ambiguity**
   - **Why**: All 42 repos on HuggingFace (`21 words × {narrow, 50_mix}`) were verified via the HF API. One word (`adversarial`) has ambiguous secret-word vs regime semantics (`adversarial_50_mix`), and pairing `snow_narrow` + `snow_50_mix` means off-diagonal FP must exclude same-secret-word organisms.
   - **Mitigation already implemented**: Default `--taboo-preset paired` uses 5 unambiguous concrete nouns (`snow, smile, blue, gold, ship` × `{narrow, 50_mix}`); `--taboo-preset distinct` uses 10 distinct words; `phase4_judge_and_report.py` excludes any same-secret-word organism when computing off-diagonal FP.

10. **Arm 0 Warm-Start Domain Confounder & Shared vs Disjoint 20k SFT Rows**
    - **Why**: Using `asher577/easynla-warmstart-data` (371k FineFineWeb prose rows) for Arm 0 while Arm 1/2 use 20k chat SFT rows confounds target representation with warm-start domain and 18× SFT data volume. Meanwhile, splitting 20k SFT explanations into 10k AV + 10k AR halves SFT data per module.
    - **Mitigation already implemented**: `phase2_contrastive_sft.py --arm arm0` supports matched 20k chat SFT for Arm 0, and `phase2_build_corpus.py` defaults to `--av-ar-split shared` (~19k train + 1k doc-disjoint val for both AV and AR, 100% doc-disjoint from `rl_shuf` and `val`), with `--av-ar-split disjoint` available if strict AV/AR SFT separation is desired.

11. **`tok1` Leakage Metric Role**
    - **Why**: At `asst_last`, every chat prompt ends on the identical `<|im_start|>assistant\n<think>\n\n</think>\n\n` token, making `tok1` Ridge FVE trivially 0 at `asst_last`.
    - **Mitigation already implemented**: Both Phase 0 and Phase 4 compute `tok1` Ridge FVE specifically on `user_content` tokens (where token identity varies richly and raw `h_target` leaks heavily).

12. **Anthropic Judge Model IDs**
    - **Why**: Default CLI flags use `--judge-model claude-sonnet-4-6` and `--taboo-judge-model claude-opus-4-8` (matching `nla/datagen/providers.py` and `nla/utils/text_judges.py`).
    - **Mitigation**: Verify active Anthropic API model IDs on your key before launching large Phase 2b / Phase 4b batches; override with `--model` / `--judge-model` / `--taboo-judge-model` as needed.

---

## 5. Troubleshooting Cheat Sheet

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `train_rl_vllm` crashes on IPC weight sync | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in RunPod base image | `unset PYTORCH_CUDA_ALLOC_CONF` (`run_phase3_arm.sh` already does this automatically) |
| `[fatal] installed vllm-lens is missing required patch hunks` | `vllm-lens` venv rebuilt without running patcher | `"$VLLM_PYTHON" utils/patch_vllm_lens.py` |
| OOM during AR SFT or RL on 80GB H100 | Full-FT fp32 AR needs ~88 GB | Pass `--hardware h100` to `run_phase3_arm.sh` (switches AR SFT to LoRA 16×4 and RL to `rl_vllm_h100_80gb.yaml` with `ar_lora: true`) |
| `av/steer_apply_rate < 0.99` | Stale `vllm-lens` or sidecar/tokenizer mismatch | Check sidecar `injection_token_id` (`149705` / `㈎` for `Qwen3-8B`) and re-run `utils/patch_vllm_lens.py` |
| `eval/extraction_rate` drops during Arm 2 RL | Antipodal centered-diff MSE > 2.0 makes `-2.0` fail floor attractive | Re-run with `--extra-rl-args "--fail-reward -4.0"` |
| `phase2_contrastive_sft.py` or `phase4_judge_and_report.py` interrupted mid-batch | Network disconnect / pod preemption | Re-run the exact same command — both scripts use append-only JSONL caches (`sft_explanation_cache.jsonl` / `judge_cache.jsonl`) and only submit uncached prompts |
