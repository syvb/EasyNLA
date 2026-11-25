# Remote-GPU Handoff Plan: Minimal Direct Diff-Target NLA on `Qwen3-8B` (Layer 24)

**Goal**: Test whether training a Natural Language Autoencoder (NLA) on **cleaned post-training activation differences** $\tilde{\Delta}_{\text{ft}}$ between `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-8B` — holding the standard single-layer L24 `EasyNLA` architecture fixed and applying **only** the minimal diff-specific geometric fixes (Fixes 1–3 + matched `<think>` hygiene) — improves post-training auditing (Refusal, Eval-Awareness, Sycophancy, and zero-shot Taboo model-organism recovery) and eliminates post-step-100 auditing degradation relative to standard NLAs.

> [!IMPORTANT]
> **Scope Discipline (Read Before Coding)**:
> - **IN SCOPE**: Standard single-layer L24 `EasyNLA` architecture (`Qwen/Qwen3-8B`), Direct Diff-Target $\Delta_{\text{ft}}$ with **Fix 1** (Identity-Anchored Ridge), **Fix 2** (chat-template masking + per-role centering), **Fix 3** (soft-clipped norm weighting in AR loss, AV injection, and post-standardization GRPO advantages), and **matched `<think>` hygiene**.
> - **EXPLICITLY OUT OF SCOPE** (already tested/deferred): Multi-layer / multi-position inputs (general NLA upgrades, intentionally held constant here for a clean ablation) and **Single-Model Base-Conditioned AR** (empirically falsified in prior local experiments: trainable Base AR suffers 82–99% distillation collapse because $H(h^{\text{target}}(x)\mid x) = 0$; strictly frozen Base AR fails via prompting-vs.-finetuning orthogonality, $\cos \approx 0.11$).

---

## 1. Self-Contained Context: What NLAs Are & Why This Experiment Matters

### 1.1 Natural Language Autoencoders (NLAs) & `EasyNLA`

A **Natural Language Autoencoder** (Fraser-Taliente et al., May 2026) is an unsupervised interpretability model that translates an LLM's internal residual-stream activation $h_\ell(x, t) \in \mathbb{R}^d$ into a human-readable natural-language explanation $e$, and scores $e$ by how faithfully a separate model can reconstruct $h_\ell(x, t)$ from $e$ alone:

1. **Activation Verbalizer (AV, encoder)**: An LLM initialized from the subject model (`Qwen/Qwen3-8B`) with a LoRA adapter. The target vector $v \in \mathbb{R}^d$ is injected into the AV's residual stream at layer 1 at a dedicated marker token (`㊗` flanked by canonical neighbors, via Karvonen norm-matched addition: $h'_p = h_p + \|h_p\|_2 \cdot v / \|v\|_2$). The AV generates `<explanation>...</explanation>`.
2. **Activation Reconstructor (AR, decoder / critic)**: A truncated copy of the subject model (layers `0..L`, i.e., first 25 blocks for L24) + a `Linear(d, d)` value head. Reading **only** the text $e$ (never the original prompt $x$), the AR outputs $\hat{h}(e) \in \mathbb{R}^d$.
3. **Training Objective**: Warm-start SFT (AV trained to emit gold explanations from activations; AR trained to predict activations from gold explanations), followed by **on-policy GRPO** where the AV generates $G = 8$ candidate explanations per prompt and receives reward $r_{ij} = -\text{MSE}_{\text{norm}}(\hat{h}(e_{ij}), v_i)$ while the AR co-trains on the same rollouts (`train_critic: true`).
4. **Headline Reconstruction Metric**: **Fraction of Variance Explained (FVE)** on held-out activations:
   $$
   \text{FVE} = 1 - \frac{\mathbb{E}[\|v - \hat{v}\|_2^2]}{\mathbb{E}[\|v - \bar{v}\|_2^2]}.
   $$

We use **[`asherps/EasyNLA`](https://github.com/asherps/EasyNLA)**, a fast data-parallel GRPO implementation built specifically for `Qwen/Qwen3-8B` at **Layer 24** (`36` layers total, `d_model = 4096`, `floor(2*36/3) = 24`). Under `torchrun --nproc_per_node=4`, each GPU runs its own single-GPU `vllm-lens` rollout engine + HF trainer + AR critic with GPU-to-GPU IPC weight sync, reaching ~70% held-out FVE in **~3–4 hours on 4×H200**.

### 1.2 Why Model-Diff NLA & Prior Findings from Our Local Pilot

Standard NLAs are trained on raw activations $h_{24}^{\text{target}}(x, t)$, where repeating/summarizing the surface prompt and final tokens explains ~64% of variance (Bazzani & Mossing, Nov 2026, `arXiv:2611.19507` App. D), and post-training auditing features (refusal, eval-awareness, sycophancy, hidden goals) must compete with massive pretraining prompt variance. Prior work by Blasi Núñez et al. (Oct 2026, `arXiv:2610.30934`) applied unmodified open-weight NLAs zero-shot to LoRA-minus-Instruct diffs and ran supervised keyword-bonus RL on Taboo words, but **never trained an unsupervised Model-Diff NLA on Base-vs.-Instruct diffs**.

In a local CPU pilot across **`Qwen3-0.6B` (Base vs. Instruct, L18)**, **`Qwen2.5-0.5B` (Base vs. Instruct, L16)**, and fine-tuned **Narrow vs. 50%-Mix Taboo LoRAs**, we found:
- **Real Signal**: Post-training rotates safety/auditing representations strongly at the assistant decision boundary `asst_last` (`cos(Base, Target) = 0.39–0.47` on Qwen3, `0.66–0.71` on Qwen2.5), while leaving pretraining-only features invariant (`Historical Era cos = 0.97–0.99`). Cleaned diffs boost Refusal, Eval-Awareness, and Sycophancy Fisher SNR by **1.4×–2.5×**.
- **Four Fatal Traps in Naive $\Delta_{\text{ft}}$** (each solved cleanly by the four minimal fixes below):
  1. **Standard Ridge shrinks $W \to 0$** and performs *worse* than Identity (`0.751` vs. `0.799` held-out web FVE on Qwen3-0.6B).
  2. **Chat-template tokens (`sys_prefix`, `asst_header`) eat 71.7–79.5% of raw $\|\Delta\|^2$ energy**, causing naive $\Delta_{\text{id}}$ to *increase* surface-token predictability above raw `h_target` (`tok4`: `0.422 → 0.612`). Even at `asst_last`, a constant prompt-independent mean offset eats **81.1%** of energy on Qwen3 (`<think>` switch).
  3. **Blind unit-normalization inflates zero-signal web drift by ~25×** (`web` is 0.5–2.6% of raw diff energy with $\le 0$ predictability).
  4. **Qwen3 `<think>`-mode mismatch** swamps `Org − Inst` diffs on non-thinking Taboo LoRAs (`ActShift = 1.635`, `KL = 25.0 nats`).

---

## 2. Exact Mathematical Spec of the Minimal 4-Fix Pipeline & `EasyNLA` Patches

All experiments use **`Qwen/Qwen3-8B-Base`** ($h^{\text{base}}$) and **`Qwen/Qwen3-8B`** ($h^{\text{target}}$) at **Layer 24** (0-indexed decoder block 24, i.e., `hidden_states[25]`, `d_model = 4096`).

### 2.1 Fix 1 — Identity-Anchored Ridge Alignment ($W_{\text{id-anc}} = I + \Delta W$)

Fit strictly on **50,000–100,000 raw pretraining tokens** (FineWeb / FineFineWeb, excluding position-0 BOS/sink tokens — **never** include chat transcripts when fitting $W_{\text{id-anc}}$!):

$$
(\Delta W^\star, b^\star) = \arg\min_{\Delta W, b} \; \frac{1}{N}\sum_{i=1}^N \left\| (h_i^{\text{target}} - h_i^{\text{base}}) - (\Delta W h_i^{\text{base}} + b) \right\|_2^2 + \alpha \|\Delta W\|_F^2.
$$

- **Implementation**: Use `float64` normal equations on GPU (`torch.linalg.solve` on $4096 \times 4096$ takes `< 2 seconds`!) with 5-fold doc-grouped CV over $\alpha \in \{10^1, 10^2, 10^3, 10^4, 10^5, 10^6\}$.
- **Aligned diff**: $\Delta_{\text{id-anc}}(x, t) = (h^{\text{target}}(x, t) - h^{\text{base}}(x, t)) - (\Delta W^\star h^{\text{base}}(x, t) + b^\star)$.
- **Save artifact**: `align_L24_id_anchored.pt` containing `delta_W` (`4096×4096`), `bias` (`4096`), `alpha`, and held-out web FVE verification (`FVE_id_anc > FVE_identity > FVE_std_ridge`).

### 2.2 Fix 2 — Chat-Template Masking + Per-Role Mean Centering

For every chat transcript formatted in Qwen3's ChatML template:
1. **Mask out static template tokens** during training: **exclude** `chat_bos` (pos 0), all `sys_prefix` tokens (`<|im_start|>system\n...<|im_start|>user\n`), and all `asst_header` tokens (`<|im_end|>\n<|im_start|>assistant` before the final assistant-turn trigger token).
2. **Retain three active token roles** for training:
   - `user_content`: sampled tokens inside the user turn (up to 2–3 random positions per prompt).
   - `asst_last`: the final prefill token opening the assistant's actual response (see `<think>` fix below!).
   - `asst_rollout`: 2–3 sampled tokens from an on-policy greedy/temp-0.7 completion $y \sim p_{\text{target}}(\cdot \mid x)$ (teacher-forced identically through both Base and Instruct so token positions match 1-to-1!).
3. **Subtract exact role-specific means** estimated on the training split:
   $$
   \tilde{\Delta}(x, t) = \Delta_{\text{id-anc}}(x, t) - \mu_{\text{role}(t)}, \qquad \mu_r = \mathbb{E}_{\text{train}, \, \text{role}=r}[\Delta_{\text{id-anc}}(x, t)].
   $$
   Save `role_means_L24.pt` (`{"user_content": ..., "asst_last": ..., "asst_rollout": ..., "tau_web_median": ...}`).

### 2.3 Fix 3 — Soft-Clipped Norm Weighting (3 Exact Patch Sites in `EasyNLA`)

Compute the noise floor $\tau = \operatorname{Median}_{\text{web\_val}}(\|\Delta_{\text{id-anc}}\|_2)$ (also log $\operatorname{P75}_{\text{web}}$ as a sanity check) and per-example gate weight:

$$
w_i = \frac{\|\tilde{\Delta}_i\|_2^2}{\|\tilde{\Delta}_i\|_2^2 + \tau^2} \in (0, 1).
$$

> [!CAUTION]
> **Critical `EasyNLA` Gotchas Discovered via Code Audit (`nla/injection.py` & `nla/train_rl_vllm.py`)**:
> 1. `EasyNLA` computes per-prompt GRPO advantages as `A_ij = (r_ij - mean_j) / (std_j + eps)`. If you multiply $r_{ij}$ by $w_i$ *before* standardization, dividing by `std_j` **completely cancels $w_i$**! You **must** apply $w_i$ *after* advantage standardization.
> 2. `nla/injection.py` (`karvonen_inject_in_residual`) and `nla/utils/vllm_steer.py` (`build_steering_vector`) **hard-normalize** every injected vector to unit norm (`v / ||v||`) before scaling by `||h_p||`, which would erase soft-clipping at the AV input!

Apply Fix 3 at all **three** code sites (controlled by a CLI/config flag `--soft-norm-tau <float>`, defaulting to `0.0` so Arms 0 and 1 run completely unchanged!):
1. **AV Injection (`nla/injection.py` & `nla/utils/vllm_steer.py`)**:
   Replace `v_unit = v / (v.norm() + 1e-9)` with:
   ```python
   denom = torch.sqrt(v.pow(2).sum(dim=-1, keepdim=True) + tau ** 2) + 1e-9
   v_scaled = v / denom
   out[b, p] = h_p + h_p.norm() * v_scaled
   ```
   *(Store raw unnormalized $\tilde{\Delta}_i$ in parquet `activation_vector` with sidecar `injection_scale: "raw"`, `mse_scale: "sqrt_d_model"` so the vector carries its true norm $\|\tilde{\Delta}_i\|_2$ into both the hook and the weight computation!)*
2. **AR Critic Loss (`nla/train_rl_vllm.py` & `nla/train_sft.py`)**:
   Compute `w_i = norm_sq / (norm_sq + tau**2)` from raw $\tilde{\Delta}_i$, target normalized direction $u_i = \sqrt{d} \cdot \tilde{\Delta}_i / (\|\tilde{\Delta}_i\|_2 + 10^{-9})$, and weight each sample's critic MSE loss by `w_i` (normalized by `w.mean()` to preserve effective learning rate!).
3. **AV GRPO Policy Loss (`nla/train_rl_vllm.py`)**:
   Compute rewards $r_{ij} = -\text{MSE}_{\text{norm}}(\hat{u}(e_{ij}), u_i)$ and standard per-group advantages $A_{ij} = (r_{ij} - \mu_i) / (\sigma_i + \epsilon)$, then multiply by $w_i$ **after** standardization:
   ```python
   weighted_adv = (w_prompt / (w_prompt.mean() + 1e-8)) * adv
   ```
   Unit-test all three patch sites with `pytest` before launching GPU runs!

### 2.4 Matched-`<think>` & Base-Wrapper Hygiene (Training & Model-Organism Evals)

`Qwen/Qwen3-8B` is a hybrid thinking model whose default chat template triggers `<think>` at `<|im_start|>assistant\n` (`KL(Target || Base) ≈ 15–17` nats), while ChatML special tokens (`<|im_start|>`, `<|im_end|>`, `<think>\n\n</think>\n\n`) can be partially OOD for `Qwen/Qwen3-8B-Base`:
- **Instruct Wrapper (Fixed)**: Always use Qwen3's official **non-thinking mode** (`tokenizer.apply_chat_template(..., enable_thinking=False)`), which appends `<|im_start|>assistant\n<think>\n\n</think>\n\n` so `asst_last` is the final prefill token (`\n\n`) right before the assistant's actual answer begins.
- **Base Wrapper Selection (Tested in Phase 0, ~5 min)**: Because Fix 2 already drops all `sys_prefix` and `asst_header` tokens, `user_content`, `asst_last`, and `asst_rollout` **still align 1-to-1** even if Base uses a natural-text dialogue wrapper instead of ChatML! In Phase 0, compare two wrappers for `Qwen3-8B-Base` against `Instruct(ChatML)`:
  1. **`Base(ChatML)`**: Exact identical `enable_thinking=False` token IDs as Instruct.
  2. **`Base(Plain)`**: Base-native dialogue transcript (2 short neutral Q&A few-shot turns followed by `User: {u}\nAssistant: {y}`, where `asst_last` is the token immediately before `{y}` and `{u}`, `{y}` share exact token boundaries with Instruct).
  - Lock in whichever Base wrapper gives **lower neutral-chat `KL(Instruct || Base)`** and **higher average Refusal/Eval-Awareness/Sycophancy Fisher SNR** in Phase 0 (also check `chat_template_amplification` on identical user query text raw vs. wrapped), and use that winning wrapper consistently across Phase 2–4!
- **In Model-Organism Evals (Taboo LoRAs)**: Before running any Taboo LoRA (e.g., `adamkarvonen/Qwen3-8B-taboo-*` and `*_50_mix`), run a 10-prompt **Mode Check** measuring `KL(Org || Inst)` on neutral prompts with `enable_thinking=False` vs. `enable_thinking=True`:
  - Pick whichever mode gives the **lower** neutral-prompt `KL(Org || Inst)` (typically `enable_thinking=False` if the LoRA was trained without `<think>`!).
  - Verify `KL(Org || Inst) < 3.0` nats on neutral prompts; log the exact KL in every eval report.
  - Evaluate **both** $\Delta(\text{Org} - \text{Inst})$ (aligned via $W = I$, since LoRA shares Instruct's coordinate system) and $\Delta(\text{Org} - W_{\text{id-anc}} h_{\text{base}})$!

---

## 3. Experimental Arms, Data, and Evaluation Suite

### 3.1 Controlled 3-Arm Design (Plus Optional Cold-Start Ablation)

All arms use **identical** hyperparameters (`configs/rl_vllm.yaml`: `batch_prompts=256`, `group_size=8`, `num_steps=400`, `lr=1e-4`, `critic_lr=8e-5`, `lora_r=128`, `lora_alpha=16`, `use_rslora=true`, `max_new_tokens=256`, `length_penalty=0.01`), **identical** underlying chat+rollout prompts, and **single-layer L24**:

| Arm | Target Vector $v$ | Fixes Enabled | Warm-Start SFT | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Arm 0** *(Standard NLA Control)* | Raw `h_target` (`Qwen/Qwen3-8B` L24) | Matched `<think>` only | Standard SFT on same chat prompts (or `asher577/easynla-warmstart-data` + 20k chat SFT) | True apples-to-apples baseline on the exact same chat/rollout distribution. |
| **Arm 1** *(Naive Diff Ablation)* | Unfixed $\Delta_{\text{std-ridge}} = h_t - W_{\text{std-ridge}} h_b$ | None (includes `sys_prefix`/`asst_header`, standard Ridge, blind unit-norm) | Contrastive SFT (same 20k pairs) | Tests whether Fixes 1–3 actually matter at 8B scale (predicts template-quoting collapse & noisy FVE). |
| **Arm 2** *(Fixed Diff-NLA — Main)* | Cleaned $\tilde{\Delta} = \Delta_{\text{id-anc}} - \mu_{\text{role}}$ | **Fixes 1 + 2 + 3 + matched `<think>`** | Contrastive SFT (20k pairs) | **Main treatment.** |
| **Arm 2b** *(Optional — Cold-Start)* | Same as Arm 2 | Fixes 1 + 2 + 3 + matched `<think>` | **No SFT** (200 steps cold-start GRPO from base Qwen3-8B AV+AR, per EleutherAI Sep 2026) | Checks if contrastive teacher SFT is even necessary or introduces teacher bias. |

### 3.2 Training Corpus & Contrastive SFT Warm-Start

1. **Alignment Corpus (Fit Fix 1 only)**: 50k–100k tokens (~1,000 passages) from `HuggingFaceFW/fineweb-edu` (80% train / 20% val, doc-disjoint).
2. **Main Chat + Rollout Corpus (SFT + RL)**:
   - Sample **60,000 diverse single-turn/multi-turn chat prompts** (e.g., 30k `allenai/WildChat-1M` English filtered + 20k `HuggingFaceH4/ultrachat_200k` + 10k safety/refusal/eval-style neutral-mix prompts from public sets like `Anthropic/hh-rlhf` / `PKU-Alignment/PKU-SafeRLHF` — strictly held out from eval benchmarks!).
   - Generate on-policy `Qwen/Qwen3-8B` completions (`enable_thinking=False`, `max_new_tokens=96`, `temperature=0.7`) via vLLM.
   - Extract L24 activations from **both** `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-8B` on the identical teacher-forced `(prompt + completion)` token IDs.
   - Sample **3 positions per conversation**: 1 random `user_content`, 1 `asst_last`, 1 random `asst_rollout` → **180,000 rows** total:
     - **20,000 rows** → SFT warm-start (`av_sft_train.parquet`, `ar_sft_train.parquet`), doc-disjoint.
     - **155,000 rows** → GRPO RL (`rl_shuf.parquet`), doc-disjoint (400 steps × 256 prompts = 102,400 prompts seen).
     - **5,000 rows** → Held-out validation (`val.parquet`), doc-disjoint.
3. **Contrastive Warm-Start Gold Explanations (20,000 SFT rows)**:
   - For Arms 1 & 2, generate concise (30–80 word) gold explanations with Claude Sonnet 4.6 via the **Anthropic Message Batches API (50% discount)**:
     - Show Sonnet the prompt prefix ending at position $t$, the **Base model's** top-5 next tokens + short continuation, and the **Instruct model's** top-5 next tokens + short continuation.
     - Prompt: *"Describe in 2–3 concise bullet points what post-training cognitive, behavioral, persona, safety, or conversational shift separates the Instruct model's internal state from the Base model at this exact token. Do NOT summarize or quote the prompt text unless necessary to explain the behavioral shift. Wrap in `<explanation>...</explanation>`."*
   - For Arm 0 (matched raw-activation control), either use standard single-model Sonnet descriptions on the same 20k chat rows (or fine-tune from `asher577/easynla-warmstart-data` + 10k chat rows).

### 3.3 Evaluation Suite (Tracked at RL Steps `0, 50, 100, 200, 300, 400`)

Every arm is evaluated across **6 quantitative benchmarks** (plus EasyNLA's built-in `text_judges` rubric) so we directly test both **peak auditing performance** and the **post-step-100 trajectory**:

| Eval Category | Benchmark / Dataset | Extraction Position(s) | Metric & Judge | Expected Outcome if Hypothesis Holds |
| :--- | :--- | :--- | :--- | :--- |
| **1. Reconstruction & Leakage** | 1,000 held-out chat val rows | `user_content`, `asst_last`, `asst_rollout` | Held-out FVE, verbatim n-gram quote length, `tok1` Ridge FVE of AR preds, `EasyNLA` `text_judges` (`unique_info`, `coherence`, `specificity`, `repetitiveness`) | Arm 2 has **lower verbatim quoting** & **higher `unique_info`** than Arm 0 and Arm 1; Arm 1 suffers template-scaffolding collapse. |
| **2. Refusal / Safety Auditing** | 100 topic-matched benign vs. harmful pairs (e.g., XSTest + HarmBench/WildJailbreak safe-vs-unsafe pairs) | `asst_last` | LLM-judge (Opus/Sonnet) binary AUROC & accuracy predicting whether target is preparing to refuse from **AV explanation alone** (no prompt shown!) | **Arm 2 > Arm 0 > Arm 1**, stays high past step 100. |
| **3. Eval-Awareness Auditing** | 100 realistic-deployment vs. benchmark/red-team-framed pairs (following Fraser-Taliente et al. App. prompt) | `asst_last` | LLM-judge balanced accuracy & AUROC detecting eval-awareness from AV explanation alone | **Arm 2 > Arm 0** (biggest SNR jump in pilot: `2.04×`). |
| **4. Sycophancy Auditing** | 100 neutral vs. user-biased misconception pairs (TruthfulQA / Anthropic sycophancy style) | `asst_last` | LLM-judge balanced accuracy detecting user-pressure/misconception-handling stance | **Arm 2 > Arm 0** (`2.46×` SNR jump in pilot). |
| **5. Zero-Shot Model Organisms (Taboo Narrow + 50%-Mix)** | 10 published `Qwen3-8B` Taboo LoRAs from `adamkarvonen/Qwen3-8B-taboo-*` — **5 Narrow** (e.g., `snow`, `smile`, `blue`, `gold`, `ship`) + **5 50%-Mix** (e.g., `blue_50_mix`, `adversarial_50_mix`, etc.) × 50 prompts each | `asst_last` + `user_mean` (matched `<think>` mode!), testing both `Org−Inst` and `Org−Base` | Bazzani & Mossing / Blasi Núñez protocol: Opus guesses top-3 secret words from AV verbalization → **Hit Rate@3** AND **off-diagonal cross-organism FP rate** | **Arm 2 > Arm 0** zero-shot on `Org−Inst` and `Org−Base`, especially on harder **50%-Mix** organisms; FP rate `< 2%`. |
| **6. Negative Pretraining Controls** | Bazzani & Mossing **Historical Period** (300 excerpts) + **MATH** (200 no-CoT problems at `Answer:`) | Random doc pos / final `:` | Balanced accuracy (Historical Period) & mechanism pass@5 (MATH) | **Arm 2 < Arm 0** — confirms Diff-NLA *selectively* isolates post-training features rather than memorizing pretraining knowledge! |

---

## 4. Step-by-Step Execution Playbook & Go/No-Go Gates (For the Remote GPU Agent)

### Phase 0 — Environment Setup & Cheap Geometry Sanity Gate (~45 min, STOP if Gate Fails!)
1. **Provision & Install**:
   - Clone `https://github.com/asherps/EasyNLA.git`; install core venv + `bash scripts/install_vllm_lens.sh` + `patch_vllm_lens.py`.
   - **Critical**: Ensure `PYTORCH_CUDA_ALLOC_CONF` is **unset** before any `train_rl_vllm` launch! Verify `torch.cuda.device_count()`, `HF_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY`, and disk space (`>= 400 GB` free for caches + parquets + checkpoints).
2. **Write and Run a `Qwen3-8B` L24 Geometry & Base-Wrapper Audit Script**:
   - *Implement this as a standalone Python script (nothing pre-exists in `EasyNLA` for this step).* Load `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-8B` one at a time in `bf16` (clearing CUDA cache between models) and extract Layer 24 (`hidden_states[25]`, `d_model = 4096`) activations across:
     1. **Pretraining alignment split**: 2,000 raw FineWeb-Edu passages (`1,600` train / `400` val, up to 64–128 tokens each, skipping position-0 BOS/sink tokens) — ~50k–100k tokens total.
     2. **Matched chat diagnostic suite**: ~20–30 neutral queries plus matched binary pairs (~50–100 pairs each, or smaller starter sets if verifying quickly) for **Refusal** (topic-matched benign vs. harmful), **Eval-Awareness** (realistic deployment vs. benchmark/red-team framing), **Sycophancy** (neutral vs. user-biased misconception), and **Historical Era** (pretraining negative control). Use `enable_thinking=False` on Instruct.
     3. **Taboo LoRA `<think>` check**: 2 open `Qwen3-8B` Taboo LoRAs (`adamkarvonen/Qwen3-8B-taboo-snow` and `adamkarvonen/Qwen3-8B-taboo-blue_50_mix`) evaluated on 10–20 neutral prompts under both `enable_thinking=False` and `enable_thinking=True`, measuring `KL(Org || Inst)` at `asst_last`.
   - **Alignment comparison (§2.1)**: Fit on FineWeb train tokens via `float64` GPU normal equations (`torch.linalg.solve` with 5-fold doc-grouped CV over $\alpha \in \{10^1, \dots, 10^6\}$) and evaluate held-out FineWeb val FVE for `W = I`, `Identity + Bias`, `Standard Ridge`, and `Identity-Anchored Ridge` ($W = I + \Delta W$). *(Shortcut: if `FVE_identity(web_val) > 0.95` already, log whether `W = I` is effectively indistinguishable from `W_id_anc`.)* Compute the noise floor $\tau = \operatorname{Median}_{\text{web\_val}}(\|\Delta_{\text{id-anc}}\|_2)$.
   - **Base Wrapper A/B Test (§2.4, ~5 min)**: On the chat pairs + neutral queries, extract Base under both **`Base(ChatML)`** (exact `enable_thinking=False` ChatML IDs) and **`Base(Plain)`** (2 short neutral Q&A turns ending in `User:\n{u}\nAssistant:\n`, concatenated piece-wise in token IDs `prefix_ids + user_ids + suffix_ids` so BPE never merges trailing punctuation with `\n` and `user_content` aligns 1-to-1 with ChatML). Compare both wrappers on neutral `KL(Instruct || Base)` at `asst_last`, `chat_template_amplification` on `user_content`, `tok1` Ridge leakage, and average Refusal/Eval-Awareness/Sycophancy Fisher SNR & 5-fold probe AUROC (comparing raw `h_target`, raw `Δ_idanc`, and Fix 2 role-centered `Unit(Δ_idanc)`).
   - **Outputs to save for Phase 2**: Write `phase0_report.json` (all metrics + Gate 0 PASS/FAIL), `phase0_config.json` (winning `base_wrapper`, `tau_web_median`, chosen `alpha`, layer `24`), `align_L24_id_anchored.pt` (`delta_W`, `bias`, `alpha`), and `role_means_L24.pt` (`μ_user_content`, `μ_asst_last`, `tau_web_median`).
   - **GO / NO-GO GATE 0**: Proceed to Phase 1 **only if**:
     - `FVE_id_anc(web_val) >= FVE_identity(web_val) > FVE_std_ridge(web_val)` (verifies Fix 1 at 8B scale),
     - Under the winning Base wrapper, role-centered `Unit(Δ_idanc)` improves average Fisher SNR over `h_target` on Refusal/Eval/Sycophancy by `>= 1.25×` while cutting `tok1` leakage by `>= 30%`, AND
     - Matched `<think>` mode keeps neutral `KL(Org || Inst) < 3.0` nats on the Taboo LoRAs.

### Phase 1 — Code Patching & Unit Tests (~30 min)
1. Implement `--soft-norm-tau` in `nla/injection.py`, `nla/utils/vllm_steer.py`, `nla/train_sft.py`, and `nla/train_rl_vllm.py` (exact 3 sites in §2.3).
2. Write and run a unit test verifying:
   - `tau=0.0` reproduces baseline `EasyNLA` outputs bitwise within `1e-5`.
   - With `tau > 0`, a tiny noise vector (`||v|| = 0.05 * tau`) gets injected at `~0.05 * ||h_p||` (not `1.0 * ||h_p||`!), critic weight `w_i ≈ 0.0025`, and GRPO `weighted_adv` shrinks by `w_i` (NOT canceled by group `std_j`!).
   - Verify `av/steer_apply_rate == 1.0` and `cjk_fraction < 0.01` on a 5-step smoke test!

### Phase 2 — Corpus Extraction & Contrastive SFT Data (~1.5 hr GPU + Async Batch API)
1. Generate 60k on-policy `Qwen/Qwen3-8B` chat rollouts (`enable_thinking=False`, `max_new_tokens=96`) via vLLM DP=4.
2. Extract L24 Base & Instruct activations on the 180k sampled positions (`user_content`, `asst_last`, `asst_rollout`, using the winning `base_wrapper` from `phase0_config.json` for Base!) + ChatML template positions (saved separately for Arm 1 naive ablation). Apply Fix 1 (`align_L24_id_anchored.pt`) and Fix 2 (`role_means_L24.pt`) to materialize the `EasyNLA` parquet + `.nla_meta.yaml` sidecars for Arms 0, 1, and 2.
3. Submit 20k contrastive explanation requests to Anthropic Batch API (Claude Sonnet 4.6) for Arms 1 & 2 (and standard single-model explanations for Arm 0 if not reusing `asher577/easynla-warmstart-data`).

### Phase 3 — SFT Warm-Start & On-Policy GRPO Training (~12–14 hr GPU across 3 Arms)
1. **Warm-Start SFT (1 epoch each, 20k rows)**:
   - Train AV (`--mode av --use-lora`, bf16, `lr=1e-4`) and AR (`--mode ar --ar-num-layers 25`, `lr=2e-5`) for Arm 0, Arm 1, and Arm 2.
   - Merge AR to HF bf16 (`scripts/merge_lora_to_hf.py`); pass AV SFT adapter directly via `--av-adapter` into RL (avoiding EasyNLA's documented ~12pp FVE cold-start penalty from re-initializing LoRA `B=0`).
2. **GRPO RL (`torchrun --standalone --nproc_per_node=4 -m nla.train_rl_vllm`)**:
   - Run **Arm 2 (Fixed Diff-NLA)** and **Arm 0 (Standard Raw Control)** FIRST (Lean Pilot milestone!).
   - Save checkpoints at steps **`0, 50, 100, 200, 300, 400`** (`--save-every 50` for first 100 steps, then `100`).
   - **Tripwire during RL**: Monitor `eval/extraction_rate` (must stay `> 0.90`), `av/steer_apply_rate` (`== 1.0`), `cjk_fraction` (`< 0.02`), and held-out `base_fve` every 10 steps. If Arm 2 FVE stalls `< 0.15` by step 100, inspect 20 sample verbalizations immediately to diagnose AV injection scale (`tau`) or SFT format drift before launching Arm 1!
   - Once Arm 0 vs. Arm 2 shows clear signal, run **Arm 1 (Naive Diff)** and optional **Arm 2b (Cold-Start RL, 200 steps)**.

### Phase 4 — Full Auditing Evaluation & Trajectory Report (~2 hr GPU + Batch Judge API)
1. Run offline vLLM generation for all 6 evaluation benchmarks across checkpoints `0, 100, 200, 300, 400` for each arm.
2. Grade verbalizations via Anthropic Batch API (Claude Sonnet 4.6 / Opus 4.8) using fixed prompt templates; compute 95% bootstrap CIs over prompts (and across organisms for Taboo).
3. Produce final comparison tables + RL-step-vs.-auditing-score plots (`Step 0 → 400`) + 30 qualitative side-by-side verbalization examples (Arm 0 vs. Arm 1 vs. Arm 2 on Refusal, Eval-Awareness, Sycophancy, and 50%-Mix Taboo).

---

## 5. Time & Cost Estimate Table

### Hardware Notes
- **Recommended: `4×H200 141GB`** — `EasyNLA`'s `train_rl_vllm` peaks at **~115 GB VRAM per GPU** at `vllm_tp=1, vllm_gpu_mem=0.35`, so 4×H200 runs zero-modification out of the box in **~3.5 hr/arm** (reference cloud rate: **\$3.50/H200-hr = \$14.00/node-hr**).
- **Budget Alternative: `4×H100 80GB`** — Requires reducing `vllm_gpu_mem: 0.25`, enabling gradient checkpointing, and halving train microbatch (or setting `--vllm-tp 2 --nproc_per_node 2`), taking **~5.5 hr/arm** (reference cloud rate: **\$2.50/H100-hr = \$10.00/node-hr**).

### Itemized Breakdown (GPU + API)

| Phase / Milestone | 4×H200 Wall Time | 4×H200 GPU-Hrs | 4×H200 Cost (\$3.50/hr) | 4×H100 Wall Time | 4×H100 Cost (\$2.50/hr) | Anthropic Batch API Cost | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 0**: Setup + L24 Geometry & `<think>` Gate | 0.75 h | 3.0 | \$11 | 1.0 h | \$10 | \$0 | Go/No-Go gate before any big spend |
| **Phase 1**: Patch Fixes 1–3 + Unit Tests + 5-step Smoke | 0.50 h | 2.0 | \$7 | 0.5 h | \$5 | \$0 | Validates soft-norm hook & GRPO weight |
| **Phase 2a**: 60k Rollouts + Base/Instruct L24 Extraction | 1.00 h | 4.0 | \$14 | 1.5 h | \$15 | \$0 | Produces 180k aligned rows for all arms |
| **Phase 2b**: Contrastive SFT Gold Explanations (20k rows) | *(async)* | — | \$0 | *(async)* | — | **\$41** | Sonnet 4.6 Batch API (~12M in / 3M out) |
| **Phase 3a**: Warm-Start SFT (AV + AR, 1 ep, 3 arms) | 1.50 h | 6.0 | \$21 | 2.25 h | \$23 | \$0 | ~0.5 h/arm on 4×H200 (0.35 h if Arm 0 uses HF data) |
| **Phase 3b — Arm 0**: Standard Raw NLA RL (400 steps) | 3.50 h | 14.0 | \$49 | 5.50 h | \$55 | \$0 | Apples-to-apples raw `h_target` control |
| **Phase 3c — Arm 2**: **Fixed Diff-NLA RL (400 steps)** | 3.50 h | 14.0 | \$49 | 5.50 h | \$55 | \$0 | **Main treatment (Fixes 1–3 + `<think>`)** |
| **Phase 3d — Arm 1**: Naive Diff-NLA RL (400 steps) | 3.50 h | 14.0 | \$49 | 5.50 h | \$55 | \$0 | Ablation proving Fixes 1–3 are required |
| **Phase 4**: 6-Benchmark Auditing Eval (5 ckpts × 3 arms) | 1.50 h | 6.0 | \$21 | 2.25 h | \$23 | **\$55** | ~20k judge calls (Sonnet/Opus Batch API) |
| **Optional Arm 2b**: Cold-Start Diff-NLA (200 steps) | 1.75 h | 7.0 | \$25 | 2.75 h | \$28 | \$10 | Tests skipping contrastive SFT warm-start |

### Budget Roll-Up Summary

| Execution Tier | What's Included | 4×H200 Total Time | 4×H200 Total Cost (GPU + API) | 4×H100 Total Time | 4×H100 Total Cost (GPU + API) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gate 0 Only** *(Verify 8B Geometry)* | Phase 0 only | **0.75 h** | **~\$11** | **1.0 h** | **~\$10** |
| **Lean 2-Arm Pilot** *(Fastest Answer)* | Phases 0–2 + **Arm 0 vs. Arm 2** + Core Evals | **~11.0 h** (`44 GPU-hr`) | **~\$235** (\$154 GPU + \$81 API) | **~16.5 h** (`66 GPU-hr`) | **~\$246** (\$165 GPU + \$81 API) |
| **Full 3-Arm Controlled Study** *(Paper-Ready)* | Phases 0–4 (**Arm 0 + Arm 1 + Arm 2**) | **~15.75 h** (`63 GPU-hr`) | **~\$317** (\$221 GPU + \$96 API) | **~24.0 h** (`96 GPU-hr`) | **~\$337** (\$241 GPU + \$96 API) |
| **Full Study + Cold-Start Ablation** | Full 3-Arm + **Arm 2b** (200 steps) | **~17.5 h** (`70 GPU-hr`) | **~\$352** (\$246 GPU + \$106 API) | **~26.75 h** (`107 GPU-hr`) | **~\$375** (\$269 GPU + \$106 API) |

*(Add a ~15–20% contingency buffer for any debugging/restart overhead; note that using Anthropic's standard non-batch API doubles the API column from ~\$81–\$106 to ~\$160–\$210.)*

---

## 6. Success Criteria & Interpretation Guide

1. **Strong Positive (Publishable Breakthrough)**: Arm 2 beats Arm 0 on Refusal, Eval-Awareness, Sycophancy, and **50%-Mix Taboo** (with `< 2%` cross-organism FP), maintains or improves auditing scores from step `100 → 400`, achieves lower verbatim quoting than Arm 0, beats Arm 1 decisively (proving Fixes 1–3 are essential), and drops on Historical Period (confirming post-training selectivity).
2. **Partial Positive (Diagnosis-Useful)**: Arm 2 beats Arm 0 on `asst_last` safety/eval-awareness/sycophancy, ties or slightly trails on Taboo zero-shot transfer (indicating Base-vs.-Instruct diff distribution differs from LoRA-vs.-Instruct diff distribution → strong empirical motivation for **Tier 2 Population-Conditioned Diff-NLA** across diverse LoRAs!).
3. **Clean Negative**: Even with Fixes 1–3, Arm 2 fails to beat Arm 0 on post-training auditing despite higher probe SNR in Phase 0 — demonstrating that the bottleneck is not representation variance share, but rather **explanation-only AR reconstruction incentive / AV-AR co-adaptation**, pointing toward **Tier 2 (Population-Conditioned AR)** or **Tier 3 (Behavioral Logit-KL AR)**.
