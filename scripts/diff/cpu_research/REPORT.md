Do not launch the three-arm GPU run as designed. On the real `Qwen/Qwen3-8B` and `Qwen/Qwen3-8B-Base` pair at layer 24, the plan's own Gate 0 fails, Fix 1's alignment is fitted to about 0.1% of tokens that have massive activations, and the Taboo mode check passes without measuring anything.

The idea is not dead. At user-content tokens the cleaned diff carries 1.3 to 2.0 times the audit SNR of raw `h_target`, and content controls fall to 0.54 to 0.68 times. Gate 0 and Benchmarks 2 to 4 read `asst_last` instead, where the diff carries 0.57 to 0.71 times, and no layer, wrapper or alignment I tried changes that. Layer 24 is a sound place to read, because swapping Instruct's layer-24 residuals into Base closes 56% to 91% of the next-token gap between the two models. In no-think mode the Taboo diffs carry the secret word at user tokens, where an untrained logit-lens readout picks it from the five candidates on 92% of single prompts.

What I need from you: pick an option under "Decision". My default is to make the four fixes below, gate on user-content positions, and run only the lean two-arm pilot.

## Fix before any GPU run

- [ ] Drop massive-activation rows wherever a row is read: the alignment fit, tau, the Fix 1 FVE check, Phase 2 sampling and Phase 4 pooling (finding 2)
- [ ] Replace the Taboo mode rule and run Benchmark 5 in no-think mode (finding 3)
- [ ] Score Gate 0 and Benchmarks 2 to 4 at user-content positions as well as `asst_last`, and refusal at rollout tokens (findings 1, 4, 7 and 10)
- [ ] Add a logit-lens baseline to Benchmark 5, and also score its user-token diffs rescaled to the training norm (finding 11)

## What I ran

Every number is from Qwen3-8B against Qwen3-8B-Base at layer 24 unless it says otherwise, computed on CPU in fp32. `scripts/diff/phase0_geometry_gate.py` ran unmodified with the runbook's arguments. Only model loading was replaced, by a loader that holds half the model at a time and caches activations on disk; it matches the repo's extractor to 3e-4 relative error on Qwen3-0.6B.

- Benchmark pairs: 100 each for refusal, eval-awareness and sycophancy, plus 60 historical-era items. Refusal pairs come from JailbreakBench because XSTest is gated.
- Chat corpus: 840 conversations (480 UltraChat, 360 PKU-SafeRLHF) with Qwen3-8B's own 48-token rollouts from `phase2_build_corpus.py`, and 20,821 sampled positions.
- "Audit SNR ratio" is Gate 0's statistic: Fisher SNR of the centred, unit-normalised diff divided by Fisher SNR of raw `h_target`. Gate 0 needs a mean of 1.25 and a minimum of 0.85 across the three audit tasks.

## Evidence against

### 1. Gate 0 fails at 8B, and the gate is not what is wrong

The repo script with runbook arguments prints `passed: false`. The alignment leg and the token-leakage leg pass; the SNR leg fails under both Base wrappers.

| Base wrapper | Neutral KL (nats) | Refusal | Eval-awareness | Sycophancy | Mean (needs 1.25) | Min (needs 0.85) | tok1 cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Plain (chosen) | 1.04 | 0.59 | 0.62 | 0.84 | 0.68 | 0.59 | 46% |
| ChatML | 3.06 | 0.55 | 0.54 | 0.63 | 0.57 | 0.54 | 55% |

I then recomputed the statistic on all 100 pairs with pair-bootstrap intervals and tried each way the result could be an artifact.

| Check at `asst_last` | Refusal | Eval-awareness | Sycophancy | Mean |
| --- | --- | --- | --- | --- |
| Planned pipeline, Plain | 0.53 [0.48, 0.59] | 0.61 [0.58, 0.66] | 0.93 [0.86, 1.02] | 0.69 |
| Planned pipeline, ChatML | 0.55 [0.53, 0.58] | 0.56 [0.53, 0.62] | 0.64 [0.60, 0.71] | 0.58 |
| `h_target` alone, same centring and normalisation | 1.00 | 1.10 | 1.05 | 1.05 |
| Alignment refitted without massive tokens, Plain | 0.51 | 0.73 | 0.89 | 0.71 |
| Best of four alignments fitted on chat activations, Plain | 0.32 | 1.10 | 0.54 | 0.65 |
| Best of four alignments fitted on chat activations, ChatML | 0.66 | 0.76 | 0.62 | 0.68 |

- The normalisation is not the cause. Centring and unit-normalising `h_target` without any Base model gives 1.05.
- A better alignment makes it worse. Maps fitted on chat activations explain 72% to 80% of `h_target` at `asst_last` under Plain, against 16% for the web-fitted map, and they keep only 11% to 14% of the refusal signal.
- No layer fixes it. With Base on the same ChatML tokens as Instruct, the naive diff never exceeds 0.78 at any of the 36 layers. Under Plain it clears both SNR legs at layers 3 to 13, but sycophancy carries that pass (2.1 to 3.5, with refusal near 1.0), and there the Base model alone has 3.6 to 6.3 times Instruct's sycophancy SNR. The wrapper produces that pass, not post-training.

![Diff SNR by layer](https://ampcode.com/artifacts/visuals/diff-snr-by-layer-06d4c753-f38d-40d1-b243-478047c16737)

- It holds for behaviour, not only prompt category. On PKU-SafeRLHF prompts I labelled Qwen3-8B's own rollouts as explicit refusals (155) or plain compliance (95) and left 110 hedged answers out. At `asst_last` the diff's SNR is 0.54 times raw `h_target` with the planned alignment and 0.61 with the masked refit.
- The diff is not selective at `asst_last`. Content labels (topic halves of the same prompts, historical era) get ratios of 0.76 to 0.84 under Plain, above the audit mean of 0.69.
- The diff does not lose decodable information either. A difference-of-means probe trained on four pairs scores 0.91, 0.96 and 0.96 on `h_target` and 0.89, 0.96 and 0.96 on the diff.

The mechanism differs by wrapper. Base under Plain already separates the classes (85% of the refusal signal and 93% of the sycophancy signal, in a direction with cosine 0.58 to 0.71 to Instruct's), so subtraction removes signal. Base under ChatML carries 2% to 20% of the signal, so the diff keeps it, but within-class spread rises 1.75 to 2.13 times.

The 0.6B pilot result in the plan does not reproduce with the committed benchmark pairs: at layer 18 the repo script gives 0.90 under Plain (chosen) and 1.12 under ChatML, where the plan reports 1.4 to 2.5.

### 2. Fix 1 is fitted to massive activations and mostly writes one coordinate

At layer 24, 36 of 25,200 web validation tokens (and 116 of 100,797 training tokens) have norms above five times the median. The validation ones have norms of 12,000 to 16,000 against a median of 264, are mostly the token `2` inside dates, and carry 82% of `h_target`'s variance. Base and Instruct fire on different tokens: 9 of the 36 are Instruct-only and 15 are Base-only.

| Map | Web-val FVE, all tokens | Web-val FVE, ordinary tokens | Median residual norm (tau) |
| --- | --- | --- | --- |
| Identity | 0.195 | 0.886 | 62.5 |
| Identity-anchored ridge, as Phase 0 fits it | 0.395 | 0.851 | 63.8 |
| Identity-anchored ridge, massive tokens masked | 0.293 | 0.926 | 49.5 |

Phase 0's map is worse than no map on ordinary tokens. On the chat corpus it explains 74% of `h_target` at rollout tokens, where plain identity explains 82%.

The damage reaches the Arm 2 target. 87.2% of the fitted correction's squared norm writes to residual coordinate 2276. In the centred, unit-normalised Arm 2 target that one coordinate is 96% to 98% of the largest direction, which holds 20% of the variance on `user_content` rows and 28% on `asst_rollout` rows. The target's participation ratio is 21 and 12 there, against 158 and 160 for `h_target`.

With massive tokens masked, the largest coordinate takes 1.6% of the correction and the participation ratios are 84 and 114. None of the 20,821 sampled chat rows is itself a massive activation, so the artifact enters through the web-fitted map.

The masked refit leaves a smaller nuisance direction. On `user_content` rows its largest direction holds 9% of the variance and follows token position (Spearman 0.88), not the size of the behavioural gap (Spearman -0.26 with `KL(Instruct || Base)`).

The massive rows are one vector at a few sink tokens. Position 0 is massive in every web passage and every chat conversation in both models, and the repo already skips it. The 116 training rows are the same thing at later positions: 99% of their energy is in coordinate 2276, their pairwise cosine averages 0.997, and the direction is the same in both models (cosine 0.9998). No training row has a norm between 1.5 and 21 times its model's median, so any threshold from 2 to 20 selects the same rows.

The two models disagree on where it fires, and that disagreement is the fit's objective. Both fire on 62 of the 116 rows, only Instruct on 14 and only Base on 40. On those 54 rows the regression target `h_target - h_base` is about 13,000 in coordinate 2276, with either sign, against a median norm of 62 on ordinary rows. The 54 rows carry 92.6% of the objective, all 116 carry 95.6%, and the other 100,681 rows carry 4.4%.

The map cannot predict where a sink appears, and every ordinary token pays for the attempt. Its row for coordinate 2276 has 21 times the norm it has after masking, and it still leaves about 8,400 of the 13,000 unexplained on the rows it was fitted to. On ordinary validation tokens it adds a correction with a standard deviation of 54.5 to that coordinate, where the Instruct-minus-Base difference has a standard deviation of 8.5. Half of the planned diff's variance on ordinary tokens is that one coordinate.

Qwen3-0.6B at layer 18 has one such row in 100,797 web tokens and none in the chat corpus after position 0, so the 0.6B pilot could not show this.

I fitted six treatments of these rows with the same ridge and the same alpha.

| Treatment | FVE, ordinary validation tokens | tau | Share of the diff's variance in coordinate 2276 | Norm of a massive row that reaches the diff, one model firing | Arm 2 participation ratio (`user_content`, `asst_rollout`) |
| --- | --- | --- | --- | --- | --- |
| None, as Phase 0 fits it | 0.851 | 63.8 | 50.3% | 145 tau | 21, 12 |
| Drop rows above 5 times the median norm | 0.926 | 49.5 | 1.3% | 218 tau | 84, 114 |
| Scale high-norm rows down to 2 times the median | 0.926 | 49.5 | 1.3% | 162 tau | 84, 115 |
| Huber loss on the residual, no norm information | 0.926 | 49.5 | 1.4% | 265 tau | 84, 114 |
| Trim the 0.5% of rows with the largest residual, no norm information | 0.926 | 49.4 | 1.3% | 219 tau | 84, 113 |
| Project the massive direction out of both models | 0.925 | 49.3 | 0% | 3.9 tau | 85, 123 |

Every treatment gives the same map on ordinary tokens, so the choice does not rest on fit quality. The treatments differ in what a massive row looks like when it reaches the diff after the fit: 145 to 265 tau under the row-level treatments, and 3.9 tau when the direction is projected out.

Massive rows do reach the diff. Scanning every position of the 840 chat conversations finds 4 to 6 massive tokens after position 0 per model and wrapper, in 94,000 to 120,000 tokens, with 1 or 2 at positions Phase 2 can sample. One of the 660 benchmark prompts has one in its user turn. Removing that token moves the sycophancy user-token ratios in finding 7 by at most 0.08.

Fix: drop rows above five times the median norm, in either model, everywhere a row is read. That means the alignment fit and its CV, tau, the FVE that Fix 1 is judged on, the role means, Phase 2's sampled rows, and the token rows Phase 4 pools into a user-token mean. It is the repo's position-0 rule extended to sinks at later positions. Projecting the direction out is the alternative if you prefer not to carry a row filter through Phases 2 and 4; it costs the 1.3% of the diff's variance that ordinary tokens have in that coordinate.

### 3. The Taboo mode check passes vacuously and sends Benchmark 5 off-distribution

With `enable_thinking=True`, Instruct and both LoRAs emit `<think>` at `asst_last` with probability above 0.9999 on the neutral prompts, so their KL is near zero whatever the LoRA has learned.

| Organism | KL, no-think | KL, think at `asst_last` | KL, think at first reasoning token |
| --- | --- | --- | --- |
| `taboo-snow` | 2.83 | 0.0003 | 0.03 |
| `taboo-blue_50_mix` | 4.66 | 0.0001 | 1.52 |

Phase 0 therefore recommends thinking mode and reports `taboo_kl_lt_3: true`. In no-think mode, which is what every arm trains on, `blue_50_mix` fails the 3.0 nat limit.

Phase 4 uses the same rule (`best_et = kls[True] < kls[False]`, `phase4_eval_suite.py` line 499), so it would extract Benchmark 5 in thinking mode. Instruct's thinking-mode `asst_last` is 3.7 tau from its no-think `asst_last`, with cosine 0.43, and the mean `Org - Inst` diff in the two modes has cosine 0.42 (snow) and 0.28 (blue_50_mix). No arm trains on that position.

Fix: compare modes at the first content token, or fix Benchmark 5 to no-think and report the 4.66 nat mismatch for `blue_50_mix` as a caveat.

### 4. The headline benchmarks read the position where the diff is weakest

`phase4_eval_suite.py` extracts Benchmarks 2 to 4 at `asst_last` only (line 389), and both negative controls at `asst_last` (lines 423 and 431). That is the position in finding 1. Training samples one row per role, so one third of Arm 2's rows are `asst_last`.

### 5. Fix 3 changes almost nothing, and the diff's norm does not track behaviour

After centring, Arm 2's chat rows sit at 1.4, 1.8 and 1.4 times tau (median, `user_content`, `asst_last`, `asst_rollout`). The soft-norm weights run from 0.33 to 0.97, and the effective sample size is 98%, 100% and 96%. With the masked refit it is 99%, 100% and 98%.

Fix 3 treats the norm of the cleaned diff as the amount of post-training signal at a position. I computed both models' exact next-token distributions at all 20,821 corpus positions to test that.

| Role | Median `KL(Instruct \|\| Base)`, nats | Spearman of diff norm with KL | Same, masked refit | Fix 3 weight on the quarter of rows with the highest KL |
| --- | --- | --- | --- | --- |
| `user_content` | 0.42 | 0.21 | 0.28 | 26% |
| `asst_last` | 1.55 | -0.05 | 0.06 | 25% |
| `asst_rollout` | 0.18 | 0.04 | 0.10 | 26% |

Uniform weights would put 25% on that quarter. The diff's direction does not predict the size of the gap better than raw activations either: held-out R² for log KL is 0.51, 0.59 and 0.44 from the Arm 2 target, and 0.60, 0.58 and 0.55 from centred `h_target`.

The norm does not track the causal effect of the residual difference either. When Instruct's upper blocks read Base's layer-24 residuals (finding 9), the Spearman correlation of the diff's norm with the KL that swap produces is 0.22, 0.04 and -0.01 by role.

Qwen3-0.6B at layer 18 gives the same answer: Spearman 0.25, 0.15 and 0.15 with `KL(Instruct || Base)`, and 27%, 25% and 26% of the Fix 3 weight on the top quarter.

### 6. The diff depends on how Base is prompted

The cosine between the centred diff under ChatML and under Plain, at identical Instruct tokens, is 0.69 on `user_content`, 0.41 at `asst_last` and 0.86 on `asst_rollout`. At `asst_last` the two diffs mostly disagree, so the target there depends on the wrapper choice. Base under ChatML predicts `h_target` there worse than a constant does (within-role FVE of -0.43).

## Evidence for

### 7. At user-content tokens the diff favours audit concepts over content

The same pairs, scored at the last user-content token and at the mean over user-content tokens. Intervals are pair-bootstrap.

| Alignment | Base wrapper | Position | Refusal | Eval-awareness | Sycophancy | Mean audit | Mean content control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `h_target` alone | | last user token | 1.01 | 1.06 | 1.04 | 1.03 | 1.03 |
| Planned | Plain | last user token | 1.08 [0.90, 1.35] | 2.21 [1.88, 2.55] | 1.04 [0.96, 1.16] | 1.45 | 0.63 |
| Planned | Plain | user-token mean | 0.60 [0.54, 0.89] | 1.39 [1.27, 1.59] | 2.03 [1.79, 2.33] | 1.34 | 0.68 |
| Planned | ChatML | last user token | 1.68 [1.15, 2.06] | 2.50 [2.16, 2.94] | 1.73 [1.41, 2.13] | 1.97 | 0.59 |
| Planned | ChatML | user-token mean | 0.87 [0.70, 1.19] | 1.26 [1.10, 1.52] | 2.83 [2.19, 3.47] | 1.65 | 0.54 |
| Masked refit | Plain | last user token | 1.46 [1.18, 1.66] | 1.97 [1.77, 2.17] | 1.01 [0.93, 1.09] | 1.48 | 0.66 |
| Masked refit | Plain | user-token mean | 0.72 [0.67, 0.95] | 1.89 [1.72, 2.09] | 2.46 [2.22, 2.66] | 1.69 | 0.60 |
| Masked refit | ChatML | last user token | 2.09 [1.49, 2.33] | 2.02 [1.76, 2.33] | 0.81 [0.73, 0.93] | 1.64 | 0.65 |
| Masked refit | ChatML | user-token mean | 1.02 [0.86, 1.23] | 1.69 [1.46, 1.97] | 1.97 [1.80, 2.18] | 1.56 | 0.55 |

This is the pattern the plan predicts: the mean audit ratio is above 1.25 in all eight rows and the content controls are below 0.7 in all eight. At the user-token mean the diff removes 64% to 80% of the within-class spread and keeps 32% to 91% of the eval-awareness and sycophancy signal.

Three things limit it:

- Refusal is not helped at the user-token mean (0.60 to 1.02), and single-token results move between alignments. Refusal behaviour on the model's own rollouts is not helped at user tokens either (finding 10).
- The raw signal is small at these positions. `h_target`'s Fisher SNR at the last user token is 0.04, 0.14 and 0.52, against 1.64, 0.58 and 0.75 at `asst_last`. The best diff value anywhere beats raw `h_target` at `asst_last` only for sycophancy (0.90 against 0.75).
- These labels are prompt categories, and Base encodes them too. Base alone has 0.9 to 2.4 times `h_target`'s SNR at user positions.

The gain is not specific to layer 24. With the naive diff at the last user token, both SNR legs clear at 30 of 36 layers under ChatML and at 33 of 36 under Plain, and the content controls stay between 0.62 and 0.86 from layer 3 to layer 34. The chart under finding 1 shows this on its "Last user token" tab.

The same sweep sharpens the third limit. At layer 24 the naive diff gives 1.89 (ChatML) and 1.65 (Plain), and Base alone gives 1.69 and 1.60. From layer 7 to layer 22 Base alone scores higher than the diff under both wrappers. A user-position gate therefore shows that the diff is selective, and it does not show that the signal comes from post-training.

### 8. On content tokens the diff removes most prompt variance and most token-identity leakage

- The aligned Base model explains 64% and 74% of `h_target`'s within-role variance on `user_content` and `asst_rollout` rows (72% and 84% with the masked refit). At `asst_last` it explains 14% (27%).
- Token-identity (`tok1`) FVE falls from 0.29 to 0.11 on `user_content` and from 0.23 to 0.09 on `asst_rollout`.
- A separate reader model on the same text (Qwen3-0.6B, layer 18) linearly recovers 57% of the Arm 0 target and 28% of the Arm 2 target on `user_content`, held out by conversation. That is less surface text to quote, and also a lower ceiling for the reconstructor.
- The same holds for what the contrastive teacher is shown. The two models' next-token states linearly explain 37%, 71% and 35% of the Arm 2 target by role, against 61%, 85% and 59% of the Arm 0 target. Expect a lower warm-start FVE in Arm 2 for that reason alone.

### 9. Most of the behavioural gap is already in the layer-24 residual stream

I ran both models on identical ChatML tokens and swapped whole layer-24 residual streams between them. KL is measured from Instruct's own next-token distribution, in nats, as a mean over rows.

| Role | Rows | Full gap (Base) | Base's upper blocks on Instruct's residuals | Gap left | Gap left at 0.6B, layer 18 |
| --- | --- | --- | --- | --- | --- |
| `user_content` | 9,901 | 0.48 | 0.21 | 44% | 28% |
| `asst_last` | 840 | 3.91 | 0.34 | 9% | 3% |
| `asst_rollout` | 10,080 | 0.41 | 0.18 | 42% | 48% |

The layer-24 difference accounts for 56% to 91% of what the two models do differently at the next token, so a description of that difference has behaviour to describe. The blocks above layer 24 add the rest, which no layer-24 target can show.

The share is largest at `asst_last`, where the diff has the least concept-level SNR (finding 1). The two results measure different things: the swap measures the whole next-token gap, and the gate measures how well one concept separates.

### 10. On the answer's tokens the diff separates refusals from compliance, unless Phase 0's alignment is used

Labels are Qwen3-8B's own rollouts on PKU-SafeRLHF prompts (155 refusals, 95 compliances), with one pooled vector per conversation and role. Cells are the diff's Fisher SNR over raw `h_target`'s.

| Role | `h_target` Fisher SNR | Planned, Plain | Masked refit, Plain | No map, Plain | Planned, ChatML | Masked refit, ChatML | No map, ChatML |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `user_content` mean | 0.09 | 0.45 | 0.59 | 0.59 | 0.39 | 0.57 | 0.58 |
| `asst_last` | 1.86 | 0.54 | 0.61 | 0.53 | 0.53 | 0.57 | 0.51 |
| `asst_rollout` mean | 0.66 | 0.81 [0.73, 0.91] | 1.11 [1.01, 1.20] | 1.15 [1.03, 1.25] | 1.00 [0.82, 1.25] | 1.31 [1.15, 1.46] | 1.42 [1.23, 1.58] |

- On rollout tokens the diff beats raw `h_target` by 11% to 42% with the masked refit or with no map at all. Phase 0's map loses that gain (0.81 and 1.00), which is a second reason for the masking fix.
- Before the answer starts, the diff does not predict the model's own refusal better than `h_target` (0.39 to 0.61). The user-token gain in finding 7 is for eval-awareness and sycophancy prompts.
- Raw `h_target` at `asst_last` (1.86) is still the best single place to read refusal behaviour. The best diff value is 0.94, on rollout tokens.

### 11. In no-think mode the Taboo diffs carry the secret word at user tokens

For all ten default organisms (five words, each as a narrow and a 50%-mix LoRA) I took the `Org - Inst` diff that Benchmark 5 feeds Arm 2, on the first 20 elicitation prompts with `enable_thinking=False`. Norms are under the planned alignment's tau.

| Position | Norm / tau | Arm 2 training rows at that role, 5th to 95th percentile | Organism ID, 10-way | Same word across the two regimes, 5-way | Logit lens on one prompt: secret word first of 5 | Energy in one direction shared by all organisms |
| --- | --- | --- | --- | --- | --- | --- |
| User-token mean | 0.49 | 1.00 to 2.22 | 100% | 100% and 100% | 91.5% | 44% |
| `asst_last` | 2.79 | 1.44 to 2.27 | 94.5% | 73% and 95% | 48.0% | 46% |

- The secret is linearly present at user tokens. I read one prompt's diff through Instruct's final-norm gain and unembedding, with no training and no other organism for reference. The secret word ranks first of the five on 91.5% of prompts, where chance is 20%.
- For the narrow LoRAs the mean diff's top tokens name the concept: `笑容` and `微笑` for smile, `船只` and `maritime` for ship, `prized` and `金价` for gold, `的颜色` and `色彩` for blue, `glaciers` and `storms` for snow. The 50%-mix LoRAs are noisier, and `ship_50_mix` is the one organism the readout mostly misses (35%).
- The narrow and 50%-mix LoRAs of a word move activations the same way. Centroids from one regime classify the other at 100%.
- The diffs are partly inside the training distribution in direction. The top 100 principal directions of the Arm 2 target hold 30% of a Taboo diff, against 55% of a held-out training target and 2.4% of a random direction.

Benchmark 5 needs three adjustments to use this:

- At the user-token mean the Taboo diffs are smaller than every one of the 9,901 training rows. Fix 3's hook injects them at 0.44 of full strength, against 0.82 for the median training row. Score them a second time rescaled to the median training norm.
- Report the two positions separately and lead with the user-token mean. At `asst_last` the readout gives 48%, or 36% without the two blue organisms, and the diffs are larger than 99.8% of training rows there.
- Add the logit-lens readout as a baseline. Without it, an Arm 2 win at user tokens does not show that the verbalizer adds anything.

The readout is a five-way choice, and Benchmark 5 asks for the word open-vocabulary. On one prompt the secret word's median rank among about 152,000 tokens is 24 to 392 for the narrow LoRAs and 1,174 to 28,749 for the 50%-mix LoRAs.

The `org_minus_base` mode that Benchmark 5 also scores puts the same LoRA change on top of an Instruct-minus-Base diff about three times its size (0.49 against a training median of 1.41 tau), if the Taboo prompts behave like the chat corpus.

## The 0.6B pair shows the same split between positions

Qwen3-0.6B against Qwen3-0.6B-Base at layer 18, same 100 pairs per task, Phase 0's alignment.

| Position | Base wrapper | Refusal | Eval-awareness | Sycophancy | Mean audit | Mean content control |
| --- | --- | --- | --- | --- | --- | --- |
| `asst_last` | `h_target` alone | 1.10 | 1.13 | 1.07 | 1.10 | 1.04 |
| `asst_last` | Plain | 0.85 [0.74, 0.96] | 1.25 [1.15, 1.35] | 0.80 [0.74, 0.87] | 0.96 | 0.84 |
| `asst_last` | ChatML | 1.07 [1.00, 1.13] | 1.36 [1.26, 1.45] | 1.08 [1.03, 1.13] | 1.17 | 1.03 |
| last user token | Plain | 0.92 [0.76, 1.23] | 2.79 [2.53, 3.05] | 0.98 [0.90, 1.07] | 1.56 | 0.83 |
| last user token | ChatML | 0.78 [0.71, 1.17] | 3.25 [2.91, 3.65] | 0.79 [0.73, 0.90] | 1.61 | 0.70 |
| user-token mean | Plain | 0.69 [0.59, 0.95] | 1.94 [1.77, 2.10] | 1.85 [1.70, 2.01] | 1.50 | 0.49 |
| user-token mean | ChatML | 0.59 [0.54, 0.83] | 1.62 [1.45, 1.80] | 2.09 [1.85, 2.32] | 1.43 | 0.39 |

At `asst_last` the diff does no better than centring and normalising `h_target` (1.10), and under ChatML it is not selective (content 1.03). At user tokens the mean audit ratio is 1.43 to 1.61 with content controls at 0.39 to 0.83. Refusal is not helped at any position (0.59 to 1.07). Alignments fitted on chat activations do not help at `asst_last` here either: the best mean is 1.00.

## Decision

| Option | What it costs | Why |
| --- | --- | --- |
| A. Four fixes, gate on user positions, lean two-arm pilot (my default) | The plan's lean pilot, about $235 and 44 H200 GPU-hours | The run is cheap, and with the fixes a negative result means something. Expect no Arm 2 advantage on the `asst_last` benchmarks. Benchmark 5 at user tokens is where Arm 2 is most likely to show one. |
| B. Drop the experiment | Nothing | Defensible if the `asst_last` benchmarks are the result you care about: nothing here predicts an Arm 2 win there. |
| C. Run as designed with `--no-fail-exit` | About $317 and 63 GPU-hours | Rejected. Arm 2 would train on a target whose largest direction is an alignment artifact, and Benchmark 5 would read an untrained position. |

Without an answer I will leave `diff-stuff` as it is and make no further changes.

## Risks

- The GPU pipeline runs in bf16 and I ran fp32. Measured afterwards with `bf16_fidelity.py` on CPU: bf16 moves the aligned diff by a median of 2.6% at user-content tokens and 5.4% on web tokens (cosine 0.9997 and 0.9986 with the fp32 diff), and the refusal SNR ratio by less than 1%. That used CPU bf16 kernels, so the remaining check is Phase 0 on GPU (about $11), which prints the same table.
- Refusal pairs are JailbreakBench, not XSTest. The earliest signal is the per-task refusal ratio in that Phase 0 report.
- Fisher SNR measures variance share, not what a trained verbalizer writes. No NLA was trained here, so finding 7 may not turn into better explanations, and finding 1 may not turn into worse ones.
- The chat corpus is 840 conversations with HF sampling in place of vLLM. Per-role numbers rest on 840 to 10,080 rows.
- The layer sweep uses the naive diff, since the fitted map exists only for layer 24. At layer 24 the two agree (0.65 and 0.69 under Plain).
- The residual swap in finding 9 needs both models on identical tokens, so it uses the ChatML wrapper for Base. Phase 0 chooses Plain, where the tokens do not line up.

## Reproduce

Scripts, result JSONs and this report are on branch `diff-stuff-cpu-research` under `scripts/diff/cpu_research/`, on top of `diff-stuff` at `89ede1b`. Nothing in the pipeline is changed there.

```bash
python scripts/diff/cpu_research/make_report_tables.py scripts/diff/cpu_research/results
```

That prints every table above from the JSONs. The README in that directory explains how the 8B models were run in 31 GB of memory and lists each script.

The massive-activation scripts (`massive_anatomy.py`, `massive_fix_variants.py`, `massive_coordinate_detail.py`, `massive_in_benchmarks.py`) and their result JSONs are committed in the same directory.
