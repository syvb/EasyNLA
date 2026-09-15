# Future Oracles: reading a model's next tokens from one activation vector

*Project report, 2026-09-14 to 2026-09-15. Code: EasyNLA branch `sv/future-rl` (`nla/future_lens/`,
`scripts/runpod_future_lens.py`, `spaces/future_oracle/`). Data, adapters, evals and logs:
[`syvb/rl-future-lens-qwen3-8b`](https://huggingface.co/datasets/syvb/rl-future-lens-qwen3-8b) (public).
Demo: [`syvb/future-oracle`](https://huggingface.co/spaces/syvb/future-oracle). Metrics: wandb project
`rl-future-lens`.*

## 1. Question and approach

Future Lens (Pal et al., 2023) asked how much of a language model's *future* output is already
present in a single hidden state, and answered it with a learned soft prompt that reads the state
into the next token. We asked the same question with a much stronger decoder: the model itself,
fine-tuned with a LoRA to verbalise a vector injected into its input, in the style of EasyNLA's
activation verbalizers. We call the resulting decoders **Future Oracles**.

Setup, identical across every run unless stated:

- **Target = decoder.** Qwen3-Base checkpoints (0.6B, 1.7B, 8B), plain-text prompt, no chat template.
  A LoRA (r=64, alpha=16, rsLoRA) on attention and MLP projections is the only trained part.
- **Injection.** The residual-stream vector `h` (output of block `layer` at token `t`) is
  unit-normalised, scaled to the layer's p75 activation norm times 2, and written into the input
  embedding of a marker token inside the prompt
  `Here is an activation vector from layer {layer} of a language model: <concept>㈎</concept>\nOutput the next {k} tokens the model will produce after this point.\n`.
- **Labels.** The target's *own* greedy continuation from token `t` (the Future Lens convention),
  not the document text; distillation of the target's stored top-64 next-token distribution at each
  label position (soft cross-entropy with an exact remainder bucket) as the training objective.
- **Data.** fineweb sample-10BT, 5.5k training documents and 300 held-out documents, 40 positions per
  document, positions kept only where the target's next-token prediction is correct (as in Future
  Lens). Held-out documents are disjoint by construction; 32-gram overlap between splits is 0.05%.
- **Training.** 8000 steps at effective batch 64 (about 2.5 epochs of one layer's positions),
  lr 1e-4 cosine to 1e-5, 600 warmup steps, one seed.
- **Metrics.** `p1`: free-running precision at offset N (greedy decode of N+1 tokens, score token N
  against the label). `tf_p1`: teacher-forced precision (label prefix fed, argmax at position N),
  Future Lens's number. Controls: `shuffled` (another position's vector), `none` (marker embedding
  untouched), `wrong_layer` (an early layer the decoder never trained on), `cross_layer` (the
  farthest trained layer's vector). Baselines: n-grams, linear probes, and `target_window`: the
  frozen target given only its last m tokens, which converts oracle precision into "tokens of
  context the vector is worth".

## 2. What was done, in order

1. **8B collection and alpha check.** Collected layers 4..32 (step 4). The injection scale was
   swept over 0.5 to 4 times the p75 norm with true and shuffled labels: the true-minus-shuffled
   readout-CE gap was flat at about 4.4 nats everywhere, so the scale does not matter in this
   range; 2 times was kept.
2. **Filter ablation.** Training and evaluating with versus without the top-1-correct filter (on
   the chat model with text labels, before the design was finalised): the filter halves N=0 by
   construction and changes N>=1 by 1 to 7 points. Kept, to match Future Lens.
3. **Design changes** made along the way, all approved: base model with a plain prompt instead of
   the chat model; greedy labels instead of text labels; distillation of the full top-64
   distribution instead of hard labels; every eval position subsampled with a fixed seed.
4. **8B multi-layer decoder** (one adapter, seven layers, hard-label and distilled variants) with
   full evals, controls and baselines.
5. **Future Lens reproduction.** Ported the paper's learned-prompt decoder to Qwen3-8B (10 soft
   tokens per layer, transplant hook, KL training) at matched data, and a compute-matched 10x
   variant.
6. **GRPO on top of the SFT decoder** (Dr. GRPO, exact-match reward, KL to the SFT adapter).
   Stopped after 90 minutes on the pre-declared "is it beating SFT per GPU-hour" check.
7. **Single-layer size scan** (0.6B, 1.7B, 8B), one decoder per model at matched relative depth,
   scored on shared held-out positions.
8. **Public release**: dataset repo made public after an automated secret scan, a ZeroGPU Space
   that auto-discovers every trained oracle, and a full sync before deleting the RunPod volume.

Total compute spend about $115 on RunPod H100s.

## 3. Results

### 3.1 The 8B multi-layer decoder (data_base_v2, 2000 positions per layer)

| layer | N=0 p1 / tf | N=1 p1 / tf | N=2 p1 / tf | N=3 p1 / tf |
|---|---|---|---|---|
| 8 | 0.608 / 0.608 | 0.341 / 0.469 | 0.202 / 0.471 | 0.152 / 0.507 |
| 16 | 0.718 / 0.719 | 0.433 / 0.539 | 0.293 / 0.549 | 0.205 / 0.559 |
| 20 | 0.753 / 0.750 | 0.482 / 0.569 | 0.326 / 0.565 | 0.232 / 0.557 |
| **24** | 0.845 / 0.846 | 0.555 / 0.617 | **0.386 / 0.588** | **0.259 / 0.569** |
| 28 | 0.871 / 0.870 | 0.565 / 0.617 | 0.353 / 0.554 | 0.247 / 0.564 |
| 32 | 0.904 / 0.903 | 0.573 / 0.607 | 0.341 / 0.530 | 0.230 / 0.540 |

Layer 24 of 36 (two thirds depth) is best for every offset beyond the next token; N=0 keeps
improving to the last layer, as Future Lens found. Shuffled and none controls sit at about 0.01,
the wrong-layer control at 0.31 / 0.13 / 0.07, the cross-layer control at 0.50 / 0.25 / 0.15 for
N=1..3. Distillation matched hard labels on precision and cut the decoder's KL to the target from
1.7 to 0.9 nats. Teacher-forced precision is nearly flat in N (0.55 to 0.62) while free-running
precision decays, the usual exposure effect.

**Against Future Lens.** The paper's prompt decoder, ported and trained on the same data, reaches
tf_p1 0.46 at N=1 on layer 24, matching the paper's GPT-J numbers; ten times more steps only moves
it to 0.476. The oracle reaches 0.617 at matched data, so the paper's method was capacity-limited
by about 15 points, not data-limited.

**Baselines (8B).** Bigram and 4-gram: 0.19 to 0.23 at N=1. Linear probe on the layer-24 vector:
0.28 / 0.14 / 0.09. Frozen target given its last 8 tokens: 0.34 / 0.21 / 0.16; last 32 tokens:
0.62 / 0.48 / 0.39. Probes for the *past* recover x_{t-1} at 0.41 and x_{t-2} at 0.20, so the vector
does not hold the literal context.

### 3.2 GRPO

From the distilled adapter, exact-match reward, 1700 steps in 90 minutes: greedy precision was
flat (0.49 / 0.33 / 0.23 to 0.48 / 0.34 / 0.24), the sampled hit rate rose (the policy sharpened),
and KL to the target's top-k grew from 0.90 to 1.03. The last hour of SFT had given 5 to 8 points.
Killed; not pursued further.

### 3.3 Size scan (single-layer oracles, shared held-out positions)

Each model gets one oracle at two-thirds depth (layer 24/36 for 8B, layer 19/28 for 0.6B and
1.7B), trained identically. The held-out split was collected unfiltered so that the same 40
positions per document exist for every model; each oracle is scored on the positions where its own
model is top-1 correct, and all three on the intersection.

Intersection (4058 positions, standard error about 0.007):

| oracle | N=0 | N=1 | N=2 | N=3 |
|---|---|---|---|---|
| Qwen3-0.6B, layer 19 | 0.811 | 0.509 | 0.321 | 0.210 |
| Qwen3-1.7B, layer 19 | 0.866 | 0.543 | 0.355 | 0.245 |
| Qwen3-8B, layer 24 | 0.923 | 0.606 | 0.408 | 0.294 |

Effective context window m*, the number of recent tokens the frozen model needs to match the
oracle's free-running precision:

| oracle | N=1 | N=2 | N=3 |
|---|---|---|---|
| 0.6B | 19 | 15 | 13 |
| 1.7B | 19 | 17 | 16 |
| 8B | 26 | 20 | 17 |

Every step in size improves every offset by 4 to 10 standard errors. Controls are at 0.01 for all
sizes; the wrong-layer control collapses past N=0. The single-layer 8B oracle beats the multi-layer
adapter at the same layer on the same split (0.584 / 0.399 / 0.285 vs 0.561 / 0.383 / 0.274 at
N=1..3), so the joint adapter had been capacity-limited by about two points. Training saturates by
step 6000 for every size.

Full tables, JSON and the figure: `evals/size_scan/` in the dataset repo; generated by
`scripts/fl_size_scan_report.py`.

## 4. Interpretation

One mid-depth residual vector carries a real, size-dependent amount of the model's own future: for
the 8B it is worth about 26 tokens of context one step ahead and 17 tokens three steps ahead. The
oracle is a much better reader than the Future Lens prompt, so earlier estimates of "how much future
is in a state" were decoder-limited. What the experiment does not settle is whether that future is a
*plan* the model executes or a compact *summary* of the context from which the decoder, being the
model, re-predicts. The weak past-token probes lean toward the former; the decisive test is causal
(patch the vector, see whether the model's real continuation follows the oracle's reading), and it
has not been run.

## 5. Lessons and pitfalls (for whoever runs this next)

- **Collection is the expensive stage.** Greedy rollouts with top-64 targets on every training
  position cost 0.6 to 2 documents per second, about 1.7 h for 5.5k documents on the 8B. An early
  estimate based on a rollout-free collection was off by 10x and cost $8 in aborted pods.
- **Positions must be matched across models.** The top-1 filter selects different positions for
  different models; collect the eval split unfiltered and let the loader filter per model.
- **doc_idx is a per-collection counter**, not a corpus index; join collections on the doc_id
  string in `docs.parquet`. An eval-only collection that starts at the wrong corpus offset overlaps
  the training documents.
- **H100 NVL pods are power-capped** (310 W) and ran the 8B SFT 2.4x slower than SXM. The launcher
  now prefers SXM and can hand evals to a second pod.
- **The network volume was the bottleneck** for GPU stock, not the GPUs. Pods that fetch inputs
  from HF and push outputs back can land anywhere; the volume was synced and deleted.
- **wandb is metrics-only; every file goes to the HF dataset repo**, uploaded by each pod as it
  finishes a stage, and the repo layout doubles as the Space's oracle registry.

## 6. Open questions worth a next push

1. **Causal test.** Patch a layer-24 vector from context A into context B and check whether the
   model's actual continuation follows what the oracle reads from it.
2. **Answer-commitment timing.** Read the state at every token of a question or chain of thought
   and detect when the final answer is already present; poetry, code and arithmetic are where plan
   and text diverge.
3. **Higher-level readouts.** Train the same decoder to describe what the response is going to do,
   not just its next tokens.
4. **Scaling curve.** More sizes and layer profiles at about $10 each; effective window as the
   y-axis; instruct and thinking-mode variants against base.
5. **Not worth pursuing:** RL on the readout and the teacher-forcing gap; it is decoder engineering
   and bought nothing.

## 7. Artefacts

| what | where |
|---|---|
| data splits | `data_base_v2/` (8B, 8 layers), `data_0.6b_L19/`, `data_1.7b_L19/`, `data_8b_evalu/` (unfiltered 8B eval) |
| oracles | `ckpts/sft_replace_embed_a2.0_greedy_distill{,_8b_L24,_1.7b_L19,_0.6b_L19}/iter_0008000` |
| other adapters | alpha sweep, filter ablation, hard-label 8B, the killed RL run |
| evals | `evals/` (8B multi-layer, Future Lens port, baselines), `evals_{size}_L{layer}/`, `evals/size_scan/` |
| logs | `logs/` (every pod's stdout) |
| demo | Space `syvb/future-oracle`: click a token, every oracle reads it; `/probe` API |
| code | `nla/future_lens/` (collect, train_rl, eval, baselines, futurelens_prompt, plots), `nla/train_sft.py --future-lens --distill`, `scripts/runpod_future_lens.py`, `spaces/future_oracle/` |
