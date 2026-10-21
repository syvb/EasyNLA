# Head-contribution NLA data (steps 1–2)

Data for an NLA that explains **what one attention head added at one token**:
per-head writes and residuals from Qwen3-8B, plus warm-start descriptions from
Claude Haiku 5.5. Code: `nla/datagen/heads/`. Config:
`configs/datagen/heads_qwen3_8b_finefineweb_100k.yaml`. Tests:
`tests/test_heads_datagen.py` (CPU, no network).

## Running it

```bash
# 1. local stages — nothing paid (corpus download is a few hundred MB)
python -m nla.datagen.heads.run_pipeline --config configs/datagen/heads_qwen3_8b_finefineweb_100k.yaml
#    = --stages corpus,extract,stats,select,prepare; prints the estimated API cost

# 2. PAID — Haiku 5.5 Batch API
python -m nla.datagen.heads.run_pipeline --config ... --stages submit                       # dry run
python -m nla.datagen.heads.run_pipeline --config ... --stages submit --confirm-spend --max-shards 1   # pilot
python -m nla.datagen.heads.run_pipeline --config ... --stages wait                         # poll + collect
python -m nla.datagen.heads.run_pipeline --config ... --stages submit --confirm-spend       # the rest
python -m nla.datagen.heads.run_pipeline --config ... --stages wait
python -m nla.datagen.heads.run_pipeline --config ... --stages retry,submit --confirm-spend # failures only
```

Every stage resumes or is idempotent. `extract` writes durable parts and skips
finished ones on restart. With `extract.num_gpus: N`, it runs one process per
GPU over interleaved parts. `submit` refuses to proceed if a previous run died
mid-submission, because blindly resubmitting could pay for a shard twice. It
also creates batches with SDK retries off for the same reason. Batch results
stay downloadable for 29 days. Credentials come from `ANTHROPIC_API_KEY_BATCH`
if set, else the SDK default.

**Expected scale** (default config, 1× H200): ~4M records (100k docs × 10
positions × 4 layers), ~110k records per head, ~65–85 GB on disk, extraction
~2–3 h. The warm-start is 1,152 heads × 400 = ~460k descriptions; ~8% are
answered "little or nothing" by rule, and the rest go to the API, ~$55–95 at
batch prices.

## Stages

| stage | module | output |
|---|---|---|
| corpus | `build_corpus.py` | `corpus.parquet`: 200 random FineFineWeb shard files × first 500 docs ≥1000 chars (domain-mixed; only the head of each file is downloaded), shuffled |
| extract | `extract.py` + `capture.py` | `docs/`, `records/L{ll}/`, `head_meta.yaml`, `wo.safetensors` |
| stats | `stats.py` | `head_stats.safetensors`, `devnorms/L{ll}/` |
| select | `split_select.py` | `splits.parquet` (doc-level av_sft / ar_sft / rl / eval), `warmstart/selection.parquet` |
| prepare | `batch_api.py` | `warmstart/requests/shard_*.jsonl` (+ `index.parquet`, `prepare.json`), `warmstart/rule_descriptions.parquet` |
| submit / wait / collect / retry | `batch_api.py` | `warmstart/batches/state.json`, `warmstart/results/`, **`warmstart/descriptions.parquet`** |

## What a record holds

One row of `records/L{ll}/` = one sampled (document, token, layer):

| column | meaning |
|---|---|
| `doc_id`, `pos`, `token_id`, `layer` | where |
| `resid_pre` | residual stream entering the layer (bf16) |
| `head_out` | o_proj input = all heads' outputs, head `h` at `[h*D:(h+1)*D]` (bf16) |
| `attn_top_idx`, `attn_top_w` | per head: top-16 source positions and weights |
| `attn_first_w`, `attn_self_w`, `attn_entropy` | per head: weight on token 0 (the sink), on the current token, entropy |

`docs/` holds each document's token ids, so every prompt and window can be
rebuilt. All of a layer's heads share a row. Writes, the post-attention
residual, and "residual with head h's write removed" all derive from
`resid_pre`, `head_out` and `wo.safetensors`:

```python
import numpy as np
from nla.datagen.heads import store

meta, stats = store.read_meta(out), store.load_stats(out)
H, D = meta.n_heads, meta.head_dim
rec = store.read_records(out, layer)
z = store.from_bf16_bits(store.fixed_numpy(rec["head_out"], np.uint16, H * D)).view(-1, H, D)
W = store.load_wo(out, layer)                                # [d_model, H*D]
w = z[:, h] @ W[:, h * D:(h + 1) * D].T                      # head h's write  [N, d_model]
li = stats.layer_pos(layer)
w_norm = (w - stats.mean_write[li, h]) / stats.dev_rms[li, h]   # "subtract the average write, scale by typical size"
resid_pre = store.from_bf16_bits(store.fixed_numpy(rec["resid_pre"], np.uint16, meta.d_model))
resid_without_h = resid_pre + z.flatten(1) @ W.T - w         # post-attention residual minus head h
```

`stats.dev_quantiles` gives the per-head quantiles for choosing an outlier cap.
`devnorms/` is row-aligned with `records/` and stores each record's per-head
‖w − mean‖ and ‖w‖.

## Warm-start descriptions (attention-based; superseded)

> **Superseded.** This describer conditions on the attention pattern and the document text. The
> rule now is that warm-start descriptions must be a function of the head's write alone, with no
> attention pattern, document or residual. Use the soft-token readouts and explanations below.
> This path is kept because the quick SFT test (below) used it.

Selection: exactly 200 av_sft + 200 ar_sft examples per head, from disjoint
documents, at most one per document per head. Sampling weight is
`0.25 + 0.75·min(dev/rms, 4)`, so large deviations come up more often.

The describer sees only what a person could see: the head name, the text up
to the current token (last 1,024 tokens), and up to 6 attended sources, each
with its weight and ±12 tokens of context. The fixed instructions and
examples live in `prompts.INSTRUCTIONS` and are the cached system prompt.
They ask for one or two clauses, literal tokens only in quotes, nothing about
the document or genre, and "little or nothing" when attention sits on the
first token. Records with ≥95% of attention on the first token get that
answer by rule, without an API call.

`descriptions.parquet` keeps the selection columns and adds `description`,
`ok`, `fail_reason`, plus flags for the filter step: `is_little`, `quote_ok`
(every quoted span appears in the prompt), `mentions_document` (heuristic),
and `n_words`. Rows are not dropped here. The cross-fitted probe filter
(step 3) decides.

## Choices worth knowing

- **Conditioning vector.** The data stores the layer-input residual. The
  recipe's "residual with the write removed" is derivable, as above, but it
  still contains the other heads' same-layer writes.
- **bf16 storage.** This is lossless for a bf16 forward. Vectors are raw
  little-endian bytes in fixed-size binary columns (`store.fixed_numpy`).
- **Attention weights.** These are recomputed in fp32 for sampled queries
  only, from the exact q/k the model uses, through a registered attention
  implementation wrapping SDPA. `tests/test_heads_datagen.py` checks them
  against eager attention.

## Soft-token readouts and explanations (current warm-start path)

The write deviation v = W_h(z_h − z̄_h), normalized per head (÷ dev_rms, capped at norm 4), is
added to the model's own attention output at the head's own layer, at a placeholder token in a
fixed few-shot prompt. This is exactly where head writes enter the residual. What the model then
predicts is the readout. Nothing else about the example reaches the model, so every readout is a
function of the write (and of its layer, which picks the injection site).
Code: `nla/datagen/heads/soft_token.py`.

```bash
# sweep recipes (prompt × β × readout × carrier): probe FVE + examples per recipe
python scripts/heads_soft_token_sweep.py --output-dir <heads data> --work-dir <dir> ...
# learning curves for the best readouts plus continuous references (run from scripts/)
python heads_soft_token_diag.py --output-dir <heads data> --work-dir <dir>
# rewrite readouts into prose explanations with Haiku 5.5 (PAID, ~$4 per 12k)
python scripts/heads_soft_token_explain.py --texts <dir>/texts.parquet --out <dir>/explain.parquet
# prompt lab used to choose the explanation prompt (PAID, see its docstring)
EXPLAIN_LAB_DIR=<out> EXPLAIN_LAB_TEXTS=<dir>/texts.parquet python scripts/heads_explain_lab/lab.py run --big v26
```

On a pod, `scripts/runpod/launch_heads_quick_sft.py --remote-script scripts/runpod/heads_soft_token_remote.sh
--hold` runs datagen with `configs/datagen/heads_qwen3_8b_softtoken.yaml` and then the sweep.

**Readouts.**
- `greedy`: the continuation's first line. It collapses to the placeholder's default reading.
- `delta`: the top-k tokens by log-prob gain over the uninjected prompt.
- `avg16`: 16 tokens by gain averaged over the `assoc` and `continue` prompts × the " something" and
  " it" carriers. This is the best text readout.
- `lens`: the write's direct W_U projection. It is a reference only.

**Probe results (Qwen3-8B).** These are from five heads (L6H27, L12H20, L18H16, L24H8, L30H9) with
2,400 examples each. Each entry is the held-out FVE of a cross-fitted ridge from features to the
write's coordinates in the head's write basis, by examples per head:

| features | 200 | 400 | 800 | 1600 | 2400 |
|---|---|---|---|---|---|
| delta | 0.031 | 0.058 | 0.090 | 0.112 | 0.125 |
| avg16 | 0.042 | 0.076 | 0.125 | 0.163 | 0.190 |
| greedy | 0.006 | 0.022 | 0.037 | 0.059 | 0.067 |
| lens | 0.042 | 0.066 | 0.110 | 0.147 | 0.177 |
| final hidden state (dense) | 0.279 | 0.401 | 0.531 | 0.635 | 0.679 |

- The model keeps much of the write (hidden-state ceiling 0.68), and the text bottleneck loses most of it.
- Every curve is still rising, so 200 examples per head under-measures every method.
- The legacy self-description (`self_describe.py`, l1_add) scored −0.003 / −0.031 (tfidf/embed)
  at 20 heads × 200.

**Explanations.** `heads_soft_token_explain.py` has Haiku 5.5 rewrite each example's avg16 and lens
readouts into one or two sentences about what the write does. The explanations are concepts in prose: they never
quote or list tokens, never mention the readouts, and do not open with stock phrases. avg16 is shown
split into distinctive tokens and common filler. A token counts as filler when it appears in at least 8%
of all lists, a fixed table per dataset. Answers that break a rule are sent back for a rewrite.

The prompt was chosen in `scripts/heads_explain_lab/`, which tried about 25 variants for $6.88 total.
Prompts were scored on 500 dev examples with two metrics:
- a second Haiku picks the true readout from the explanation, among 10 same-head candidates
  (5 near-duplicates); chance is 0.10 and the raw list scores 0.96
- an Opus 4–5-way quality judge

| prompt | 10-way pick (dev / held-out) | Opus quality |
|---|---|---|
| plain list, no split (v1) | 0.60 / 0.64 | 5.6 |
| **final (v26: split, concept-first openings, thinking medium)** | **0.68 / 0.70** | **6.8** |

What helped:
- the filler split, about +5 points
- the stock-opening rule plus the rewrite check
- telling the model to ignore weak evidence silently, since commenting on noisy evidence was the main rule break
- thinking

A per-request reminder, label-style notes and a third readout did not help.

## L24 RL on the standard NLA recipe (latest)

- **Setup:** AV + AR SFT redone per `docs/train_new_model.md` (norm-matched injection, direction-only
  normalized MSE, 1 epoch), then 400 GRPO steps on `configs/rl_sgpu.yaml` defaults (single H100,
  co-trained AR, k3 KL 0.01). Reward = 2·cos − 2.
- **Result** (held-out mean cosine of the reconstruction; skill vs the head-mean direction):
  SFT pair 0.071 (T=1) → RL step 400 **0.596 (57% skill)**; shuffled control 0. Each model alone ≈ 2×
  (RL AV → frozen SFT AR 0.178; SFT AV → RL AR 0.198), so most of the gain is AV↔AR co-adaptation.
- **But the explanations break the rules:** 93% quote tokens at T=1 (often tokens the head attended
  to, or the current token), plus stray markup and invented strings. A copy/quote penalty or a frozen
  reader is needed before using RL here.
- Details, curves, incidents: `docs/head_nla_l24_std_plan.md`. The earlier raw-magnitude RL run
  (`docs/head_nla_l24_rl_plan.md`) was stopped at step 44.
- Artifacts: models `syvb/head-nla-qwen3-8b-l24-std-{av,ar,rl-ckpts}`; dataset `l24/rl_data/`,
  `l24/std_rl/results/`, `l24/std_rl/faithfulness/` (blinded judge: RL AV no more faithful than SFT, 1.99 vs 1.98).
- **Viewer:** `l24/std_rl/viewer/head_nla_viewer.html` (offline, single file): all 32 heads at 32 positions of 8
  held-out docs with RL / SFT / gold explanations, cos and FVE, attended tokens and readouts. Rebuild with
  `scripts/runpod/heads_l24_viewer_remote.sh` + `scripts/heads_viewer_html.py`.

## L24 run: probe test + SFT on soft-token explanations

- **Setup:** all 32 heads of L24, 1,500 examples per head.
- **Probe:** the explanations keep 42–47% of the word lists' FVE (0.067–0.075 vs 0.159), and only about 10% of
  their content words are copied from the readouts.
- **SFT:**
  - The AR reaches 0.063 FVE on gold explanations (shuffled: −0.082).
  - The AR reaches 0.049 on the **AV's own** explanations (AV given another write: −0.067).
  - Both pre-RL checks pass, with CIs far from 0.
- Details, costs and incidents: `docs/head_nla_l24_plan.md`.
- Artifacts: dataset `l24/`, models `syvb/head-nla-qwen3-8b-l24-av` / `-ar`.

## Quick SFT test (attention-based Claude descriptions; superseded)

64 heads (L12, L24) × 200 examples. The AV and AR were trained on the attention-based Haiku
descriptions, with residuals in neither model. The AR learned a little (held-out FVE 0.019 vs −0.025
shuffled). The AV did not beat a shuffled control. Launcher: `scripts/runpod/launch_heads_quick_sft.py`.
Checks: `scripts/heads_sft_checks.py`. Description comparison: `scripts/heads_compare_descriptions.py`.

## Artifacts on Hugging Face (public since 2026-10-21)

Scanned before release: no credentials (exact-value and pattern search over every text file, every
parquet string column and the safetensors headers); wandb host/entity names and dead pod SSH
endpoints were redacted from logs and the dataset history squashed.

- Dataset `syvb/head-nla-qwen3-8b`. See its README for which parts matter:
  - `l24/`: the L24 probe test + SFT run, and `l24/rl_data/` + `l24/std_rl/` for the standard-recipe RL (highest)
  - `soft_token/`: sweep and diagnostic readouts, curves, logs
  - `explanations/`: the prompt-lab outputs and the final pilot
  - `quick_sft_run1/`: SFT data, Claude descriptions, checks
  - `dev_qwen3_0.6b/`: small local dev runs, unimportant
- Models `syvb/head-nla-qwen3-8b-l24-std-av` / `-l24-std-ar` (standard-recipe SFT) and
  `-l24-std-rl-ckpts` (RL checkpoints every 50 steps with optimizer state).
- Models `syvb/head-nla-qwen3-8b-l24-av` / `-l24-ar`: the earlier L24 SFT (raw-magnitude contract).
- Model `syvb/head-nla-qwen3-8b-ar-quick`: the AR from the quick SFT test (LoRA + value head, 400 steps).
  It is marginal and kept for reference. The matching AV adapter was lost when its pod was deleted.

The 8B head dataset itself (`records/`, ~GBs) was not kept, but it regenerates in about 5 minutes
on one H100 from `configs/datagen/heads_qwen3_8b_softtoken.yaml` (seeded). The diag script reselects
the same examples from it.
