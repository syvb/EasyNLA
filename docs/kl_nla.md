# KL-NLA: experiment plan (Qwen3-8B, layer 24)

Branch `sv/kl-nla`. Code: `--recon-loss kl` in `nla/train_sft.py` and `nla/train_rl_vllm.py`,
helper `nla/utils/kl_splice.py`. Status 2026-09-25: CPU-tested only; nothing launched.

## Question

Today an NLA is scored by how well the AR rebuilds the activation *vector* (MSE / FVE), which
weights every direction equally. The SAE literature's alternative (end-to-end / KL-trained SAEs)
scores a reconstruction by what it does to the model: splice it back in and measure
KL(p_orig ‖ p_splice). Here that is the next-token distribution at the extraction position t,
with the AR's prediction (rescaled to the true norm) written over the layer-24 residual at t.

1. Does the KL metric rank explanations sensibly (gold > SFT AV > wrong document), and how far
   does it disagree with FVE?
2. Does training the AR on KL make it extract more *functionally relevant* content from the same
   explanations than an MSE-trained AR does?
3. Does GRPO with a −KL reward give better explanations, or a degenerate objective?

## Assets (all checked 2026-09-25)

| what | where | notes |
|---|---|---|
| SFT verbalizer (AV) | `syvb/nanonla-qwen3-8b-L24-av` | merged bf16, 36 layers; SFT only, no RL |
| SFT reconstructor (AR) | `syvb/nanonla-qwen3-8b-L24-ar` | 25 layers + `value_head.safetensors`; loads through `NLACriticModel.from_pretrained` + `critic_predict` (fve_cmp reproduced nanoNLA's FVE with it) |
| recon-RL reference | `syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0` | only p0.0; the other p* are length-penalty runs |
| eval rows | `asher577/easynla-warmstart-data:av_sft_val.parquet` → 2,844 rows / 287 docs unseen by the AV and AR | same rows as the Sept fvecmp eval |
| eval explanations | `syvb/pred-nla-qwen3-8b:evals/fvecmp/explanations.json` | gold, quote, av_greedy, av_sample, rl_greedy, rl_sample, wrong for those 2,844 rows. **No AV generation needed for Phases 0–1.** |
| AR training rows | `syvb/nanonla-qwen3-8b-L24-data-full:ar_sft_full.parquet` (50k) | the AR's own training docs; excluded from the eval rows |
| RL rows | `…-data-full:rl_full.parquet` (30k) | doc overlap with the eval rows NOT checked: filter the 287 eval doc_ids out in prep |

Every parquet above has `detokenized_text_truncated` + `n_raw_tokens`, and 6,000/6,000 sampled
warm-start rows re-tokenize exactly, so the KL path can use all of them. Prefixes: p50 ~180
tokens, p90 ~750, max ~4k.

FVE reference points on the eval rows (fvecmp, AR above): gold 0.641, av_greedy 0.491,
rl_greedy 0.533, quote 0.383, av_sample 0.366, wrong −0.84.

## The main risk: a next-token objective

KL is measured at ONE position (the datasets keep only the prefix, not the continuation), so it
mostly scores whether the spliced state produces the right next-token distribution. Earlier work
here found SFT explanations already end with a confident next-token guess ~97% of the time. So the
cheap optimum for a −KL reward may be "state the next token", which is a
text-to-logits autoencoder, not an activation explainer. Every phase measures this directly,
and Phase 1b is the fix if it bites.

## Phase 0: KL audit of the existing SFT NLA (no training)

One H100, ~30 min including downloads, ≈ $2 (SECURE H100 bills $3.49/h).
New script `scripts/kl_audit.py`. It rebuilds the fvecmp row set and asserts the doc_id sequence
matches `explanations.json`. Each row is one `SpliceKL.kl` call with all conditions as query
copies, so each prefix runs once.

**Gates (stop and fix if any fails):**
- G1: splicing the *stored* activation gives KL ≈ 0 (median < 1e-2 and p90 < 5e-2 nats) at 8B in bf16.
  The threshold was set at 1e-3 in the first draft and loosened before any 8B data existed: the stored
  vectors came from a batched, right-padded bf16 extraction, so recompute noise is expected, whereas a real
  bug gives nats. G1 checks
  layer indexing, the re-tokenization and bf16 at full scale. The CPU tests covered a tiny model and 0.6B.
- G2: ≥ 99% of the 2,844 prefixes round-trip.

**Conditions:**
- Text, via the AR: gold, av_greedy, av_sample, rl_greedy, rl_sample, quote, wrong.
- Vector: mean direction (the predict-the-mean baseline, for "KL recovered %" = 1 − KL/KL_mean),
  orthogonal-to-gold (the RL failure floor), and gold rotated to cos ∈ {0.95, 0.9, 0.8, 0.7, 0.5},
  which maps FVE onto KL.

**Report**, with doc-bootstrap 95% CIs throughout:
- KL, KL recovered %, FVE and top-1 agreement per condition.
- Per-row Spearman ρ(KL, MSE).
- The cos→KL curve.
- **Next-token diagnostic:** does the explanation contain the target's greedy next token (a
  content word, ≥ 3 chars)? Report the rate per condition, and how much of the per-row KL
  recovered that indicator explains.

**Decisions:**
- Ordering fails (gold ≤ wrong, or av ≈ wrong on KL): single-position KL is too noisy at L24. Go to 1b first.
- ρ(KL, MSE) > 0.9 per row: KL adds little over MSE. Run Phase 1 only, as a cheap confirmation, then stop.
- The next-token indicator explains most of the KL gain: expect RL to degenerate. Phase 1b before Phase 2.
- Otherwise go to Phase 1.

## Phase 1: AR only, KL vs MSE, same start (matched)

One H100, ~1.5 h for both arms plus evals, ≈ $5–8.

- Both arms start from `syvb/nanonla-qwen3-8b-L24-ar` (`--base-ckpt` takes the prepared critic),
  LoRA r128 (the `train_sft` default; a bf16 AR plus the bf16 target fits in 80 GB, full-FT fp32 does not).
  Same data (`ar_sft_full`, 1 epoch = 782 steps at effective batch 64 = 16 × 4 accumulation), lr 5e-5
  (warmup 50, cosine to 2e-6), seed 0. The only difference is `--recon-loss kl` vs `mse`.
  The micro-batch is 16 because `train_sft` does not enable gradient checkpointing for an unquantized
  LoRA AR. The AR must be passed as a local directory: `train_sft` treats `--base-ckpt` as a prepared
  critic only if `value_head.safetensors` is on disk.
- How to run: `scripts/runpod_kl_nla.py launch --stages audit0` then
  `--stages ar_kl,ar_mse,audit1`. `scripts/pod_kl_nla.sh` is the pod side. Results go to the private HF
  dataset `syvb/kl-nla-qwen3-8b` (`evals/audit0`, `ckpts/ar_kl`, `ckpts/ar_mse`, `evals/audit1`).
  Training curves go to W&B project `kl-nla`.
- Gradient clipping at 1.0 binds on both losses (the smoke run's grad norms were 90–270), so the
  ~10× loss-scale difference does not need its own lr.
- Eval: rerun the Phase 0 audit with each new AR. Primary contrast: KL recovered on **av_greedy**,
  AR-KL minus AR-MSE, doc-bootstrap CI. Secondary: FVE cost, whether `wrong` stays near the mean
  baseline (the AR is not just learning a prior), and whether `quote` gains more than gold does
  (KL rewarding copied text).
- **Go to Phase 2 only if** the primary contrast is > 0 with CI > 0 *and* AR-KL's gain is not
  explained by the next-token indicator. If KL training doesn't change what the AR pulls out of the same
  explanations, a KL reward would just chase the MSE-AR's signal.
- Optional extra arm, only if Phase 1 is flat: AR-KL from the truncated base instead of from the
  MSE-trained AR (~$3). That rules out "the MSE solution is a basin KL can't leave in one epoch".

## Phase 1b (conditional): multi-position KL

Needed if the next-token diagnostic dominates. Generate a greedy 16-token continuation of each prefix
with the target (cheap, no grad), keep the splice at t, and sum the KL over positions t..t+15.
This is closer to e2e-SAE practice, since later positions see the spliced state only through
attention to it. Code: extend `SpliceKL.kl` to take continuation ids. The query copies become
blocks of (1+m) tokens with a block-diagonal mask, ~40 lines, plus a test against the unpadded reference.
Then redo Phase 0's diagnostic, and Phase 1 if needed.

## Phase 2: RL pilot, −KL reward vs −MSE reward

One H200 per arm, needed because vLLM + actor + AR + a second 8B base is ~131 GB of 141. Budget ~5 h
per arm for 250 steps at 64 prompts × G 8. That's a guess: measure it in the smoke. I haven't seen
an H200 bill, so check the listing at launch.

- Prerequisite: a 10-step GPU smoke of `train_rl_vllm --recon-loss kl` (~$2). Its vLLM loop has
  never executed; only `score_with_critic`'s KL path is unit-tested.
- Arms, both from the merged SFT AV with a fresh LoRA r16/α32 (the p0.0 recipe), 250 steps:
  A = reward −MSE, AR co-trained on MSE (starts from AR-MSE of Phase 1);
  B = reward −KL, AR co-trained on KL (starts from AR-KL).
  Also rerun A in this trainer rather than comparing with nanoNLA's p0.0.
- Eval every 25 steps on the 2,844 rows, with each arm's final explanations **cross-scored by both ARs**.
  That removes the home-AR advantage, the lesson of fvecmp, where an AR trained on gold favours gold.
  Report KL recovered, FVE, extraction rate, length, the next-token indicator rate, and verbatim
  4-gram overlap with the prefix (quote drift). Text judges need `ANTHROPIC_API_KEY`, which we
  don't have; skip them or port them to OpenRouter.
- B wins if its cross-scored KL recovered beats A's (CI > 0) *without* its next-token or quote
  rates rising above A's. If those rates rise, the objective is being gamed, whatever the KL says.

## Code to write before launching

1. `scripts/kl_audit.py` (Phases 0/1 eval). CPU smoke on 0.6B with the tmp smoke set, fake explanations.
2. Pod launcher. Port `scripts/runpod_pred_nla.py` + `scripts/pod_pred_nla.sh` from `sv/pred-nla`:
   `--volume none`, HF as the store, stages `audit | ar_kl | ar_mse | eval | rl_smoke | rl_kl | rl_mse`,
   self-terminating. Results go to HF under a new dataset repo (e.g. `syvb/kl-nla-qwen3-8b`).
3. Phase 1b code, only if Phase 0 says so.

## Budget and order

| step | GPU | est. | gate to next |
|---|---|---|---|
| Phase 0 audit | 1×H100 | ~$2 | G1, G2, ordering |
| Phase 1 AR-KL vs AR-MSE | 1×H100 | $5–8 | primary contrast CI > 0 |
| (1b multi-position) | 1×H100 | ~$3 + redo | next-token diagnostic |
| RL smoke | 1×H200 | ~$2 | runs, memory fits |
| Phase 2 two arms | 1×H200 each | ~10 h total | — |

Every paid step needs the user's go-ahead; nothing has been launched.
