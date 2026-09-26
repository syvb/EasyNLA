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

### Phase 0 result (2026-09-25, pod 02n7d5wlluz4iy, H100, ~20 min)

HF `syvb/kl-nla-qwen3-8b:evals/audit0` (summary.md / summary.json / rows.npz). 2,844 rows, 287 docs.
Brackets are 95% doc-bootstrap intervals.

- **Gates pass.** G1: stored-activation KL median 5.0e-4, p90 1.8e-3 nats (max 0.045). G2: 100%.
  FVE reproduces fvecmp exactly (gold 0.641, av_greedy 0.491, wrong −0.841), so the rows and the AR
  loading match.
- **Ordering holds, same ranking as FVE.** KL recovered: gold 0.842 [0.826, 0.857] > rl_greedy 0.764 >
  rl_sample 0.743 > av_greedy 0.720 [0.697, 0.741] > av_sample 0.627 > quote 0.605 ≫ wrong −1.075
  (KL 11.3 nats, as bad as the orthogonal vector and worse than the mean direction's 5.4). KL is heavy-tailed:
  av_greedy median 0.45, mean 1.53, p90 4.2 nats.
- **Per-row KL and MSE agree only loosely:** Spearman 0.29 (gold) to 0.60 (quote), well below the 0.9
  "KL adds nothing" bar.
- **The AR's error is concentrated on output-relevant directions.** Gold explanations reach FVE 0.641
  with 0.86 nats of KL. The gold vector randomly rotated to the same FVE (cos 0.9) costs only 0.115 nats.
  At equal vector error the AR's miss is ~7× costlier to the model than an isotropic one. This is the gap a
  KL-trained AR could close.
- **Next-token diagnostic: not dominant (yet).** On the 1,690 rows whose greedy next token is a content
  word, av_greedy names it 60% of the time. Those rows carry 56% of av_greedy's KL gain: proportional, not
  concentrated. Recovered is 0.82 named vs 0.71 not named. Quote names it 34% of the time and gets 24% of its gain there.
- Decision per the rules above: proceed to Phase 1. Phase 1b is not triggered.

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

### Phase 1 result (2026-09-25, pod 92gm22kd38x5vu, H100, 1.4 h ≈ $5)

HF `syvb/kl-nla-qwen3-8b`: `ckpts/ar_kl`, `ckpts/ar_mse` (LoRA + value head, `iter_0000782`), and
`evals/audit1` (summary.md, contrasts.md from `scripts/kl_phase1_contrasts.py`, rows.npz).
W&B `kl-nla` / `kl-nla-phase1`. Speed: 15 steps/min for KL (52 min), 70 steps/min for MSE (11 min).
Gates pass again (G1 median 4.9e-4).

| AR (same 2,844 rows) | KL recovered, av_greedy | FVE, av_greedy | KL recovered, gold | FVE, gold |
|---|---|---|---|---|
| sft (start) | 0.720 | 0.491 | 0.842 | 0.641 |
| mse (+1 epoch MSE) | 0.721 | 0.499 | 0.845 | 0.651 |
| kl (+1 epoch KL) | **0.860** | 0.343 | **0.915** | 0.464 |

- **Primary contrast passes.** KL recovered, kl − mse, on av_greedy: **+0.139 [+0.126, +0.153]**. It is positive
  for every text condition: gold +0.071, rl_greedy +0.110, av_sample +0.165, quote +0.189. The matched MSE
  control is flat (mse − sft +0.001 [−0.001, +0.004]), so the gain comes from the loss, not the extra epoch.
  Top-1 agreement is +0.109 on av_greedy (0.560 → 0.668).
- **The cost is FVE:** av_greedy 0.499 → 0.343, gold 0.651 → 0.464. KL and vector reconstruction pull apart.
  The KL-trained AR from the *SFT AV's* explanations (0.860) now beats the MSE AR from *gold* explanations (0.845).
- **The gain sits in the tail.** Split rows by the mse-AR's own KL: the worst quartile goes 4.97 → 2.31 nats
  (kl better on 96% of rows), Q3 0.79 → 0.44, Q2 0.28 → 0.22. The best quartile gets slightly worse, 0.047 → 0.073
  (kl better on only 48%).
- **Not a next-token effect.** The gain on rows whose explanation does *not* name the greedy next token
  (0.712 → 0.840) is at least as large as on rows that do (0.821 → 0.911).
- **Caveat: part of it is hedging.** A wrong document's explanation now beats the mean-direction
  baseline on 30% of rows (sft/mse AR: 5%). Wrong-document KL falls from 11.3 to 6.9 nats, and the wrong − matched
  KL gap shrinks from 9.8 to 6.2 nats. The KL AR has learned to avoid confidently wrong predictions: less committal
  vectors, a better "prior" than the mean direction. So some of the +0.139 is not extra information read from the
  explanation. The top-1 gain and the not-named-rows gain say some of it is, but these numbers can't split the two.
- **Decision per the rules above:** the Phase 2 go criteria are met (CI > 0, not next-token-explained). But the
  hedging caveat has to be controlled first, because an RL reward that pays for hedging would teach the AV
  to be vague. Cheap control (~$1, one H100 audit), not yet run:
  1. each AR's **own no-information baseline**: an empty and a generic explanation → KL, and report KL recovered
     against that instead of against the mean direction;
  2. the mse-AR's prediction **shrunk toward the mean direction** with the blend weight fit on half the docs,
     scored on the other half. If shrinkage alone closes most of the +0.139, the KL AR is mostly a better-calibrated MSE AR.

### Hedging control: pre-registration (written 2026-09-26, before running)

`scripts/kl_hedge_control.py`, pod stage `hedge`, same 2,844 rows restricted to those where av_greedy,
gold and wrong all have a prediction. An explanation-free input gives an AR the same prompt on every row,
so its "no-info" prediction is one constant vector, the AR's learned prior (three texts: empty, short
generic, long generic). Shrinkage blends an AR's predicted direction with a prior direction,
normalize(α·pred + (1−α)·prior), α ∈ {0.1, …, 1.0}. α is fit on half the documents by pooled KL recovered
on av_greedy, applied to the other half (2-fold cross-fit), and reused unchanged for gold and wrong. The
MSE AR is shrunk toward the mean direction and toward each of the KL AR's three no-info constants, the
strongest available prior. The KL AR is shrunk toward the mean direction only.

**Rule.** gap closed = (best cross-fit shrunk mse − raw mse) / (raw kl − raw mse), pooled KL recovered on
av_greedy, where "best" is the prior that does best:
- ≥ 50% → **mostly calibration**: the Phase 1 gain is hedging. Don't use −KL as an RL reward without a
  calibration-neutral normalization.
- < 25% **and** raw kl − best shrunk mse has CI > 0 → **mostly content**: Phase 2 can use −KL.
- otherwise → **mixed**.

Also reported: each AR's KL gain in nats over its own best no-info constant, and whether shrinkage
reproduces the KL AR's milder wrong-document KL.

### Hedging control result (2026-09-26, pod dwxs48t06k4e5y, H100, ~25 min)

HF `syvb/kl-nla-qwen3-8b:evals/hedge` (summary.md / summary.json / rows.npz). 2,794 rows (all three
text conditions scorable), 287 docs.

- **Pre-registered verdict: mostly calibration, narrowly.** The best shrunk MSE AR (toward the KL AR's
  long-generic prior, α = 0.6 in both folds) closes **54.7% [50.6%, 58.7%]** of the gap. The other two KL-AR
  priors close 48.8% and 51.1%. The mean direction closes only 7.4%: it is a poor prior for KL.
- **Half the gain is content.** Raw KL AR − best shrunk MSE AR = **+0.063 [+0.056, +0.069]** KL recovered on
  av_greedy, and +0.052 [+0.046, +0.057] on gold. Shrinking the KL AR itself gains nothing (best α = 1.0 in
  both folds): it is already calibrated.
- **Shrinkage reproduces the wrong-document effect.** Wrong-document KL recovered goes −1.082 (raw MSE AR) →
  −0.323 (shrunk), vs −0.277 for the KL AR. So the KL AR's gentler errors are almost entirely calibration.
- **The MSE AR's predictions are overconfident.** Its KL-vs-α curve peaks at α ≈ 0.6–0.7 for every prior,
  and its own no-info constant is terrible (KL 8.2–10.7 nats vs 5.4 for the mean direction). The KL AR's
  constants sit at 4.9–5.7. Gain over each AR's own prior (KL AR 4.13 nats, MSE AR 6.69) mostly reflects how bad
  each prior is, so it doesn't compare ARs.

**What this means.**
- About half of Phase 1's +0.139 is the KL AR being calibrated. You get that for free by shrinking an MSE AR's
  output toward a good prior.
- The other half is extra output-relevant content read from the same explanations.
- Per the rule, −KL shouldn't be the RL reward without a calibration-neutral normalization. The natural
  candidate is scoring each explanation against the AR's own no-info prediction on the same row (a per-row gain).
  An alternative is using a post-hoc-calibrated AR for both arms of any comparison.
- Note on the rule's premise: calibration is a property of the fixed AR, not of the AV's explanations. A
  hedging AR still pays more for specific, correct explanations (0.76 nats) than for vague ones (≈ its prior,
  4.9 nats). So the risk to RL is miscredited comparisons between arms, more than a direct push toward
  vagueness. That's an argument about design, not a result.

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
