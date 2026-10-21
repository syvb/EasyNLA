# L24 head-NLA on the standard NLA recipe (SFT redo + RL)

Status (2026-10-21): **run complete** (SFT redo + 400 RL steps + post-run checks); results below.
Supersedes the raw-magnitude RL run of `head_nla_l24_rl_plan.md`, which was stopped at step 44 by
decision (see "Why").

## Results (summary)

Held-out, mean cosine between the AR's reconstruction and the true write; "skill" =
(cos − 0.06) / 0.94, where 0.06 is the cosine of each head's mean write direction (0 = knows only the
head, 100% = perfect). 6,360 eval rows (cross pairings: 3,200).

| AV → AR | decoding | AV on true write | skill | AV on another write | gold → AR |
|---|---|---|---|---|---|
| SFT → SFT | greedy | 0.095 | 3.7% | 0.009 | 0.116 |
| SFT → SFT | T=1 | 0.071 | 1.2% | 0.006 | 0.116 |
| RL step 400 → RL step 400 | greedy | 0.551 | 52.2% | 0.004 | 0.231 |
| **RL step 400 → RL step 400** | **T=1** | **0.596** | **57.0%** | 0.000 | 0.231 |
| RL step 400 → frozen SFT AR | greedy | 0.178 | 12.6% | 0.004 | 0.117 |
| SFT → RL step-400 AR | greedy | 0.198 | 14.6% | 0.017 | 0.231 |

- **Large, write-specific gain**: 1% → 57% skill (same T=1 decoding), shuffled control stays at 0,
  all 32 heads improve.
- **Both models improved on their own**: the RL AV read by the frozen SFT AR ≈ 2× the SFT pair (and
  above gold text for that reader); the RL AR reads ordinary (SFT-AV, gold) text ≈ 2× better (partly
  the training the 1-epoch SFT AR lacked).
- **Most of the gain is co-adaptation**: the pair (0.55–0.60) is far above either model alone
  (≈ 0.18–0.20). On grammatical heads (H3, H4) the RL AR scores RL-AV text at 0.21–0.25 but gold at
  0.02–0.03 — text that beats gold 10× where gold carries nothing is a code. Caveat: the "frozen reader
  recovers ≈ 18%" figure is not a bound — that cross pairing was greedy with 74% of outputs cut to 300
  characters and never run at T=1, the 1-epoch SFT AR is under-trained, and quotes hurt it.
- **The explanations broke the warm-start rules** (pre-registered criterion 4 fails): at T=1, 93%
  quote tokens (SFT: 0%), 114 words (SFT: 34), 79% contain stray markup (`</p>`, `</turns>`), 14%
  non-Latin fragments. Typical step-390 text: one real sentence about what the head does, then a tail
  of quoted words, markup and invented strings ("DIF (Differential Individuation Framework)",
  "Cryptonorrow").
- **Where the quotes come from** (greedy step-400 outputs, 12,456 quotes; chance = another write of
  the same head): content-word quotes match a token **the head attended to** 14.8% (chance 3.0%), the
  **current token** 8.2% (0.3%), the write's avg16 / lens readouts 10.4% / 7.5% (4.0% / 1.2%); any
  source 31% (9.8%) — rule-sensitive by ±10 points (`scripts/heads_quote_sources.py`). The write carries
  the identity of tokens the head copies (OV), and the AV learned to name them. Of the 69% that match
  no source, a Fable review found ~55% are generic repeated words ("first", "next", "future" —
  plausibly next-token promotions beyond the stored readout lists), ~12% morphological variants, ~15%
  semantic neighbours and only ~15% truly invented. Function-word quotes are ≈ chance. (Attention data
  used for this analysis only, never for warm-start data.)
- **Quotes and markup are not what earns the reward**: within head, quote-free T=1 rows score
  *higher* (cos 0.73 vs 0.59; ρ(cos, #quotes) = −0.21) and `</p>` count tracks short prose (filler).
  The co-adapted channel is the templated prose: 66% of greedy outputs open "Statements about…", 281
  distinct two-word openers vs 3,158 for SFT, top-50 words = 50% of content tokens (SFT 25%).
- **Faithfulness did not improve** (blinded Sonnet judge, 480 eval rows, `heads_faithfulness_judge_rl.py`):
  against the write's readouts, gold 4.06 / SFT 1.98 / **RL 1.99** / control 1.28 (RL first paragraph
  only: 2.02); with context and attended tokens added: 2.82 / 1.59 / 1.58 / 1.09. Within each text type
  faithfulness and cosine correlate (ρ ≈ +0.41; within head +0.27), but RL rows judged unrelated (score
  1) still reach cos 0.51, and 33% of RL rows have cos ≥ 0.6 with faithfulness ≤ 2. The most faithful
  RL heads are H16–H19, H5, H26 (2.5–3.1).
- **Greedy decoding degenerates** at step 400: 74% of greedy outputs run to the 256-token cap (T=1:
  0.1%); the checks keep the first 300 characters of capped text, so the greedy T=1 gap (+0.046) is a
  truncation artifact (−0.005 on rows that were not truncated).

Training curve (trainer eval, 128 held-out RL prompts, T=1): cosine 0.076 (step 0) → 0.229 (50) →
0.338 (100) → 0.469 (200) → 0.530 (250) → 0.550 (300) → 0.592 (350) → 0.608 (390). KL to the SFT AV
rose to ≈ 1 nat/token by step 20 and stayed 1.1–1.4 (main Qwen3-8B NLA baseline: 0.41 at step 15,
0.72 at 100); entropy 1.8 → 2.5–3.4; length 65 → ~105 tokens; quote rate 0.2% → 75% of rollouts by
step 70.

**Pre-registered criteria** (Fable review): 1 RL pair beats SFT — pass; 2 RL AV via the frozen SFT AR
beats the SFT pair — nominal pass (0.178 vs 0.095; confounded, see above); 3 true − shuffled — pass;
4 quote ≤ 5% — **fail** (93%); 5 faithfulness — **fail** (RL = SFT); no-degeneration (KL bounded,
length < cap) — **fail** (KL ≈ 1.2 with β 0.01 ⇒ the KL term was ≈ 1% of the reward signal).

**Verdict.** RL on the direction-only reward works as optimization and produces some real improvement
(both models individually ~2×), but under this setup most of the gain is a private AV↔AR code carried
by templated prose, while the explanations are no more faithful and break the no-token rule.
Likely drivers, ranked: a co-trained LoRA AR starting under-fit (its "finishing SFT" and "learning
the dialect" are one gradient), an effectively-off KL, a 1-epoch AV, and a 4× smaller batch than the
main setup. Next, cheapest first: offline re-scoring (quotes/markup stripped, first sentence only,
T=1 frozen-reader cross pairing, frozen-AR reads of checkpoints 50–300); the multi-epoch AR retrain
(below); a frozen-AR GRPO run with β 0.05–0.1 and a penalty for quotes/markup/non-Latin.

**Tools for looking at it** (public, `syvb/head-nla-qwen3-8b`): `l24/std_rl/examples_100.md`,
`fve_distribution.png`, `viewer/` (all heads at 32 positions of 8 held-out docs) and `doc_viewer/` (every
token × every head of 4 held-out docs with full attention, heads overview and FVE heatmap).

## Why

The first L24 RL run (`l24-rl-sgpu`, wandb `oscgpslq`) carried over two choices from the head SFT
that the standard NLA recipe doesn't make, and that nobody had argued for:

- **Magnitude-preserving injection** (`h + ‖h‖·v`) instead of the standard norm-matched
  `h + ‖h‖·v/‖v‖`.
- **Raw reconstruction MSE** (targets √d·v, magnitude included) instead of the standard MSE on
  normalized vectors (direction only).

Raw MSE weights big writes most (in the reward, the AR loss and the pooled FVE), ties the reward to
how well the AR guesses a write's strength (hard to verbalize, and not what the explanations are
about), and made the training curves unreadable: a 64-prompt batch's raw predict-zero MSE has a
standard deviation of 0.38 across random batches, against an observed step-to-step SD of 0.32 in
`reward/mean`. The SFT recipe also differed from `docs/train_new_model.md` (2 epochs, which that
doc says overfits; AR lr 1e-4 with a ×10 value-head lr). So both SFTs are redone on the standard
recipe, and RL runs on `configs/rl_sgpu.yaml`'s defaults.

The stopped run's 44 steps (eval FVE 3.7 → 11.8% raw at step 30; KL to the SFT AV ≈ 1 nat/token,
about 3× faster than the main Qwen3-8B NLA baseline) are kept in `/home/syvb/l24_rl` for reference.

## Recipe

| | standard NLA (`docs/train_new_model.md`, `configs/rl_sgpu.yaml`) | this run |
|---|---|---|
| AV SFT | LoRA r128/α16, batch 64, 1 epoch, lr 1e-4 → 1e-5, warmup 50, clip 1.0, norm-matched injection | same (micro-batch 32 × 2 if 64 OOMs) |
| AR SFT | same LoRA, batch 64, 1 epoch, lr 2e-5 → 2e-6, warmup 50, clip 1.0, normalized MSE, `layer+1` blocks | same (25 blocks) |
| RL | rl_sgpu defaults: AV lr 1e-4, AR 8e-5 co-trained, 64 × 8, 256 new tokens, k3 KL β 0.01, failure −2 | same |
| quantization | 4-bit QLoRA on small GPUs | bf16 LoRA (memory only; H100 80 GB) |
| AR input | the explanation | `Attention head L24Hh: ` + explanation (one AR serves 32 heads; the AV prompt names no head) |
| RL evals | base_fve + text_judges | base_fve (text_judges needs document text this data lacks) |
| ops | — | rollout batch 256, micro-batches 8/16, eval gen batch 64, save every 50 (mirrored to HF), `--max-hours 5`, OOM retry at half batch |

Same data as before: the saved L24 selection (20,690 AV / 20,702 AR / 6,360 eval rows; 32 heads)
plus 1,000 RL prompts per head appended by the seeded datagen (selection check: the first 48k rows
must be identical, `dev_norm` within 1e-3).

One known risk: the standard AR recipe (lr 2e-5, one epoch) was tuned on ~250k rows; here one epoch
is 324 steps. If the AR underfits, check 1 in `checks_sft` shows it; RL co-trains the AR at 8e-5
from there.

## Metrics

`heads_sft_checks.py` now reports both spaces from the same generations:

- **direction** (primary): both vectors scaled to norm √d; baseline = the head's mean normalized
  AR-training write. Written to `checks.json["direction"]` and `row_errors_direction.npz`.
- **raw** (secondary, unchanged; byte-identical to the previous version on the CPU fixture).

`heads_compare_checks.py --metric direction|raw` (default direction) pairs two runs on shared rows.

## Pipeline (`scripts/runpod/heads_l24_std_remote.sh`)

install → download SFT data + saved selection → datagen (+RL split) → selection check → rl.parquet
→ **AR SFT → AV SFT → upload both to HF** (`syvb/head-nla-qwen3-8b-l24-std-{av,ar}`) →
**checks_sft** (6,360 rows; the bar) → RL (checkpoints mirrored to
`syvb/head-nla-qwen3-8b-l24-std-rl-ckpts` as written) → keep last + best → checks on the RL pair
(last, best) and both cross pairings (3,200 rows) → collect. wandb group `l24-std`
(`l24-std-ar-sft`, `l24-std-av-sft`, `l24-std-rl`).

Time ≈ 7.2 h (setup 0.2, SFT 0.6, checks_sft 0.35, RL ≤ 5, post-checks 1.1) ≈ $29 at $3.99/hr.
Launcher `--max-hours 8` (watchdog 8.5 h, ≤ $34). RunPod balance before launch ≈ $36.

Launch:

```bash
cd ~/EasyNLA && mkdir -p /home/syvb/l24_std
GPUS="NVIDIA H100 80GB HBM3|3.5|4.5" setsid nohup scripts/runpod/launch_with_retry.sh /home/syvb/l24_std heads-l24-std \
    --remote-script scripts/runpod/heads_l24_std_remote.sh --secrets hf,wandb \
    --fetch results --no-checkpoints --max-hours 8 --avoid-machine bdh8atjj57i7 \
    > /home/syvb/l24_std/retry.log 2>&1 < /dev/null &
```

## Monitoring

Report-only (never stopping the run): `checks_sft` results as soon as they land (if check 1 or 2
fails there, RL will likely be uninformative — the user decides), then `eval/fve_pct` (now a
direction FVE on a fixed set, the curve to watch), `av/kl_to_ref`, entropy, length, quote rate.
The step-50 checkpoint is verified from HF with `heads_l24_ckpt_verify_remote.sh` on a separate
small pod (structure on CPU here first).

## Success criteria (direction metric)

1. RL pair (`checks_rl` or `checks_rl_best`) beats `checks_sft`'s `av/true`, paired CI excluding 0.
2. The RL AV scored by the SFT AR (`checks_rlav_sftar`) also beats the SFT pair: the AV improved, not
   just a private AV/AR code.
3. Check 2 (true − shuffled write) still holds for the RL pair.
4. No token-listing drift: quote rate ≤ 5% and copying ≤ 0.23 (same pre-registered lines as before).
5. The faithfulness judge on the same rows: RL AV not less faithful than the SFT AV.

## AR retrain (2026-10-20, during the first std run)

The standard AR recipe's "one epoch" means ~3,900 updates on its 250k-row split; here one epoch is
324 steps (20.7k rows). The 1-epoch AR ended still improving: training-batch cosine
0.01 → 0.08 → 0.11 → 0.117 at steps 0/100/200/300, the same trajectory as the old raw-loss AR
(0.14 at 300), which kept climbing to 0.19 over its second epoch. So the AR is retrained on the
standard recipe (same lr / schedule / LoRA) for up to 12 epochs (3,888 steps ≈ the standard update
count), saving each epoch and keeping the checkpoint with the best **held-out** cosine on 2,000 rows
of the AV split (doc-disjoint from the AR's data; the 6,360 eval rows stay untouched for the checks).
`train_sft.py --heldout-head-prefix` gives the held-out text the AR's head prefix, and the held-out
log now includes the cosine (normalized MSE = 2 − 2·cos).

`scripts/runpod/heads_l24_std_ar2_remote.sh` reuses the standard AV (`syvb/head-nla-qwen3-8b-l24-std-av`)
and the RL prompts (`l24/rl_data/` in the dataset repo, from the first std pod), so no datagen:
AR SFT (≈ 35 min) → upload `-std-ar2` → checks_sft → RL (`-std-ar2-rl-ckpts`, wandb
`l24-std-ar2-rl`) → checks → collect. ≈ 7 h, ≈ $28.

Note on reading direction metrics: with both vectors at norm √d the normalized MSE is 2 − 2·cos and
every baseline (zero, global mean, per-head mean: cos 0.016 / 0.06) scores ≈ 1.0, so a normalized
prediction only beats the baseline above cos 0.5 — far above what text-to-head-write reaches. The
FVE in this space is mostly that handicap; mean cosine (and comparisons between conditions) is the
readable number. The RL reward is exactly 2·cos − 2.

## What happened during the run (2026-10-20 → 21)

Pod `v2xudlsul9ka52` (secure H100, $3.99/hr, machine `6s1wqc1gf9go`; the first raw-run pod's machine
had been flagged by RunPod, so launches use `--avoid-machine`), 13:02 → 01:46 UTC (≈ 12.7 h, ≈ $51).

- **SFT** (13:10–13:30): AR 1 epoch (324 steps, training-batch cosine 0.01 → 0.12, still rising as the
  lr decayed); AV at micro-batch 64 OOMed and the built-in retry ran 32 × 2 (held-out ppl 6.09 → 5.26;
  the old 2-epoch AV reached 4.90). Uploaded as `syvb/head-nla-qwen3-8b-l24-std-{av,ar}`.
- **checks_sft**: AV cosine 0.095 vs 0.009 shuffled (check 2 CI [+0.164, +0.180] direction FVE),
  28/32 heads significantly above shuffled; the signal is concentrated in H16–19, H25–26, H5, H13, H24.
- **AR retrain** (multi-epoch with held-out selection, `heads_l24_std_ar2_remote.sh`) was prepared and
  CPU-tested but **not run**: RL started first and was let continue.
- **Step-50 checkpoint verification** on a second pod (A100, 0.41 h): structure (keys, shapes, finite,
  moved from SFT, optimizer state) + functional checks on 960 rows; the step-50 pair already quoted
  in 81% of greedy outputs.
- **No early stopping by the operator; the 5 h trainer cap was bypassed on request**: the pipeline
  script was paused (SIGSTOP) so it would not move on to the checks; at the cap (step 260, 18:59) a
  watcher (`scripts/runpod/rl_continue_watch.sh`) resumed from the cap's checkpoint with the full
  state (policy, co-trained AR, both Adam states, data cursor) into the same wandb run. The first
  resume crashed at `wandb.config.update` (a float-repr ConfigError on a re-logged baseline); fixed
  with `allow_val_change=True` (`7c37904`), tested against the server, relaunched a minute later.
  The trainer gained `--stop-file` for a clean operator stop (`9240399`).
- **Watchdog / launcher**: the 8 h launcher was SIGKILLed (no cleanup) and the on-pod watchdog
  replaced twice with longer ones (new one started before the old shell, then the old shell, then its
  sleep), so only the operator deleted the pod.
- **Checkpoints** mirrored to HF every 50 steps by `scripts/runpod/hf_ckpt_uploader.py`, each with
  `optim_latest.pt` (step 400's verified loadable at step 400): resume past 400 with
  `--resume-from-lora iter_000400`.
- **Post-run checks** took ≈ 2 h instead of 1: greedy step-400 outputs mostly run to the 256-token cap.
  `heads_sft_checks.py --temperature 1` (`b095c40`) added the T=1 checks for the RL and SFT pairs.

## Artifacts (public)

- Models: `syvb/head-nla-qwen3-8b-l24-std-av`, `-std-ar` (SFT), `-std-rl-ckpts` (`iter_000050` …
  `iter_000400` and the cap's `iter_000260`; each: AV adapter, `critic/` AR LoRA + value head,
  `optim_latest.pt`, configs).
- Dataset `syvb/head-nla-qwen3-8b`: `l24/rl_data/` (the 32k RL prompts) and `l24/std_rl/results/`
  (all checks greedy + T=1 with generations and per-row errors, `eval_ctx.parquet` with the eval rows'
  context and attended tokens for the quote analysis, training/run logs).
- wandb: group `l24-std` (`l24-std-ar-sft`, `l24-std-av-sft`, `l24-std-rl` = run `hv887ru2`).
