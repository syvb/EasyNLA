# Head NLA L24: continuing RL with the matryoshka objective

Goal: make the L24 head NLA put the most reconstruction-relevant content first, following
[Matryoshka NLAs](https://www.lesswrong.com/posts/aWYQHh2pEFCLYh85C/matryoshka-nlas-training-activation-verbalizers-to-frontload-1),
by continuing training from the std pair's step-400 checkpoint
(`syvb/head-nla-qwen3-8b-l24-std-rl-ckpts/iter_000400`: AV LoRA, co-trained AR, both Adam states).

## Objective (as in the post)

- **Truncation.** Each GRPO group draws L ~ U[1, 120] and every sample in it keeps only its first L
  explanation-content tokens (counted after the `<explanation>\n` opening). The AR scores that prefix
  (behind its usual `Attention head L24Hh: ` lead-in) and co-trains on it. The group shares L, so no
  sample gets a lucky longer cut. Rollouts are generated in cap order, each `generate()` call running
  only to its chunk's largest cap. Tokens past the cap are never generated or trained. A capped
  rollout is a valid explanation, not the −2 failure.
- **KL.** k3, β tripled to 0.03, tapered so later tokens still learn from the reconstruction reward:
  full β for content tokens < 16, linear down to 0.1β at token 120. The post gives no schedule;
  this one is our choice.
- **AR warm-up.** The post's AR warm-start saw truncated text, because otherwise "the AR is really bad
  at the start of RL". Here the AR has only ever read whole explanations, so the first 40 steps
  update the AR alone, on prefixes of the current AV's rollouts, with the AV frozen. The checkpoint at
  their end (`iter_000440`) is also the fair baseline: the unchanged step-400 AV with an AR that reads
  prefixes.
- Everything else is the std run's recipe (`configs/rl_sgpu.yaml`: AV lr 1e-4, AR lr 8e-5, 64 × 8,
  256-token cap, hinged length penalty, direction-only reward).

Context: the step-400 AV writes about 147 tokens at T=1 (p10–p90 137–158) and the gold texts about 42.
98% of its explanations are longer than 120 tokens, so nearly every rollout is cut (`trunc/capped_frac`
≈ 1), with these consequences:
- The run trains only the first 120 content tokens; the AV is never again trained on how it ends.
- The 256-token failure reward and the hinged length penalty (past 192 tokens) are inactive by
  construction (`av/truncated_count`, `av/len_pen_mean` stay 0).
- Whole-explanation behaviour (up to 256 tokens) is untrained; it is reported (`eval/closed_frac`,
  `eval/len_mean`, the checks, the sweep's "full" column) but is not what the objective optimizes.
This matches the post's setup.

## Run

`scripts/runpod/heads_l24_mtry_remote.sh` (wandb `l24-mtry` / `l24-mtry-rl`). Steps 400 → 440 are
the AR-only warm-up; 440 → 840 use the matryoshka objective. The script:
- runs a three-step smoke test first (one AV-frozen step with the held-out eval, then two
  matryoshka steps);
- mirrors checkpoints every 50 steps, plus the warm-up's last step, to
  `syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts`;
- logs held-out prefix FVE and cosine (`eval/fve_pct_t{4,16,64}`, `eval/cos_t{4,16,64}`) next to the
  usual `eval/fve_pct`, plus `eval/closed_frac`, `eval/len_mean`, `trunc/noncanon_opening_frac` and
  the within-group reward spread by cap (`trunc/group_std_cap_*`). Under truncation `eval/fve_pct` also
  counts unclosed explanations, so it shifts at step 400 for a metric reason, not a model one;
- ends with the standard T=1 checks on all 6,360 eval rows (`--no-fallback-cut`, because a matryoshka
  AV may not close its explanation).

There is no trainer time cap. The pod watchdog is set at 16 h, about twice the expected time.

## Evaluation (`scripts/heads_matryoshka_sweep.py`, `heads_l24_mtry_sweep_remote.sh`)

On 2,000 held-out eval rows (seeded, all heads), the step-400 AV and the matryoshka AV each write one
T=1 explanation; the gold texts are included too. Every AR (step 400, end of warm-up, matryoshka) scores
every AV's first L ∈ {0, 1, 2, 4, 8, 16, 32, 64, 120} tokens and the whole explanation. Metrics are mean
direction cosine and FVE against the head's mean write. The questions:

1. **Ordering (primary).** With the matryoshka AR fixed, do the matryoshka AV's prefixes score
   higher than the same explanations with their sentences reversed (the post's reversal check, a
   re-scoring of the same text)? The same check on the step-400 AV shows how much it already
   frontloads.
2. Does the matryoshka pair beat the warm-up baseline (step-400 AV + warm-up AR) at short prefixes?
   Reading rule, fixed before the run: if `eval/cos_t16` rises by more than 0.01 between steps 430
   and 440, the warm-up AR had not converged; then question 2 is inconclusive and only question 1
   counts.
3. Cost at length: at L = 120 (the longest trained prefix), and separately at the whole 256-token
   output, read by the step-400 AR (the matryoshka AR never saw text past 120 tokens).

## Time and cost (secure H100, $3.99/hr)

The aborted first launch measured about 18–20 s per AV-frozen step and about 45 s per matryoshka
step. Estimate:
- run: setup + smoke 0.4 h, warm-up 0.2 h, 400 steps ≈ 5 h, held-out evals every 10 steps ≈ 0.8 h,
  checks ≈ 1 h; ≈ 7.5–8 h, ≈ $31;
- sweep: ≈ 1 h, ≈ $4.

## History

The first launch (2026-10-22) was stopped during the warm-up, before any AV update or checkpoint.
Generation in each chunk ran exactly to the chunk's largest cap, so the group at that cap looked
"stopped at the cap", was not marked as cut, and failed the closing-tag extraction (−2; 3–6% of
rollouts). Fixed in `5d1a06a` (generate one token past the largest cap), with a regression test.
