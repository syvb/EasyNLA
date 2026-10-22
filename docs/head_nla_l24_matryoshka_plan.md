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

Context: the step-400 AV writes about 147 tokens at T=1 (p10–p90 137–158) and the gold texts about 42,
so U[1, 120] covers most of what the AV says.

## Run

`scripts/runpod/heads_l24_mtry_remote.sh` (wandb `l24-mtry` / `l24-mtry-rl`). Steps 400 → 440 are
the AR-only warm-up; 440 → 840 use the matryoshka objective. The script:
- runs a three-step smoke test first (one AV-frozen step with the held-out eval, then two
  matryoshka steps);
- mirrors checkpoints every 50 steps, plus the warm-up's last step, to
  `syvb/head-nla-qwen3-8b-l24-mtry-rl-ckpts`;
- logs held-out prefix FVE (`eval/fve_pct_t{4,16,64}`) next to the usual `eval/fve_pct`;
- ends with the standard T=1 checks on all 6,360 eval rows (`--no-fallback-cut`, because a matryoshka
  AV may not close its explanation).

There is no trainer time cap. The pod watchdog is set at 16 h, about twice the expected time.

## Evaluation (`scripts/heads_matryoshka_sweep.py`, `heads_l24_mtry_sweep_remote.sh`)

On 2,000 held-out eval rows (seeded, all heads), the step-400 AV and the matryoshka AV each write one
T=1 explanation; the gold texts are included too. Every AR (step 400, end of warm-up, matryoshka) scores
every AV's first L ∈ {0, 1, 2, 4, 8, 16, 32, 64, 120} tokens and the whole explanation. Metrics are mean
direction cosine and FVE against the head's mean write. The questions:

1. Does the matryoshka pair beat the warm-up baseline (step-400 AV + prefix AR) at short prefixes?
2. What does it cost at full length relative to the step-400 pair?
3. With the AR held fixed (cross-reads), is the gain in the AV's ordering or only in the AR?

## Time and cost (secure H100, $3.99/hr)

The aborted first launch measured about 18–20 s per AV-frozen step and about 45 s per matryoshka
step. Estimate:
- run: setup + smoke 0.6 h, warm-up 0.2 h, 400 steps ≈ 5 h, checks ≈ 0.7 h; ≈ 6.5 h, ≈ $26;
- sweep: ≈ 1 h, ≈ $4.

## History

The first launch (2026-10-22) was stopped during the warm-up, before any AV update or checkpoint.
Generation in each chunk ran exactly to the chunk's largest cap, so the group at that cap looked
"stopped at the cap", was not marked as cut, and failed the closing-tag extraction (−2; 3–6% of
rollouts). Fixed in `5d1a06a` (generate one token past the largest cap), with a regression test.
