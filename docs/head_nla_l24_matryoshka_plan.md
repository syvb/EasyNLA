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

## Runs

| run | script | steps | wandb |
|---|---|---|---|
| matryoshka | `scripts/runpod/heads_l24_mtry_remote.sh` | 400 → 440 AR-only → 840 | `l24-mtry` / `l24-mtry-rl` |
| control | `scripts/runpod/heads_l24_ctrl_remote.sh` | 400 → 800, std objective | `l24-mtry` / `l24-ctrl-rl` |

The control separates the objective's effect from simply training 400 more steps. Both runs:
- resume the same step-400 state;
- run a three-step smoke test first;
- mirror checkpoints every 50 steps to `syvb/head-nla-qwen3-8b-l24-{mtry,ctrl}-rl-ckpts`;
- log held-out prefix FVE (`eval/fve_pct_t{4,16,64}`) next to the usual `eval/fve_pct`;
- end with the standard T=1 checks on all 6,360 eval rows (`--no-fallback-cut`, because a matryoshka AV
  may not close its explanation).

Neither run has a trainer time cap. The pod watchdog is set at 16 h, about twice the expected time.

## Evaluation (`scripts/heads_matryoshka_sweep.py`, `heads_l24_mtry_sweep_remote.sh`)

On 2,000 held-out eval rows (seeded, all heads), each AV writes one T=1 explanation:
- the step-400 AV;
- the matryoshka AV;
- the control AV;
- plus the gold texts.

Every AR (step 400, end of warm-up, matryoshka, control) scores every AV's first L ∈ {0, 1, 2, 4, 8,
16, 32, 64, 120} tokens and the whole explanation. Metrics are mean direction cosine and FVE against
the head's mean write. The questions:

1. Does the matryoshka pair beat the warm-up baseline (step-400 AV + prefix AR) at short prefixes?
2. What does it cost at full length relative to the control?
3. With the AR held fixed (cross-reads), is the gain in the AV's ordering or only in the AR?

## Time and cost (secure H100, $3.99/hr)

Estimates:
- matryoshka: setup + smoke 0.6 h, 40 warm-up steps ≈ 0.4 h, 400 steps ≈ 5–6 h (capped rollouts, est. 45–55 s/step), checks ≈ 0.7 h; ≈ 7.5 h, ≈ $30;
- control: 0.6 + 400 × ~70 s ≈ 7.8 h + 0.7 h; ≈ 9 h, ≈ $36;
- sweep: ≈ 1.2 h, ≈ $5.

Total ≈ $70.
