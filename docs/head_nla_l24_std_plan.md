# L24 head-NLA on the standard NLA recipe (SFT redo + RL)

Status (2026-10-20): implemented, CPU-tested, not yet run. Supersedes the raw-magnitude RL run of
`head_nla_l24_rl_plan.md`, which was stopped at step 44 by decision (see "Why").

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
