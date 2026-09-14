# RL Future Lens (`nla/future_lens/`)

Read out **future tokens** from a **single hidden state** of a frozen Qwen3 model with
an EasyNLA-style activation verbalizer, trained by SFT and then GRPO, with the
controls needed to tell "reads the state" from "learned better priors". Spec:
`~/plan-rl.md`; execution plan: `~/plan-rl-run.md`. There is no reconstructor.

## Pipeline

```
collect ──> train.parquet / eval.parquet / docs.parquet (+ sidecars)
   │
   ├─> train_sft --future-lens  (alpha sweep: 8-10 short runs, true vs shuffled labels)
   │        └─> SFT adapter (iter_*/ + future_lens.json)
   ├─> future_lens.train_rl --av-ckpt <sft>  (reward exact_match | target_logp; 3 seeds)
   │        └─> RL adapter (iter_*/), evals.jsonl (real vs shuffled every N steps)
   ├─> future_lens.eval --adapter <sft|rl>  (real / shuffled / none / wrong_layer; surprisal)
   ├─> future_lens.baselines ngram | probe --leakage
   └─> future_lens.plots evals/*.jsonl
```

Everything is model-size generic: the whole chain runs on CPU with Qwen3-0.6B
(`scripts/smoke_future_lens_cpu.sh`, ~10 min) and on one 80 GB GPU with Qwen3-8B
(`scripts/runpod_future_lens.py`, one pod per stage).

## Conventions

* **Readout.** K = N+1 tokens starting at `x_{t+1}`. "precision@1 at N" scores the
  readout token at offset N, i.e. `x_{t+1+N}` (Future Lens: N=1 ↔ `x_{t+2}`). K is drawn
  per row from `{1,2,3,4,5,9}` at collection time (the spec's N ∈ {1,2,4,8}, plus N=0 as the
  logit-lens sanity check and N=3 because the success criterion is stated at N=2 or 3);
  the prompt states K and the layer. A readout shorter than K is a miss at the missing
  offsets, in training logs and in eval alike.
* **Labels are token ids** (`target_ids[:k]`), never re-tokenised text.
* **Layer index** = output of decoder block K (`hidden_states[K+1]`), as in datagen.
* **Injection** (default `replace_embed`): `alpha_layer * h/‖h‖` replaces the marker's
  input embedding; `alpha_layer` = 75th-percentile norm at that layer (sidecar
  `future_lens.injection_scale_by_layer`), times `--alpha-mult`. `--injection karvonen`
  is EasyNLA's additive norm-matched hook (ablation arm). `--affine` adds an
  identity-init `Linear(d,d)` before the write. The choice travels with the checkpoint
  (`future_lens.json`), so RL and eval pick it up automatically.
* **Thinking is off** (`enable_thinking=False`) at every chat-template call in
  future-lens code, which makes Qwen3's template emit an empty `<think></think>` block
  before the readout — the decoder input therefore differs from the spec's literal
  template by that block. The legacy NLA path is untouched.
* **Layers.** The collector stores 4/8/12/16/20/24; SFT and RL train on 8–24 only
  (`layers:` in the configs). Layer 4 exists for the wrong-layer control, which injects the
  layer-4 vector of the same position under the row's own prompt and alpha.
* **Rewards.** `exact_match` = matches / K, −0.1 on a length violation. `target_logp` =
  [Σ log p_target over produced slots − 4 nats per empty slot] / K: summed, not averaged,
  so stopping early is never the best move. The eval `surprisal` is the plain per-token
  mean over non-empty readouts (read it together with `len_ok`).
* **Eval cost.** `future_lens.eval` merges the LoRA into the base weights (1.7x faster
  generation) unless `--surprisal` is on, which needs the adapter-disabled base as the frozen
  target. Surprisal is opt-in (a true-prefix forward per readout costs ~5x the generation) and
  scored on `--surprisal-rows` per cell with a `--surprisal-ctx`-token prefix. `--max-rows` is a
  seeded random subsample of positions shared across layers.
* **Preceding tokens** (`prev_ids`) and document token ids (`docs.parquet`) exist for
  the leakage probe and the target-logprob reward / surprisal only. No decoder code
  path reads them.

## Commands (Qwen3-8B)

```bash
# 1. data (1x H100, ~1.5 h): ~220k positions x 6 layers, 6k eval positions, top-1-correct filter
python -m nla.future_lens.collect --base-ckpt Qwen/Qwen3-8B --corpus HuggingFaceFW/fineweb \
    --corpus-config sample-10BT --n-train-docs 5500 --n-eval-docs 300 --layers 4,8,12,16,20,24 \
    --positions-per-doc 40 --eval-positions-per-doc 20 --batch-size 8 --out-dir $D

# 2. alpha / injection sweep (each ~10 min): pick the largest heldout/ce_gap
for m in 0.5 1 2 4; do for s in "" --shuffle-activations; do
  python -m nla.train_sft --config configs/future_lens/sft_alpha_sweep.yaml --base-ckpt Qwen/Qwen3-8B \
      --parquet $D/train.parquet --heldout-parquet $D/eval.parquet --alpha-mult $m $s --save-dir $C/sweep_a${m}${s}
done; done   # + --injection karvonen (x2)

# 3. SFT warm-start (1 epoch, ~2 h). Gate: heldout/p1_off0, p1_off1 > bigram baseline
python -m nla.train_sft --config configs/future_lens/sft.yaml --base-ckpt Qwen/Qwen3-8B \
    --parquet $D/train.parquet --heldout-parquet $D/eval.parquet --alpha-mult $ALPHA --save-dir $C/sft

# 4. GRPO (4k steps, ~4 h/seed). Gate: reward/group_std_mean > 0; eval/p1_gap grows
python -m nla.future_lens.train_rl --config configs/future_lens/rl.yaml --base-ckpt Qwen/Qwen3-8B \
    --av-ckpt $C/sft/iter_XXXXXXX --parquet $D/train.parquet --eval-parquet $D/eval.parquet \
    --save-dir $C/rl_em_s0 --reward exact_match --seed 0

# 5. controls + baselines
python -m nla.future_lens.eval --base-ckpt Qwen/Qwen3-8B --adapter $C/rl_em_s0/iter_004000 \
    --parquet $D/eval.parquet --out $E/rl_em_s0.jsonl --conditions real,shuffled,none,wrong_layer \
    --layers 8,12,16,20,24 --group rl_em --seed 0 --dump-readouts $E/readouts_rl_em_s0.jsonl
python -m nla.future_lens.baselines ngram --parquet $D/eval.parquet --out $E/baselines.jsonl \
    --hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000   # never counts eval docs
python -m nla.future_lens.baselines probe --train-parquet $D/train.parquet --parquet $D/eval.parquet --leakage --out $E/baselines.jsonl
python -m nla.future_lens.baselines leakage --parquet $D/eval.parquet --readouts $E/readouts_rl_em_s0.jsonl --out $E/leakage.jsonl
python -m nla.future_lens.plots $E/*.jsonl --out plots/ --sft sft --rl rl_em      # pools seeds by --group
```

## What to watch

| Signal | Where | Healthy |
|---|---|---|
| `heldout/ce_gap` = L(shuffled) − L(true) | SFT | clearly > 0; the alpha with the largest gap wins |
| `heldout/p1_off0`, `p1_off1` | SFT | above the bigram baseline before starting RL |
| `reward/group_std_mean`, `reward/frac_zero_var_groups` | RL | std > 0, zero-variance fraction well below 1 |
| `av/kl_to_ref`, `av/len_violation_frac` | RL | KL small and stable; violations → 0 |
| `eval/p1_gap/L{l}_N{n}` = real − shuffled | RL | grows with training; the number that matters |
| `av/marker_bad_count` | RL | 0 (template/tokeniser drift otherwise) |
| `leak_agree_ngram_when_ngram_wrong` vs `leak_agree_future` | leakage | the decoder should side with the future, not with a wrong past-only prior |

## Files

| File | Role |
|---|---|
| `nla/future_lens/inject.py` | embedding-replacement hook, Karvonen dispatch, affine map |
| `nla/future_lens/data.py` | parquet columns, sidecar `future_lens:` block, prompt builder, shuffle control |
| `nla/future_lens/collect.py` | extraction (replaces datagen stages 0-3) |
| `nla/train_sft.py --future-lens` | warm-start; held-out CE gap + precision monitor |
| `nla/future_lens/rewards.py` | exact-match / target-logprob rewards, Dr. GRPO advantages |
| `nla/future_lens/train_rl.py` | GRPO trainer (no critic) |
| `nla/future_lens/eval.py` | controlled eval → JSONL |
| `nla/future_lens/baselines.py` | n-gram, linear probes, leakage probes |
| `nla/future_lens/plots.py` | tables, success check, PNGs |
| `configs/future_lens/*.yaml` | tuned run configs |
| `scripts/smoke_future_lens_cpu.sh` | end-to-end CPU smoke (Qwen3-0.6B) |
| `scripts/runpod_future_lens.py` | pod-per-stage launcher (never runs on its own) |
| `tests/test_future_lens.py` | unit tests (injection exactness, rewards, surprisal, n-gram) |
