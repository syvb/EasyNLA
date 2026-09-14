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
(`scripts/smoke_future_lens_cpu.sh`, ~10 min) and on one 80 GB GPU with Qwen3-8B-Base
(`scripts/runpod_future_lens.py`, one pod per stage).

## Conventions

* **Readout.** K = N+1 tokens starting at `x_{t+1}`. "precision@1 at N" scores the
  readout token at offset N, i.e. `x_{t+1+N}` (Future Lens: N=1 ↔ `x_{t+2}`). K is drawn
  per row from `{1,2,3,4,5,9}` at collection time (the spec's N ∈ {1,2,4,8}, plus N=0 as the
  logit-lens sanity check and N=3 because the success criterion is stated at N=2 or 3);
  the prompt states K and the layer. A readout shorter than K is a miss at the missing
  offsets, in training logs and in eval alike.
* **Model.** `Qwen/Qwen3-8B-Base` plays target and decoder (one copy of the weights, LoRA
  on the decoder), as in Future Lens (base GPT-J). Prompts use `prompt_format: plain`
  (template text + newline, then the readout; recorded in the sidecar by the collector),
  because the chat template is meaningless to a base checkpoint. Post-trained checkpoints
  use `chat` (thinking off). The base decoder is worse at following the prompt before SFT;
  it learns the format from the 200k SFT examples.
* **Labels are token ids** (`target_ids[:k]`), never re-tokenised text. `--label greedy`
  (the default in the run configs, and Future Lens's convention: "the generated tokens outputs
  of GPT-J through greedy decoding") reads out the **target's own greedy continuation** from
  x_<=t; `--label text` reads out the corpus continuation. The loader swaps the columns, so
  SFT labels, the exact-match reward, p@1/p@5 and the n-gram baselines all follow the choice,
  which travels with the checkpoint (`future_lens.json`). The model's greedy continuation
  agrees with the text only ~48/30/20% of the time at N=1/2/3, so under text labels most of
  the target at N>=2 is unrecoverable from any state.
* **Two precision conventions.** `p1` is free-running: the decoder generates all K tokens
  itself. `tf_p1` is Future Lens's teacher-forced version (their Eq. 11): the label prefix is
  fed after the prompt and the argmax at offset N is scored. The bigram baseline has both too;
  its teacher-forced value (~0.23 for Qwen3's tokenizer on FineWeb, the paper's 20.1%) is the
  right gate for `tf_p1`, the free-running one (~0.05 at N=1) for `p1`.
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
* **Layers.** The collector stores 4/8/12/16/20/24/28/32; SFT and RL train on 8–32
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
  seeded random subsample of positions shared across layers, and so is the SFT held-out set.
  Under greedy labels the loader drops rows whose greedy first token disagrees with the
  corpus token (bf16 near-tie flips between the filter pass and generate) or whose
  continuation contains EOS, and prints the counts.
* **Controls.** `shuffled` (another position's vector, same layer), `none` (no injection),
  `wrong_layer` (the layer-4 vector, which the decoder never trained on: out-of-distribution),
  `cross_layer` (the vector of the farthest *trained* layer at the same position: in-distribution,
  wrong layer). Baselines: n-grams (free-running and teacher-forced), linear/leakage probes,
  and `target_window`: the frozen target greedy-continues only the last m ∈ {1,2,4,8,32} tokens
  before t. That last one bounds what a decoder could score by recovering a few recent tokens
  from the state and re-simulating the model; `real − shuffled` alone cannot rule that out
  because shuffling removes token identity along with everything else.
* **What H1 can and cannot show.** GRPO continues the SFT adapter for 4k steps × 128
  rollouts, and there is no SFT-continued arm with matched steps (deliberately skipped for
  now), so a GRPO gain is "RL vs. stopping SFT here", not "RL vs. more SFT". The paper's own
  Table 2 is flat beyond N=1 (48.4 / 43.7 / 46.9 at N=1/2/3), so there is no "N=2/3 gap" to
  close; the question is simply whether the sequence-level objective reads more out at N=2,3.
  The success check therefore reports both `p1` (free-running, what the reward optimises) and
  `tf_p1` (teacher-forced, the paper's metric); a gain in `p1` without one in `tf_p1` means the
  decoder became more self-consistent, not that the state became more readable.
* **Preceding tokens** (`prev_ids`) and document token ids (`docs.parquet`) exist for
  the leakage probe, the target-window baseline and the target-logprob reward / surprisal only.
  No decoder code path reads them.

## Commands (Qwen3-8B-Base)

```bash
# 1. data (1x H100, ~1.5 h): ~220k positions x 6 layers, 6k eval positions, top-1-correct filter
python -m nla.future_lens.collect --base-ckpt Qwen/Qwen3-8B-Base --corpus HuggingFaceFW/fineweb \
    --corpus-config sample-10BT --n-train-docs 5500 --n-eval-docs 300 --layers 4,8,12,16,20,24 \
    --positions-per-doc 40 --eval-positions-per-doc 20 --batch-size 8 --greedy all --prompt-format plain --out-dir $D

# 2. alpha / injection sweep (each ~7 min): pick the largest heldout/readout_ce_gap among true-label runs
L=8,12,16,20,24,28,32
for m in 1 2 4; do for s in "" --shuffle-activations; do
  python -m nla.train_sft --config configs/future_lens/sft_alpha_sweep.yaml --base-ckpt Qwen/Qwen3-8B-Base \
      --parquet $D/train.parquet --heldout-parquet $D/eval.parquet --layers $L --label greedy \
      --injection replace_embed --alpha-mult $m $s --save-dir $C/sweep_replace_embed_a${m}${s:+_shuf}
done; done   # launcher: `launch alpha_sweep --sweep "replace_embed:1,2,4;karvonen:1" --sweep-shuffle-mults 2`

# 3. SFT warm-start (8000 steps ~ half an epoch, ~1.7 h). Gate: heldout/p1_off0, p1_off1 > bigram
#    (free-running p1 vs the free-running bigram, tf_p1 vs the teacher-forced bigram ~0.23)
python -m nla.train_sft --config configs/future_lens/sft.yaml --base-ckpt Qwen/Qwen3-8B-Base \
    --parquet $D/train.parquet --heldout-parquet $D/eval.parquet --layers $L --label greedy \
    --alpha-mult $ALPHA --num-steps 8000 --save-dir $C/sft_replace_embed_a${ALPHA}_greedy

# 4. GRPO (4k steps, ~4 h/seed). Gate: reward/group_std_mean > 0; eval/p1_gap grows.
#    label / injection / alpha are read from the SFT checkpoint's future_lens.json
python -m nla.future_lens.train_rl --config configs/future_lens/rl.yaml --base-ckpt Qwen/Qwen3-8B-Base \
    --av-ckpt $C/sft_replace_embed_a${ALPHA}_greedy/iter_0002000 --parquet $D/train.parquet \
    --eval-parquet $D/eval.parquet --layers $L --save-dir $C/rl_exact_match_s0 --reward exact_match --seed 0

# 5. controls + baselines (eval reads label/injection/alpha from the adapter too)
python -m nla.future_lens.eval --base-ckpt Qwen/Qwen3-8B-Base --adapter $C/rl_exact_match_s0/iter_004000 \
    --parquet $D/eval.parquet --out $E/rl_exact_match_s0.jsonl --layers $L --wrong-layer 4 \
    --conditions real,shuffled,none,wrong_layer,cross_layer --surprisal --surprisal-rows 256 \
    --tag-kv checkpoint=rl_exact_match_s0_iter_004000,group=rl_exact_match,seed=0 \
    --dump-readouts $E/readouts_rl_exact_match_s0_iter_004000.jsonl
python -m nla.future_lens.baselines --label greedy ngram --parquet $D/eval.parquet --out $E/baselines.jsonl \
    --hf-corpus HuggingFaceFW/fineweb --hf-config sample-10BT --hf-docs 20000   # never counts eval docs
python -m nla.future_lens.baselines --label greedy probe --train-parquet $D/train.parquet --parquet $D/eval.parquet \
    --layers $L --leakage --out $E/baselines.jsonl
python -m nla.future_lens.baselines --label greedy target_window --parquet $D/eval.parquet --out $E/baselines.jsonl
python -m nla.future_lens.baselines --label greedy leakage --parquet $D/eval.parquet \
    --readouts $E/readouts_sft_*.jsonl $E/readouts_rl_*.jsonl --out $E/leakage.jsonl
python -m nla.future_lens.plots $E/*.jsonl --out plots/ --sft sft_replace_embed_a2.0_greedy --rl rl_exact_match
#   pools seeds by --group, never across labels/eval sets; success check pooled over layers + per layer, p1 and tf_p1
```

## Relation to Future Lens (Pal et al. 2023)

Same: base checkpoint; positions filtered to top-1-correct next-token prediction; the target
frozen; labels = the target's greedy continuation; confidence buckets. Different: our decoder
is a LoRA'd copy of the target reading a natural-language prompt with the state written into
one token (EasyNLA-style) rather than a 10-token soft prefix; our primary metric is
free-running. Their appendix says a prompt optimised for N=1 "generalizes surprisingly well
to other N, but optimizing for other N does not perform well for N=1" — it does *not* say
N=2/3 training fails at N=2/3, which the spec's motivation overstates.

## What to watch

| Signal | Where | Healthy |
|---|---|---|
| `heldout/readout_ce_gap` = L(shuffled) − L(true) on readout tokens | SFT | clearly > 0; the alpha with the largest gap wins (`ce_gap` includes the EOS token) |
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
