# Predictive-readout NLA: training the verbalizer for a frozen reader

A pilot experiment. It asks one question:

> If you train an activation verbalizer to help an *independent, frozen* language
> model predict what the target model does next, do you get explanations that
> carry more usable information than reconstruction-trained NLA explanations —
> and does that improvement survive changing the reader?

Everything here lives in `nla/pred/` and runs from this repo. It assumes you know
what an NLA is: a verbalizer (AV) that reads a residual-stream activation and
writes text about it, and a reconstructor (AR) that maps that text back to the
activation, with reconstruction error supplying the RL reward.

---

## 1. The idea in one paragraph

Standard NLA rewards the AV for text the AR can invert. That is a closed loop:
the AR is trained alongside the AV and has every incentive to become good at
reading exactly this AV's dialect. An explanation could score well while being
useless to anybody else. This experiment replaces the AR with a **frozen reader**
that is never trained on anything: show it the explanation, then ask how well it
predicts the continuation the *target model itself* produced from that position.
An explanation earns reward only by telling an outsider something true about what
the model was about to do.

```
  standard NLA           activation ──AV──▶ text ──AR──▶ activation'
                                               reward = −‖activation − activation'‖²
                                               (AR co-trains: the loop can close on itself)

  this experiment        activation ──AV──▶ text ──▶ [frozen reader] ──▶ P(continuation)
                                               reward = nats saved on the target
                                               model's own next 24 tokens
                                               (reader never trains: nothing to collude with)
```

---

## 2. What gets measured

### Continuations

For each activation position we take the document prefix that produced it, hand
it back to the frozen target model, and sample **4 continuation branches of 24
tokens** at temperature 1.0. These are fixed once and reused by every later
stage, so every candidate explanation for a position is judged against the same
sampled futures.

### Predictive gain

A reader is shown a short framing, the explanation, and then the continuation,
and we take the teacher-forced log-probability it assigns to the continuation
tokens.

```
gain(z) = score(reader | explanation z) − score(reader | no explanation)
```

reported in **nats per target token**. It is a difference between two prompts
over an identical set of scored tokens.

### Bridges come free

Teacher forcing scores every continuation token in one forward pass, and token
*j* is scored conditioned on the explanation plus continuation tokens `0..j-1`.
So splitting the 24 tokens into three buckets gives three bridge lengths from a
single forward, at no extra cost:

| bucket | reader has already seen | what a gain here means |
|---|---|---|
| tokens 0–8 | the explanation only | the explanation helps from a standing start |
| tokens 8–16 | explanation + 8 real tokens | it still helps with some real context |
| **tokens 16–24** | explanation + 16 real tokens | **headline**: it says something 16 tokens of the actual text did not |

The last bucket is the reward for RL and the headline number in every table. The
earlier two are reported because a method that only helps before the reader has
seen any real text is much weaker than one that still helps after 16 tokens.

### Two readers

| role | model | used for |
|---|---|---|
| training reader | `Qwen/Qwen3-4B-Base` | the RL reward, and checkpoint selection |
| held-out reader | `google/gemma-3-4b-pt` | evaluation only |

The held-out reader is a different model family with a different tokenizer, and
**the trainer never loads it** — not as a metric, not for early stopping. If a
gain appears on Qwen and vanishes on Gemma, the AV found phrasing that suits one
reader rather than words that communicate, and that is a negative result worth
having.

Base (pretrained) readers are used rather than instruction-tuned ones, because
the task is plain continuation likelihood on web text, which is what a base model
is best at and what it was trained to do.

---

## 3. Conditions compared

All three sit on the **same merged AV checkpoint**, differing only by the LoRA on
top. That is what makes the comparison clean, and it also makes the KL reference
exact: the policy is a fresh LoRA, so disabling the adapter returns the SFT
policy. (If you instead start the policy from an existing adapter with
`--init-adapter`, the bare base is no longer the warm start, and the trainer
loads a frozen copy of that adapter as the anchor instead.)

| condition | what it is |
|---|---|
| `sft` | the supervised warm start: `syvb/nanonla-qwen3-8b-L24-av`, no adapter |
| `recon_rl` | published reconstruction-GRPO adapter on that AV: `syvb/nanonla-qwen3-8b-L24-rl-lora#p0.0` (held-out FVE 0.532 → 0.585 at unchanged length) |
| `behavioral_rl` | this experiment: same init, GRPO on the frozen-reader gain |
| `recon_matched` | *optional 4th arm*: the reconstruction objective run through **this** trainer at identical steps, batch, LoRA and rollouts |

The optional fourth arm exists because the published comparator is a rank-16
attention-only adapter from a run of unknown length. Comparing against it alone,
"behavioral beat reconstruction" could just mean the two runs were configured
differently. `configs/pred/rl_recon_matched.yaml` removes that objection for
roughly the cost of the behavioral run.

### The control that actually matters

A generic note ("this is encyclopaedic prose; expect a date next") can help a
reader predict text without saying anything about *this* activation. So the
headline control is **matched versus shuffled**: score checkpoint C's explanation
for position *j* against position *i*'s continuation.

Because the AV prompt is a fixed template whose only per-position content is the
injected vector, "an explanation generated from a mismatched activation" and
"another position's explanation" are the same object. The shuffled arm therefore
reuses the generated texts under a derangement that never pairs two positions
from the same document. That is not a shortcut — it is strictly better, because
both arms then draw from one identical pool of explanation texts and differ only
in which continuation each is paired with.

---

## 4. Hypotheses and what would falsify them

| # | claim | measured by | a failure means |
|---|---|---|---|
| H1 | the objective is learnable | behavioral > sft on the training reader | the reward is too noisy or too weak in this form |
| H2 | **it transfers** | behavioral > sft on Gemma | reader-specific phrasing, not communication |
| H3 | it is activation-specific | matched > shuffled | generic predictive boilerplate |
| H4 | it competes with reconstruction | behavioral vs recon on held-out gain | direct behavioral supervision may not be worth the swap |

H2 is the result. A large Qwen gain with a flat Gemma gain is a clean negative
and should stop the line of work rather than start a bigger run.

Two failure modes get explicit machinery rather than trust:

**Length.** The reward rises with explanation length essentially for free — more
words, more chances to help. A behavioral checkpoint could top the table by
writing longer. So the report always prints median explanation length beside
every gain and breaks gain down by length tercile, and RL runs with a hinged
length penalty scaled to *this* reward (~0.01–0.3 nats), not the reconstruction
trainer's (~0.5).

**Legibility.** RL on any proxy can drift into text that works on the metric and
reads badly. `report.md` carries representative explanations per checkpoint with
the source text and the true continuation beside them, for the qualitative pass.

---

## 5. The pipeline

Five stages, each a CLI, each logging to the `pred-nla` wandb project under one
group so prep → gate → rl → eval read as one experiment.

### 1. Continuations — `nla.pred.continuations`

Streams an NLA dataset parquet, reconstructs each position's prefix from
`detokenized_text_truncated`, and samples the target model's continuations.

A position is **dropped unless its prefix re-tokenizes to exactly
`n_raw_tokens`** — a prefix that does not round-trip is not the context the
activation came from. (On the public warm-start data this passes on 2000/2000
rows checked.) Positions are also capped at 1024 prefix tokens, because the
continuation must come from the *full* context, so a long prefix costs time
rather than being truncated.

Output: one parquet with the activation, the prompt, the branches' text, their
target-model token ids, and the character offset of every token boundary — which
is what lets a different reader bucket the same characters later.

### 2. Gate — `nla.pred.gate`

Before spending anything on RL, check the reward already responds to which
activation an explanation came from. On ~500 held-out validation positions it
scores the SFT and reconstruction explanations, matched and shuffled, on both
readers, and **exits non-zero unless matched − shuffled is positive with a 95%
interval excluding zero on at least one reader**. The launcher honours that exit
code and stops the pod before RL.

This is the cheapest decision point in the experiment. If the score cannot tell a
correct explanation from a mismatched one *before* optimization, no amount of
optimization will fix it, and the thing to fix is the scoring task.

### 3. RL — `nla.pred.train_rl`

GRPO with group-relative advantages, a k3 KL toward the SFT init, and the
reward behind a seam (`nla/pred/rewards.py`) so the reconstruction arm runs
through the identical trainer.

Rollouts are batched **across prompts**. The stock single-GPU trainer generates
one prompt at a time, which on this workload is the entire step; batching turns
32×8 rollouts into a handful of `generate()` calls and is the difference between
a two-hour pilot and a two-day one.

Injection health is checked every step (marker well-formedness and a CJK-output
canary), because a silently broken injection looks exactly like "the method does
not work".

### 4. Eval — `nla.pred.eval`

Generate for every checkpoint first (verbalizer resident), then free it and score
with one reader at a time. Peak memory is one 8B model or one 4B model rather
than all three, and every condition is scored against identical continuations.
Comparisons are paired over positions; intervals bootstrap over **documents**,
since positions from one document are correlated.

### 5. Report — `nla.pred.report`

Writes `report.md`: headline table, bridge-length breakdown, paired checkpoint
comparisons, the transfer ratio, the length check, and representative
explanations.

---

## 6. Two correctness decisions worth knowing about

**The continuation is tokenized on its own and appended to the prefix's tokens**,
rather than tokenizing the joined string. This costs a little naturalness at the
seam and buys the thing the whole comparison rests on: the scored token set is
bit-identical across conditions, so a paired difference is attributable purely to
the prefix. Tokenizing the joined string would let the explanation's last
character change how the first continuation token splits, quietly changing what
is being compared. `tests/test_pred_nla.py` asserts this property directly.

**Bucket membership is decided by character offsets, not token counts.** Qwen and
Gemma tokenize the same continuation differently; characters are the only shared
coordinate system. Reported gain is normalized by the *target model's* token
count for the bucket, which is a fixed denominator, so the two readers' numbers
are directly comparable.

---

## 7. Running it

### Locally, first — no GPU, no spend

```bash
.venv/bin/python -m pytest tests/test_pred_nla.py -q     # ~30 s
bash scripts/smoke_pred_cpu.sh /tmp/pred_smoke           # ~25 min
```

The smoke script runs the real modules on Qwen3-0.6B: it builds a tiny NLA
dataset through datagen's own marker picker and sidecar serializer, warm-starts
an AV so it actually emits `<explanation>` tags, samples continuations, runs the
gate with two readers from different tokenizer families, takes GRPO steps on the
frozen-reader reward, and writes a report. It proves the plumbing, not the
science: a 0.6B verbalizer trained for 40 steps on six documents is evidence of
nothing.

### On a pod

```bash
python scripts/runpod_pred_nla.py plan                        # live prices, no spend
python scripts/runpod_pred_nla.py launch --stages prep,gate --dry-run
python scripts/runpod_pred_nla.py launch --stages prep,gate,rl,eval
python scripts/runpod_pred_nla.py status
python scripts/runpod_pred_nla.py terminate <pod_id>
```

One GPU is enough: the policy is 8B bf16 and the reader 4B bf16, where the stock
reconstruction trainer already fits an 8B policy plus a co-trained 5.9B critic on
one card. There is **no network volume** — HuggingFace is the store, so each
stage pulls what it needs and pushes what it made and the pod can run wherever
there is stock.

### Data

The pool is the FineFineWeb half of `asher577/nla-rl-data-free8` (the other half
is chat transcripts, a different distribution). Its sidecar records a different
corpus file (`corpus_fresh_130k.parquet`) from the `finefineweb_100k.parquet`
the SFT warm start and the published reconstruction-RL checkpoint were built
from. Both are samples of the same underlying corpus, and document-level
disjointness between the two files is **not** byte-verified here — but every
checkpoint is evaluated on the same positions, so residual overlap would inflate
all three arms' absolute gains together rather than favour one of them, and the
comparisons are all paired differences.

Splits are by document hash: ~6% validation, ~12% evaluation, the rest for RL,
with no document crossing a split.

The injection contract was checked across all four artifacts before any of this
was written: the merged AV, the frozen AR, the RL pool and the warm-start data
agree byte-for-byte on the marker character, its token id, both neighbour ids and
the actor prompt template, and all record Qwen3-8B layer 24 at d_model 4096. A
mismatch there would silently inject the vector in the wrong place, which looks
exactly like a method that does not work.

| split | positions | used by |
|---|---|---|
| rl | 12,000 | RL rollouts |
| val | 1,000 | the gate, and RL's in-loop eval (training reader only) |
| eval | 2,000 | the final tables |

---

## 8. Cost

Measured against live RunPod rates (`plan` prints the current numbers):

| stage | wall clock | notes |
|---|---|---|
| continuations | ~0.6 h | 15k positions × 4 branches × 24 tokens |
| gate | ~0.5 h | 500 positions × 2 checkpoints × 2 readers |
| behavioral RL | ~4.6 h | 300 steps at 32 prompts × 8 samples |
| eval + report | ~1.0 h | 2000 positions × 3 checkpoints × 2 readers |
| **total** | **~6.7 h + ~0.35 h startup** | one GPU |

The RL figure is a per-step cost model, not a guess: ~9 s of rollout decode,
~14 s of reader forwards, and ~32 s for the update's forward, backward and
reference passes, so ~55 s/step on an H100-class card. 300 steps at 32 prompts
is 9,600 draws, which is 0.8 epochs of a 12,000-position pool — deliberately
under one epoch. The compute-matched reconstruction arm is cheaper per step
(~45 s, the AR is one forward where the reader is four) and adds ~3.7 h.

At the rates seen on 2026-09-17 that is roughly **$19 on an H100** or **$25 on
an H200** for the four-stage chain, and about **$29 / $39** with the
reconstruction arm. Budget two to three times that for reruns. Startup is not
free either: each pod spends 15–25 minutes pulling the image, installing, and
downloading four models, which is where rerun money actually goes.

`scripts/runpod_pred_nla.py plan` prints the current rates, the per-stage
breakdown, and your balance, and warns when the balance is below the estimate.

---

## 9. What this does not show

Predictive gain is a difference in log-probability between two prompts. It is
not an information-theoretic quantity, and this experiment does not claim:

- that the explanation is *sufficient* for the activation,
- that any fraction of the activation has been "recovered",
- that the explanation is a causal account of the model's computation.

It measures one thing: whether an independent reader predicts the target model's
next tokens better when shown the explanation, and whether that depends on the
explanation matching the activation it came from.

---

## 10. Layout

```
nla/pred/
  reader.py         frozen reader + character-offset span alignment (the core)
  scoring.py        explanations -> gain, with the no-explanation baseline cached
  rewards.py        the reward seam: reader_gain | recon
  continuations.py  stage 1 - sample the target model's futures
  gate.py           stage 2 - the pre-RL activation-specificity gate
  train_rl.py       stage 3 - GRPO
  evaluate.py       shared generate/score/summarize machinery
  eval.py           stage 4 - the final tables
  report.py         stage 5 - report.md
  data.py           positions parquet, document-level splits, derangement
  stats.py          document-clustered and paired bootstrap
configs/pred/       rl_behavioral.yaml, rl_recon_matched.yaml
scripts/            runpod_pred_nla.py, smoke_pred_cpu.sh,
                    make_pred_smoke_data.py, pred_hf_sync.py
tests/              test_pred_nla.py
```
