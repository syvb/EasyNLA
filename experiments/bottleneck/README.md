# Capability degradation under a continuous NLA language bottleneck

During generation, replace the residual stream at the NLA layer (24) with its
round-trip reconstruction `ĥ = AR(AV(h))` for **every generated token**, and
measure which capability areas degrade. `AR ∘ AV` is a lossy codec whose
channel is natural language; substituting the round-trip at layer 24 forces
the *generated-position* state that the model carries past that depth through
the channel. Capability retention by domain is a behavioral map of what the
codec preserves — a stronger, domain-stratified probe than FVE on generic text.

**Scope caveat (registered up front):** the bottleneck is not total. Prefill
is left clean, so the prompt's KV at every layer — including above the tap —
remains directly attendable: prompt-resident information (exact operands, code
identifiers, non-English question text) can be re-fetched from clean KV even
if the codec drops it; only *generated-position* state and "what to fetch"
must survive the channel. The optional `nla_prompt` condition (substitute
prompt positions too) quantifies this bypass on a short-prompt task.

Checkpoint: `asher577/nla-qwen-3-8b` (AV LoRA on the `asher577/nla-warmstart-2x`
AV-SFT base; 25-layer AR critic; held-out FVE ~78-79%, max_new 150,
extraction 100% / truncation 0%). Target M: stock `Qwen/Qwen3-8B`,
**non-thinking mode**, greedy — every generated token costs one ~150-token AV
explanation (~150× overhead), so thinking traces are off and output lengths
stay comparable across conditions.

## The intervention

Every generated token must be sampled from a substituted stream, so the tap
fires (a) at the **last real prompt position during prefill** — that stream
produces the first generated token; without this, single-token answers
(MMLU-Pro letters) would bypass the codec entirely — and (b) at the current
position of every decode step. Per substitution: layers 0..24 run normally →
h; AV verbalizes h (vLLM, Karvonen norm-matched injection at the marker, temp
1.0 as trained); AR reconstructs ĥ from the explanation; **ĥ is rescaled to
‖h‖** (the codec is direction-only by construction: injection norm-matches and
the AR loss normalizes both sides to √d — without the rescale we'd measure
scale miscalibration, not information loss; note this hands the codec the true
norm for free — ‖AR(z)‖ is logged so norm-information content stays
analyzable); layers 25..35 compute and cache KV from the substituted stream;
next token sampled greedily. Layers ≤24 only ever see clean streams — h is
therefore a pure function of token history, which makes every run exactly
replayable offline from the stored output token ids.

## Conditions

- **C0 clean** — validates the harness (expect scores near published
  Qwen3-8B non-thinking numbers; treat ±2-3pp as a pass — harness/extraction
  differences make exact parity a time sink; C0′≡C0 is the real check).
- **C0′ identity** — h substituted for itself through the full interception
  machinery (serialization round-trip + same write path, no arithmetic), at
  the same positions C1 substitutes. Must match C0 **token-for-token**.
- **C1 nla** — the measurement. Stochastic through z-sampling → 2-3 codec
  seeds on gsm8k/triviaqa gauge gross instability (3 seeds bound variance
  only loosely — don't cite as "variance was checked").
- **C1p nla_prompt** — C1 plus substitution of every prompt position
  (quantifies the clean-prompt-KV bypass; run on triviaqa).

Deferred (slot into the same tap later, replaying from stored output_ids):
**shuffled-explanation first** — AR(z) from a different position is exactly
manifold-matched to C1's substitutions, needs only the stored z corpus —
then a noise control matched per-task on the empirical per-position cosine
distribution. Also: every-k-th-token substitution — the *contingency dial* if
C1 floors all long-horizon tasks (pick k from Stage A flip rates, and note it
leaks clean KV on unsubstituted steps).

## Two-stage evaluation

**Stage A** (cheap dense map, `stage_a.py`): ~64 seqs × 256 tok × 12 domains
(fineweb, wikipedia, code, openwebmath, arxiv, pubmed, chat, zh/fr/hi wiki,
synthetic JSON/tables, **on-policy** = M's own greedy outputs — open-ended /
math / code / QA prompts, train splits only — teacher-forced; the codec
trained only on finefineweb, so chat-mode activations are off-distribution and
that gap gets its own domain). One clean forward captures h everywhere; every
position verbalized+reconstructed (>95% of cost); then three codec-free
patched forwards reusing the same ĥ:
1. **full** — all positions replaced (= teacher-forced C1; position i's
   metrics include reconstructed history, i.e. compounding);
2. **nosink** — position 0 + top-1%-‖h‖ (Qwen massive-activation sinks) kept
   clean — the real robustness check, since a mangled sink direction
   contaminates every position through attention;
3. **one-position-only** calibration (first batch/domain, 8 strided
   positions) — single-step damage with clean history, decomposing full into
   per-step vs compounded.
Metrics per token: ΔNLL, KL(clean‖patched), top-1 flip, cosine (free).

**Stage B** (`stage_b.py` + offline `score_stage_b.py`): generation-mode
benchmarks — gsm8k 200, **gsm8k_short 200** (same problems, answer-only,
~16 substituted steps — the horizon control: long-vs-short math on the same
items separates "codec drops math content" from "long chains compound
per-step damage"), math500 150, humaneval 164, mbpp 150, triviaqa 300,
popqa 300, mmlu_pro 40/category (±15pp per-category CIs — aggregate to
clusters), mgsm 5×100 (**en**+fr+zh+ru+sw; en gives a same-item cross-language
contrast), ifeval 150 (doubles as the instruction-following/format-compliance
internal control), 50 open-ended fluency prompts. Retention =
score(C1)/score(C0), paired bootstrap CIs; fluency reported as (ΔNLL under
clean M, rep3, length, hit-cap rate) **jointly** — PPL alone rewards
degenerate loops; format-compliance rates and chance-adjusted MMLU-Pro
retention reported alongside.

## Order of operations

1. `setup_box.sh` — venv (pinned vllm 0.19 + patched vllm-lens, peft 0.19.1,
   datasets 5.0.0), downloads, AV LoRA merge.
2. `sanity_checks.py` — custom loop ≡ batched HF generate; C0′ ≡ C0; tiny C1
   smoke incl. the prefill-substitution log check.
3. `norm_scatter.py` — reproduce held-out FVE (~0.78) through our plumbing +
   confirm AR norms are uncalibrated. **Both 2 and 3 must pass first.**
4. `stage_a.py` → `analysis_stage_a.py` (go/no-go + expectations calibration).
5. Stage B pilot spanning horizons (gsm8k, gsm8k_short, triviaqa, mmlu_pro,
   fluency; small slices). Decision rule: abort/redesign only if floored
   **across horizons** — a long-horizon-only floor is itself a result.
6. Full Stage B: C0, C0′, C1, +2 codec seeds, + nla_prompt on triviaqa.
   `run_all.sh` does 2-6 in order with sentinels.

## Analysis & priors (registered up front)

1. Domain map: Stage B retention and Stage A ΔNLL/KL side by side; retention
   vs mean steps-to-answer across tasks (the horizon axis); absolute deltas
   alongside ratios wherever clean < ~0.4 (popqa).
2. Attribution: correlate per-domain Stage B retention with Stage A cosine
   (noting Stage A "full" is compounded — use the one-step calibration to
   bridge). Off-trend domains (good cosine, big drop) are the story.
3. Read the explanations on the worst domains (the z corpus is stored in
   full): does the AV describe math vaguely while dropping operands? Is
   non-English paraphrased into English with specifics lost? Also: per-step
   cosine-vs-step curves from the C1 steplogs (does the codec degrade as
   prefixes drift off-distribution?).
4. Caveats for the writeup (mandatory): prefill-KV bypass; horizon confound
   (partially resolved by gsm8k_short); information-loss vs off-manifold
   fragility undecided until the shuffled-z control; norm handed to the codec
   by the rescale; format-compliance conflation (report compliance rates);
   only registered contrasts get confirmatory language — off-trend domains
   are exploratory; sub-15pp retention orderings are below resolution.
5. Priors: worst — long-horizon multi-step tasks, then exact-token-identity
   content generated (not prompt-resident) — intermediate arithmetic results,
   generated code identifiers, non-English generation; mildest — style,
   fluency, broad-topic knowledge, short answers fetched from clean prompt
   KV. Also expected: on-policy/chat Stage A domain worse than fineweb.

## Initial run (halved) vs full design

The config + run_all.sh as committed run the **initial (halved)** matrix; the
full design is restored by deleting `task_sizes`/`mgsm_langs`/`max_new_caps`
from config.yaml and uncommenting the extension lines in run_all.sh. Cost
logic: a cohort runs until its slowest row finishes, so *steps* (cohorts ×
max_new), not problem counts, are the unit — sizes are cohort-aligned to 192
(200 problems would cost two full cohorts).

Initial run keeps one benchmark per capability axis: gsm8k + gsm8k_short (the
horizon pair, 192 each), humaneval (code, capped 384), triviaqa + popqa
(recall, 192 each), mmlu_pro (40/cat), mgsm en/fr/zh × 64 (same-item
cross-language contrast kept), fluency; +1 codec seed on gsm8k/triviaqa;
Stage A at 32 seqs/domain (all 12 domains — domain breadth is the point,
per-token n stays ~8k/domain). **Deferred to the extension run** (per-cohort
shards make it incremental): math500, mbpp, ifeval, mgsm ru/sw + full 100/
lang, QA n→300, codec seed 2, nla_prompt, Stage A → 64 seqs. Known trade-offs:
per-language mgsm CIs widen to ±12pp (directional only); no dedicated
instruction-following benchmark (format-compliance rates partially cover it);
no hard-math point (gsm8k+short still carries the math/horizon story).

## Cost envelope (1× H200, ~$3/hr)

~150-token explanations (the checkpoint's cap); C1 cohort decode steps are a
synchronous vLLM generate() over active rows, ~5-7s each at cohort 192 (eager).

Initial run: setup 1-1.5h; gates 0.3-0.5h; Stage A (~100k round-trips)
0.5-0.8h; pilot ~0.5h; C0+C0′ 0.3-0.5h; C1 ≈ 1,400 steps ≈ **2-2.7h**;
+1 seed 0.4-0.6h → **~5-7 GPU-h ≈ $14-22** clean path, **$21-30** with
debugging reserve. Extension run (deferred set, ~6-8h more) brings the total
to the full-design **~15-25 GPU-h ≈ $45-75**. vLLM gets 0.35 of the card; HF
side (M 16G + AR 11.5G) shares the rest.

## Gotchas already encoded

- adapter_config's base path is a staging artifact → base passed explicitly.
- vLLM prefix caching stays OFF (identical prompts, per-request steering).
- Injection verified per-request via the patched vllm-lens steer log.
- Marker is ㈎ U+320E; sidecar asserts catch tokenizer drift.
- Don't name checkpoint dirs `av`/`ar` etc. (package shadowing).
- Never `hash()` for sampling seeds — salted per process; conditions run in
  separate processes and would silently sample different problem sets.
- Script-based HF datasets are dead on datasets≥3 — all loaders parquet-native.
- Prefill logits capped via `logits_to_keep=1` (full [B,S,V] prefill logits at
  cohort 192 would be ~17GB).
- Stage B resume is per-cohort (shards); steplogs written before the shard
  sentinel so a crash can't lose them.
