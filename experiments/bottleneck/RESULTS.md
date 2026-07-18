# Initial-run results (2026-07-17) — audited

Post-run review: three independent audits (results/scoring validity with raw-output
reading, figure-vs-data verification, skeptical claims review). Every number below
survived recomputation from the raw artifacts; corrections from the audits are
applied and flagged. Raw outputs, z corpus, steplogs, logs, figures:
`hf://datasets/syvb/nla-bottleneck-results` (initial_run/). Run: 1× H100 NVL,
halved matrix per README §Initial run.

## Retention table (C1/C0, paired bootstrap 95% CI)

| Task | C0 | C1 | Retention |
|---|---|---|---|
| HumanEval (code) | 0.841 | 0.000 | 0/164 — 95% upper ≈ 0.03. Genuine: 0/164 outputs parse as Python at all; total non-code degeneration, not failed tests |
| GSM8K (math CoT) | 0.542 | 0.047 | 0.087 [0.036, 0.146]; seed 1: 0.077 |
| MGSM (5-lang math) | 0.536 | 0.062 | 0.117 — en 0.047 / fr 0.109 / zh 0.031 |
| GSM8K answer-only | 0.089 | 0.016 | 0.18 [0.00, 0.47] — **uninformative**: clean baseline at floor (17/192); the horizon confound is NOT resolved by this cell |
| TriviaQA (recall) | 0.519 | 0.307 | **0.592 [0.485, 0.708]**; seed 1: 0.602 (audit-repaired EM¹; strict first-line EM gave 0.39/0.44) |
| PopQA (long-tail recall) | 0.167 | 0.099 | **0.594 [0.367, 0.880]** (repaired¹; strict gave 0.41) |
| MMLU-Pro (MC) | 0.366 | 0.248 | 0.678 [0.589, 0.766]; chance-adj 0.557 — **partly an option-prior artifact²** |
| Fluency | NLL 0.204 | NLL 1.644 | rep3 0.011→0.409, hit-cap 0.72→0.92: locally fluent but **degenerate/repetitive** (low NLL partly rewards the loops) |

¹ The degraded model often produces the correct entity glued to stray
`</think>` tags or preamble; strict first-line EM scored those 0 (54/189
"wrong" C1 TriviaQA rows contain a gold alias vs 13/189 clean). The repaired
extractor (scoring.py: strip think tags, accept an "answer is X" span)
recovers ZERO additional clean rows — it removes an asymmetric format
artifact, not a loosening. Lenient "alias appears anywhere" upper bounds:
TriviaQA 0.83, PopQA 0.95.

² Under C1, MMLU-Pro predictions collapse toward "A" (30% vs 11% gold base
rate; accuracy 0.42 on gold=A vs 0.23 otherwise; no such split in clean) —
the extractor grabs the first letter of echoed option lists, which *favors*
C1, so the drop is if anything understated, and "preserved recognition"
should be discounted accordingly.

**Harness control:** C0′ (identity substitution) is **byte-identical** to C0 —
output_ids equal element-wise for every pid on all 8 tasks. The entire measured
effect passes through the codec.

**Codec health over ~350k AV calls:** truncation 0.19%, extraction failure
0.025% (after retries; 2.6% needed a retry), unverified injections 0.
Held-out FVE through our plumbing: 0.713 (checkpoint card claims 0.78-0.79 on
its own pool); cosine mean 0.903. ‖AR(z)‖ is uncorrelated with ‖h‖
(r = 0.012; AR under-scales ~3.4×) — the rescale-to-‖h‖ hands the codec the
true norm for free, a limit on how "pure" the information-loss reading is.

## Stage A domain map (teacher-forced, all positions substituted)

ΔNLL ordering: en prose (wikipedia/fineweb/chat/pubmed/arxiv) 1.9-2.2 <
OpenWebMath 3.3 < on-policy 3.9 < fr 4.9 < JSON/code 5.5-5.7 < hi 6.9 <
**zh 10.0 (94% top-1 flips)**. corr(cosine, ΔNLL) = −0.71 (n=12), but both
tails have outliers: zh is +5.0 ΔNLL above the fit (2× predicted; the codec
paraphrases meaning while flipping 94-96% of CJK tokens) and code is −2.0
below it — cosine is a coarse, not reliable, predictor. One-position
calibration: compounded KL is 1.3–2.8× single-step (largest multipliers on
easy prose). Per-step cosine during generation is flat (0.82-0.84, no drift),
but the **first substituted position (prefill) is the worst of the run at
0.60** — and it is the position that decides single-token answers.

## Supported claims (audited wording)

1. Long-horizon generation through the codec collapses: code 0%, CoT math
   ~9%, regardless of language (fr 0.109 ≥ en 0.047 on the same items — the
   damage axis is math/horizon and CJK token identity, **not** "non-English").
2. Recognition and short retrieval partially survive — MC ~0.56 chance-adj
   (discounted by the option-prior artifact²), recall ~0.6 (repaired¹) —
   but the **mechanism is unresolved**: correct entities appear in the first
   1-2 generated tokens, prompt KV above layer 24 is attendable by
   construction, and the `nla_prompt` control that would separate
   codec-carried from prompt-KV-bypass retention was deferred.
3. Format compliance collapses in lockstep with accuracy on long-form tasks
   (`####` 55%→1%, code fences 100%→0%) while short formats survive.
   Mechanistic link from Stage A: whitespace/punctuation/delimiter tokens are
   the worst-reconstructed class in every domain (e.g. code-domain spaces:
   cosine 0.64, flip 0.88) — format loss looks like a *channel* of the
   damage, not an independent additive term.
4. The z corpus qualitatively confirms the registered prior: AV explanations
   describe math structurally while mangling operands, confabulating entities
   (France→"Canada/Ottawa"), and leaking CJK fragments.
5. Two codec seeds gave overlapping CIs (gsm8k 0.087/0.077, triviaqa
   0.592/0.602) — no gross instability; not a variance estimate.

## Error taxonomy (from reading C1-fail/C0-pass samples + the z corpus)

The channel transmits *descriptors* reliably and *bindings* unreliably. Five
recurring kinds, with prevalence:

1. **Same-category entity substitution** — type slot survives, filler doesn't
   (Kriek fruit → "blackcurrant"; Apennines → "the Alps"; z: France →
   "Canada/Ottawa"). 46 triviaqa items are C1-wrong/C0-right; misses are
   near-neighbors, not non-sequiturs.
2. **Operand corruption/drift** — numbers unstable *within one equation*
   ("12 − 2 = 12 − 7 = 2"; "2/5 of 60 = 12"); question quantities vanish.
   Already present in z: digits appear in 95% of math explanations but are
   internally inconsistent ("alleged difference 600, but actual answer 320").
3. **State-stall loops** — corrupted intermediates stall the computation and
   the model re-asserts the last stable phrase (rep3>0.3: humaneval 69%,
   fluency 66%, gsm8k 38%, triviaqa 13%, mmlu 3% — loops track output length).
4. **Structural-glue corruption** — code starts with correct signatures/bodies
   then newline/space fusion breaks syntax ("return x + ySo the function…",
   "largest_divdivisor") and drifts into prose-about-code; matches Stage A's
   worst-class = whitespace/punct tokens. HumanEval 0/164 = cannot sustain
   exact syntax, not absent knowledge.
5. **Register/mode confusion** — non-thinking-mode state doesn't survive:
   "Okay, so I need to…" planning voice in 8-46% of C1 outputs (0% clean),
   stray </think> in 45% of triviaqa C1; answer-then-"Okay"×40 = EOS/
   completion state lost (hit-cap 40-92%). Plus artifacts: mmlu option-echo
   ("A. …" → the option-prior bias) and CJK leakage (27% of z's, 3-6% of
   outputs).

## Standing caveats (until the extension run)

Prefill-KV bypass unquantified (`nla_prompt` not run) — a live confound for
ALL short-answer retention; the rescale gives the codec ‖h‖ for free;
information-loss vs off-manifold fragility undecided until the shuffled-z
control; greedy non-thinking decoding caps clean baselines (gsm8k_short at
floor); n=12 Stage A domains with outlier-driven correlation; sub-15pp
retention orderings below resolution.
