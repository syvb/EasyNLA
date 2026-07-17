# Capability degradation under a continuous NLA language bottleneck

During generation, replace the residual stream at the NLA layer (24) with its
round-trip reconstruction `ĥ = AR(AV(h))` at **every generated token**, and
measure which capability areas degrade. `AR ∘ AV` is a lossy codec whose
channel is natural language; substituting the round-trip at layer 24 forces
everything the model carries past that depth through the channel. Capability
retention by domain is a behavioral map of what the codec preserves — a
stronger, domain-stratified probe than FVE on generic text.

Checkpoint: `asher577/nla-qwen-3-8b` (AV LoRA on the `asher577/nla-warmstart-2x`
AV-SFT base; 25-layer AR critic; held-out FVE ~78-79%, max_new 150,
extraction 100% / truncation 0%). Target M: stock `Qwen/Qwen3-8B`,
**non-thinking mode**, greedy — every generated token costs one ~150-token AV
explanation (~150× overhead), so thinking traces are off and output lengths
stay comparable across conditions.

## The intervention

For each generated token step (prefill left clean): layers 0..24 run normally →
h; AV verbalizes h (vLLM, Karvonen norm-matched injection at the marker, temp
1.0 as trained); AR reconstructs ĥ from the explanation; **ĥ is rescaled to
‖h‖** (the codec is direction-only by construction: injection norm-matches and
the AR loss normalizes both sides to √d — without the rescale we'd measure
scale miscalibration, not information loss); layers 25..35 compute and cache KV
from the substituted stream; next token sampled greedily. Layers ≤24 only ever
see (and cache) clean streams, layers >24 only reconstructed ones — internally
consistent, and errors compound through attention, which is the regime under
study.

## Conditions

- **C0 clean** — validates the harness (expect scores near published
  Qwen3-8B non-thinking numbers; treat ±2-3pp as a pass — harness/extraction
  differences make exact parity a time sink; C0′≡C0 is the real check).
- **C0′ identity** — h substituted for itself through the full interception
  machinery (serialization round-trip + same write path, no arithmetic).
  Must match C0 **token-for-token**; catches dtype/indexing/off-by-one-layer.
- **C1 NLA round-trip** — the measurement. Stochastic through z-sampling →
  2-3 codec seeds on gsm8k/triviaqa gauge the variance.

Deferred (slot into the same tap later): distortion-matched noise control,
shuffled-explanation, interpolation, every-k-th-token substitution — the last
is the *contingency* if C1 floors every benchmark (FVE 0.78 ≈ cosine ~0.9 per
step, compounded over hundreds of tokens, may leave no domain contrast).

## Two-stage evaluation

**Stage A** (cheap dense map, `stage_a.py`): ~64 seqs × 256 tok × 12 domains
(fineweb, wikipedia, code, openwebmath, arxiv, pubmed, chat, zh/fr/hi wiki,
synthetic JSON/tables, **on-policy** = M's own greedy outputs teacher-forced —
the codec trained only on finefineweb, so chat-mode activations are
off-distribution and that gap gets its own domain). One clean forward captures
h everywhere; every position verbalized+reconstructed (batched — >95% of
cost); one patched forward with all positions replaced; per-token ΔNLL,
KL(clean‖patched), top-1 flip, cosine (free). Caveat: verbalizes clean-history
activations — compounding is Stage B's job.

**Stage B** (`stage_b.py` + offline `score_stage_b.py`): generation-mode
benchmarks — gsm8k 200, math500 150, humaneval 164, mbpp 150, triviaqa 300,
popqa 300, mmlu_pro 40/category (per-category CIs at n=40 are ±15pp —
aggregate to clusters in analysis), mgsm 4×100, ifeval 150, 50 open-ended
fluency prompts (PPL under clean M + repetition). Retention =
score(C1)/score(C0), paired bootstrap CIs.

## Order of operations

1. `setup_box.sh` — venv (pinned vllm 0.19 + patched vllm-lens), downloads,
   AV LoRA merge.
2. `sanity_checks.py` — custom loop ≡ HF generate; C0′ ≡ C0; tiny C1 smoke.
3. `norm_scatter.py` — reproduce held-out FVE (~0.78) through our plumbing +
   confirm AR norms are uncalibrated. **Both 2 and 3 must pass before
   anything downstream.**
4. `stage_a.py` (go/no-go: if ΔNLL is catastrophic everywhere, expect floors)
   → `analysis_stage_a.py`.
5. Stage B pilot (24 gsm8k + 8 fluency, C0 vs C1) — abort/redesign if floored.
6. Full Stage B: C0, C0′, C1, +2 codec seeds on gsm8k/triviaqa. `run_all.sh`
   does 2-6 in order with sentinels.

## Analysis & priors (registered up front)

1. Domain map: Stage B retention and Stage A ΔNLL/KL side by side.
2. Attribution: correlate per-domain behavioral degradation with Stage A
   reconstruction cosine. Off-trend domains (good cosine, big drop) are the
   interesting ones — cosine weights all directions equally, behavior doesn't.
3. Read the explanations on the worst domains (the z corpus is stored in
   full): does the AV describe math vaguely while dropping operands? Is
   non-English paraphrased into English with specifics lost?
4. Caveat until the noise control runs: C1 degradation conflates information
   loss with off-manifold fragility at layer 24.
5. Priors: worst — exact-token-identity domains (arithmetic operands, code
   identifiers, non-English, verbatim recall); mildest — style, fluency,
   broad-topic knowledge. Also expected: on-policy/chat Stage A domain
   measurably worse than fineweb (train-distribution gap).

## Cost envelope (1× H200, ~$3/hr)

~150-token explanations (not ~500 — the checkpoint's cap): Stage A ≈ 197k
positions ≈ 30M AV tokens ≈ 1-2h. Stage B C1 ≈ 440k generated tokens ≈ 66M AV
tokens ≈ 3-5h at batch 64-128; C0/C0′ ≈ 1h. Whole first pass incl. setup:
**8-13 GPU-h ≈ $25-40** (budget ~2× for first-run debugging). vLLM gets 0.35
of the card; HF side (M 16G + AR 11.5G) shares the rest.

## Gotchas already encoded

- adapter_config's base path is a staging artifact → base passed explicitly.
- vLLM prefix caching stays OFF (identical prompts, per-request steering).
- Injection verified per-request via the patched vllm-lens steer log.
- Marker is ㈎ U+320E; sidecar asserts catch tokenizer drift.
- Don't name checkpoint dirs `av`/`ar` etc. (package shadowing).
- Qwen3 massive-activation positions: report Stage A with/without top-1%-norm.
