# Plan: L24 probe test + SFT with soft-token explanations

Goal: find out whether the Haiku explanations of soft-token readouts (`scripts/heads_soft_token_explain.py`)
are good warm-start data for a head-contribution NLA. Two questions, in order:

1. **Probe test.** Do the explanations still carry the write? Cross-fitted ridge from explanation features to
   the write's coordinates, per head, against the raw readouts and a dense ceiling.
2. **SFT.** Train an AV (write → explanation) and an AR (explanation → write) on them and run the pre-RL checks.

Constraints, unchanged:
- Warm-start text is a function of the write alone: no attention pattern, no document, no residual.
- The AV prompt names no head.
- RunPod secure cloud only.

## Scope

- Qwen3-8B, **layer 24, all 32 heads**.
- **1,500 examples per head** (48k total), drawn by the existing `select` stage: doc-level splits, at most
  one example per document per head, weighted toward larger-than-typical writes
  (`0.25 + 0.75·min(dev/rms, 4)`).
  - Per head: 650 av_sft + 650 ar_sft + 200 eval.
  - The probe uses all 1,500 per head. The SFT uses the splits.
- Corpus: 10k FineFineWeb docs (200 files × 50), 10 positions per doc, L24 only, which gives 100k records.
  - Split fractions are av 0.35 / ar 0.35 / eval 0.15 / rl 0.15.
  - That leaves ≥1,500 docs in every split for 650 / 650 / 200 per head.

## Pipeline (one held pod, `scripts/runpod/heads_l24_remote.sh`)

| stage | what | est. time |
|---|---|---|
| install | `pip install -e .` | 3 min |
| datagen | `run_pipeline --stages corpus,extract,stats,select` (`configs/datagen/heads_qwen3_8b_l24.yaml`; no `prepare`: the attention-based describer is not used) | 10 min |
| readouts | `scripts/heads_soft_token_readouts.py` on `warmstart/selection.parquet`. The listed outputs follow it. | 12 min |
| explain | `scripts/heads_soft_token_explain.py` (Haiku 5.5, v26 prompt, thinking medium, concurrency 120; passes capped at 75 + 20 min; needs ≥95% coverage) | 25–50 min |
| probe | `scripts/heads_explain_probe.py` (learning curves at 200/400/800/1500 per head; features below) | 15–20 min |
| export | `export_sft --descriptions explain.parquet` | 2 min |
| ar_sft | LoRA r128 + value head, `--ar-raw-loss --value-head-lr-mult 10 --ar-num-layers 25`, 2 epochs, bs 64, lr 1e-4 | 5 min |
| av_sft | LoRA, `--inject-preserve-magnitude`, 2 epochs, bs 32, lr 1e-4, held-out loss on 2,000 eval rows | 40–50 min |
| checks | `scripts/heads_sft_checks.py` on the 6,400 eval rows | 15 min |

The readouts stage writes:
- **v:** export_sft's `normalized_writes`, so the readouts, the probe targets and the SFT vectors are
  the same vectors.
- **avg16:** the best text readout from the diagnostics.
- **lens:** the write's direct vocabulary projection.
- **targets:** v projected onto the head's orthonormal write basis, 128-d.
- **hidden:** the final hidden state of the injected prompt, a dense ceiling, stored as float16.

**Probe features:**
- `avg16` and `lens` (TF-IDF), the inputs the explanation sees
- `explain` (TF-IDF) and `explain_embed` (Qwen3-8B mean-pooled)
- `hidden` (dense)
- the controls `explain_shuf` (explanations permuted within each head) and `empty`

The expected reference points come from the 5-head diagnostics: avg16 0.163 at 1,600 per head, hidden 0.635.

Reference caveats:
- The diagnostics' 1,600 column trains on 800 per fold, against 750 here.
- Their normalization capped ‖v‖ at 4. Here the cap is min(q0.999/rms, 4), which differs on ≤0.3% of
  the selected rows.

The probe also reports a copying measure: the share of each explanation's content words that are one
of its own readout tokens. A high explain/avg16 ratio with high copying would mean near-lists rather
than a real rewrite.

**Pass criteria (the SFT runs either way):**
- **Probe:** the better of `explain` and `explain_embed` keeps most of `avg16`'s FVE. A ratio of at least
  0.5 at 1,500 per head is a pass. A ratio under 0.3 means the rewriting loses too much. TF-IDF works
  better on token lists than on prose, so the embedding is the fairer view. The per-head gap is reported
  with its SE over heads.
- **SFT:** `ar/gold` (the AR on held-out gold explanations) beats `ar/gold_shuf` and `ar/empty`. `av/true`
  (the AR on AV-generated explanations) beats `av/shuf` (the AV given another example's write).
  - Both checks get 95% bootstrap CIs, over rows and over heads.
  - An AR-free companion runs a per-head ridge from the text alone to the write, on the gold, AV-true
    and AV-shuf texts. A weak AR then can't hide a good AV.

**Total:** about 2 h 15 min to 3 h 10 min on one secure H100 (~$9–13), plus about $15 of Haiku
(range $12–32, depending on thinking length).
- The launcher runs **without `--hold`**, so it fetches and deletes as soon as DONE or FAILED appears.
  `--max-hours` is 6, and the watchdog self-deletes at 6.5 h.
- The launcher runs detached (`nohup setsid`) on this always-on VM.

## Decisions and their reasons

- **AV and AR train on disjoint documents.** This is the repo's split design. It keeps RL and the
  checks clean and costs us half the data per model. With 650 per head per model, that's about 3× the
  quick test.
- **Two epochs.** The quick test used two, and the data is still small. The repo's one-epoch default
  was tuned for the much larger residual-stream datasets.
- **The AR text still names the head** (`Attention head L24H7: …`). With one layer the name is all it
  needs to know which subspace to write into. The AV prompt names none.
- **AV injection stays at layer 1, as in the standard NLA.** The readouts inject at the head's own layer,
  but a LoRA AV should learn to read the vector either way. Changing both at once would confound.
- **The AV prompt** asks for "one or two sentences", to match the explanation length (up to 45 words). It
  previously said "short clauses".
- **Rule-breaking explanations are dropped.** Rows that still break the token/opening rules after 4
  rewrites are saved with `ok=False`, and both the export and the probe drop them.
- **Disjoint AV/AR documents are a choice.** That matters for RL, not for these checks. If the AV is
  data-starved, a rerun can train it on av_sft + ar_sft.
- **First ablation if check 2 fails:** inject the AV at L24, the soft-token site, before concluding
  the data is bad.
- **The filler table** for the explanation prompt (tokens in at least 8% of avg16 lists) is computed over
  the 48k L24 lists. It is one fixed table per dataset, like dev_rms.

## Review (3 Fable reviewers: code, method and constraints, ops)

- No blockers in the experiment logic. All four constraints were verified in the code.
- The ops blockers were fixed:
  - `set -E`
  - the pgrep self-match
  - fetch-before-hold
  - a best-effort fetch on launcher death
- Other changes adopted:
  - a cache loader that survives a line cut off by a killed pass
  - an `ok` column
  - a leak check that matches only the token lists, with plurals
  - CIs and the text-only probe in the checks
  - the copying measure and per-head SEs in the probe
  - concurrency 120 and shorter explain timeouts

## Risks

- **Explanation stage failures** (rate limits, refusals, empty replies):
  - Each example is retried by the SDK, then skipped and logged, never fatal.
  - A second pass covers the gaps.
  - Rows still missing are dropped from the probe and the SFT, and counted.
- **The export ran 20 min in the quick test.** That was per-row matvecs, fixed since then by one matmul per
  head. The readouts stage exercises the same code first, so a regression shows up early.
- **Idle pod after DONE.** There is no hold mode.
- **Dead-script detection:** the launcher's `pgrep` used to match its own ssh shell, so `__DEAD__` could
  never fire. It now uses a `[h]eads_…` pattern, and the remote script uses `set -E` so a failure inside
  a function still writes FAILED.
- **Launcher killed mid-run:** it now does a best-effort fetch before the delete.
