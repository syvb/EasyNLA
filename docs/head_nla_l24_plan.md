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

## Results (2026-10-19)

**Both pre-RL checks pass, by wide margins.** The probe lands between the pass and fail lines.

### Probe test

Held-out FVE, mean over the 32 heads, about 1,490 usable examples per head (47,752/48,000 explanations
have `ok=True`):

| features | 200 | 400 | 800 | all (~1,490) |
|---|---|---|---|---|
| avg16 | 0.060 | 0.085 | 0.121 | 0.159 |
| lens | 0.063 | 0.096 | 0.137 | 0.172 |
| explain (TF-IDF) | 0.004 | 0.025 | 0.049 | 0.067 |
| explain_embed (Qwen3-8B) | 0.009 | 0.033 | 0.055 | 0.075 |
| hidden (ceiling) | 0.383 | 0.538 | 0.663 | 0.737 |
| explain_shuf / empty | −0.015 | −0.008 | −0.004 | −0.002 |

- **Ratio:** explain / avg16 = 0.42 (TF-IDF), 0.47 (embedding). That is between the plan's fail line (0.3) and pass line
  (0.5).
  - The per-head gap is −0.091 ± 0.009 (TF-IDF) and −0.084 ± 0.012 (embedding).
  - The explanations beat avg16 on 0 heads (TF-IDF) and 4 heads (embedding).
- **Copying:** a median of 10% of an explanation's content words are its own readout tokens (p90 25%), so these
  are real rewrites, not near-lists.
- **The explanation curve is still steep:** ×17 from 200 to 1,490 per head, against ×2.7 for avg16.
- **Heads vary a lot:**
  - L24H25: avg16 0.39, explain 0.29, hidden 0.85.
  - L24H1: avg16 0.02, explain 0.01.
- **lens beats avg16 at L24.** The 5-layer diagnostics had them about equal.

### SFT checks

Eval split, 6,360 rows, 32 heads. FVE of the AR's reconstruction (√d-scaled normalized write; baseline = the
per-head AR-training mean):

| condition | pooled | per-head |
|---|---|---|
| ar/gold | 0.063 | 0.068 |
| ar/empty | −0.065 | −0.064 |
| ar/gold_shuf | −0.082 | −0.086 |
| **av/true** | **0.049** | **0.056** |
| av/shuf | −0.067 | −0.071 |

- **Check 1 (AR):** +0.063, 95% CI [+0.058, +0.068] pooled; [+0.039, +0.097] per-head.
  - The quick test on attention-based Claude descriptions got 0.019 here.
- **Check 2 (AV):** +0.116, 95% CI [+0.109, +0.123] pooled; [+0.077, +0.182] per-head.
  - The quick test's AV did not beat its shuffled control.
  - Here AV text scores about 80% of what the gold explanations score through the same AR.
- **Text-only probe** (per head, 200 rows, low power): gold 0.003, AV-true 0.022, AV-shuf −0.015.
- **AV distinct outputs:** 80%. There is a visible fallback mode ("Hedged, tentative phrasing …") for writes
  it can't place.
- **Examples** (`checks/examples.md` on HF), gold → AV:
  - "immunization and the control of infectious disease, especially vaccination campaigns" → "Pandemic and
    infectious-disease subject matter, especially the COVID-19 outbreak"
  - "Time and timing as a concept …" → "Time as a concept, especially the passing of moments …"

### What happened operationally

- **Run 1** (2.66 h) did datagen, readouts, explanations and most of the probe.
- **The probe took ~2.5 h on the pod, against under 1 s per fit locally.** BLAS threads oversubscribed on the
  shared 64-core host. The remote script now caps them at 8.
- **Host maintenance:** RunPod announced a driver-upgrade window with host reboots (16:00–17:30 UTC) during the
  probe. The artifacts were pulled to the VM and the pod stopped.
- **The probe was finished locally** in a few minutes from the saved data.
- **Run 2** (`heads_l24_sft_remote.sh`, 1.22 h) regenerated the seeded data. Its selection was identical
  (48,000/48,000). It then trained and checked on the saved explanations.
  - No secure H100 was free at first. The launcher gained `--gpu-id` and a retry wrapper (H100 → A100 SXM → A100 PCIe,
    secure only), and the first retry got an H100.
- **Launcher PID gotcha:** `setsid nohup … & echo $!` records a wrapper PID, not the launcher's. Find the launcher with
  `pgrep -f`. The launcher's signal path (best-effort fetch, then delete) worked when sent to the right PID.
- **wandb:**
  - AR and AV losses were streamed live by `scripts/wandb_tail_log.py`, a sidecar tailing run.log, because the run
    had started with `--no-wandb`.
  - Future runs pass `--secrets …,wandb` and log natively.
  - Results summary: `scripts/wandb_log_results.py`.
- **Cost:**
  - Pods: 3.88 h × $3.99 = $15.50.
  - Haiku: 86.3M input (most cached) + 17.2M output tokens, at most $17 at list price.

### Next

- **RL is the obvious next step:** the AV already produces text the AR can use.
- **More examples per head:** the explanation probe curve is steep.
- **The AV injection site:** layer 1 vs L24, as an ablation.
- **The AV's fallback mode:** watch it under RL.

## Faithfulness of the AV's explanations (2026-10-19)

**Question:** are AV explanations that are more faithful to the write also the ones that reconstruct better?

**Method:**
- `scripts/runpod/heads_l24_regen_remote.sh` reran the checks from the saved checkpoints (0.44 h pod) and saved
  every generation (`generations.parquet`). The reproduction was exact: same FVEs, same per-row errors.
- `scripts/heads_faithfulness_judge.py`: a blinded Claude Sonnet 5.5 judge (effort low) scored 823 eval rows (25 per
  head plus 25 hand-labelled rows), in three conditions, interleaved. The judge saw the row's readouts and one text,
  and never which condition it came from.
- Faithfulness to the readouts (1–5), plus flags for unsupported specifics and generic templates.
- 2,460/2,469 judgments parsed.
- The judge agrees with a hand rating of 25 examples: Spearman 0.89.

| text | faithful (1–5) | ≥4 | unsupported specifics | generic template | reconstruction gain |
|---|---|---|---|---|---|
| gold (Haiku) | 4.06 | 88% | 7% | 3% | +0.060 |
| AV, true write | 2.69 | 24% | 68% | 50% | +0.053 |
| AV, another write (control) | 1.80 | 4% | 86% | 55% | −0.074 |

Gain = (baseline error − error) ÷ the head's mean baseline error, per row.

**Yes: faithfulness predicts reconstruction.**
- Spearman between AV faithfulness and AV gain is +0.41 over 817 rows.
- **+0.27 controlling for the gold text's gain on the same row,** that is, for how reconstructible the write is at all.
  So it isn't only the write type.
- The within-head mean is +0.31 over 32 heads. Across heads, mean AV faithfulness vs AV FVE gives Spearman 0.71.

**AV gain by faithfulness score:**

| score | n | AV gain | gold gain, same rows |
|---|---|---|---|
| 1 | 66 | −0.006 | −0.010 |
| 2 | 323 | +0.003 | +0.020 |
| 3 | 234 | +0.041 | +0.043 |
| 4 | 189 | +0.174 | +0.174 |

- When the AV is faithful, it matches the gold explanation's usefulness.
- At score 2 it loses most of what the gold had.
- The residual after regressing on gold gain rises from −0.02 (score 2) to +0.04 (score 4).

**Flags:**
- **Generic template** (50% of AV outputs): mean gain +0.008, against +0.097 without.
- **Unsupported specifics** (68%): +0.041, against +0.078 without.
  - The judge flags embellishment heavily. Part of this is the gold style: 7% of the gold is flagged too, and the
    AV amplifies it.

**By head:**
- Best: H19 (faithfulness 3.89, FVE 0.22), H16, H26.
- Worst: H2 (1.96, −0.01), H3, H30, H7, H21.
- The gold faithfulness is ~4 on every head, so the AV's per-head gap is learning, not data.
- The worst heads are the grammatical ones, where even gold reconstructs little (gold FVE ≤ 0.03).

**Implications for RL:**
- Reconstruction reward lines up with faithfulness. Templates and unfaithful text earn ~0, so RL should push away
  from them where the AR can use better text.
- Unsupported specifics cost reward only mildly, so RL may not remove them. A judge-based penalty or a cleaner gold
  style could.
- On grammatical heads, gold itself barely helps the AR. A stronger AR is needed there before RL can reward
  faithfulness.

**Cost:** pod $1.76; Sonnet ≤ $8 (1.84M input incl. cache reads + 0.44M output).
