# CPU-only research harness for the Diff-NLA experiment

These scripts ran the Phase 0 gate and follow-up geometry checks on the real
`Qwen/Qwen3-8B` / `Qwen/Qwen3-8B-Base` pair at layer 24 on a 16-vCPU, 31 GB, no-GPU
machine, to gather evidence for or against the experiment in
`docs/diff_nla_experiment_plan.md` before any GPU run. Result JSONs are in `results/`;
`make_report_tables.py results` prints every table from them. `REPORT.md` is the write-up.

Nothing here changes the pipeline. The queue scripts (`run_after_*.sh`) hard-code the
paths of the machine they ran on and are kept as a record of the run order.

## How an 8B model fits in 31 GB

`cached_io.py` replaces `load_causal_lm`, `extract_at_positions` and `unload_model`
in a repo script's namespace (`run_patched.py`); the script's metrics, gates and
thresholds are untouched.

- Layer-24 activations come from a **bottom half** (embedding + blocks 0..24, fp32, 22 GB).
- Logits come from a **top half** (blocks 25..35 + norm + lm_head, 11 GB) fed the cached
  layer-24 residual stream through `inputs_embeds`.
- Every whole-sequence activation is cached on disk as fp16, keyed by model, layer and token ids.
- Taboo LoRAs are merged into the half being loaded (`W += alpha/r * B @ A`).
- `jit_full.py` loads the whole model (generation, all-layer sweeps) with bf16-resident weights
  that are cast to fp32 per block during its own forward call.

The machine's memory limit cannot hold a model half plus the tensors a script accumulates, so
each repo script runs three times:

```bash
R=scripts/diff/cpu_research
# 1. record every extraction request (no model is loaded; activations are random)
NLA_RECORD=m.jsonl NLA_RECORD_STOP_AT=evaluate_alignment_suite \
  python $R/run_patched.py scripts/diff/phase0_geometry_gate.py --output-dir /tmp/throwaway ...
# 2. fill the cache with only the model in memory
python $R/warm_cache.py m.jsonl
# 3. the real run reads everything from the cache
python $R/run_patched.py scripts/diff/phase0_geometry_gate.py --output-dir out ...
```

## Checks that the harness matches the repo

| Script | Compares | Result |
| :--- | :--- | :--- |
| `verify_split.py` | split loader vs `extract_at_positions` on the full fp32 model (0.6B, L18) | hidden rel. error 3.1e-4 (fp16 storage), KL 2.3e-7 nats |
| `verify_lora.py` | merged-LoRA split loader vs PEFT (0.6B, L18) | hidden rel. error 2.4e-4, KL 9.0e-8 nats (adapter effect 1.85 nats) |
| `verify_jit.py` | bf16-resident JIT loader vs fp32 `generate` (0.6B) | identical greedy tokens |
| `layer_sweep.py` | JIT whole-model pass vs split loader at the planned layer (8B, L24) | max rel. difference 3.9e-5 |
| `behaviour_link.py` | logits from fp16 residuals vs logits from fp32 residuals | reported in each result JSON as `fp16_feed_check` |

Differences from the GPU pipeline: forward passes are fp32 rather than bf16, and rollouts use
HF sampling rather than vLLM.

## Scripts

Gate 0 and its controls, on the benchmark pairs:

- `gate_controls.py`: the gate's SNR statistic on all pairs with pair-bootstrap CIs, plus
  normalisation-only, Base-only and content controls.
- `snr_decomposition.py`: class signal and within-class spread of Base and of the diff, relative to `h_target`.
- `gate_positions.py`: the same statistic at user-content positions, which the gate does not score.
- `layer_sweep.py`: the statistic at every layer (`--position asst_last|user_last`).
- `alignment_variants.py`: the statistic when the alignment is fitted on chat activations instead of web text.
- `robust_alignment.py`: Phase 0's alignment suite refitted with massive-activation tokens masked.
- `dw_coordinate_energy.py`: which residual coordinates the fitted correction writes to.
- `massive_anatomy.py`: which tokens have massive activations, which coordinate carries them, and how much of the
  alignment objective they hold; also scans every chat position for them.
- `massive_fix_variants.py`: six treatments of those rows in the alignment fit, on the same metrics.
- `massive_coordinate_detail.py`: what the fitted map does in the massive coordinate, with and without masking.
- `massive_in_benchmarks.py`: massive tokens in the benchmark prompts, and the user-token-mean scores without them.
- `ar_init_gap.py`: how far each arm's training target is from where the reconstructor starts (identity read of the
  Instruct Layer-K state), and how much a linear head could recover from a perfect Instruct or Base activation.
- `bf16_fidelity.py`: rounding noise of the stock bf16 loader against the fp32 reference, on h and on the aligned diff.
- `taboo_mode_check.py`: what the `<think>` mode check measures and what Phase 4 does with it.
- `taboo_transfer.py`: size, shape and organism information of the Taboo `Org - Inst` diffs Benchmark 5 reads,
  and how far they sit from the Arm 2 training target.

On a chat corpus with Qwen3-8B's own rollouts (`phase2_build_corpus.py --step rollouts`):

- `corpus_rows.py`: shared loader for sampled Target / Base activations.
- `corpus_geometry.py`: noise floor and Fix 3 weights, composition, text recoverability, wrapper
  agreement and refusal-behaviour separation of each arm's targets.
- `top_direction.py`: what the largest direction of the Arm 2 target is.
- `behaviour_link.py`: whether the cleaned diff tracks `KL(Instruct || Base)` and how much of each
  arm's target the two models' next-token states explain.
- `cross_patch.py`: how much of the behavioural gap is already in the Layer-K residual stream.
- `corpus_behaviour_positions.py`: refuse-versus-comply separation at each role.

Helpers: `summarize_phase0.py`, `make_chat_prompts_mix.py`, `make_report_tables.py`,
`make_layer_sweep_view.py`, `bench_8b.py`, `sanity_8b.py`, `warm_cache.py`.
