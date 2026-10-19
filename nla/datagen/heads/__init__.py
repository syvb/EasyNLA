"""Head-contribution NLA datagen — per-attention-head writes + warm-start descriptions.

A separate pipeline from the residual-stream one in nla/datagen (different
row unit, different targets), sharing its conventions: stage CLIs that
resume from durable part-files, an orchestrator that shells out to them
(run_pipeline.py), and a manifest the downstream stages assert against.

    corpus      build_corpus.py   domain-mixed text sample → corpus.parquet
    extract     extract.py        forward the model; at sampled (token, layer)
                                  pairs record the layer-input residual, every
                                  head's output (o_proj input), and a top-k
                                  summary of each head's attention pattern
    stats       stats.py          per-head average write + typical deviation
                                  size; per-record deviation norms
    select      split_select.py   doc-level split; warm-start examples spread
                                  evenly across heads, larger deviations
                                  sampled more often
    prepare     batch_api.py      warm-start prompts → Batch API request shards
    submit/wait batch_api.py      PAID: submit shards, poll, collect results →
                                  warmstart/descriptions.parquet

Layout under output_dir (see store.py for the schemas):

    head_meta.yaml          contract: model dims, sampling params, layers
    wo.safetensors          o_proj weights per layer (write = W_O[:, h] @ head_out[h])
    docs/part_*.parquet     token ids per document
    records/L{ll}/part_*    one row per sampled (token, layer)
    head_stats.safetensors  per-head means / deviation scales
    devnorms/L{ll}/part_*   row-aligned with records: per-head deviation norms
    splits.parquet          doc_id → av_sft | ar_sft | rl | eval
    warmstart/              selection, request shards, batch state, results
"""
