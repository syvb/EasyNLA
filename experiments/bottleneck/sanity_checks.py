"""Sanity checks #3: the generation harness itself (run BEFORE burning GPU-hours).

  1. Custom greedy loop (condition=clean) must be token-identical to HF
     model.generate(do_sample=False) — validates left-padding/position_ids/KV.
  2. C0′ (identity substitution through the full interception machinery) must
     be token-identical to C0 — validates the tap (dtype, indexing, layer).
  3. Tiny C1 smoke: codec round-trip on a couple of prompts × few tokens;
     prints the z texts and per-token cosines so a human can eyeball them.

    python -m experiments.bottleneck.sanity_checks --config ... [--skip-codec]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from experiments.bottleneck.patched import BottleneckModel, StepLog

PROMPTS = [
    "What is the capital of France?",
    "Write one sentence about the ocean.",
    "Compute 17 + 25 and give just the number.",
    "Name three programming languages.",
    "What year did the Apollo 11 mission land on the Moon?",
    "Translate 'good morning' to Spanish.",
    "What is 12 * 12?",
    "Give a synonym for 'happy'.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip-codec", action="store_true")
    p.add_argument("--max-new", type=int, default=64)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    m = BottleneckModel(cfg.get("m_ckpt", "Qwen/Qwen3-8B"),
                        layer_index=cfg.get("layer_index", 24))
    ids_list = [m.build_chat_ids([{"role": "user", "content": q}]) for q in PROMPTS]

    # --- 1. custom loop == HF generate (greedy), SAME left-padded batch ---
    # (batched-vs-unbatched bf16 GEMMs can legitimately flip a near-tie argmax,
    # so the reference must use identical batch shapes for a bitwise claim)
    ours = m.generate(ids_list, args.max_new, condition="clean")
    B = len(ids_list)
    maxp = max(len(p) for p in ids_list)
    ids = torch.full((B, maxp), m.pad_id, dtype=torch.long, device=m.device)
    attn = torch.zeros((B, maxp), dtype=torch.long, device=m.device)
    for r, p_ in enumerate(ids_list):
        ids[r, maxp - len(p_):] = torch.tensor(p_, dtype=torch.long, device=m.device)
        attn[r, maxp - len(p_):] = 1
    with torch.no_grad():
        out = m.model.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=args.max_new,
            do_sample=False, pad_token_id=m.pad_id)
    for i in range(B):
        ref = out[i, maxp:].tolist()
        # trim ref at first EOS (inclusive) to match our loop's stopping rule
        for j, tok in enumerate(ref):
            if tok in m.eos_ids:
                ref = ref[:j + 1]
                break
        assert ours[i] == ref, (
            f"CHECK 1 FAILED (prompt {i}): custom loop diverges from HF generate\n"
            f"ours={ours[i][:20]}\nref ={ref[:20]}")
    print(f"CHECK 1 PASSED: custom greedy loop == batched model.generate on "
          f"{len(PROMPTS)} prompts")

    # --- 2. C0' == C0 ---
    ident = m.generate(ids_list, args.max_new, condition="identity")
    for i in range(len(PROMPTS)):
        assert ident[i] == ours[i], (
            f"CHECK 2 FAILED (prompt {i}): identity substitution changed tokens\n"
            f"clean={ours[i][:20]}\nident={ident[i][:20]}")
    print(f"CHECK 2 PASSED: C0' identity-substitution == C0 on {len(PROMPTS)} prompts "
          f"x {args.max_new} tokens")

    # --- 3. tiny C1 smoke ---
    if args.skip_codec:
        print("(codec smoke skipped)")
    else:
        from experiments.bottleneck.codec import NLACodec
        codec = NLACodec(
            av_merged_dir=cfg["av_merged_dir"], ar_dir=cfg["ar_dir"],
            vllm_gpu_mem=cfg.get("vllm_gpu_mem", 0.35),
            vllm_max_len=cfg.get("vllm_max_len", 1024),
            av_max_tokens=cfg.get("av_max_tokens", 150),
            av_temperature=cfg.get("av_temperature", 1.0),
        )
        logs: list[StepLog] = []
        gen = m.generate(ids_list[:2], 8, codec=codec, condition="nla", step_logs=logs)
        for i in range(2):
            print(f"\n--- C1 smoke prompt {i}: {PROMPTS[i]!r}")
            print(f"    output: {m.tokenizer.decode(gen[i], skip_special_tokens=True)!r}")
        assert logs, "CHECK 3 FAILED: no codec step logs recorded"
        assert any(l.step == -1 for l in logs), (
            "CHECK 3 FAILED: no prefill-last-position substitution logged — "
            "the first generated token would bypass the codec")
        print(f"\nCHECK 3: {len(logs)} codec calls, "
              f"cosine mean {sum(l.cosine for l in logs)/len(logs):.3f}, "
              f"z_len mean {sum(l.z_len for l in logs)/len(logs):.0f}")
        print(f"    first z: {logs[0].z_text[:300]!r}")
        print(f"    codec stats: {codec.report_stats()}")
        bad = [l for l in logs if not l.steer_verified]
        assert not bad, f"CHECK 3 FAILED: {len(bad)} unverified injections"
        print("CHECK 3 PASSED")

    print("SANITY_DONE", flush=True)


if __name__ == "__main__":
    main()
