"""Merge the RL AV LoRA into its AV-SFT warmstart base for vLLM serving.

The checkpoint's adapter_config.json records a cluster staging path in
base_model_name_or_path (kept byte-identical by the uploader), so the base is
passed EXPLICITLY. The base is asher577/nla-warmstart-2x subfolder av/ (full
model + tokenizer + nla_meta.yaml); the LoRA is asher577/nla-qwen-3-8b av/.

Output dir carries model + tokenizer + nla_meta.yaml — everything NLACodec
needs. Do NOT name output dirs after installed packages ('av' shadows PyAV).

    python -m experiments.bottleneck.merge_av \
        --base-dir /workspace/warmstart/av --lora-dir /workspace/rl_ckpt/av \
        --out /workspace/av_merged
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, help="AV-SFT warmstart base (full model dir)")
    p.add_argument("--lora-dir", required=True, help="RL AV LoRA adapter dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    assert out.name not in ("av", "ar", "utils", "tokenizers"), (
        f"dir name {out.name!r} can shadow an installed package (ENV gotcha)")
    if (out / "config.json").exists():
        print(f"[merge_av] {out} already exists — skipping")
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge_av] base={args.base_dir} + lora={args.lora_dir}")
    base = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(base, args.lora_dir)
    merged = peft.merge_and_unload()
    # Atomic: save_pretrained writes config.json before the 16GB shards, so a
    # crash mid-save would leave a dir that passes the resume skip-check.
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    merged.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(args.base_dir).save_pretrained(tmp)
    sidecar = Path(args.base_dir) / "nla_meta.yaml"
    assert sidecar.exists(), f"warmstart base lacks nla_meta.yaml ({sidecar})"
    shutil.copy2(sidecar, tmp / "nla_meta.yaml")
    tmp.rename(out)
    print(f"[merge_av] done -> {out}")
    print("MERGE_AV_DONE")


if __name__ == "__main__":
    main()
