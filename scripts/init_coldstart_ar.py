"""Initialize a cold-start (no-SFT) truncated NLACriticModel HF checkpoint.

Required for Arm 2b (Cold-Start RL, no warm-start SFT): `nla.train_rl_vllm`
loads `--ar-ckpt` via `NLACriticModel.from_pretrained`, which requires a
truncated `K+1`-layer HF model + `value_head.safetensors`. This script creates
that identity-initialized AR checkpoint from `--base-ckpt` without running SFT.

Example:
    python scripts/init_coldstart_ar.py \
        --base-ckpt Qwen/Qwen3-8B \
        --ar-num-layers 25 \
        --output-dir /ckpts/diff_nla/arm2b_coldstart_ar \
        --sidecar /data/diff_nla/arm2/rl_shuf.parquet
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nla.schema import sidecar_path_for
from nla.train_sft import init_critic_from_base


def init_coldstart_ar(
    base_ckpt: str,
    ar_num_layers: int,
    output_dir: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    sidecar: str | None = None,
    strip_final_norm: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    critic = init_critic_from_base(
        base_ckpt,
        ar_num_layers,
        dtype,
        quant_config=None,
        device_map=None,
        max_memory=None,
        strip_final_norm=strip_final_norm,
    )
    critic.save_pretrained(str(output_dir))
    try:
        tok = AutoTokenizer.from_pretrained(base_ckpt)
        tok.save_pretrained(str(output_dir))
    except Exception as exc:
        print(f"[init_coldstart_ar] note: tokenizer not copied from {base_ckpt}: {exc}")
    if sidecar is not None:
        sc = sidecar_path_for(sidecar)
        if sc.exists():
            shutil.copy2(sc, output_dir / "nla_meta.yaml")
    print(f"[init_coldstart_ar] saved cold-start {ar_num_layers}-layer AR → {output_dir}")
    return output_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", required=True, help="HF model id or path (e.g. Qwen/Qwen3-8B)")
    p.add_argument("--ar-num-layers", type=int, default=25, help="Truncated layer count (layer_index + 1)")
    p.add_argument("--output-dir", required=True, help="Directory to save the cold-start AR checkpoint")
    p.add_argument("--sidecar", default=None, help="Optional dataset parquet/sidecar to copy nla_meta.yaml from")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--strip-final-norm", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    init_coldstart_ar(
        args.base_ckpt,
        args.ar_num_layers,
        Path(args.output_dir),
        dtype=dtype,
        sidecar=args.sidecar,
        strip_final_norm=args.strip_final_norm,
    )


if __name__ == "__main__":
    main()
