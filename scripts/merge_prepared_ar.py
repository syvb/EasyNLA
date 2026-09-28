"""Merge an AR LoRA trained ON TOP OF A PREPARED critic into a full critic dir.

train_sft --use-lora --base-ckpt <prepared AR dir> saves only the LoRA + value
head (ar_lora_value_head.safetensors + ar_meta.json). train_rl_vllm needs a full
NLACriticModel dir for --ar-ckpt. scripts/merge_lora_to_hf.py can't do this: it
rebuilds the AR from a freshly truncated RAW base, which is the wrong starting
weights for a LoRA fit on the prepared AR.

    python scripts/merge_prepared_ar.py --base <prepared AR dir> --lora <iter dir> --out <dir>

Checks that the merged critic predicts what the LoRA-loaded one does (cosine per
test prompt >= 0.999) before saving.
"""
import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kl_audit import ar_predict, load_ar, log  # noqa: E402

TEST_TEXTS = ["Scientific explanation of how bacteria coordinate via chemical signals.",
              "A recipe list: the next item is likely an ingredient quantity.",
              "Legal contract clause; the sentence continues with an obligation.", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="prepared AR dir (value_head.safetensors)")
    ap.add_argument("--lora", required=True, help="train_sft --use-lora iter dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import AutoTokenizer
    from nla.config import load_nla_config
    from nla.models import NLACriticModel

    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = load_nla_config(a.base, tok)
    _, critic = load_ar(f"x={a.base}:{a.lora}", a.device)
    before = ar_predict(critic, tok, cfg.critic_prompt_template, TEST_TEXTS, cfg.mse_scale, a.device)

    n = 0
    for m in critic.backbone.modules():
        if isinstance(m, BaseTunerLayer):
            m.merge()
            n += 1
    sd = {k.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias"): v
          for k, v in critic.state_dict().items() if "lora_" not in k}
    del critic
    log(f"merged {n} LoRA layers")

    fresh = NLACriticModel.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    miss, unexp = fresh.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"merged state dict mismatch: missing {miss[:3]} unexpected {unexp[:3]}"
    fresh = fresh.to(a.device).eval()
    after = ar_predict(fresh, tok, cfg.critic_prompt_template, TEST_TEXTS, cfg.mse_scale, a.device)
    cos = torch.nn.functional.cosine_similarity(before, after, dim=-1)
    log(f"merged vs LoRA-loaded prediction cosine per prompt: {[round(c, 5) for c in cos.tolist()]}")
    assert (cos >= 0.999).all(), "merged AR does not reproduce the LoRA AR"

    os.makedirs(a.out, exist_ok=True)
    fresh.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    shutil.copy2(os.path.join(a.base, "nla_meta.yaml"), os.path.join(a.out, "nla_meta.yaml"))
    log(f"saved merged AR -> {a.out}")


if __name__ == "__main__":
    main()
