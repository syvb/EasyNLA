"""Generate held-out explanations + AR predictions and dump them as artifacts.

The whitening 2×2 analysis (docs/whitening_experiment.md) needs, per model:
full explanation text and the critic's d-dim prediction for each held-out row
— which the RL trainer's eval loop does not persist. This script produces
them offline from any AV/AR checkpoint pair, and prints the native-space FVE
while it's at it.

Outputs:
  {out}.npz    pred (n,d f32) · gold (n,d f32) · extracted (bool) · row_idx
  {out}.jsonl  one line per row: {idx, extracted, mse_norm, explanation, response}
               (line 0 is a meta header: checkpoints, sidecar, git commit)

Cross-space FVE (mapping preds between raw/whitened spaces with the
whitening-stats npz) is analysis-time math on these files — see the
"pinned convention" in docs/whitening_experiment.md.

Examples:
  # free cell: the raw-trained NLA (merged AV + RL LoRA + merged AR from HF)
  python scripts/dump_nla_eval.py \
      --av-ckpt syvb/nanonla-qwen3-8b-L24-av \
      --av-adapter syvb/nanonla-qwen3-8b-L24-rl-lora --av-adapter-subfolder p0.0 \
      --ar-ckpt syvb/nanonla-qwen3-8b-L24-ar \
      --sidecar <data>/av_sft_val.parquet --parquet <data>/av_sft_val.parquet \
      --n 1000 --out /artifacts/raw_nla_on_raw_val

  # a LoRA-format AV/AR pair (EasyNLA SFT/RL output dirs)
  python scripts/dump_nla_eval.py \
      --base-ckpt Qwen/Qwen3-8B --av-lora <ckpts>/av_sft/iter_XXXX \
      --ar-ckpt <ckpts>/ar_sft/iter_XXXX \
      --sidecar <data_w>/av_sft_val.parquet --parquet <data_w>/av_sft_val.parquet \
      --n 1000 --out /artifacts/whitened_sft_on_whitened_val
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.config import load_nla_config
from nla.schema import (
    EXPLANATION_RE,
    compute_predict_mean_baselines,
    normalize_activation,
)
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook


def _load_actor(args, device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    assert bool(args.av_ckpt) != bool(args.av_lora), (
        "pass exactly one of --av-ckpt (merged model) or --av-lora (adapter on --base-ckpt)"
    )
    base_id = args.av_ckpt or args.base_ckpt
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    if args.av_lora:
        model = PeftModel.from_pretrained(model, args.av_lora)
    if args.av_adapter:
        kw = {"subfolder": args.av_adapter_subfolder} if args.av_adapter_subfolder else {}
        model = PeftModel.from_pretrained(model, args.av_adapter, **kw)
    return model.eval()


def _load_critic(ar_ckpt, base_ckpt, device):
    """EasyNLA LoRA-format dir (ar_meta.json) or a merged NLACriticModel
    checkpoint (local dir or HF id with value_head.safetensors)."""
    if (Path(ar_ckpt) / "ar_meta.json").exists():
        from show_nla_generations import load_ar_critic

        return load_ar_critic(ar_ckpt, base_ckpt, device)
    from nla.models import NLACriticModel

    critic = NLACriticModel.from_pretrained(
        ar_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    return critic.eval()


def _load_rows(parquet, n, skip_rows):
    pf = pq.ParquetFile(parquet)
    prompts, acts, taken, seen = [], [], 0, 0
    for rg_idx in range(pf.num_row_groups):
        if taken >= n:
            break
        rg = pf.read_row_group(rg_idx, columns=["prompt", "activation_vector"])
        if seen + rg.num_rows <= skip_rows:
            seen += rg.num_rows
            continue
        start = max(0, skip_rows - seen)
        seen += rg.num_rows
        pr = rg.column("prompt").to_pylist()[start:]
        ac = (
            rg.column("activation_vector")
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(rg.num_rows, -1)[start:]
        )
        take = min(n - taken, len(pr))
        prompts += pr[:take]
        acts.append(ac[:take].astype(np.float32))
        taken += take
    assert taken > 0, f"no rows read (skip_rows={skip_rows} too large?)"
    return prompts, np.concatenate(acts, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--av-ckpt", default=None, help="merged AV model (dir or HF id)")
    p.add_argument("--av-lora", default=None, help="AV LoRA adapter dir (used with --base-ckpt)")
    p.add_argument("--av-adapter", default=None, help="extra LoRA on top (e.g. an RL adapter)")
    p.add_argument("--av-adapter-subfolder", default=None)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--sidecar", required=True, help="dataset parquet or ckpt dir (the NLA contract)")
    p.add_argument("--parquet", required=True, help="held-out rows (prompt + activation_vector)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--skip-rows", type=int, default=0, help="use >0 when --parquet is a train file")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out", required=True, help="output prefix → {out}.npz + {out}.jsonl")
    args = p.parse_args()

    device = "cuda"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tok)
    prompts, gold = _load_rows(args.parquet, args.n, args.skip_rows)
    n, d = gold.shape
    assert d == cfg.d_model, f"rows are {d}-wide, sidecar says {cfg.d_model}"
    print(f"{n} rows · d={d} · sidecar {args.sidecar}")

    actor = _load_actor(args, device)
    vref = [None]
    register_karvonen_hook(
        actor, vref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1,
    )
    critic = _load_critic(args.ar_ckpt, args.base_ckpt, device)

    # Every row uses the same actor prompt (only the injected vector differs),
    # so a batch is the same token row repeated — no padding needed.
    ptxt0 = build_prompt_text(prompts[0], cfg.injection_char, tok)
    ids0 = tok.encode(ptxt0, add_special_tokens=False)
    for i in range(1, min(n, 50)):
        assert tok.encode(build_prompt_text(prompts[i], cfg.injection_char, tok),
                          add_special_tokens=False) == ids0, f"prompt {i} differs from prompt 0"

    responses = []
    for lo in range(0, n, args.batch_size):
        b = min(args.batch_size, n - lo)
        pt = torch.tensor([ids0] * b, dtype=torch.long, device=device)
        vref[0] = torch.from_numpy(gold[lo : lo + b]).to(device)
        try:
            with torch.no_grad():
                out = actor.generate(
                    input_ids=pt, attention_mask=torch.ones_like(pt),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id, return_dict_in_generate=True,
                )
        finally:
            vref[0] = None
        for row in out.sequences[:, pt.shape[1]:]:
            responses.append(tok.decode(row, skip_special_tokens=True))
        done = lo + b
        if done % (args.batch_size * 8) == 0 or done == n:
            print(f"  generated {done}/{n}")

    expls = []
    for r in responses:
        m = EXPLANATION_RE.search(r)
        expls.append(m.group(1).strip() if m else None)
    extracted = np.array([e is not None for e in expls])
    print(f"extraction rate: {extracted.mean():.3f}")

    template = cfg.critic_prompt_template
    tok.padding_side = "right"  # critic_predict anchors at attention_mask.sum-1
    preds = np.zeros((n, d), dtype=np.float32)
    idxs = [i for i in range(n) if extracted[i]]
    for lo in range(0, len(idxs), args.batch_size):
        chunk = idxs[lo : lo + args.batch_size]
        enc = tok([template.format(explanation=expls[i]) for i in chunk],
                  add_special_tokens=False, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = critic_predict(critic, enc["input_ids"], enc["attention_mask"], cfg.mse_scale)
        preds[chunk] = out.float().cpu().numpy()

    # Native-space FVE on extracted rows (paper def: 1 − mse/raw-variance baseline)
    gold_t = torch.from_numpy(gold[extracted])
    pred_t = torch.from_numpy(preds[extracted])
    pn = normalize_activation(pred_t, cfg.mse_scale)
    gn = normalize_activation(gold_t, cfg.mse_scale)
    mse_rows = F.mse_loss(pn, gn, reduction="none").mean(dim=1)
    _, baseline = compute_predict_mean_baselines(torch.from_numpy(gold), cfg.mse_scale)
    fve = 1.0 - mse_rows.mean().item() / baseline
    print(f"native FVE (extracted rows): {fve:.4f}  "
          f"(mse {mse_rows.mean().item():.4f} / baseline {baseline:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f"{out_path}.npz", pred=preds, gold=gold, extracted=extracted,
        row_idx=np.arange(n), mse_scale=np.float64(cfg.mse_scale or 0.0),
    )
    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    with open(f"{out_path}.jsonl", "w") as f:
        f.write(json.dumps({
            "meta": True, "av_ckpt": args.av_ckpt, "av_lora": args.av_lora,
            "av_adapter": args.av_adapter, "av_adapter_subfolder": args.av_adapter_subfolder,
            "ar_ckpt": args.ar_ckpt, "sidecar": args.sidecar, "parquet": args.parquet,
            "n": n, "extraction_rate": float(extracted.mean()), "native_fve": fve,
            "activation_norm": cfg.activation_norm, "git_commit": git_commit,
        }) + "\n")
        mrow = 0
        for i in range(n):
            mse_i = float(mse_rows[mrow]) if extracted[i] else None
            if extracted[i]:
                mrow += 1
            f.write(json.dumps({
                "idx": i, "extracted": bool(extracted[i]), "mse_norm": mse_i,
                "explanation": expls[i], "response": responses[i],
            }) + "\n")
    print(f"wrote {out_path}.npz + {out_path}.jsonl")


if __name__ == "__main__":
    main()
