"""How much does bf16 arithmetic perturb the Layer-K diff compared with fp32?

Every other script here computes activations in fp32, but the GPU runbook extracts
with `--dtype bf16`. The diff is a difference of two large, similar vectors, so
rounding noise that is negligible on h can still matter on h_target - h_base.

For the same token sequences this compares:
  * fp32 reference: the disk-cached split-model activations (`cached_io`)
  * bf16: the stock repo loader `load_causal_lm(dtype=bfloat16)` + `extract_at_positions`
          (blocks above K dropped to save RAM; they do not affect Layer K)

and reports, per role, the noise on h and on the aligned diff.

    NLA_ACT_CACHE=/home/user/act_cache python bf16_fidelity.py \
        --phase0-dir /home/user/research/p0_8b_L24 --out /home/user/research/bf16_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io  # noqa: E402
from nla.diff.alignment import apply_alignment, load_alignment  # noqa: E402
from nla.diff.benchmarks import (  # noqa: E402
    NEUTRAL_QUERIES,
    flatten_binary_pairs,
    load_fineweb_passages,
    load_refusal_pairs,
)
from nla.diff.metrics import compute_fisher_snr  # noqa: E402
from nla.diff.model_io import extract_at_positions  # noqa: E402
from nla.diff.wrappers import encode_aligned_pair  # noqa: E402


def load_bottom_bf16(name: str, layer_index: int, attn: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[: layer_index + 1])
    model.lm_head = torch.nn.Identity()
    model.eval()
    return model


def stack(exs: list[Any]) -> torch.Tensor:
    return torch.cat([e.hidden for e in exs if e.hidden.shape[0] > 0], dim=0)


def noise_stats(a_ref: torch.Tensor, a_test: torch.Tensor) -> dict[str, float]:
    """Row-wise comparison of a test matrix against its reference."""
    ref = a_ref.double()
    err = a_test.double() - ref
    rn = ref.norm(dim=-1)
    en = err.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(a_test.double(), ref, dim=-1)
    return {
        "n": int(ref.shape[0]),
        "median_ref_norm": float(rn.median()),
        "median_err_norm": float(en.median()),
        "median_rel_err": float((en / rn.clamp_min(1e-9)).median()),
        "p95_rel_err": float(torch.quantile(en / rn.clamp_min(1e-9), 0.95)),
        "noise_energy_frac": float(en.pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)),
        "median_cos": float(cos.median()),
        "p05_cos": float(torch.quantile(cos, 0.05)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-neutral", type=int, default=10)
    ap.add_argument("--n-web", type=int, default=24)
    ap.add_argument("--web-max-tokens", type=int, default=64)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    p0 = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text())
    layer = int(p0["layer_index"])
    wrapper = str(p0["base_wrapper"])
    art = load_alignment(p0["align_id_anchored_path"])
    tau = float(p0["tau_web_median"])
    tok = AutoTokenizer.from_pretrained(p0["target_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad = int(tok.pad_token_id)

    # Load the same 50 pairs Phase 0 used and keep the first n, so the fp32 side is a cache hit.
    prompts, labels, _ = flatten_binary_pairs(load_refusal_pairs(50, source="auto", seed=0)[: args.n_pairs])
    n_ref = len(prompts)
    prompts = list(prompts) + list(NEUTRAL_QUERIES[: args.n_neutral])
    inst, base = [], []
    for p in prompts:
        ie, be = encode_aligned_pair(tok, p, base_wrapper=wrapper, enable_thinking=False, max_total_tokens=512)
        inst.append(ie)
        base.append(be)
    inst_pos = [e.user_content_positions + [e.asst_last_idx] for e in inst]
    base_pos = [e.user_content_positions + [e.asst_last_idx] for e in base]
    roles: list[str] = []
    row_prompt: list[int] = []
    for i, e in enumerate(inst):
        roles += ["user_content"] * len(e.user_content_positions) + ["asst_last"]
        row_prompt += [i] * (len(e.user_content_positions) + 1)
    roles_a = np.array(roles)
    row_prompt_a = np.array(row_prompt)

    web = load_fineweb_passages(n_passages=args.n_web)
    web_seqs = [tok.encode(d["text"], add_special_tokens=False)[: args.web_max_tokens] for d in web]
    web_seqs = [s for s in web_seqs if len(s) >= 2]
    web_pos = [list(range(1, len(s))) for s in web_seqs]

    jobs = {
        "target": (p0["target_model"], [e.input_ids for e in inst], inst_pos),
        "base": (p0["base_model"], [e.input_ids for e in base], base_pos),
    }
    H: dict[str, dict[str, torch.Tensor]] = {"fp32": {}, "bf16": {}}
    timing: dict[str, float] = {}

    # fp32 reference through the disk cache (one model resident at a time).
    for key, (name, seqs, pos) in jobs.items():
        t0 = time.time()
        lm = cached_io.lazy_load_causal_lm(name)
        H["fp32"][f"{key}_chat"] = stack(cached_io.cached_extract(lm, seqs, pos, layer_index=layer, pad_token_id=pad))
        H["fp32"][f"{key}_web"] = stack(
            cached_io.cached_extract(lm, web_seqs, web_pos, layer_index=layer, pad_token_id=pad)
        )
        cached_io.lazy_unload_model(lm)
        timing[f"fp32_{key}_s"] = round(time.time() - t0, 1)
        print(f"[fp32] {key}: chat {tuple(H['fp32'][f'{key}_chat'].shape)} in {timing[f'fp32_{key}_s']}s", flush=True)

    for key, (name, seqs, pos) in jobs.items():
        t0 = time.time()
        model = load_bottom_bf16(name, layer, args.attn)
        timing[f"bf16_{key}_load_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        H["bf16"][f"{key}_chat"] = stack(
            extract_at_positions(model, seqs, pos, layer_index=layer, pad_token_id=pad, batch_size=args.batch_size)
        )
        H["bf16"][f"{key}_web"] = stack(
            extract_at_positions(model, web_seqs, web_pos, layer_index=layer, pad_token_id=pad,
                                 batch_size=args.batch_size)
        )
        timing[f"bf16_{key}_fwd_s"] = round(time.time() - t0, 1)
        del model
        import gc

        gc.collect()
        print(f"[bf16] {key}: load {timing[f'bf16_{key}_load_s']}s fwd {timing[f'bf16_{key}_fwd_s']}s", flush=True)

    out: dict[str, Any] = {
        "layer": layer, "base_wrapper": wrapper, "tau_web_median": tau, "attn": args.attn,
        "n_prompts": len(prompts), "n_web_docs": len(web_seqs), "timing": timing, "sections": {},
    }

    def diffs(dtype: str, part: str) -> tuple[torch.Tensor, torch.Tensor]:
        ht, hb = H[dtype][f"target_{part}"], H[dtype][f"base_{part}"]
        return apply_alignment(hb, ht, art), (ht - hb).float()

    for part in ("chat", "web"):
        d32, n32 = diffs("fp32", part)
        d16, n16 = diffs("bf16", part)
        ht32 = H["fp32"][f"target_{part}"]
        norms = ht32.norm(dim=-1)
        massive = norms > 5.0 * norms.median()
        masks: dict[str, torch.Tensor] = {"all": torch.ones_like(massive), "non_massive": ~massive, "massive": massive}
        if part == "chat":
            masks["user_content_non_massive"] = torch.from_numpy(roles_a == "user_content") & ~massive
            masks["asst_last"] = torch.from_numpy(roles_a == "asst_last")
        sec: dict[str, Any] = {"n_massive_rows": int(massive.sum())}
        for mname, m in masks.items():
            if int(m.sum()) < 2:
                continue
            sec[mname] = {
                "h_target": noise_stats(ht32[m], H["bf16"][f"target_{part}"][m]),
                "h_base": noise_stats(H["fp32"][f"base_{part}"][m], H["bf16"][f"base_{part}"][m]),
                "diff_id_anchored": noise_stats(d32[m], d16[m]),
                "diff_naive": noise_stats(n32[m], n16[m]),
                "median_diff_norm_over_tau": float(d32[m].norm(dim=-1).median() / tau),
                "median_diff_err_over_tau": float((d16[m] - d32[m]).norm(dim=-1).median() / tau),
            }
        out["sections"][part] = sec

    # Does the noise move the statistic the gate reads? Refusal Fisher SNR, both dtypes.
    snr: dict[str, Any] = {}
    lab = np.array(labels)
    for dtype in ("fp32", "bf16"):
        d, _ = diffs(dtype, "chat")
        ht = H[dtype]["target_chat"]
        is_ref = row_prompt_a < n_ref
        al = torch.from_numpy((roles_a == "asst_last") & is_ref)
        y_al = lab[row_prompt_a[al.numpy()]]
        um_d, um_h, y_um = [], [], []
        for i in range(n_ref):
            m = torch.from_numpy((row_prompt_a == i) & (roles_a == "user_content"))
            if int(m.sum()) == 0:
                continue
            um_d.append(d[m].mean(dim=0))
            um_h.append(ht[m].mean(dim=0))
            y_um.append(int(lab[i]))
        snr[dtype] = {
            "asst_last_h_target": compute_fisher_snr(ht[al], y_al),
            "asst_last_diff": compute_fisher_snr(d[al], y_al),
            "user_mean_h_target": compute_fisher_snr(torch.stack(um_h), y_um),
            "user_mean_diff": compute_fisher_snr(torch.stack(um_d), y_um),
        }
    out["refusal_fisher_snr"] = snr

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
