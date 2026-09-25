"""Do normal NLA explanations carry more than a text-only description would?

A reconstruction NLA is judged by how much of the activation the AR critic can rebuild from the
explanation (FVE). Most of an activation is a function of the input text, so a description of
the text alone should already reconstruct a lot. This compares, on warm-start VALIDATION rows
whose documents neither the SFT verbalizer nor the AR trained on:
    av_greedy / av_sample  the SFT verbalizer's explanation, generated FROM THE ACTIVATION
    gold                   Claude Sonnet 4.6's explanation written from the TEXT ONLY (the warm-start
                           target, same prompt) -- the input-only baseline, in-distribution for the AR
    quote                  the last 40 words of the text, verbatim
    wrong                  av_greedy of a different document (chance)
FVE = 1 - MSE(AR(expl), activation) / predict-the-mean MSE (the NLA paper definition), per row.
av - gold > 0: the verbalizer conveys activation content a strong reader of the text could not infer.
Intervals bootstrap over documents (rows from one document are correlated)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trunc_check import log  # noqa: E402


def boot_docs(x, docs, B=4000, seed=0):
    """Mean of x with a 95% bootstrap interval over documents."""
    ok = ~np.isnan(x); x, docs = x[ok], docs[ok]
    u, inv = np.unique(docs, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(u)); cnts = np.bincount(inv, minlength=len(u))
    r = np.random.default_rng(seed); bs = []
    for _ in range(B):
        j = r.integers(0, len(u), len(u)); bs.append(sums[j].sum() / cnts[j].sum())
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(u))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True, help="av_sft_val.parquet (response = text-only gold explanation)")
    ap.add_argument("--exclude", nargs="*", default=[], help="parquets whose documents the models trained on")
    ap.add_argument("--av", default="syvb/nanonla-qwen3-8b-L24-av")
    ap.add_argument("--ar", default="syvb/nanonla-qwen3-8b-L24-ar")
    ap.add_argument("--rl-adapter", default=None,
                    help="reconstruction-RL LoRA on the SFT verbalizer, 'repo#subfolder' (e.g. ...-rl-lora#p0.0)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true", help="CPU smoke: skip the models, synthetic MSEs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from nla.config import load_nla_config
    from nla.schema import compute_predict_mean_baselines, extract_explanation, resolve_target_scale

    T = pq.read_table(a.val, columns=["prompt", "response", "activation_vector", "doc_id",
                                      "detokenized_text_truncated"]).to_pandas()
    seen = set()
    for p in a.exclude:
        seen |= set(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
    keep = [i for i in range(len(T)) if T.doc_id[i] not in seen and extract_explanation(T.response[i])]
    if a.n:
        keep = list(np.random.default_rng(0).permutation(keep)[:a.n])
    docs = np.array([T.doc_id[i] for i in keep])
    log(f"{len(T)} validation rows; {len(keep)} kept from {len(set(docs))} documents unseen by the models "
        f"(excluded {len(seen)} training documents)")
    ACT = torch.tensor(np.stack([np.asarray(T.activation_vector[i], dtype=np.float32) for i in keep]))
    E = {"gold": [extract_explanation(T.response[i]) for i in keep],
         "quote": [" ".join(T.detokenized_text_truncated[i].split()[-40:]) for i in keep]}
    prompts = [[dict(m) for m in T.prompt[i]] for i in keep]
    del T

    tok = AutoTokenizer.from_pretrained(a.av)
    cfg = load_nla_config(a.val, tok)
    heads = [("av", None, None)]
    if a.rl_adapter:
        repo, _, sub = a.rl_adapter.partition("#"); heads.append(("rl", repo, sub or None))
    if not a.fake:
        from nla.pred.av import generate_explanations, load_av
        rows = [{"prompt": p, "activation": v} for p, v in zip(prompts, ACT.numpy())]
        for head, repo, sub in heads:
            model, vref = load_av(a.av, repo, adapter_subfolder=sub, device=a.device, dtype=torch.bfloat16,
                                  inj_ids=(cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                           cfg.injection_right_neighbor_id))
            for mode, temp in (("greedy", 0.0), ("sample", 1.0)):
                name = f"{head}_{mode}"; t0 = time.time(); torch.manual_seed(0)
                E[name] = generate_explanations(model, tok, vref, rows, cfg.injection_char, max_new_tokens=a.max_new_tokens,
                                                temperature=temp, batch_size=a.gen_batch, device=a.device)
                log(f"{name}: extraction {np.mean([e is not None for e in E[name]]):.1%}, median words "
                    f"{np.median([len(e.split()) for e in E[name] if e]):.0f} ({(time.time() - t0) / 60:.1f} min)")
            del model, vref
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
    else:
        for head, _, _ in heads:
            E[f"{head}_greedy"] = [g[: len(g) // 2] for g in E["gold"]]; E[f"{head}_sample"] = list(E[f"{head}_greedy"])
    # wrong: av_greedy of a row from a DIFFERENT document
    rng = np.random.default_rng(3); n = len(keep); other = np.empty(n, int)
    for i in range(n):
        j = int(rng.integers(0, n))
        while docs[j] == docs[i]:
            j = int(rng.integers(0, n))
        other[i] = j
    E["wrong"] = [E["av_greedy"][j] for j in other]
    json.dump({"doc_id": docs.tolist(), **E}, open(os.path.join(a.out, "explanations.json"), "w"))

    mse_scale = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    _, base = compute_predict_mean_baselines(ACT, mse_scale)
    log(f"predict-the-mean baseline MSE on these rows: {base:.4f} (nanoNLA held-out used 0.6704)")
    if a.fake:
        M = {k: np.array([np.nan if e is None else base * (0.5 + 0.1 * (len(e) % 3)) for e in v]) for k, v in E.items()}
    else:
        from nla.models import NLACriticModel
        from nla.pred.rewards import ReconReward
        critic = NLACriticModel.from_pretrained(a.ar, torch_dtype=torch.bfloat16).to(a.device).eval()
        rr = ReconReward(critic, tok, cfg.critic_prompt_template, mse_scale, a.device, fve_baseline=base)
        M = {}
        for k, v in E.items():
            m = rr._mse(v, list(ACT))
            M[k] = np.array([np.nan if x is None else x for x in m], float)
            log(f"scored {k}")
    F = {k: 1.0 - v / base for k, v in M.items()}
    np.savez_compressed(os.path.join(a.out, "fve.npz"), doc_id=docs, **F)
    res = {"n_rows": n, "n_docs": int(len(set(docs))), "baseline_mse": base}
    lines = [f"FVE on {n} validation rows from {len(set(docs))} unseen documents (95% CI bootstrapped over documents)"]
    for k in [f"{h}_{m}" for h, _, _ in heads for m in ("greedy", "sample")] + ["gold", "quote", "wrong"]:
        d = boot_docs(F[k], docs); res[k] = d
        fn = 1.0 - M[k] / 0.6704; dn = boot_docs(fn, docs); res[f"{k} (nanoNLA baseline 0.6704)"] = dn
        lines.append(f"  {k:<10} FVE {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]   (on the nanoNLA baseline: {dn[0]:.3f})"
                     f"   (scored {int((~np.isnan(F[k])).sum())})")
    lines.append("paired differences:")
    pairs = [("av_greedy", "gold"), ("av_sample", "gold"), ("gold", "quote"), ("av_greedy", "quote"),
             ("av_greedy", "wrong"), ("gold", "wrong")]
    if a.rl_adapter:
        pairs = [("rl_greedy", "gold"), ("rl_sample", "gold"), ("rl_greedy", "av_greedy"), ("rl_sample", "av_sample")] + pairs
    for x, y in pairs:
        d = boot_docs(F[x] - F[y], docs); res[f"{x} - {y}"] = d
        lines.append(f"  {x} - {y}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    for h, _, _ in heads:
        both = ~np.isnan(F[f"{h}_greedy"]) & ~np.isnan(F["gold"])
        res[f"frac_rows_{h}_beats_gold"] = float(np.mean(F[f"{h}_greedy"][both] > F["gold"][both]))
        lines.append(f"  rows where {h}_greedy reconstructs better than gold: {res[f'frac_rows_{h}_beats_gold']:.1%}")
    lines.append("examples (gold text-only | av_greedy from activation):")
    for i in range(min(3, n)):
        lines.append(f"  [{(E['gold'][i] or '')[:260]}]\n    [{(E['av_greedy'][i] or '')[:260]}]")
    print("\n".join(lines), flush=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
