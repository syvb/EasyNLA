"""Checkpoint curves for the KL-NLA RL pilot (docs/kl_nla_phase2.md).

For every saved RL checkpoint of one arm (step 0 = the SFT AV, no adapter), generate a
greedy explanation for each of N audit rows (docs no AV/AR trained on), then score all
of them with FROZEN judges that don't move during RL:
    A_kl   the Phase 1 KL-trained AR (the KL arm's starting AR: favours the KL arm)
    A_mse  the Phase 1 MSE-trained AR (the MSE arm's starting AR: favours the MSE arm)
    A_cal  A_mse's direction shrunk toward A_kl's long-generic no-info prior at alpha=0.6
           (the hedging control's calibrated MSE AR; built from A_mse, so a KL-arm win
           under it is conservative)
Per checkpoint: KL recovered (vs the mean direction) under A_kl and A_cal, FVE under
A_mse, extraction rate, and drift measures: explanation length (tokens), names the
target's greedy next token, word-4-gram overlap with the prefix (quoting), vagueness
(cosine of A_kl's prediction to A_kl's no-info prior), specificity (KL of A_kl's
prediction spliced into a DIFFERENT row's prefix minus on its own row).
Logs one W&B run (x axis = RL step) and writes curves.json / curves.md / explanations.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kl_audit import ar_predict, boot_docs, boot_ratio, load_ar, load_eval_rows, log, names_token  # noqa: E402

PRIOR_TEXT = ("Web text continuing a document about a general topic, written in a neutral "
              "informative register.")   # kl_hedge_control's "generic_long"
ALPHA_CAL = 0.6


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def ngram_overlap(expl, prefix, n=4):
    if not expl:
        return np.nan
    w = re.findall(r"\w+", expl.lower())
    grams = {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    if not grams:
        return np.nan
    p = re.findall(r"\w+", prefix.lower())
    pg = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
    return len(grams & pg) / len(grams)


@torch.no_grad()
def generate(model, tok, vref, prompt_text, acts, bs, max_new, device):
    """Greedy explanations, one per activation (left-padded batches)."""
    from nla.schema import extract_explanation
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out, lens = [], []
    for c0 in range(0, len(acts), bs):
        ca = acts[c0:c0 + bs]
        enc = tok([prompt_text] * len(ca), return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        vref[0] = torch.stack(ca).to(device).float()
        try:
            g = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                               max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        new = g[:, enc.input_ids.shape[1]:]
        for i in range(len(ca)):
            ids = [t for t in new[i].tolist() if t != tok.eos_token_id]
            text = tok.decode(ids, skip_special_tokens=True)
            out.append(extract_explanation(text))
            lens.append(len(ids))
    return out, lens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--explanations", required=True, help="fvecmp explanations.json (fixes the row set)")
    ap.add_argument("--av", required=True, help="merged SFT AV (the RL arms' base)")
    ap.add_argument("--run-dir", required=True, help="train_rl_vllm --save-dir (iter_XXXXXX adapters)")
    ap.add_argument("--ar-kl", required=True, help="DIR[:LORA_DIR] Phase 1 KL AR")
    ap.add_argument("--ar-mse", required=True, help="DIR[:LORA_DIR] Phase 1 MSE AR")
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=500, help="rows (seeded, spread over docs)")
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--av-dtype", default="bfloat16")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-fake-gen", action="store_true",
                    help="CPU smoke only: skip generation and use explanations.json conditions "
                         "(rotating per checkpoint) so the scoring path runs on a non-NLA model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.config import load_nla_config
    from nla.schema import INJECT_PLACEHOLDER, compute_predict_mean_baselines, normalize_activation
    from nla.utils import build_prompt_text, register_karvonen_hook
    from nla.utils.kl_splice import load_splice_kl, tokenize_prefixes

    # ---- rows: a seeded subset of the audit rows with usable prefixes ----
    T, keep, E, docs, gold = load_eval_rows(a.val, a.exclude, a.explanations)
    tok = AutoTokenizer.from_pretrained(a.av)
    cfg = load_nla_config(a.av, tok)
    pids = tokenize_prefixes(tok, [T.detokenized_text_truncated[i] for i in keep],
                             [T.n_raw_tokens[i] for i in keep])
    ok = [i for i in range(len(keep)) if pids[i] is not None]
    sel = sorted(np.random.default_rng(0).choice(ok, size=min(a.n, len(ok)), replace=False).tolist())
    R = len(sel)
    P = [pids[i] for i in sel]
    prefix_text = [T.detokenized_text_truncated[keep[i]] for i in sel]
    d_rows = docs[sel]
    acts = [gold[i] for i in sel]
    # fixed partner row from a DIFFERENT document, for the specificity measure
    other = np.empty(R, dtype=int)
    for i in range(R):
        j = (i + R // 2) % R
        while d_rows[j] == d_rows[i]:
            j = (j + 1) % R
        other[i] = j
    mean_hat = unit(normalize_activation(gold, 1.0).mean(0))
    prompt_msgs = pq.read_table(a.val, columns=["prompt"]).column("prompt")[keep[sel[0]]].as_py()
    assert INJECT_PLACEHOLDER in prompt_msgs[0]["content"]
    prompt_text = build_prompt_text(prompt_msgs, cfg.injection_char, tok)

    # ---- checkpoints ----
    steps = sorted(int(d.split("_")[1]) for d in os.listdir(a.run_dir) if re.fullmatch(r"iter_\d+", d))
    ckpts = [(0, None)] + [(s, os.path.join(a.run_dir, f"iter_{s:06d}")) for s in steps]
    log(f"{R} rows / {len(set(d_rows))} docs; checkpoints: {[s for s, _ in ckpts]}")

    # ---- generation: one base, adapters swapped in ----
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(a.av, torch_dtype=getattr(torch, a.av_dtype),
                                                attn_implementation="sdpa").to(a.device).eval()
    model = base
    if len(ckpts) > 1:
        model = PeftModel.from_pretrained(base, ckpts[1][1], adapter_name=f"s{ckpts[1][0]}")
        for s, path in ckpts[2:]:
            model.load_adapter(path, adapter_name=f"s{s}")
    vref = [None]
    register_karvonen_hook(model, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    gens = {}
    fake = ["av_greedy", "quote", "wrong", "av_sample", "gold", "rl_greedy"]
    for k, (s, path) in enumerate(ckpts if not a.debug_fake_gen else []):
        t0 = time.time()
        if path is None:
            if model is base:
                gens[s] = generate(model, tok, vref, prompt_text, acts, a.gen_batch, a.max_new_tokens, a.device)
            else:
                with model.disable_adapter():
                    gens[s] = generate(model, tok, vref, prompt_text, acts, a.gen_batch, a.max_new_tokens, a.device)
        else:
            model.set_adapter(f"s{s}")
            gens[s] = generate(model, tok, vref, prompt_text, acts, a.gen_batch, a.max_new_tokens, a.device)
        ext = np.mean([e is not None for e in gens[s][0]])
        log(f"  step {s}: generated ({time.time() - t0:.0f}s), extraction {ext:.1%}")
    if a.debug_fake_gen:
        for k, (s, _) in enumerate(ckpts):
            ex = [E[fake[k % len(fake)]][i] for i in sel]
            gens[s] = (ex, [len(tok.encode(e)) if e else 0 for e in ex])
        log(f"  DEBUG: fake generations from explanations.json for steps {list(gens)}")
    json.dump({str(s): {"explanation": g[0], "gen_tokens": g[1]} for s, g in gens.items()} | {"doc_id": d_rows.tolist()},
              open(os.path.join(a.out, "explanations.json"), "w"))
    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- frozen judges ----
    preds = {}
    for name, spec in [("kl", a.ar_kl), ("mse", a.ar_mse)]:
        _, critic = load_ar(f"{name}={spec}", a.device)
        for s in gens:
            preds[(name, s)] = ar_predict(critic, tok, cfg.critic_prompt_template, gens[s][0],
                                          cfg.mse_scale, a.device)
        if name == "kl":
            prior = ar_predict(critic, tok, cfg.critic_prompt_template, [PRIOR_TEXT],
                               cfg.mse_scale, a.device)[0]
        del critic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    prior_hat = unit(prior)
    for s in gens:
        m = preds[("mse", s)]
        preds[("cal", s)] = unit(ALPHA_CAL * unit(m) + (1 - ALPHA_CAL) * prior_hat.expand_as(m))

    splice = load_splice_kl(a.target, a.ar_kl.partition(":")[0], cfg.extraction_layer_index, a.device,
                            a.micro_batch, dtype=getattr(torch, a.target_dtype))
    # one splice pass: per row, the mean-direction baseline + every (judge, step) prediction,
    # plus A_kl's prediction spliced into the deranged row (specificity)
    pref, vecs, who = [], [], []
    for i in range(R):
        pref.append(P[i]); vecs.append(mean_hat); who.append(("mean", None, i))
    for (j, s), v in preds.items():
        if j == "mse":
            continue
        for i in range(R):
            if not torch.isnan(v[i, 0]):
                pref.append(P[i]); vecs.append(v[i]); who.append((j, s, i))
                if j == "kl":
                    pref.append(P[other[i]]); vecs.append(v[i]); who.append(("kl_other", s, i))
    kl_all, top_o, _ = splice.kl(pref, torch.stack(vecs).float().to(a.device), return_argmax=True)
    kl_all = kl_all.float().cpu().numpy()
    # every (judge, step) array exists even if a checkpoint extracted nothing (all NaN)
    KL = {(j, s): np.full(R, np.nan) for j in ("cal", "kl", "kl_other") for s in gens}
    top_orig = np.zeros(R, dtype=np.int64)
    for k, (j, s, i) in enumerate(who):
        KL.setdefault((j, s), np.full(R, np.nan))[i] = kl_all[k]
        if j == "mean":
            top_orig[i] = int(top_o[k])
    base_kl = KL[("mean", None)]
    tok_str = [tok.decode([t]).strip() for t in top_orig]
    content = np.array([len(x) >= 3 and any(c.isalpha() for c in x) for x in tok_str])
    gn = normalize_activation(torch.stack(acts), cfg.mse_scale)
    _, fve_base = compute_predict_mean_baselines(torch.stack(acts), cfg.mse_scale)

    curves = []
    for s in gens:
        ex, ln = gens[s]
        mse = ((normalize_activation(preds[("mse", s)], cfg.mse_scale) - gn) ** 2).mean(-1).numpy()
        named = np.array([names_token(ex[i], tok_str[i]) if content[i] else np.nan for i in range(R)])
        spec = KL[("kl_other", s)] - KL[("kl", s)]
        vague = torch.nn.functional.cosine_similarity(preds[("kl", s)], prior.expand_as(preds[("kl", s)]), dim=-1).numpy()
        row = {
            "step": s,
            "kl_recovered/A_cal": boot_ratio(KL[("cal", s)], base_kl, d_rows),
            "kl_recovered/A_kl": boot_ratio(KL[("kl", s)], base_kl, d_rows),
            "fve/A_mse": boot_docs(1 - mse / fve_base, d_rows),
            "extraction_rate": float(np.mean([e is not None for e in ex])),
            "gen_tokens": boot_docs(np.array(ln, dtype=float), d_rows),
            "names_next_token": boot_docs(named, d_rows),
            "quote_4gram_overlap": boot_docs(np.array([ngram_overlap(ex[i], prefix_text[i]) for i in range(R)]), d_rows),
            "vagueness_cos_prior": boot_docs(vague, d_rows),
            "specificity_nats": boot_docs(spec, d_rows),
        }
        curves.append(row)
        log(f"  step {s}: KLrec cal {row['kl_recovered/A_cal'][0]:.3f} kl {row['kl_recovered/A_kl'][0]:.3f} "
            f"| FVE {row['fve/A_mse'][0]:.3f} | ext {row['extraction_rate']:.0%} | len {row['gen_tokens'][0]:.0f} "
            f"| next-tok {row['names_next_token'][0]:.2f} | quote {row['quote_4gram_overlap'][0]:.3f} "
            f"| vague {row['vagueness_cos_prior'][0]:.3f} | spec {row['specificity_nats'][0]:.2f}")
    json.dump({"n_rows": R, "n_docs": int(len(set(d_rows))), "alpha_cal": ALPHA_CAL, "curves": curves},
              open(os.path.join(a.out, "curves.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "rows.npz"), doc_id=d_rows, top_orig=top_orig,
                        **{f"kl/{j}/{s}": v for (j, s), v in KL.items()})
    cols = ["kl_recovered/A_cal", "kl_recovered/A_kl", "fve/A_mse", "names_next_token",
            "quote_4gram_overlap", "vagueness_cos_prior", "specificity_nats", "gen_tokens"]
    L = [f"# RL checkpoint curves ({R} rows / {len(set(d_rows))} docs)", "",
         "| step | " + " | ".join(cols) + " | extraction |", "|" + "---|" * (len(cols) + 2)]
    for r in curves:
        L.append(f"| {r['step']} | " + " | ".join(f"{r[c][0]:.3f}" for c in cols) + f" | {r['extraction_rate']:.1%} |")
    open(os.path.join(a.out, "curves.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)

    if a.wandb_project:
        import wandb
        run = wandb.init(project=a.wandb_project, group=a.wandb_group, name=a.wandb_name,
                         job_type="curve_eval", config=vars(a))
        for r in curves:
            wandb.log({k: (v[0] if isinstance(v, list) else v) for k, v in r.items() if k != "step"}
                      | {f"{k}_lo": v[1] for k, v in r.items() if isinstance(v, list)}
                      | {f"{k}_hi": v[2] for k, v in r.items() if isinstance(v, list)}, step=r["step"])
        run.finish()


if __name__ == "__main__":
    main()
