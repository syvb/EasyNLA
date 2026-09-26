"""KL audit of NLA reconstructors (docs/kl_nla.md, Phases 0 and 1).

On the fvecmp validation rows (docs no AV/AR trained on), with the explanations
fvecmp saved, score every condition two ways:
    FVE  1 - MSE(pred, gold) / predict-the-mean MSE           (the NLA paper metric)
    KL   KL(p_orig || p_splice): the target's next-token distribution at the
         extraction position with the prediction spliced over layer K there
         (nla/utils/kl_splice.py). "KL recovered" = 1 - KL / KL(mean direction).
Text conditions go through each AR given (--ar name=DIR[:LORA_DIR]); vector
conditions (mean direction, orthogonal-to-gold, gold rotated to cosine c) do not
depend on the AR. Also: the next-token diagnostic (does the explanation name the
target's greedy next token, and how much of the KL gain sits on rows that do).

Gates (exit 3 if either fails, so the pod stops before training):
    G1  splicing the stored activation gives KL ~ 0: median < 1e-2 and p90 < 5e-2 nats
        (bf16 recompute noise; a layer/tokenization bug gives nats)
    G2  >= 99% of the prefixes re-tokenize to n_raw_tokens
Intervals: 95% bootstrap over documents (rows from one document are correlated).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXT_CONDS = ["gold", "av_greedy", "av_sample", "rl_greedy", "rl_sample", "quote", "wrong"]
COSINES = [0.95, 0.9, 0.8, 0.7, 0.5]


def log(*a):
    print(f"[kl_audit {time.strftime('%H:%M:%S')}]", *a, flush=True)


def boot_docs(x, docs, B=4000, seed=0):
    """Mean of x with a 95% bootstrap interval over documents."""
    ok = ~np.isnan(x)
    x, docs = x[ok], docs[ok]
    if len(x) == 0:
        return [float("nan")] * 3 + [0]
    u, inv = np.unique(docs, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u))
    r = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        j = r.integers(0, len(u), len(u))
        bs.append(sums[j].sum() / max(cnts[j].sum(), 1))
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(u))]


def boot_ratio(num, den, docs, B=4000, seed=0):
    """1 - sum(num)/sum(den) with a doc bootstrap (KL recovered, pooled)."""
    ok = ~(np.isnan(num) | np.isnan(den))
    num, den, docs = num[ok], den[ok], docs[ok]
    u, inv = np.unique(docs, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(u))
    sd = np.bincount(inv, weights=den, minlength=len(u))
    r = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        j = r.integers(0, len(u), len(u))
        bs.append(1 - sn[j].sum() / sd[j].sum())
    return [float(1 - num.sum() / den.sum()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)), int(len(u))]


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ok.sum() > 2 else float("nan")


def load_ar(spec, device):
    """name=DIR or name=DIR:LORA_DIR (a train_sft --use-lora AR checkpoint on top of DIR)."""
    from nla.models import NLACriticModel
    name, path = spec.split("=", 1)
    base, _, lora = path.partition(":")
    critic = NLACriticModel.from_pretrained(base, torch_dtype=torch.bfloat16)
    if lora:
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file
        meta = json.load(open(os.path.join(lora, "ar_meta.json")))
        inject_adapter_in_model(LoraConfig(
            r=meta["lora_r"], lora_alpha=meta["lora_alpha"], lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", use_rslora=True, target_modules=meta["target_modules"],
        ), critic.backbone)
        sd = load_file(os.path.join(lora, "ar_lora_value_head.safetensors"))
        own = dict(critic.named_parameters())
        missing = [k for k in own if ("lora_" in k or k.startswith("value_head")) and k not in sd]
        extra = [k for k in sd if k not in own]
        assert not missing and not extra, f"LoRA load mismatch: missing {missing[:3]} extra {extra[:3]}"
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].dtype))
        log(f"AR {name}: {base} + LoRA {lora} ({len(sd)} tensors)")
    else:
        log(f"AR {name}: {base}")
    return name, critic.to(device).eval()


@torch.no_grad()
def ar_predict(critic, tok, template, texts, mse_scale, device, bs=64):
    """[N, d] predictions; NaN rows where the text is missing or the prompt > 1024 tokens."""
    from nla.utils import critic_predict
    out = torch.full((len(texts), critic.value_head.weight.shape[0]), float("nan"))
    ids = [None if t is None else tok.encode(template.format(explanation=t), add_special_tokens=False)
           for t in texts]
    ok = [i for i, x in enumerate(ids) if x is not None and 0 < len(x) <= 1024]
    for cs in range(0, len(ok), bs):
        ch = ok[cs:cs + bs]
        T = max(len(ids[i]) for i in ch)
        bx = torch.full((len(ch), T), tok.eos_token_id, dtype=torch.long, device=device)
        at = torch.zeros((len(ch), T), dtype=torch.long, device=device)
        for r, i in enumerate(ch):
            bx[r, :len(ids[i])] = torch.tensor(ids[i], device=device)
            at[r, :len(ids[i])] = 1
        out[ch] = critic_predict(critic, bx, at, mse_scale).float().cpu()
    return out


def names_token(expl, tok_str):
    """Does the explanation contain the next token as a whole word (case-insensitive)?"""
    if expl is None:
        return np.nan
    return float(re.search(rf"(?<![A-Za-z0-9]){re.escape(tok_str)}(?![A-Za-z0-9])", expl, re.I) is not None)


def load_eval_rows(val, exclude, explanations, n=None):
    """Exactly fvecmp's rows (same filter, same order), checked against its saved doc_ids.
    Returns (table, kept row indices, explanations dict, doc_id array, gold [N, d])."""
    from nla.schema import extract_explanation
    T = pq.read_table(val, columns=["response", "activation_vector", "doc_id",
                                    "detokenized_text_truncated", "n_raw_tokens"]).to_pandas()
    seen = set()
    for p in exclude:
        seen |= set(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
    keep = [i for i in range(len(T)) if T.doc_id[i] not in seen and extract_explanation(T.response[i])]
    E = json.load(open(explanations))
    assert [T.doc_id[i] for i in keep] == E["doc_id"], "row set differs from explanations.json"
    if n:
        keep = keep[:n]
        E = {k: v[:n] for k, v in E.items()}
    docs = np.array([T.doc_id[i] for i in keep])
    log(f"{len(keep)} rows from {len(set(docs))} docs (excluded {len(seen)} training docs); "
        f"matches explanations.json")
    gold = torch.tensor(np.stack([np.asarray(T.activation_vector[i], dtype=np.float32) for i in keep]))
    return T, keep, E, docs, gold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True, help="av_sft_val.parquet (+ its .nla_meta.yaml)")
    ap.add_argument("--exclude", nargs="*", default=[], help="training parquets (fvecmp's exclusion set)")
    ap.add_argument("--explanations", required=True, help="fvecmp explanations.json")
    ap.add_argument("--ar", action="append", required=True, help="name=DIR[:LORA_DIR]")
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=None, help="first n rows only (smoke)")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from nla.config import load_nla_config
    from nla.schema import compute_predict_mean_baselines, normalize_activation
    from nla.utils.kl_splice import load_splice_kl, orthogonal_fail_vectors, tokenize_prefixes

    T, keep, E, docs, gold = load_eval_rows(a.val, a.exclude, a.explanations, a.n)
    N = len(keep)

    ar_dir0 = a.ar[0].split("=", 1)[1].partition(":")[0]
    tok = AutoTokenizer.from_pretrained(ar_dir0)
    cfg = load_nla_config(ar_dir0, tok)
    mse_scale = cfg.mse_scale
    _, base_mse = compute_predict_mean_baselines(gold, mse_scale)

    # ---- G2: prefixes ----
    pids = tokenize_prefixes(tok, [T.detokenized_text_truncated[i] for i in keep],
                             [T.n_raw_tokens[i] for i in keep])
    g2 = sum(p is not None for p in pids) / N
    log(f"G2 round-trip: {g2:.2%}")
    rows = [i for i in range(N) if pids[i] is not None]
    P = [pids[i] for i in rows]

    # ---- vectors: AR-independent conditions ----
    g_rows = gold[rows]
    g_hat = g_rows / g_rows.norm(dim=-1, keepdim=True)
    mean_dir = normalize_activation(gold, 1.0).mean(0)
    rng = torch.Generator().manual_seed(0)
    noise = torch.randn(g_hat.shape, generator=rng)
    u = noise - (noise * g_hat).sum(-1, keepdim=True) * g_hat
    u = u / u.norm(dim=-1, keepdim=True)
    vec_conds = {"stored": g_rows, "mean_dir": mean_dir.expand_as(g_rows),
                 "orthogonal": orthogonal_fail_vectors(g_rows, mean_dir)}
    for c in COSINES:
        vec_conds[f"cos{c}"] = c * g_hat + (1 - c * c) ** 0.5 * u

    # ---- text conditions through each AR ----
    text_preds = {}
    for spec in a.ar:
        name, critic = load_ar(spec, a.device)
        for c in TEXT_CONDS:
            text_preds[f"{name}/{c}"] = ar_predict(critic, tok, cfg.critic_prompt_template,
                                                   E[c], mse_scale, a.device)[rows]
            log(f"  AR {name} {c}: {(~text_preds[f'{name}/{c}'][:, 0].isnan()).sum().item()} scored")
        del critic
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- one splice pass: every condition of a row shares its cached prefix ----
    splice = load_splice_kl(a.target, ar_dir0, cfg.extraction_layer_index, a.device,
                            a.micro_batch, dtype=getattr(torch, a.target_dtype))
    conds = list(vec_conds) + list(text_preds)
    allv = {**vec_conds, **text_preds}
    R = len(rows)
    KL = {c: np.full(R, np.nan) for c in conds}
    TOP = {c: np.full(R, np.nan) for c in conds}
    top_orig = np.zeros(R, dtype=np.int64)
    t0 = time.time()
    chunk = 256
    for cs in range(0, R, chunk):
        idx = list(range(cs, min(cs + chunk, R)))
        pref, vecs, who = [], [], []
        for c in conds:
            v = allv[c][idx]
            for k, i in enumerate(idx):
                if not torch.isnan(v[k, 0]):
                    pref.append(P[i]); vecs.append(v[k]); who.append((c, i))
        with torch.no_grad():
            kl, to, ts = splice.kl(pref, torch.stack(vecs).to(a.device), return_argmax=True)
        kl = kl.float().cpu().numpy()
        for j, (c, i) in enumerate(who):
            KL[c][i] = kl[j]
            TOP[c][i] = float(ts[j] == to[j])
            top_orig[i] = int(to[j])
        log(f"  splice {min(cs + chunk, R)}/{R} rows ({time.time() - t0:.0f}s)")

    # ---- MSE (normalized, the fvecmp/FVE definition) ----
    gn = normalize_activation(g_rows, mse_scale)
    MSE = {c: ((normalize_activation(allv[c], mse_scale) - gn) ** 2).mean(-1).numpy() for c in conds}
    d_rows = docs[rows]

    # ---- G1 ----
    st = KL["stored"]
    g1 = {"median": float(np.median(st)), "p90": float(np.percentile(st, 90)), "max": float(st.max())}
    g1_ok = g1["median"] < 1e-2 and g1["p90"] < 5e-2
    log(f"G1 stored-activation KL: {g1} -> {'PASS' if g1_ok else 'FAIL'}")

    # ---- summary ----
    summ = {"n_rows": R, "n_docs": int(len(set(d_rows))), "g2_roundtrip": g2, "g1": g1,
            "g1_pass": g1_ok, "g2_pass": g2 >= 0.99, "fve_baseline_mse": base_mse,
            "kl_mean_baseline": boot_docs(KL["mean_dir"], d_rows), "conditions": {}}
    for c in conds:
        summ["conditions"][c] = {
            "kl": boot_docs(KL[c], d_rows),
            "kl_recovered": boot_ratio(KL[c], KL["mean_dir"], d_rows),
            "fve": boot_docs(1 - MSE[c] / base_mse, d_rows),
            "top1_agree": boot_docs(TOP[c], d_rows),
            "spearman_kl_mse": spearman(KL[c], MSE[c]),
        }
    # pooled rank correlation over all text conditions of each AR
    for spec in a.ar:
        name = spec.split("=", 1)[0]
        ks = np.concatenate([KL[f"{name}/{c}"] for c in TEXT_CONDS])
        ms = np.concatenate([MSE[f"{name}/{c}"] for c in TEXT_CONDS])
        summ[f"spearman_kl_mse_pooled/{name}"] = spearman(ks, ms)

    # ---- next-token diagnostic ----
    tok_str = [tok.decode([t]).strip() for t in top_orig]
    content = np.array([len(s) >= 3 and any(ch.isalpha() for ch in s) for s in tok_str])
    nt = {"content_word_rows": int(content.sum()), "per_condition": {}}
    for spec in a.ar:
        name = spec.split("=", 1)[0]
        for c in TEXT_CONDS:
            key = f"{name}/{c}"
            named = np.array([names_token(E[c][rows[i]], tok_str[i]) if content[i] else np.nan
                              for i in range(R)])
            gain = KL["mean_dir"] - KL[key]
            m = ~np.isnan(named) & ~np.isnan(gain)
            share = float(gain[m & (named == 1)].sum() / gain[m].sum()) if gain[m].sum() != 0 else float("nan")
            nt["per_condition"][key] = {
                "named_rate": boot_docs(named, d_rows),
                "kl_recovered_named": boot_ratio(np.where(named == 1, KL[key], np.nan),
                                                 np.where(named == 1, KL["mean_dir"], np.nan), d_rows),
                "kl_recovered_not_named": boot_ratio(np.where(named == 0, KL[key], np.nan),
                                                     np.where(named == 0, KL["mean_dir"], np.nan), d_rows),
                "gain_share_on_named_rows": share,
            }
    summ["next_token"] = nt

    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "rows.npz"), doc_id=d_rows, top_orig=top_orig,
                        **{f"kl/{c}": KL[c] for c in conds}, **{f"mse/{c}": MSE[c] for c in conds},
                        **{f"top1/{c}": TOP[c] for c in conds})

    lines = [f"# KL audit ({R} rows / {summ['n_docs']} docs)", "",
             f"G1 stored-activation KL median {g1['median']:.2e}, p90 {g1['p90']:.2e} -> "
             f"{'PASS' if g1_ok else 'FAIL'}; G2 round-trip {g2:.2%} -> {'PASS' if g2 >= 0.99 else 'FAIL'}", "",
             "| condition | KL (nats) | KL recovered | FVE | top-1 agree | rho(KL,MSE) |", "|---|---|---|---|---|---|"]
    f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"                          # noqa: E731
    for c in conds:
        s = summ["conditions"][c]
        lines.append(f"| {c} | {f3(s['kl'])} | {f3(s['kl_recovered'])} | {f3(s['fve'])} | "
                     f"{s['top1_agree'][0]:.3f} | {s['spearman_kl_mse']:.2f} |")
    lines += ["", f"Next-token diagnostic ({nt['content_word_rows']} rows whose greedy next token is a content word):",
              "", "| condition | names next token | KL rec. (named) | KL rec. (not named) | gain share on named |",
              "|---|---|---|---|---|"]
    for k, s in nt["per_condition"].items():
        lines.append(f"| {k} | {f3(s['named_rate'])} | {f3(s['kl_recovered_named'])} | "
                     f"{f3(s['kl_recovered_not_named'])} | {s['gain_share_on_named_rows']:.2f} |")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    sys.exit(0 if (g1_ok and g2 >= 0.99) else 3)


if __name__ == "__main__":
    main()
