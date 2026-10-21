"""Mechanistic case study of one L24 head from the doc-viewer data (CPU only, no GPU or residuals).

Uses the doc-viewer outputs (per-site writes [T,32,4096], full attention [32,T,T], explanations) plus
three Qwen3-8B weight tensors (o_proj, final norm, lm_head). For the KV group containing --head:

1. z_h,i = pinv(O_h) w_h,i recovers each head's 128-d attention output (exact up to fp16).
2. The heads of a GQA group share value vectors, so stacking their attention rows gives an overdetermined
   system A V = Z for the per-source values V [T,128]; O_h v_j is then source j's write to head h.
3. Logit lens (final-norm gain, no RMS scale) of the sink value, of top-contributing sources and of
   the mean non-sink write; a sink + doc-topic 2-direction model of the write; attention selectivity;
   query-side gating; how often each explanation source names the doc's domain.

Usage: python scripts/heads_kv_group_study.py --viewer-dir ~/l24_docs/hf/l24/std_rl/doc_viewer --head 18
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.stats import spearmanr
from transformers import AutoTokenizer

DOMAINS = {
    "crime": r"crim|police|polic|victim|suspect|investigat|court|legal|law\b|defendant|prosecut|homicid|murder|fraud|justice|jail|sentenc|charge|offen",
    "computing": r"program|code|compil|software|comput|algorithm|syntax|semantic|interpreter|LLVM|developer|engineer",
    "medical": r"clinic|medic|patient|psych|therap|treat|disorder|depress|health|symptom|diagnos|brain",
}
DOC_DOMAIN = {"01": "crime", "02": "computing", "03": "medical", "04": "crime"}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_weights(repo, layer):
    shard = lambda s: hf_hub_download(repo, f"model-0000{s}-of-00005.safetensors")  # noqa: E731
    with safe_open(shard(3), "pt") as f:
        wo = f.get_tensor(f"model.layers.{layer}.self_attn.o_proj.weight").float()
    with safe_open(shard(4), "pt") as f:
        g = f.get_tensor("model.norm.weight").float()
    with safe_open(shard(5), "pt") as f:
        wug = (f.get_tensor("lm_head.weight").float() * g).bfloat16()
    return wo, wug


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer-dir", required=True)
    p.add_argument("--head", type=int, default=18)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--group-size", type=int, default=4, help="query heads per KV head (Qwen3-8B: 32/8)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    wo, wug = load_weights(a.model, a.layer)
    hd = wo.shape[1] // 32
    group = list(range(a.head // a.group_size * a.group_size, (a.head // a.group_size + 1) * a.group_size))
    k_h = group.index(a.head)
    O = {h: wo[:, h * hd:(h + 1) * hd] for h in group}

    def lens(x, k=8):
        return [tok.decode([t]) for t in (torch.as_tensor(x).float().bfloat16() @ wug.T).float().topk(k).indices.tolist()]

    docs = {}
    for d in sorted(glob.glob(os.path.join(a.viewer_dir, "0*"))):
        n = os.path.basename(d)
        ids = np.load(f"{d}/token_ids.npy")
        A = torch.tensor(np.load(f"{d}/attn.npz")["attn"][group].astype(np.float32))
        W = torch.tensor(np.load(f"{d}/writes.npz")["v"][:, group].astype(np.float32))
        Z = [torch.linalg.lstsq(O[h], W[:, k].T).solution.T for k, h in enumerate(group)]
        V = torch.linalg.lstsq(torch.cat(list(A)), torch.cat(Z)).solution
        fit = [((A[k] @ V - Z[k]).norm() / Z[k].norm()).item() for k in range(len(group))]
        S = pd.read_parquet(f"{d}/sites.parquet")
        S = S[S["head"] == a.head].sort_values("pos")
        docs[n] = dict(s=[tok.decode([t]) for t in ids], A=A, V=V, W=W[:, k_h].numpy(), S=S, fit=fit)
        print(f"{n}: shared-value fit rel. err per head {dict(zip(group, np.round(fit, 3)))}")

    print(f"\n=== H{a.head}: top sources by total contribution (sum_i a_ij |O v_j|), and their lens")
    for n, r in docs.items():
        U = r["V"] @ O[a.head].T
        A = r["A"][k_h]
        c = (A * U.norm(dim=1)[None]).sum(0)
        c[0] = 0
        print(f"-- {n}")
        for j in c.argsort(descending=True)[:8].tolist():
            print(f"   {r['s'][j]!r:>16}@{j:<4} {c[j]:6.1f}  {lens(U[j])}")

    print("\n=== mean non-sink write lens, every head in the KV group")
    for k, h in enumerate(group):
        print(f"H{h}  sink value: {lens(np.mean([r['V'][0].numpy() for r in docs.values()], 0) @ O[h].T.numpy())}")
        for n, r in docs.items():
            U, A = r["V"] @ O[h].T, r["A"][k]
            print(f"   {n[:16]:<16} {lens(((A[1:, 1:] @ U[1:]) / A[1:, 1:].sum(1, keepdim=True)).mean(0))}")

    names = list(docs)
    w0 = unit(np.stack([docs[n]["W"][0] for n in names]))
    print(f"\n=== H{a.head} write geometry\nsink write cos across docs: min {(w0 @ w0.T).min():.3f}")
    sink = unit(w0.mean(0))
    topic = {}
    for n in names:
        w = docs[n]["W"][1:]
        topic[n] = unit((w - np.outer(w @ sink, sink)).mean(0))
    print("doc-topic direction cos:\n", np.array([[topic[x] @ topic[y] for y in names] for x in names]).round(2))
    for n in names:
        u = unit(docs[n]["W"][1:])
        cap = lambda t: ((u @ np.linalg.qr(np.stack([sink, t], 1))[0]) ** 2).sum(1).mean()  # noqa: E731
        print(f"{n[:22]:<22} mean cos^2 captured: sink {((u @ sink) ** 2).mean():.2f} | sink+own topic {cap(topic[n]):.2f}"
              f" | sink+other topic {np.mean([cap(topic[m]) for m in names if m != n]):.2f}")

    print("\n=== attention selectivity: spearman(mean attention a source gets, its value's cos to doc topic)")
    for n in names:
        r = docs[n]
        proj = unit((r["V"] @ O[a.head].T).numpy()) @ topic[n]
        an = r["A"][k_h].numpy().copy()
        an[:, 0] = 0
        an[1:] /= an[1:].sum(1, keepdims=True)
        T = len(proj)
        recv = an[1:].sum(0) / np.arange(T, 0, -1)
        m = np.arange(1, T - 5)
        print(f"{n[:22]:<22} {spearmanr(recv[m], proj[m]).correlation:+.2f}")

    Q = pd.concat([r["S"].iloc[1:].assign(tok=[r["s"][i] for i in r["S"].pos.iloc[1:]], doc=n[:2]) for n, r in docs.items()])
    g = Q.groupby("tok").agg(n=("first_w", "size"), sink=("first_w", "mean")).query("n>=4").sort_values("sink")
    print("\n=== query gating (first-token weight by query token, >=4 occurrences)")
    print("lowest sink: ", [f"{t!r}:{v:.2f}" for t, v in g.sink.head(12).items()])
    print("highest sink:", [f"{t!r}:{v:.2f}" for t, v in g.sink.tail(12).items()])
    Q["sink_q"] = pd.qcut(Q.first_w, 4)
    print(Q.groupby("sink_q", observed=True)[["v_norm", "rl_fve", "gold_rl_fve", "sft_fve"]].mean().round(2))

    print("\n=== fraction of explanations naming each domain (doc domains: 01/04 crime, 02 computing, 03 medical)")
    rows = [dict(src=src[:-5], doc=d, dom=dom, rate=Q[Q.doc == d][src].str.contains(rx, case=False, regex=True).mean())
            for src in ["rl_text", "sft_text", "gold_text"] for d in sorted(Q.doc.unique()) for dom, rx in DOMAINS.items()]
    print(pd.DataFrame(rows).pivot_table(index=["src", "doc"], columns="dom", values="rate").round(2).to_string())


if __name__ == "__main__":
    main()
