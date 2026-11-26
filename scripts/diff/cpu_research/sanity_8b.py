"""Sanity checks on the cached 8B activations/logits and a closer look at Base-vs-Instruct drift.

    python sanity_8b.py <manifest.jsonl> <phase0_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.alignment import load_alignment, predict_aligned_base
from nla.diff.benchmarks import NEUTRAL_QUERIES, load_fineweb_passages
from nla.diff.wrappers import encode_aligned_pair

manifest, p0 = sys.argv[1], Path(sys.argv[2])
K = 24
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
pad = int(tok.pad_token_id)
tgt = cached_io.LazyLM("Qwen/Qwen3-8B")
base = cached_io.LazyLM("Qwen/Qwen3-8B-Base")

# 1. Do the split-loader logits make sense as language-model predictions?
print("== next-token argmax at asst_last (Instruct ChatML | Base ChatML | Base Plain) ==")
for q in NEUTRAL_QUERIES[:6]:
    ie, bc = encode_aligned_pair(tok, q, base_wrapper="chatml", enable_thinking=False, max_total_tokens=512)
    _, bp = encode_aligned_pair(tok, q, base_wrapper="plain", enable_thinking=False, max_total_tokens=512)
    outs = []
    for m, e in ((tgt, ie), (base, bc), (base, bp)):
        ex = cached_io.cached_extract(m, [e.input_ids], [[e.asst_last_idx]], layer_index=K, pad_token_id=pad,
                                      topk_logits=3)[0]
        outs.append(" ".join(f"{tok.decode([i])!r}:{p:.2f}" for i, p in zip(ex.topk_ids[0], ex.topk_probs[0])))
    print(f"- {q[:50]!r}\n    inst: {outs[0]}\n    base/chatml: {outs[1]}\n    base/plain: {outs[2]}")

# 2. Web tokens: how far apart are the two residual streams, and where does the gap live?
docs = load_fineweb_passages(n_passages=2000)[1600:2000]
seqs = [tok.encode(d["text"], add_special_tokens=False)[:64] for d in docs]
seqs = [s for s in seqs if len(s) >= 2]
pos = [list(range(1, len(s))) for s in seqs]
ht = torch.cat([e.hidden for e in cached_io.cached_extract(tgt, seqs, pos, layer_index=K, pad_token_id=pad)]).double()
hb = torch.cat([e.hidden for e in cached_io.cached_extract(base, seqs, pos, layer_index=K, pad_token_id=pad)]).double()
print(f"\n== web-val tokens: {tuple(ht.shape)} ==")
print(f"median ||h_t||={ht.norm(dim=1).median():.1f}  median ||h_b||={hb.norm(dim=1).median():.1f}  "
      f"median ||h_t-h_b||={(ht - hb).norm(dim=1).median():.1f}")
cos = torch.nn.functional.cosine_similarity(ht, hb, dim=1)
htc, hbc = ht - ht.mean(0), hb - hb.mean(0)
cosc = torch.nn.functional.cosine_similarity(htc, hbc, dim=1)
print(f"cos(h_t, h_b): mean={cos.mean():.3f}  centred: mean={cosc.mean():.3f}")
var_t = htc.pow(2).mean(0)
order = var_t.argsort(descending=True)
print(f"top-variance dims of h_t: {[(int(i), round(float(var_t[i]), 1)) for i in order[:5]]} "
      f"| share of total variance in top 1/5/50 dims: {float(var_t[order[:1]].sum() / var_t.sum()):.3f} / "
      f"{float(var_t[order[:5]].sum() / var_t.sum()):.3f} / {float(var_t[order[:50]].sum() / var_t.sum()):.3f}")
resid_id = (ht - hb - (ht - hb).mean(0)).pow(2).mean(0)
print(f"identity+bias FVE, trace-weighted: {1 - float(resid_id.sum() / var_t.sum()):.4f} | "
      f"per-dim median: {float((1 - resid_id / var_t).median()):.4f}")
art = load_alignment(p0 / f"align_L{K}_id_anchored.pt")
pred = predict_aligned_base(hb.float(), art).double()
resid = (ht - pred).pow(2).mean(0)
print(f"id-anchored ridge FVE, trace-weighted: {1 - float(resid.sum() / var_t.sum()):.4f} | "
      f"per-dim median: {float((1 - resid / var_t).median()):.4f} | "
      f"mean cos(pred - mean, h_t - mean): "
      f"{float(torch.nn.functional.cosine_similarity(pred - ht.mean(0), htc, dim=1).mean()):.3f}")
# variance-standardised view: does the picture change once every dim counts equally?
z_t, z_b = htc / var_t.sqrt(), hbc / hbc.pow(2).mean(0).sqrt()
print(f"per-dim correlation between h_t and h_b: median={float((z_t * z_b).mean(0).median()):.3f} "
      f"p10={float((z_t * z_b).mean(0).quantile(0.1)):.3f} p90={float((z_t * z_b).mean(0).quantile(0.9)):.3f}")
ev = torch.linalg.eigvalsh(htc.T @ htc / len(htc)).flip(0)
print(f"h_t PCA: top-1 share={float(ev[0] / ev.sum()):.3f} top-10={float(ev[:10].sum() / ev.sum()):.3f} "
      f"participation ratio={float(ev.sum() ** 2 / ev.pow(2).sum()):.1f}")
d = ht - pred
dc = d - d.mean(0)
evd = torch.linalg.eigvalsh(dc.T @ dc / len(dc)).flip(0)
print(f"Delta PCA: top-1 share={float(evd[0] / evd.sum()):.3f} top-10={float(evd[:10].sum() / evd.sum()):.3f} "
      f"participation ratio={float(evd.sum() ** 2 / evd.pow(2).sum()):.1f}")
print(f"median ||Delta_idanc||={d.norm(dim=1).median():.1f} (tau) vs median ||h_t - mean||={htc.norm(dim=1).median():.1f}")
json.dump({"cos_mean": float(cos.mean()), "cos_centered_mean": float(cosc.mean())},
          open("/home/user/research/sanity_8b.json", "w"))
