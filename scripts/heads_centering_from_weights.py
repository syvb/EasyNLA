"""How much of the head-write centering can be had from the weights? (CPU only; downloads Qwen3-8B shards 1–3, ~12 GB;
peak RAM about 6–8 GB)

    python scripts/heads_centering_from_weights.py --out <json>

The centering subtracts m_h = W_O,h z̄_h, the head's mean write over 100k web-text records (l24/meta). Candidates:

  first_token   W_O,h W_V a_0: the write at a document's first token, where the head can only attend to itself. Needs the
                weights and one token, no corpus (one estimate per document; their average is reported too)
  value_mean    W_O,h W_V mean_t(a_t): the attention-free part of the mean, with a_t = RMSNorm(x_t) the normalized input to
                layer 24. One 4096-d vector for the whole layer, from the given tokens (web or chat, positions ≥ 50)

a_t comes from running layers 0–23 on the 4 doc-viewer web documents and the 4 chats (fp32, one layer in memory at a
time), checked against the stored head writes. Each candidate is scored against the true mean on the training sites
(l24/sft_data/eval.parquet):
  share         1 − |m − m̂|² / |m|²: how much of the mean's size it removes
  free_cos      the head-only constant's mean cos after centering with m̂ (raw writes: 0.26, exact centering: 0.09), and
                the share of that drop it recovers
  rotation      median cos between each write centered with m̂ and centered exactly
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import load_file

REPO = "syvb/head-nla-qwen3-8b"
BASE = "Qwen/Qwen3-8B"
WEB = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression", "04_law_gainsborough_fraud"]
CHAT = ["01_ipv4_code", "02_lighthouse_story", "03_train_problem", "04_ada_lovelace"]
LAYER, H, KV, D = 24, 32, 8, 128
GROUPS = {"KV6 (H24-27)": [24, 25, 26, 27], "parking heads (H1, H6, H20, H23)": [1, 6, 20, 23],
          "other 24 heads": [h for h in range(32) if h not in (1, 6, 20, 23, 24, 25, 26, 27)]}


def get(f: str) -> str:
    return hf_hub_download(REPO, f, repo_type="dataset")


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """x [T, n, D], rotate_half convention (HF Qwen3)."""
    t = x.shape[0]
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    ang = torch.outer(torch.arange(t, dtype=torch.float64), inv)
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().float()[:, None], emb.sin().float()[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Weights:
    def __init__(self):
        idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json")))["weight_map"]
        self.idx, self.handles = idx, {}
        self.cfg = json.load(open(hf_hub_download(BASE, "config.json")))

    def raw(self, name: str) -> torch.Tensor:
        f = self.idx[name]
        if f not in self.handles:
            self.handles[f] = safe_open(hf_hub_download(BASE, f), framework="pt")
        return self.handles[f].get_tensor(name)

    def __call__(self, name: str) -> torch.Tensor:
        return self.raw(name).float()

    def layer(self, li: int) -> dict:
        p = f"model.layers.{li}."
        return {k[len(p):]: self(k) for k in self.idx if k.startswith(p)}


def attention(L: dict, x: torch.Tensor, eps: float, theta: float):
    """One Qwen3 self-attention block on one document → (normalized input a, values, head outputs z, o_proj output)."""
    a = rms_norm(x, L["input_layernorm.weight"], eps)
    t = x.shape[0]
    q = (a @ L["self_attn.q_proj.weight"].T).view(t, H, D)
    k = (a @ L["self_attn.k_proj.weight"].T).view(t, KV, D)
    v = (a @ L["self_attn.v_proj.weight"].T).view(t, KV, D)
    q = rope(rms_norm(q, L["self_attn.q_norm.weight"], eps), theta)
    k = rope(rms_norm(k, L["self_attn.k_norm.weight"], eps), theta)
    kq, vq = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)      # query head h reads KV head h // 4
    s = torch.einsum("thd,shd->hts", q, kq) / D ** 0.5
    s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    z = torch.einsum("hts,shd->thd", s.softmax(-1), vq)                            # [T, H, D]
    return a, v, z, z.reshape(t, H * D) @ L["self_attn.o_proj.weight"].T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    W = Weights()
    eps, theta = W.cfg["rms_norm_eps"], W.cfg["rope_theta"]
    assert W.cfg["head_dim"] == D and W.cfg["num_attention_heads"] == H and W.cfg["num_key_value_heads"] == KV
    assert not W.cfg.get("attention_bias") and not W.cfg.get("rope_scaling"), "the block below assumes neither"
    docs = [("web", n, f"l24/std_rl/doc_viewer/{n}") for n in WEB] + [("chat", n, f"l24/chat_viewer/{n}") for n in CHAT]
    ids = {n: torch.tensor(np.load(get(f"{b}/token_ids.npy")).astype(np.int64)) for _, n, b in docs}
    emb = W.raw("model.embed_tokens.weight")                                           # bf16: the fp32 copy is 2.5 GB
    xs = {n: emb[t].float() for n, t in ids.items()}
    del emb
    for li in range(LAYER):
        L = W.layer(li)
        for n in xs:
            x = xs[n]
            x = x + attention(L, x, eps, theta)[3]
            b = rms_norm(x, L["post_attention_layernorm.weight"], eps)
            g, u = b @ L["mlp.gate_proj.weight"].T, b @ L["mlp.up_proj.weight"].T
            xs[n] = x + (torch.nn.functional.silu(g) * u) @ L["mlp.down_proj.weight"].T
        del L
        print(f"layer {li} done", flush=True)
    L24 = W.layer(LAYER)

    st = load_file(get("l24/meta/head_stats.safetensors"))
    m = torch.from_numpy(st["mean_write"][0]).double()                               # [H, d_model]
    rms = torch.from_numpy(st["dev_rms"][0]).double()
    ql = list(st["quantile_levels"])
    cap = np.minimum(st["dev_quantiles"][0][:, ql.index(min(ql, key=lambda q_: abs(q_ - 0.999)))] / st["dev_rms"][0], 4.0)
    Wo = L24["self_attn.o_proj.weight"].double().view(-1, H, D)                       # [d_model, H, D]
    Wv = L24["self_attn.v_proj.weight"].double().view(KV, D, -1)                       # [KV, D, d_model]

    def write(zh: torch.Tensor) -> torch.Tensor:                                      # [.., H, D] → [.., H, d_model]
        return torch.einsum("mhd,...hd->...hm", Wo, zh)

    # layer 24 on every document: check the head writes against the stored ones, collect a_t and first-token values
    check, A, first = [], {"web": [], "chat": []}, []
    for kind, n, b in docs:
        a, v, z, _ = attention(L24, xs[n], eps, theta)
        w = write(z.double())                                                          # [T, H, d]
        sv = torch.from_numpy(np.load(get(f"{b}/writes.npz"))["v"].astype(np.float64))
        ok = sv.norm(dim=-1) < torch.from_numpy(cap)[None] - 1e-3
        ref = rms[None, :, None] * sv + m[None]
        c = torch.nn.functional.cosine_similarity(w, ref, dim=-1)[ok]
        check.append({"doc": n, "median_cos": round(float(c.median()), 5), "min_cos": round(float(c.min()), 4)})
        A[kind].append(a[50:].double())
        first.append(write(v[0].double().repeat_interleave(H // KV, 0)[None])[0])     # pos 0: z = its own value
    print("check vs stored writes:", check, flush=True)

    t = pq.read_table(get("l24/sft_data/eval.parquet"), columns=["activation_vector", "head"]).to_pydict()
    E = torch.tensor(t["activation_vector"], dtype=torch.float64)
    eh = torch.tensor(t["head"])
    raw = rms[eh, None] * E + m[eh]                                                    # raw writes of the training sites
    keep = E.norm(dim=-1) < torch.from_numpy(cap)[eh] - 1e-3

    def unit(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def free_cos(c: torch.Tensor) -> np.ndarray:                                       # c: centered writes [N, d]
        return np.array([float(unit(c[(eh == h) & keep]).mean(0).norm()) for h in range(H)])

    fc_raw, fc_exact = free_cos(raw), free_cos(raw - m[eh])

    def score(mhat: torch.Tensor) -> dict:
        share = (1 - (m - mhat).pow(2).sum(-1) / m.pow(2).sum(-1)).numpy()
        cosm = torch.nn.functional.cosine_similarity(mhat, m, dim=-1).numpy()
        fc = free_cos(raw - mhat[eh])
        rec = (fc_raw - fc) / (fc_raw - fc_exact)
        rot = np.array([float(torch.nn.functional.cosine_similarity(raw[(eh == h) & keep] - mhat[h],
                                                                     raw[(eh == h) & keep] - m[h], dim=-1).median())
                        for h in range(H)])
        out = {}
        for g, hs in GROUPS.items():
            out[g] = {"share_of_mean_removed": round(float(np.median(share[hs])), 2), "cos_to_mean": round(float(np.median(cosm[hs])), 2),
                      "free_cos_after": round(float(np.median(fc[hs])), 3), "recovered_share_of_free_cos_drop": round(float(np.median(rec[hs])), 2),
                      "rotation_vs_exact": round(float(np.median(rot[hs])), 3)}
        out["all heads"] = {"share_of_mean_removed": round(float(np.median(share)), 2), "cos_to_mean": round(float(np.median(cosm)), 2),
                            "free_cos_after": round(float(np.median(fc)), 3), "recovered_share_of_free_cos_drop": round(float(np.median(rec)), 2),
                            "rotation_vs_exact": round(float(np.median(rot)), 3),
                            "per_head_share": [round(float(x), 2) for x in share], "per_head_recovered": [round(float(x), 2) for x in rec]}
        return out

    res = {"check_vs_stored_writes": check,
           "baseline": {"free_cos_raw_median": round(float(np.median(fc_raw)), 3), "free_cos_exact_median": round(float(np.median(fc_exact)), 3)}}
    res["first_token_one_doc"] = {}
    for (k, n, _), f in zip(docs, first):
        sc = score(f)
        res["first_token_one_doc"][f"{k}:{n}"] = sc["all heads"] | {"by_group": {g: x for g, x in sc.items() if g != "all heads"}}
    res["first_token_avg_8_docs"] = score(torch.stack(first).mean(0))
    for kind in ("web", "chat"):
        mu = torch.cat(A[kind]).mean(0)                                                # mean normalized input, positions >= 50
        res[f"value_mean_{kind}"] = score(write(torch.einsum("kdm,m->kd", Wv, mu).repeat_interleave(H // KV, 0)[None])[0])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    show = {k: (v if not isinstance(v, dict) or "all heads" not in v else {g: {kk: vv for kk, vv in s.items() if not kk.startswith("per_head")}
                                                                            for g, s in v.items()}) for k, v in res.items() if not k.startswith("first_token_one")}
    print(json.dumps(show, indent=1))
    print("first token, single documents (all heads):", {k: (v["share_of_mean_removed"], v["recovered_share_of_free_cos_drop"])
                                                         for k, v in res["first_token_one_doc"].items()})


if __name__ == "__main__":
    main()
