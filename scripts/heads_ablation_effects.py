"""Counterfactual relevance of every attention head of one layer at every token (GPU).

    python scripts/heads_ablation_effects.py --data <dir with <doc>/token_ids.npy (+ check_writes.npz)> --out <dir>

For each document, token t and head h, head h's output at token t ONLY is replaced, and the model is run on from
that layer; everything else (other heads, other tokens) stays clean. Two replacements:
  mean   the head's mean attention output z (mean over all tokens t >= 1 of all documents given)
  zero   zero (the head removed)
Effects, each measured against an unablated control row in the same batch (so batching noise cancels):
  resid    ||Δ final-layer residual at t|| / ||residual at t||     (pre-final-norm)
  kl       KL(clean || ablated) of the next-token distribution at t (nats)
  dlogp    log p_clean(actual next token) − log p_ablated(...) at t: > 0 when the head's output helps predict it
  later    Σ over later tokens t' > t of KL at t' (the write at t is read by later tokens through attention)
Per document <out>/<doc>/effects.parquet: one row per (pos, head) with <metric>_<mode> columns, plus the clean
model's entropy and log p(next) at each token. Only layers >= --layer are re-run (the residual entering the layer
is cached); a check reproduces the clean final residual through that shortcut before anything is measured.
Layers >= --layer, the final norm and the unembedding then run in float32 (--fp32, default): in bf16 any change at
token t, however small, scrambles the rounding of everything that attends to it, which put a head-independent floor
of ~7e-4 nats per token under the KLs (and made `later` track only the number of later tokens).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--chunk-rows", type=int, default=8, help="batch rows per logits chunk")
    p.add_argument("--max-positions", type=int, default=0, help="debug: only the first N tokens of each doc")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-fp32", dest="fp32", action="store_false", help="keep the re-run layers in bf16")
    a = p.parse_args()
    from transformers import AutoModelForCausalLM

    torch.set_grad_enabled(False)          # inference only (the write check below touches parameters)
    dev, L = a.device, a.layer
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = sorted(d for d in data.iterdir() if (d / "token_ids.npy").exists())
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    cfg = model.config
    H, hd = cfg.num_attention_heads, cfg.head_dim
    layers = model.model.layers
    o_proj, norm = layers[L].self_attn.o_proj, model.model.norm
    grab: dict = {}
    hooks = [layers[L].register_forward_pre_hook(lambda m, args, kw: grab.__setitem__("resid_in", (args[0] if args else kw["hidden_states"]).detach()), with_kwargs=True),
             o_proj.register_forward_pre_hook(lambda m, args: grab.__setitem__("z", args[0].detach())),
             norm.register_forward_pre_hook(lambda m, args: grab.__setitem__("final", args[0].detach()))]

    # ---- clean pass per doc: cache the residual entering layer L, z, final residual and next-token log-probs
    clean = {}
    with torch.no_grad():
        for d in docs:
            ids = torch.tensor(np.load(d / "token_ids.npy"), device=dev)[None]
            if a.max_positions:
                ids = ids[:, : a.max_positions]
            lp = model(input_ids=ids, use_cache=False).logits[0].float().log_softmax(-1)
            clean[d.name] = dict(ids=ids[0], resid_in=grab["resid_in"][0].clone(), z=grab["z"][0].clone(),
                                 final=grab["final"][0].float().clone(), lp=lp)
            print(f"{d.name}: {ids.shape[1]} tokens, clean pass done", flush=True)
    for h_ in hooks[:2]:
        h_.remove()
    zbar = torch.cat([c["z"][1:] for c in clean.values()]).float().mean(0).to(torch.bfloat16)   # [H*hd]

    # write check against the viewer's stored writes v = cap(W_h (z − z̄) / rms_h): the head's average write z̄ is
    # not available here, so compare differences between tokens, where it cancels (same head indexing, same vectors)
    W_o = o_proj.weight.float()
    for d in docs:
        f = d / "check_writes.npz"
        if not f.exists():
            continue
        cw = np.load(f)
        keep = [i for i, t in enumerate(cw["pos"]) if int(t) < clean[d.name]["z"].shape[0]]
        if len(keep) < 2:
            continue
        z = clean[d.name]["z"][[int(cw["pos"][i]) for i in keep]].float()                      # [n, H*hd]
        w = torch.stack([torch.stack([W_o[:, h * hd:(h + 1) * hd] @ z[i, h * hd:(h + 1) * hd] for h in range(H)])
                         for i in range(len(keep))])                                           # [n, H, d]
        v = torch.tensor(cw["v"][keep], device=dev).float()
        cs = [torch.nn.functional.cosine_similarity(w[i] - w[j], v[i] - v[j], dim=1).cpu().numpy()
              for i in range(len(keep)) for j in range(i + 1, len(keep))]
        print(f"{d.name}: write check (token differences vs viewer writes): median cos {np.median(cs):.5f}, "
              f"min {np.min(cs):.4f} over {len(cs)} pairs × {H} heads", flush=True)
        assert np.median(cs) > 0.99, "head indexing / write definition differs from the viewer's"

    # ---- shortcut: run only layers >= L from the cached residual
    full_layers = layers
    model.model.layers = torch.nn.ModuleList(list(full_layers)[L:])

    def run_from_L(resid_in: torch.Tensor) -> torch.Tensor:
        T = resid_in.shape[1]
        pos = torch.arange(T, device=dev)[None].expand(resid_in.shape[0], T)
        return model.model(inputs_embeds=resid_in, position_ids=pos, use_cache=False).last_hidden_state

    # bf16 kernels need not match bit-for-bit between the full and the shortcut pass; effects are measured against
    # an in-batch control row on the shortcut path, so this only has to show the shortcut is the same function.
    with torch.no_grad():
        for d in docs:
            c = clean[d.name]
            hid = run_from_L(c["resid_in"][None])
            rel = ((grab["final"][0].float() - c["final"]).norm(dim=-1) / c["final"].norm(dim=-1))
            lp_s = model.lm_head(hid[0]).float().log_softmax(-1)
            kl = (c["lp"].exp() * (c["lp"] - lp_s)).sum(-1)
            print(f"shortcut check {d.name}: final-residual relative error median {rel.median().item():.1e}, max "
                  f"{rel.max().item():.1e}; KL(full ‖ shortcut) median {kl.median().item():.1e}, max {kl.max().item():.1e}",
                  flush=True)
            assert rel.median().item() < 5e-3 and rel.max().item() < 5e-2, "the shortcut is not the same function"
    if a.fp32:     # the re-run part in float32 (true fp32 matmuls: TF32 stays off)
        torch.backends.cuda.matmul.allow_tf32 = False
        for m_ in (*model.model.layers, model.model.norm, model.lm_head):
            m_.float()
        model.model.rotary_emb.float()
        zbar = zbar.float()
        for c in clean.values():
            c["resid_in"] = c["resid_in"].float()
        with torch.no_grad():
            c = clean[docs[0].name]
            lp_s = model.lm_head(run_from_L(c["resid_in"][None])[0]).float().log_softmax(-1)
            kl = (c["lp"].exp() * (c["lp"] - lp_s)).sum(-1)
            print(f"fp32 re-run vs the bf16 model on {docs[0].name}: KL median {kl.median().item():.1e}, "
                  f"max {kl.max().item():.1e} (precision difference, not an error)", flush=True)

    rows_mode = [("mean", h) for h in range(H)] + [("zero", h) for h in range(H)]
    B = len(rows_mode) + 1                                   # + control row (no ablation)
    state = {"t": 0}

    def ablate(m, args):
        x = args[0].clone()
        t = state["t"]
        for b, (mode, h) in enumerate(rows_mode):
            x[b, t, h * hd:(h + 1) * hd] = zbar[h * hd:(h + 1) * hd] if mode == "mean" else 0
        return (x,)

    hk = o_proj.register_forward_pre_hook(ablate)
    meta = {"layer": L, "n_heads": H, "base": a.base, "modes": ["mean", "zero"], "fp32_rerun": a.fp32, "docs": {}}
    with torch.no_grad():
        for d in docs:
            c = clean[d.name]
            ids, T = c["ids"], c["ids"].shape[0]
            res = {k: np.full((T, H, 2), np.nan, np.float32) for k in ("resid", "kl", "dlogp", "later")}
            noise = []
            t0 = time.time()
            for t in range(T):
                state["t"] = t
                hid = run_from_L(c["resid_in"][None].expand(B, -1, -1).contiguous())    # [B, T, d] normed
                fin = grab["final"].float()                                             # [B, T, d] pre-norm
                ctrl_f = fin[-1, t]
                rel = ((fin[:-1, t] - ctrl_f).norm(dim=-1) / ctrl_f.norm()).cpu().numpy()
                if t > 0 and t % 50 == 0:   # tokens before t must be untouched by an ablation at t
                    noise.append(((fin[:-1, :t] - fin[-1:, :t]).abs().max()).item())
                lp_c = model.lm_head(hid[-1, t:]).float().log_softmax(-1)              # [T-t, V]
                kl = torch.empty(B - 1, T - t, device=dev)
                for s in range(0, B - 1, a.chunk_rows):
                    lp_a = model.lm_head(hid[s:s + a.chunk_rows, t:]).float().log_softmax(-1)
                    kl[s:s + a.chunk_rows] = (lp_c.exp()[None] * (lp_c[None] - lp_a)).sum(-1)
                    if t + 1 < T:
                        nxt = ids[t + 1]
                        dl = (lp_c[0, nxt] - lp_a[:, 0, nxt]).cpu().numpy()
                        for j, b in enumerate(range(s, min(s + a.chunk_rows, B - 1))):
                            mode, h = rows_mode[b]
                            res["dlogp"][t, h, 0 if mode == "mean" else 1] = dl[j]
                kl = kl.clamp_min(0).cpu().numpy()
                for b, (mode, h) in enumerate(rows_mode):
                    m = 0 if mode == "mean" else 1
                    res["resid"][t, h, m] = rel[b]
                    res["kl"][t, h, m] = kl[b, 0]
                    res["later"][t, h, m] = kl[b, 1:].sum() if T - t > 1 else 0.0
                if t % 50 == 0:
                    print(f"  {d.name} t={t}/{T} ({time.time() - t0:.0f}s)", flush=True)
            ent = -(c["lp"].exp() * c["lp"]).sum(-1).cpu().numpy()
            lpn = np.r_[c["lp"][torch.arange(T - 1, device=dev), ids[1:]].cpu().numpy(), np.nan]
            df = pd.DataFrame({"pos": np.repeat(np.arange(T), H), "head": np.tile(np.arange(H), T),
                               "clean_entropy": np.repeat(ent, H), "clean_logp_next": np.repeat(lpn, H)})
            for k, v in res.items():
                for m, mode in enumerate(("mean", "zero")):
                    df[f"{k}_{mode}"] = v[:, :, m].reshape(-1)
            (out / d.name).mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / d.name / "effects.parquet")
            meta["docs"][d.name] = {"tokens": T, "seconds": round(time.time() - t0), "max_noise_before_t": max(noise, default=0.0)}
            print(f"{d.name}: done in {time.time() - t0:.0f}s; max |Δ| before t (should be ~0): {max(noise, default=0.0):.2e}", flush=True)
            for k in ("kl", "resid", "dlogp", "later"):
                print(f"   {k}_mean: median {np.nanmedian(df[f'{k}_mean']):.4g}  p99 {np.nanpercentile(df[f'{k}_mean'], 99):.4g}", flush=True)
    hk.remove()
    (out / "effects_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
