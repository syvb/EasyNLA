"""Where does the soft-token readout lose the write? Learning curves of the description → write
probe, per head, for text readouts and two continuous references.

    python scripts/heads_soft_token_diag.py --output-dir <head dataset> --work-dir <dir>

One head per layer, --per-head examples each (by write size; never by attention). Features:

  delta    soft_token "delta" readout text (TF-IDF)        — the warm-start candidate
  greedy   soft_token greedy readout text (TF-IDF)
  hidden   the injected prompt's last-position final hidden state (dense) — everything the
           model's reading of the write kept, before it is squeezed into words
  delta24  the same with 24 tokens
  avg16    16 tokens by the write's log-prob gain averaged over --avg-prompts × --avg-carriers
           (every prompt and carrier is fixed, so it is still a function of the write alone)
  lens     top tokens of the write's direct vocabulary projection W_U (g ⊙ v) (TF-IDF) —
           no prompt at all; a reference, not a warm-start candidate

FVE is 2-fold cross-fitted at each per-head sample size N (train on N/2), so the curve shows
whether more data would lift a readout or whether it has plateaued.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from heads_soft_token_sweep import normalized, pick_examples  # noqa: E402

from nla.datagen.heads import soft_token  # noqa: E402
from nla.datagen.heads.probe import crossfit  # noqa: E402
from nla.datagen.heads.store import atomic_write_table, load_wo, read_meta  # noqa: E402


@torch.no_grad()
def final_states(sd: soft_token.SoftTokenDescriber, rec: soft_token.Recipe, vecs: torch.Tensor,
                 layers: list[int]) -> np.ndarray:
    ids, marks = sd.encode(rec)
    out = np.zeros((len(vecs), sd.model.config.hidden_size), dtype=np.float32)
    for layer in sorted(set(layers)):
        rows = [i for i, lay in enumerate(layers) if lay == layer]
        for s in range(0, len(rows), sd.batch_size):
            _final_batch(sd, rec, vecs, rows[s:s + sd.batch_size], sd.site_layer(rec, layer), ids, marks, out)
    return out


def _final_batch(sd: soft_token.SoftTokenDescriber, rec: soft_token.Recipe, vecs: torch.Tensor, idx: list[int],
                 site: int, ids: list[int], marks: list[int], out: np.ndarray) -> None:
    delta = vecs[idx].to(sd.device, torch.float32) * rec.beta * sd.carrier_norm(rec, site)
    x = torch.tensor([ids] * len(idx), device=sd.device)
    with sd._inject(site, marks, len(ids), delta):
        h = sd.model(x, output_hidden_states=True).hidden_states[-1][:, -1]
    out[idx] = h.float().cpu().numpy()


@torch.no_grad()
def avg_delta_tokens(sd: soft_token.SoftTokenDescriber, recipes: list[soft_token.Recipe], vecs: torch.Tensor,
                     layers: list[int], k: int = 16, pool: int = 128) -> list[str]:
    """Top-k tokens by log-prob gain averaged over recipes, among the pool most likely on average."""
    out = [""] * len(vecs)
    enc = [sd.encode(r) for r in recipes]
    for layer in sorted(set(layers)):
        rows = [i for i, lay in enumerate(layers) if lay == layer]
        site = sd.site_layer(recipes[0], layer)
        clean = []
        for r, (ids, marks) in zip(recipes, enc, strict=True):
            x = torch.tensor([ids], device=sd.device)
            clean.append(sd.model(x).logits[0, -1].float().log_softmax(-1))
        for s in range(0, len(rows), sd.batch_size):
            idx = rows[s:s + sd.batch_size]
            gain = logp_mean = 0
            for r, (ids, marks), c in zip(recipes, enc, clean, strict=True):
                delta = vecs[idx].to(sd.device, torch.float32) * r.beta * sd.carrier_norm(r, site)
                x = torch.tensor([ids] * len(idx), device=sd.device)
                with sd._inject(site, marks, len(ids), delta):
                    lp = sd.model(x).logits[:, -1].float().log_softmax(-1)
                gain = gain + (lp - c) / len(recipes)
                logp_mean = logp_mean + lp / len(recipes)
            cand = logp_mean.topk(pool, dim=-1).indices
            top = cand.gather(1, gain.gather(1, cand).topk(k, dim=-1).indices)
            for i, row in zip(idx, top.tolist(), strict=True):
                out[i] = ", ".join(repr(sd.tokenizer.decode([t])) for t in row)
    return out


@torch.no_grad()
def lens_tokens(sd: soft_token.SoftTokenDescriber, vecs: torch.Tensor, k: int = 8) -> list[str]:
    w_u = sd.model.get_output_embeddings().weight
    g = sd.model.model.norm.weight
    texts = []
    for s in range(0, len(vecs), 256):
        logits = (vecs[s:s + 256].to(w_u.device, w_u.dtype) * g) @ w_u.T
        for row in logits.float().topk(k, dim=-1).indices.tolist():
            texts.append(", ".join(repr(sd.tokenizer.decode([t])) for t in row))
    return texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--per-head", type=int, default=2400)
    p.add_argument("--sizes", default="200,400,800,1600,2400")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--carrier", default=" something")
    p.add_argument("--avg-prompts", default="assoc,continue")
    p.add_argument("--avg-carriers", default=" something, it")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device-map", default="cuda:0")
    args = p.parse_args()

    out_dir, work = Path(args.output_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out_dir)
    rng = np.random.default_rng(args.seed)
    heads = [(layer, int(rng.integers(meta.n_heads))) for layer in meta.layers]
    rows = pick_examples(out_dir, heads, args.per_head, args.seed)
    print(f"{len(rows)} examples over {[f'L{a}H{b}' for a, b in heads]}")
    vecs = normalized(out_dir, rows, 4.0)
    layers = [r["layer"] for r in rows]

    sd = soft_token.SoftTokenDescriber(meta.base_model, args.device_map, "bfloat16", args.batch_size)
    delta_rec = soft_token.Recipe("delta", soft_token.PROMPTS["assoc"], beta=1.0, carrier=args.carrier,
                                  readout="delta")
    greedy_rec = soft_token.Recipe("greedy", soft_token.PROMPTS["assoc"], beta=2.0, carrier=args.carrier)
    cache = work / "texts.parquet"
    texts = pq.read_table(cache).to_pydict() if cache.exists() else {}
    texts.pop("example_idx", None)
    if "delta" not in texts:
        texts["delta"] = sd.describe(delta_rec, vecs, layers)
    if "greedy" not in texts:
        texts["greedy"] = sd.describe(greedy_rec, vecs, layers)
    if "lens" not in texts:
        texts["lens"] = lens_tokens(sd, vecs)
    if "delta24" not in texts:
        texts["delta24"] = sd.describe(soft_token.Recipe("delta24", delta_rec.prompt, beta=1.0, carrier=args.carrier,
                                                         readout="delta", top_k=24), vecs, layers)
    if "avg16" not in texts:
        avg = [soft_token.Recipe(f"{p_}{c}", soft_token.PROMPTS[p_], beta=1.0, carrier=c, readout="delta")
               for p_ in args.avg_prompts.split(",") for c in args.avg_carriers.split(",")]
        texts["avg16"] = avg_delta_tokens(sd, avg, vecs, layers)
    atomic_write_table(pa.table({"example_idx": list(range(len(rows))), **texts}), cache)
    if (work / "hidden.npy").exists():
        hidden = np.load(work / "hidden.npy")
    else:
        hidden = final_states(sd, delta_rec, vecs, layers)
        np.save(work / "hidden.npy", hidden)
    text_names = ["delta", "delta24", "avg16", "greedy", "lens"]

    d = meta.head_dim
    sizes = [int(s) for s in args.sizes.split(",")]
    res: dict[str, dict[str, dict[int, float]]] = {}
    for layer, head in heads:
        idx = np.array([i for i, r in enumerate(rows) if (r["layer"], r["head"]) == (layer, head)])
        u = torch.linalg.svd(load_wo(out_dir, layer)[:, head * d:(head + 1) * d].float(), full_matrices=False).U
        y_all = (vecs[idx] @ u).numpy()
        hd = f"L{layer}H{head}"
        res[hd] = {}
        for n in sizes:
            if n > len(idx):
                continue
            sub = np.random.default_rng([args.seed, n]).choice(len(idx), n, replace=False)
            y = y_all[sub]
            for name in text_names:
                res[hd].setdefault(name, {})[n] = crossfit([texts[name][idx[i]] for i in sub], y, seed=args.seed)[0]
            res[hd].setdefault("hidden", {})[n] = crossfit(hidden[idx[sub]], y, seed=args.seed)[0]
        print(hd, json.dumps({k: {n: round(v, 3) for n, v in c.items()} for k, c in res[hd].items()}), flush=True)
    (work / "curves.json").write_text(json.dumps(res, indent=1))

    print(f"\nmean held-out FVE over {len(heads)} heads, by examples per head (2-fold: trained on half)")
    print(f"{'features':<10}" + "".join(f"{n:>8}" for n in sizes))
    for name in [*text_names, "hidden"]:
        print(f"{name:<10}" + "".join(f"{np.mean([res[h][name][n] for h in res if n in res[h][name]]):>8.3f}"
                                      for n in sizes))


if __name__ == "__main__":
    main()
