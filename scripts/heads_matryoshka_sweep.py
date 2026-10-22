"""Prefix-length sweep for matryoshka head NLAs: how much of the write an AR recovers from the first L
tokens of each AV's explanation.

    python scripts/heads_matryoshka_sweep.py --sft-dir /workspace/sft --base-ckpt Qwen/Qwen3-8B \\
        --av rl400=<iter dir> --av mtry=<iter dir> --ar rl400=<iter dir>/critic --ar mtry=<iter dir>/critic \\
        --out-dir <dir> [--max-rows 2000]

On --max-rows held-out eval rows (a seeded subset of eval.parquet, all heads), each AV writes one T=1
explanation (at most --max-new-tokens; one cut off before its closing tag keeps what it said, as in the
matryoshka trainer); the gold descriptions come along as AV "gold". Every AR then scores every AV's
first L tokens (L in --lengths; 0 = empty text) and the whole explanation ("full"), with the head prefix
the AR was trained on. Direction metrics as in heads_sft_checks.py: both vectors at norm √d, mean
cosine, and FVE against the head's mean normalized AR-training write.

Writes generations.parquet (row-aligned texts + token counts), sweep.json (av → ar → L → cos / fve /
fve_head_avg) and row_cos.npz (per-row cosine for every av|ar|L). --gens <generations.parquet> reuses
earlier generations (only AVs missing from it are generated).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from heads_sft_checks import ar_predict  # noqa: E402

from nla.config import load_nla_config  # noqa: E402
from nla.datagen.heads.export_sft import critic_explanation  # noqa: E402
from nla.schema import resolve_target_scale  # noqa: E402
from nla.train_rl_self_contained import open_explanation  # noqa: E402
from nla.utils import build_prompt_text, register_karvonen_hook  # noqa: E402


def named(items: list[str]) -> dict[str, str]:
    out = {}
    for s in items:
        k, _, v = s.partition("=")
        assert k and v, f"expected name=path, got {s!r}"
        out[k] = v
    return out


@torch.no_grad()
def generate(actor, vref, tok, cfg, prompts, vecs, device, batch_size, max_new_tokens) -> list[str]:
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        texts = [build_prompt_text(p, cfg.injection_char, tok) for p in prompts[s:s + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        vref[0] = torch.tensor(vecs[s:s + batch_size], dtype=torch.float32, device=device)
        try:   # the RL rollout sampler: T=1, no top-p / top-k
            gen = actor.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                                 top_p=1.0, top_k=0, repetition_penalty=1.0,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        finally:
            vref[0] = None
        for g in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            e = open_explanation(g)
            out.append("" if e is None else e)
        print(f"  generated {min(s + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--av", action="append", default=[], help="name=AV adapter dir (repeatable)")
    p.add_argument("--ar", action="append", default=[], help="name=AR dir with ar_meta.json (repeatable)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gens", default=None, help="reuse this generations.parquet (same rows)")
    p.add_argument("--lengths", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 120])
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--inject-preserve-magnitude", action=argparse.BooleanOptionalAction, default=False,
                   help="must match how the AV was trained (the standard recipe: off)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from peft import PeftModel
    from show_nla_generations import load_ar_critic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft, out = Path(a.sft_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_ckpt)
    cfg = load_nla_config(str(sft / "av_sft.parquet"), tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)

    ev_all = pq.read_table(sft / "eval.parquet").to_pylist()
    sel = np.sort(np.random.default_rng(a.seed).choice(len(ev_all), size=min(a.max_rows, len(ev_all)),
                                                       replace=False))
    ev = [ev_all[i] for i in sel]
    heads = [(r["activation_layer"], r["head"]) for r in ev]
    v = np.stack([np.asarray(r["activation_vector"], dtype=np.float32) for r in ev])
    by_head: dict[tuple, list[int]] = defaultdict(list)
    for i, h in enumerate(heads):
        by_head[h].append(i)

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * mse_scale_f
    tr = pq.read_table(sft / "ar_sft.parquet", columns=["activation_vector", "activation_layer", "head"])
    tr_u = unit(np.asarray(tr.column("activation_vector").combine_chunks().flatten(), dtype=np.float32)
                .reshape(tr.num_rows, -1))
    tr_heads = list(zip(tr.column("activation_layer").to_pylist(), tr.column("head").to_pylist(), strict=True))
    head_mean = {h: tr_u[[i for i, x in enumerate(tr_heads) if x == h]].mean(0) for h in set(tr_heads)}
    base = np.stack([head_mean[h] for h in heads]).astype(np.float32)
    g_u = unit(v)
    base_err = ((base - g_u) ** 2).sum(-1)

    texts: dict[str, list[str]] = {"gold": [r["description"] for r in ev]}
    if a.gens:
        g = pq.read_table(a.gens).to_pydict()
        assert list(g["eval_row"]) == sel.tolist(), "--gens was made on other rows"
        texts.update({k[4:]: g[k] for k in g if k.startswith("av__")})
    todo = {k: d for k, d in named(a.av).items() if k not in texts}
    if todo:
        model = AutoModelForCausalLM.from_pretrained(a.base_ckpt, dtype=torch.bfloat16).to(a.device)
        actor = None
        vref: list = [None]
        for name, d in todo.items():
            if actor is None:
                actor = PeftModel.from_pretrained(model, d, adapter_name=name).eval()
                register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                       cfg.injection_right_neighbor_id, layer_idx=1,
                                       preserve_magnitude=a.inject_preserve_magnitude)
            else:
                actor.load_adapter(d, adapter_name=name)
            actor.set_adapter(name)
            print(f"[gen] {name} <- {d}", flush=True)
            torch.manual_seed(a.seed)
            texts[name] = generate(actor, vref, tok, cfg, [r["prompt"] for r in ev], v, a.device,
                                   a.batch_size, a.max_new_tokens)
        del actor, model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    ids = {k: [tok.encode(t, add_special_tokens=False) for t in ts] for k, ts in texts.items()}
    pq.write_table(pa.table({
        "eval_row": sel, "layer": [la for la, _ in heads], "head": [hd for _, hd in heads],
        **{f"av__{k}": ts for k, ts in texts.items()},
        **{f"ntok__{k}": [len(x) for x in ids[k]] for k in texts},
    }), out / "generations.parquet")
    for k in texts:
        n = np.array([len(x) for x in ids[k]])
        print(f"[len] {k}: tokens mean {n.mean():.0f}, p10/50/90 {np.percentile(n, [10, 50, 90]).round().tolist()}")

    Ls = [*a.lengths, "full"]
    res: dict = {"n_rows": len(ev), "lengths": Ls, "sweep": {}}
    row_cos = {}
    for ar_name, ar_dir in named(a.ar).items():
        critic = load_ar_critic(ar_dir, a.base_ckpt, a.device)
        for av_name, toks in ids.items():
            cell = res["sweep"].setdefault(av_name, {}).setdefault(ar_name, {})
            for L in Ls:
                desc = texts[av_name] if L == "full" else [tok.decode(x[:L]).strip() for x in toks]
                pr = unit(ar_predict(critic, tok, [critic_explanation(la, hd, d) for (la, hd), d
                                                    in zip(heads, desc, strict=True)],
                                     mse_scale_f, a.device, a.batch_size))
                cos = (pr * g_u).sum(-1) / mse_scale_f ** 2
                err = ((pr - g_u) ** 2).sum(-1)
                cell[str(L)] = {
                    "cos": float(cos.mean()),
                    "fve": 1.0 - float(err.sum() / base_err.sum()),
                    "fve_head_avg": float(np.mean([1.0 - err[ix].sum() / base_err[ix].sum()
                                                   for ix in by_head.values()])),
                }
                row_cos[f"{av_name}|{ar_name}|{L}"] = cos.astype(np.float32)
            print(f"[{ar_name} reads {av_name}] cos " + " ".join(
                f"{L}:{cell[str(L)]['cos']:.3f}" for L in Ls), flush=True)
        del critic
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    (out / "sweep.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "row_cos.npz", head=np.array([hd for _, hd in heads]), **row_cos)

    print(f"\nmean cosine(AR prediction, write) by explanation prefix length ({len(ev)} held-out rows, T=1)")
    print(f"{'AV':<10}{'AR':<10}" + "".join(f"{L!s:>8}" for L in Ls))
    for av_name, d in res["sweep"].items():
        for ar_name, cell in d.items():
            print(f"{av_name:<10}{ar_name:<10}" + "".join(f"{cell[str(L)]['cos']:>8.3f}" for L in Ls))
    print(f"wrote {out / 'sweep.json'}")
    assert all(math.isfinite(c["cos"]) for d in res["sweep"].values() for cell in d.values() for c in cell.values())


if __name__ == "__main__":
    main()
