"""Pre-RL checks for a head-contribution NLA warm-start (the recipe's go/no-go gates).

    python scripts/heads_sft_checks.py --sft-dir <export dir> --base-ckpt Qwen/Qwen3-8B \\
        --av-ckpt <av save-dir>/iter_XXXXXXX --ar-ckpt <ar save-dir>/iter_XXXXXXX --out-dir <dir>

On the held-out eval split (documents neither model trained on), FVE of the AR's prediction of
the √d-scaled normalized write. The baseline is each head's mean target over the AR's TRAINING
split — what an AR that ignores the description can learn from the head name — so it scores
exactly 0 and never peeks at eval labels:

  ar/gold            AR given the warm-start (gold) descriptions
  ar/empty           AR given an empty description (out of distribution, for reference)
  ar/gold_shuf       gold descriptions permuted within each head
  ar/head_mean       predicting that per-head training mean — the 0 line, no model at all
  av/true            AR given the AV's descriptions of the TRUE write        } check 2:
  av/shuf            AR given the AV's descriptions of ANOTHER example's     } true ≫ shuf
                     write (same head) — scored against the true write

Check 1 passes when ar/gold is clearly above 0, ar/empty and ar/gold_shuf; check 2 when
av/true is clearly above av/shuf (checks.json has 95% bootstrap CIs over rows and over heads). An
AR-free companion: a per-head ridge from text alone to the write, for the gold, AV-true and AV-shuf
texts. Writes checks.json, row_errors.npz, generations.parquet (every eval row's gold / AV texts,
row-aligned with row_errors.npz) and examples.md to --out-dir.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nla.config import load_nla_config  # noqa: E402
from nla.datagen.heads.export_sft import CRITIC_TEMPLATE, critic_explanation  # noqa: E402
from nla.schema import extract_explanation, resolve_target_scale  # noqa: E402
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook  # noqa: E402


def fve(pred: np.ndarray, gold: np.ndarray, base: np.ndarray) -> float:
    """1 − SSE(pred) / SSE(base); base = per-row baseline prediction (the head's training mean).
    Pooled over rows, so large-norm rows weigh most — see fve_heads for the per-head average."""
    return 1.0 - float(((pred - gold) ** 2).sum()) / max(float(((base - gold) ** 2).sum()), 1e-12)


def fve_heads(pred: np.ndarray, gold: np.ndarray, base: np.ndarray, by_head: dict) -> float:
    """FVE computed within each head, then averaged over heads (every head counts equally)."""
    return float(np.mean([fve(pred[ix], gold[ix], base[ix]) for ix in by_head.values()]))


@torch.no_grad()
def ar_predict(critic, tok, texts: list[str], mse_scale_f, device, batch_size: int) -> np.ndarray:
    out = []
    for s in range(0, len(texts), batch_size):
        ids_list = [tok.encode(CRITIC_TEMPLATE.format(explanation=t), add_special_tokens=False)
                    for t in texts[s : s + batch_size]]
        T = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), T), tok.eos_token_id, dtype=torch.long, device=device)
        attn = torch.zeros_like(ids)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x, device=device)
            attn[i, : len(x)] = 1
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device != "cpu"):
            out.append(critic_predict(critic, ids, attn, mse_scale_f).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def av_generate(actor, vref, tok, cfg, rows: list[dict], vecs: np.ndarray, device, batch_size: int,
                max_new_tokens: int) -> list[str]:
    tok.padding_side = "left"   # batched generation; the injection hook finds the marker by id + neighbors
    out: list[str] = []
    n_fallback = 0
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        texts = [build_prompt_text(r["prompt"], cfg.injection_char, tok) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        vref[0] = torch.tensor(vecs[s : s + batch_size], dtype=torch.float32, device=device)
        try:
            gen = actor.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        finally:
            vref[0] = None
        for g in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            expl = extract_explanation(g)
            if expl is None:   # cut off before </explanation>: keep what was said, minus the tag
                n_fallback += 1
                expl = g.replace("<explanation>", "").strip()[:300]
            out.append(expl)
    print(f"  AV generations without a closing tag: {n_fallback}/{len(rows)}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--av-ckpt", required=True, help="AV LoRA adapter dir (train_sft --mode av iter_*)")
    p.add_argument("--ar-ckpt", required=True, help="AR iter_* dir (ar_meta.json + ar_lora_value_head)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inject-preserve-magnitude", action=argparse.BooleanOptionalAction, default=True,
                   help="must match how the AV was trained")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from peft import PeftModel
    from show_nla_generations import load_ar_critic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft, out = Path(args.sft_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(str(sft / "av_sft.parquet"), tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    root_d = math.sqrt(cfg.d_model)

    ev = pq.read_table(sft / "eval.parquet").to_pylist()
    if args.max_rows:
        ev = ev[: args.max_rows]
    heads = [(r["activation_layer"], r["head"]) for r in ev]
    v = np.stack([np.asarray(r["activation_vector"], dtype=np.float32) for r in ev])
    gold = root_d * v
    by_head: dict[tuple, list[int]] = defaultdict(list)
    for i, h in enumerate(heads):
        by_head[h].append(i)
    rng = np.random.default_rng(args.seed)
    perm = np.arange(len(ev))
    for idxs in by_head.values():   # a derangement within each head: every row gets ANOTHER example
        if len(idxs) > 1:
            rot = rng.permutation(idxs)
            perm[rot] = np.roll(rot, 1)
    print(f"{len(ev)} eval examples over {len(by_head)} heads")
    ar_train = pq.read_table(sft / "ar_sft.parquet", columns=["activation_vector", "activation_layer", "head"])
    tr_v = np.asarray(ar_train.column("activation_vector").combine_chunks().flatten()).reshape(ar_train.num_rows, -1)
    tr_heads = list(zip(ar_train.column("activation_layer").to_pylist(), ar_train.column("head").to_pylist(),
                        strict=True))
    head_mean = {h: tr_v[[i for i, x in enumerate(tr_heads) if x == h]].mean(0) for h in set(tr_heads)}
    missing = sorted(set(heads) - set(head_mean))
    assert not missing, f"eval heads {missing[:3]} have no AR training rows"
    baseline = np.stack([head_mean[h] for h in heads]).astype(np.float32)

    def ar_texts(descs: list[str]) -> list[str]:
        return [critic_explanation(lay, hd, d) for (lay, hd), d in zip(heads, descs, strict=True)]

    # --- AV generations (true vs another example's write) ---
    base = AutoModelForCausalLM.from_pretrained(args.base_ckpt, dtype=torch.bfloat16).to(args.device)
    actor = PeftModel.from_pretrained(base, args.av_ckpt).eval()
    vref: list = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1,
                           preserve_magnitude=args.inject_preserve_magnitude)
    av_true = av_generate(actor, vref, tok, cfg, ev, v, args.device, args.batch_size, args.max_new_tokens)
    av_shuf = av_generate(actor, vref, tok, cfg, ev, v[perm], args.device, args.batch_size, args.max_new_tokens)
    del actor, base
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- AR scoring ---
    critic = load_ar_critic(args.ar_ckpt, args.base_ckpt, args.device)
    gold_desc = [r["description"] for r in ev]
    conds = {
        "ar/gold": ar_texts(gold_desc),
        "ar/empty": ar_texts([""] * len(ev)),
        "ar/gold_shuf": ar_texts([gold_desc[j] for j in perm]),
        "av/true": ar_texts(av_true),
        "av/shuf": ar_texts(av_shuf),
    }
    results: dict = {"n_eval": len(ev), "n_heads": len(by_head), "fve": {}, "fve_head_avg": {}, "fve_by_layer": {}}
    preds = {}
    for name, texts in conds.items():
        preds[name] = ar_predict(critic, tok, texts, mse_scale_f, args.device, args.batch_size)
    preds["ar/head_mean"] = baseline
    layers = sorted({lay for lay, _ in heads})

    def score(gold_m: np.ndarray, preds_m: dict, base_m: np.ndarray, seed_off: int) -> tuple[dict, dict, np.ndarray]:
        """FVE per condition (pooled, per-head average, per layer), checks 1 and 2 with paired bootstrap
        CIs over rows and over heads, and the per-row squared errors, all in one metric space."""
        res: dict = {"fve": {}, "fve_head_avg": {}, "fve_by_layer": {}}
        for name, pr in preds_m.items():
            res["fve"][name] = fve(pr, gold_m, base_m)
            res["fve_head_avg"][name] = fve_heads(pr, gold_m, base_m, by_head)
            res["fve_by_layer"][name] = {}
            for lay in layers:
                idx = [i for i, (la, _) in enumerate(heads) if la == lay]
                res["fve_by_layer"][name][f"L{lay}"] = fve(pr[idx], gold_m[idx], base_m[idx])
        f_, h_ = res["fve"], res["fve_head_avg"]
        res["check1_ar_beats_empty"] = f_["ar/gold"] - max(0.0, f_["ar/empty"], f_["ar/gold_shuf"])
        res["check2_av_true_minus_shuf"] = f_["av/true"] - f_["av/shuf"]
        res["check1_per_head"] = h_["ar/gold"] - max(0.0, h_["ar/empty"], h_["ar/gold_shuf"])
        res["check2_per_head"] = h_["av/true"] - h_["av/shuf"]
        err_m = {name: ((pr - gold_m) ** 2).sum(-1) for name, pr in preds_m.items()}
        base_err_m = ((base_m - gold_m) ** 2).sum(-1)
        boot = np.random.default_rng(args.seed + seed_off)
        idx = boot.integers(0, len(ev), size=(1000, len(ev)))
        bf = {k: 1.0 - e[idx].sum(1) / base_err_m[idx].sum(1) for k, e in err_m.items()}
        head_ix = list(by_head.values())
        ph = {k: np.array([fve(preds_m[k][ix], gold_m[ix], base_m[ix]) for ix in head_ix]) for k in preds_m}
        hidx = boot.integers(0, len(head_ix), size=(1000, len(head_ix)))
        bh = {k: v_[hidx].mean(1) for k, v_ in ph.items()}

        def ci(x: np.ndarray) -> list[float]:
            return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
        z = np.zeros(1000)
        res["ci95"] = {
            "check1_pooled": ci(bf["ar/gold"] - np.maximum.reduce([z, bf["ar/empty"], bf["ar/gold_shuf"]])),
            "check2_pooled": ci(bf["av/true"] - bf["av/shuf"]),
            "check1_per_head": ci(bh["ar/gold"] - np.maximum.reduce([z, bh["ar/empty"], bh["ar/gold_shuf"]])),
            "check2_per_head": ci(bh["av/true"] - bh["av/shuf"]),
        }
        return res, err_m, base_err_m

    # Raw: the √d-scaled write, magnitude included. Direction: both sides rescaled to norm mse_scale
    # (the standard NLA reward/loss space), baseline = the head's mean normalized training write.
    raw_res, err, base_err = score(gold, preds, baseline, 1)
    results.update(raw_res)

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * mse_scale_f
    tr_u = unit(tr_v.astype(np.float32))
    head_mean_dir = {h: tr_u[[i for i, x in enumerate(tr_heads) if x == h]].mean(0) for h in set(tr_heads)}
    base_dir = np.stack([head_mean_dir[h] for h in heads]).astype(np.float32)
    preds_dir = {k: (base_dir if k == "ar/head_mean" else unit(pr)) for k, pr in preds.items()}
    results["direction"], err_dir, base_err_dir = score(unit(gold), preds_dir, base_dir, 2)
    # With both sides at norm √d, MSE = 2 − 2·cos: the direction-only reward is the cosine, and a
    # normalized prediction beats the (≈ predict-zero) baseline only above cos 0.5, so the FVE above
    # mostly measures that handicap. Mean cosine is the readable number; compare conditions on it.
    g_u = unit(gold)
    results["direction"]["cos"] = {
        k: float(((unit(pr) * g_u).sum(-1) / mse_scale_f ** 2).mean()) for k, pr in preds_dir.items()}
    results["av_distinct_fraction"] = len(set(av_true)) / len(av_true)

    import pyarrow as pa
    pq.write_table(pa.table({
        "row": np.arange(len(ev)), "example_idx": [r["example_idx"] for r in ev],
        "layer": [la for la, _ in heads], "head": [hd for _, hd in heads],
        "pos": [r["pos"] for r in ev], "dev_rel": [r["dev_rel"] for r in ev],
        "gold": gold_desc, "av_true": av_true, "av_shuf": av_shuf,
        "shuf_partner_row": perm,   # av_shuf[i] is the AV's text for row perm[i]'s write
    }), out / "generations.parquet")
    for fname, e_, b_ in (("row_errors.npz", err, base_err), ("row_errors_direction.npz", err_dir, base_err_dir)):
        np.savez(out / fname, base=b_, layer=np.array([la for la, _ in heads]),
                 head=np.array([hd for _, hd in heads]), **{k.replace("/", "__"): e for k, e in e_.items()})
    head_ix = list(by_head.values())

    # AR-free companion to check 2: per-head cross-fitted ridge from the text alone to the write
    # (nla.datagen.heads.probe.crossfit), so a weak AR can't hide a good AV or vice versa.
    from nla.datagen.heads.probe import crossfit
    text_conds = {"gold": gold_desc, "av_true": av_true, "av_shuf": av_shuf}
    results["text_probe_head_avg"] = {
        k: float(np.mean([crossfit([txt[i] for i in ix], v[ix], seed=args.seed)[0] for ix in head_ix]))
        for k, txt in text_conds.items()}
    v_dir = unit(v)
    results["direction"]["text_probe_head_avg"] = {
        k: float(np.mean([crossfit([txt[i] for i in ix], v_dir[ix], seed=args.seed)[0] for ix in head_ix]))
        for k, txt in text_conds.items()}
    (out / "checks.json").write_text(json.dumps(results, indent=1))
    f = results["fve"]

    print("\nheld-out FVE (√d-scaled normalized write; baseline = per-head AR-training mean)")
    print(f"{'condition':<16}{'pooled':>8}{'per-head':>9}" + "".join(f"{f'L{lay}':>8}" for lay in layers))
    for name in preds:
        print(f"{name:<16}{f[name]:>8.3f}{results['fve_head_avg'][name]:>9.3f}"
              + "".join(f"{results['fve_by_layer'][name][f'L{lay}']:>8.3f}" for lay in layers))
    print(f"\ncheck 1 (AR: gold − best of head-mean/empty/shuffled): {results['check1_ar_beats_empty']:+.3f} pooled, "
          f"{results['check1_per_head']:+.3f} per-head")
    print(f"check 2 (AV: true − another example's write):     {results['check2_av_true_minus_shuf']:+.3f} pooled, "
          f"{results['check2_per_head']:+.3f} per-head")
    c95 = results["ci95"]
    def fmt(c: list[float]) -> str:
        return f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    print(f"95% bootstrap CIs: check 1 pooled {fmt(c95['check1_pooled'])}, per-head {fmt(c95['check1_per_head'])}; "
          f"check 2 pooled {fmt(c95['check2_pooled'])}, per-head {fmt(c95['check2_per_head'])}")
    tp = results["text_probe_head_avg"]
    print(f"text-only probe (per-head ridge, text → write, mean over heads): gold {tp['gold']:.3f}, "
          f"AV on true write {tp['av_true']:.3f}, AV on another write {tp['av_shuf']:.3f}")
    d_ = results["direction"]
    print("\nheld-out FVE, DIRECTION ONLY (both sides normalized; baseline = per-head mean normalized write)")
    print(f"{'condition':<16}{'pooled':>8}{'per-head':>9}")
    for name in preds:
        print(f"{name:<16}{d_['fve'][name]:>8.3f}{d_['fve_head_avg'][name]:>9.3f}")
    print("mean cosine(pred, write): " + ", ".join(f"{k} {c:.3f}" for k, c in d_["cos"].items()))
    print(f"check 1: {d_['check1_ar_beats_empty']:+.3f} pooled {fmt(d_['ci95']['check1_pooled'])}, "
          f"{d_['check1_per_head']:+.3f} per-head {fmt(d_['ci95']['check1_per_head'])}")
    print(f"check 2: {d_['check2_av_true_minus_shuf']:+.3f} pooled {fmt(d_['ci95']['check2_pooled'])}, "
          f"{d_['check2_per_head']:+.3f} per-head {fmt(d_['ci95']['check2_per_head'])}")
    tpd = d_["text_probe_head_avg"]
    print(f"text-only probe (direction): gold {tpd['gold']:.3f}, AV true {tpd['av_true']:.3f}, "
          f"AV other {tpd['av_shuf']:.3f}")
    print(f"AV distinct descriptions: {results['av_distinct_fraction']:.0%}")

    lines = ["# Head NLA warm-start: held-out examples\n",
             "| condition | FVE |", "|---|---|", *[f"| {k} | {f[k]:.3f} |" for k in preds], ""]
    show = rng.choice(len(ev), size=min(25, len(ev)), replace=False)
    for i in show:
        r = ev[i]
        lines += [f"## L{r['activation_layer']}H{r['head']} · pos {r['pos']} · dev/rms {r['dev_rel']:.2f} · "
                  f"‖v‖ {np.linalg.norm(v[i]):.2f}",
                  f"- **Gold (training-style target):** {gold_desc[i]}",
                  f"- **AV, true write:** {av_true[i]}",
                  f"- **AV, another example's write:** {av_shuf[i]}", ""]
    (out / "examples.md").write_text("\n".join(lines))
    print(f"wrote {out / 'checks.json'} and {out / 'examples.md'}")


if __name__ == "__main__":
    main()
