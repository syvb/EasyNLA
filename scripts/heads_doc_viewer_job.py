"""Every token × every head of one layer, for a queue of documents: writes, full attention, readouts, and
RL / SFT / Haiku-gold explanations with their reconstruction scores. Runs on the GPU pod as a queue worker.

    python scripts/heads_doc_viewer_job.py --queue <dir> --out <dir> --meta <l24/meta dir> --sft-dir <sft data> \
        --texts-ref <48k texts.parquet> --sft-av <dir> --sft-ar <dir> --rl-ckpt <iter dir> [--hf-repo <dataset>]

Queue: JSON specs {"name", "text", optional "doc_id", "domain", "url", "source"} in <queue>/, processed in
filename order and moved to <queue>/done/. New specs can be dropped in while it runs; it exits when the
queue is empty and <queue>/CLOSE exists, or after --idle-minutes with nothing new.

Per document, in <out>/<name>/ (and uploaded to <hf-repo>:<hf-path>/<name>/ when --hf-repo is given):
  meta.json       name, source, doc_id, domain, url, n_tokens, write check against stored eval rows
  token_ids.npy   the document as the extractor tokenizes it (truncated to --max-length)
  attn.npz        attn: float16 [heads, T, T], the layer's full attention pattern (eager attention)
  writes.npz      v: float16 [T, heads, d_model], the normalized writes exactly as the AV / AR see them
  sites.parquet   one row per (pos, head): readouts, the three explanations, cos / FVE for each, attention stats
The write is (z_h − z̄_h) W_O,hᵀ / dev_rms_h, capped at min(q0.999 / rms, 4) (export_sft.normalized_writes);
z comes from the default (SDPA) forward, as in the extraction; the attention pattern is evaluation evidence only.
"""

import argparse
import asyncio
import json
import math
import shutil
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from heads_sft_checks import ar_predict, av_generate  # noqa: E402
from heads_soft_token_diag import avg_delta_tokens, lens_tokens  # noqa: E402
import heads_soft_token_explain as EX  # noqa: E402
from nla.config import load_nla_config  # noqa: E402
from nla.datagen.heads import soft_token  # noqa: E402
from nla.datagen.heads.export_sft import critic_explanation  # noqa: E402
from nla.datagen.heads.store import load_stats, load_wo, read_meta  # noqa: E402
from nla.schema import resolve_target_scale  # noqa: E402
from nla.utils import register_karvonen_hook  # noqa: E402
from show_nla_generations import load_ar_critic  # noqa: E402


def log(msg: str) -> None:
    print(f"[docjob {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


class Writer:
    """Captures layer-L head outputs (SDPA forward) and attention probabilities (eager forward)."""

    def __init__(self, model, tokenizer, layers, layer: int, meta_dir: Path, cap_quantile: float, max_norm: float,
                 max_length: int):
        self.model, self.tok, self.layer, self.max_length = model, tokenizer, layer, max_length
        self.meta = read_meta(meta_dir)
        self.stats = load_stats(meta_dir)
        self.wo = load_wo(meta_dir, layer)                                   # [d_model, H*d]
        li = self.stats.layer_pos(layer)
        self.z_bar = self.stats.mean_head_out[li].float()                    # [H, d]
        self.rms = self.stats.dev_rms[li].float()                            # [H]
        q = self.stats.quantile_levels.tolist()
        qi = min(range(len(q)), key=lambda i: abs(q[i] - cap_quantile))
        assert abs(q[qi] - cap_quantile) < 1e-6, f"cap quantile {cap_quantile} not in {q}"
        self.cap = torch.clamp(self.stats.dev_quantiles[li, :, qi].float() / self.rms, max=max_norm)   # [H]
        self.attn_mod = layers[layer].self_attn

    def tokenize(self, text: str) -> list[int]:
        self.tok.truncation_side = "right"
        return self.tok([text], truncation=True, max_length=self.max_length, add_special_tokens=True)["input_ids"][0]

    @torch.no_grad()
    def run(self, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        H, d = self.meta.n_heads, self.meta.head_dim
        dev = self.model.get_input_embeddings().weight.device
        x = torch.tensor([ids], device=dev)
        got: dict = {}
        h1 = self.attn_mod.o_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("z", a[0].detach()))
        try:
            self.model(input_ids=x, use_cache=False)
        finally:
            h1.remove()
        z = got["z"][0].to(torch.bfloat16).float().cpu().view(len(ids), H, d)        # as the records store it
        w = torch.einsum("thd,mhd->thm", z - self.z_bar, self.wo.view(-1, H, d))    # [T, H, d_model]
        v = w / self.rms[None, :, None]
        n = v.norm(dim=-1, keepdim=True)
        v = torch.where(n > 0, v * torch.clamp(self.cap[None, :, None] / n.clamp_min(1e-12), max=1.0), v)
        cfg = self.model.config
        prev = cfg._attn_implementation
        cfg._attn_implementation = "eager"
        h2 = self.attn_mod.register_forward_hook(lambda m, a, o: got.__setitem__("attn", o[1].detach()))
        try:
            self.model(input_ids=x, use_cache=False)
        finally:
            h2.remove()
            cfg._attn_implementation = prev
        attn = got["attn"][0].float().cpu().numpy()                                     # [H, T, T]
        return v.numpy().astype(np.float32), attn


def unit(x: np.ndarray, scale: float) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta", required=True, help="head dataset meta dir (head_meta.yaml, head_stats, wo)")
    p.add_argument("--sft-dir", required=True, help="SFT data dir (av_sft sidecar, ar_sft head means, eval rows)")
    p.add_argument("--texts-ref", required=True, help="the 48k readouts table: the explainer's common-filler split")
    p.add_argument("--sft-av", required=True)
    p.add_argument("--sft-ar", required=True)
    p.add_argument("--rl-ckpt", required=True, help="RL iter dir (AV adapter + critic/)")
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gen-batch", type=int, default=128)
    p.add_argument("--readout-batch", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--idle-minutes", type=float, default=20)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-path", default="l24/std_rl/doc_viewer")
    p.add_argument("--no-gold", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--debug-positions", type=int, default=0, help="explain only the last N tokens (dry runs)")
    args = p.parse_args()

    import os
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    queue, out, meta_dir, sft = Path(args.queue), Path(args.out), Path(args.meta), Path(args.sft_dir)
    (queue / "done").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # --- models (all resident) ---
    log("loading models")
    sd = soft_token.SoftTokenDescriber(args.base, args.device, "bfloat16", args.readout_batch)
    writer = Writer(sd.model, sd.tokenizer, sd.layers, args.layer, meta_dir, 0.999, 4.0, args.max_length)
    H, d_model = writer.meta.n_heads, writer.meta.d_model
    avg_recipes = [soft_token.Recipe(f"{p_}{c}", soft_token.PROMPTS[p_], beta=1.0, carrier=c, readout="delta")
                   for p_ in ("assoc", "continue") for c in (" something", " it")]
    tok = AutoTokenizer.from_pretrained(args.base)
    cfg = load_nla_config(str(sft / "av_sft.parquet"), tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(args.device)
    actor = PeftModel.from_pretrained(base, args.sft_av, adapter_name="sft").eval()
    actor.load_adapter(args.rl_ckpt, adapter_name="rl")
    vref: list = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1, preserve_magnitude=False)
    ar = {"rl": load_ar_critic(str(Path(args.rl_ckpt) / "critic"), args.base, args.device),
          "sft": load_ar_critic(args.sft_ar, args.base, args.device)}
    prompt = pq.read_table(sft / "eval.parquet", columns=["prompt"]).slice(0, 1).to_pylist()[0]["prompt"]
    # FVE baseline: each head's mean normalized AR-training write (heads_sft_checks' direction metric)
    art = pq.read_table(sft / "ar_sft.parquet", columns=["activation_vector", "activation_layer", "head"])
    tr = np.asarray(art.column("activation_vector").combine_chunks().flatten()).reshape(art.num_rows, -1)
    tr_u = unit(tr.astype(np.float32), mse_scale_f)
    keys = list(zip(art.column("activation_layer").to_pylist(), art.column("head").to_pylist(), strict=True))
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    missing = [h for h in range(H) if not by[(args.layer, h)]]
    if missing:
        log(f"WARNING: no AR-training rows for heads {missing}: their FVE baseline is the zero vector")
    head_mean = np.stack([tr_u[by[(args.layer, h)]].mean(0) if by[(args.layer, h)] else np.zeros(tr_u.shape[1], np.float32)
                          for h in range(H)])                                                # [H, d_model]
    ev = pq.read_table(sft / "eval.parquet", columns=["doc_id", "pos", "head", "activation_vector"]).to_pandas()
    ref = pq.read_table(args.texts_ref, columns=["avg16"]).to_pydict()["avg16"]
    df = EX.doc_freq(ref)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    ex_args = argparse.Namespace(model="claude-haiku-5-5", effort="medium", retries=4, concurrency=24,
                                 common_threshold=0.08)
    api = None
    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
    log(f"models ready; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    idle_since = time.time()
    while True:
        specs = sorted(f for f in queue.glob("*.json"))
        if not specs:
            if (queue / "CLOSE").exists() or time.time() - idle_since > args.idle_minutes * 60:
                log("queue empty; exiting")
                break
            time.sleep(15)
            continue
        f = specs[0]
        try:
            spec = json.loads(f.read_text())
            name = spec["name"]
            log(f"=== {name} ===")
            t0 = time.time()
            dd = out / name
            dd.mkdir(parents=True, exist_ok=True)
            ids = writer.tokenize(spec["text"])
            T = len(ids)
            v, attn = writer.run(ids)                                           # [T,H,dm], [H,T,T]
            np.save(dd / "token_ids.npy", np.array(ids, np.int32))
            np.savez_compressed(dd / "attn.npz", attn=attn.astype(np.float16))
            np.savez_compressed(dd / "writes.npz", v=v.astype(np.float16))
            check = {}
            if spec.get("doc_id"):
                m = ev[ev.doc_id == spec["doc_id"]]
                if len(m):
                    stored = np.stack(m.activation_vector.values).astype(np.float32)
                    mine = v[m.pos.values, m["head"].values]
                    cs = (stored * mine).sum(1) / np.linalg.norm(stored, axis=1) / np.linalg.norm(mine, axis=1)
                    rel = np.linalg.norm(stored - mine, axis=1) / np.linalg.norm(stored, axis=1)
                    check = {"n_stored_rows": int(len(m)), "min_cos": float(cs.min()), "max_rel_err": float(rel.max())}
                    log(f"write check vs {len(m)} stored eval rows: min cos {cs.min():.5f}, max rel err {rel.max():.2e}")
                    assert cs.min() > 0.995, f"captured writes disagree with the stored eval rows: {check}"
            positions = list(range(T))[-args.debug_positions:] if args.debug_positions else list(range(T))
            sites = [(t, h) for t in positions for h in range(H)]
            vs = np.stack([v[t, h] for t, h in sites])                           # [n_sites, d_model]
            vv = torch.from_numpy(vs)
            log(f"{T} tokens × {H} heads = {len(sites)} sites; readouts")
            avg16 = avg_delta_tokens(sd, avg_recipes, vv, [args.layer] * len(sites))
            lens = lens_tokens(sd, vv)
            # gold: Haiku from the readouts only, in the background while the GPU generates
            gold: dict = {}
            gold_cache = dd / "gold.jsonl"
            th = None
            if not args.no_gold and key:
                rows = [(i, EX.user_text(avg16[i], lens[i], df, ex_args.common_threshold)) for i in range(len(sites))]
                done = EX.load_cache(gold_cache)
                todo = [r for r in rows if r[0] not in done]
                th = threading.Thread(target=lambda: asyncio.run(EX.run(todo, gold_cache, ex_args, key)), daemon=True)
                th.start()
            texts = {}
            for kind in ("rl", "sft"):
                actor.set_adapter(kind)
                log(f"{kind} AV generation (T={args.temperature})")
                texts[kind] = av_generate(actor, vref, tok, cfg, [{"prompt": prompt}] * len(sites), vs,
                                          args.device, args.gen_batch, args.max_new_tokens, args.temperature)
            if th is not None:
                log("waiting for the gold explanations")
                th.join()
                gold = EX.load_cache(gold_cache)
            texts["gold"] = [gold.get(i, {}).get("explain", "") for i in range(len(sites))]
            g = unit(vs, mse_scale_f)
            base_err = ((head_mean[[h for _, h in sites]] - g) ** 2).sum(-1)
            cols: dict = {"pos": [t for t, _ in sites], "head": [h for _, h in sites],
                          "token_id": [ids[t] for t, _ in sites], "v_norm": np.linalg.norm(vs, axis=1),
                          "avg16": avg16, "lens": lens, "rl_text": texts["rl"], "sft_text": texts["sft"],
                          "gold_text": texts["gold"], "gold_ok": [not gold.get(i, {}).get("breaks_rules", True)
                                                                  for i in range(len(sites))]}
            for reader, text_kinds in (("rl", ("rl", "gold")), ("sft", ("sft", "gold"))):
                for kind in text_kinds:
                    ar_in = [critic_explanation(args.layer, h, s) for (_, h), s in zip(sites, texts[kind], strict=True)]
                    pred = unit(ar_predict(ar[reader], tok, ar_in, mse_scale_f, args.device, 64), mse_scale_f)
                    err = ((pred - g) ** 2).sum(-1)
                    tag = kind if kind != "gold" else f"gold_{reader}"
                    cols[f"{tag}_cos"] = 1 - err / (2 * mse_scale_f ** 2)
                    cols[f"{tag}_fve"] = 1 - err / base_err
            ar_ = attn                                                             # stats per (pos, head)
            cols["first_w"] = [float(ar_[h, t, 0]) for t, h in sites]
            cols["self_w"] = [float(ar_[h, t, t]) for t, h in sites]
            cols["entropy"] = [float(-(ar_[h, t, : t + 1] * np.log(ar_[h, t, : t + 1] + 1e-12)).sum()) for t, h in sites]
            pq.write_table(pa.table(cols), dd / "sites.parquet")
            meta = {k: spec.get(k, "") for k in ("name", "doc_id", "domain", "url", "source")}
            meta.update({"n_tokens": T, "n_heads": H, "layer": args.layer, "temperature": args.temperature,
                         "write_check": check, "minutes": round((time.time() - t0) / 60, 1),
                         "rl_fve_mean": float(np.mean(cols["rl_fve"])), "sft_fve_mean": float(np.mean(cols["sft_fve"])),
                         "rl_ckpt": str(args.rl_ckpt), "n_gold": int(sum(bool(s) for s in texts["gold"]))})
            (dd / "meta.json").write_text(json.dumps(meta, indent=1))
            log(f"{name}: {T} tokens in {meta['minutes']} min; RL FVE mean {meta['rl_fve_mean']:+.3f}, "
                f"SFT {meta['sft_fve_mean']:+.3f}; gold {meta['n_gold']}/{len(sites)}")
            if api is not None:
                for attempt in range(3):
                    try:
                        api.upload_folder(repo_id=args.hf_repo, repo_type="dataset", folder_path=str(dd),
                                          path_in_repo=f"{args.hf_path}/{name}",
                                          commit_message=f"doc viewer: {name} ({T} tokens × {H} heads)")
                        log(f"uploaded {name} to {args.hf_repo}:{args.hf_path}/{name}")
                        break
                    except Exception as e:   # noqa: BLE001 — the local copy is fetched at the end anyway
                        log(f"upload attempt {attempt + 1} failed: {e}")
                        time.sleep(30)
            (dd / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        except Exception:
            log(f"!!! {f.name} failed:\n{traceback.format_exc()}")
            (out / f"{f.stem}.FAILED").write_text(traceback.format_exc())
        shutil.move(str(f), queue / "done" / f.name)
        idle_since = time.time()


if __name__ == "__main__":
    main()
