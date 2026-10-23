"""Matryoshka explanations for the doc viewer: every (token, head) site of each doc-viewer document, from
the stored writes (writes.npz, exactly as the AV / AR see them), explained by the matryoshka AV and
scored the way heads_doc_viewer_job.py scores the std RL explanations. Runs on a GPU pod.

    python scripts/heads_doc_viewer_mtry.py --viewer-dir <doc_viewer dir> --sft-dir <sft data> \\
        --mtry-ckpt <mtry iter dir> --warm-ar <warm-up iter dir>/critic [--out <dir>]

Writes <out or viewer-dir>/<doc>/mtry.parquet, one row per (pos, head) of that doc's sites.parquet:
  mtry_text                the matryoshka AV's explanation (T=1, whole; one cut off keeps what it said)
  mtry_cos, mtry_fve       the matryoshka AR reading it (direction cosine; FVE vs the head's mean write)
  mtry_t16_cos / _fve      the matryoshka AR reading only its first 16 tokens
  rl_t16_cos / _fve        the std RL explanation's first 16 tokens (sites.parquet rl_text), read by the
                           warm-up AR (prefix-trained on the std AV's text): the fair short-prefix baseline
"""
import argparse
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

from heads_sft_checks import ar_predict, av_generate  # noqa: E402

from nla.config import load_nla_config  # noqa: E402
from nla.datagen.heads.export_sft import critic_explanation  # noqa: E402
from nla.schema import resolve_target_scale  # noqa: E402
from nla.utils import register_karvonen_hook  # noqa: E402


def unit(x: np.ndarray, s: float) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer-dir", required=True)
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--mtry-ckpt", required=True, help="matryoshka iter dir (AV adapter + critic/)")
    p.add_argument("--warm-ar", required=True, help="end-of-warm-up AR dir (ar_meta.json + weights)")
    p.add_argument("--out", default=None, help="default: next to each doc's sites.parquet")
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--prefix-tokens", type=int, default=16)
    p.add_argument("--gen-batch", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-sites", type=int, default=0, help="first N sites per doc only (dry runs)")
    a = p.parse_args()

    from peft import PeftModel
    from show_nla_generations import load_ar_critic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft = Path(a.sft_dir)
    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = load_nla_config(str(sft / "av_sft.parquet"), tok)
    s = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    prompt = pq.read_table(sft / "eval.parquet", columns=["prompt"]).slice(0, 1).to_pylist()[0]["prompt"]
    art = pq.read_table(sft / "ar_sft.parquet", columns=["activation_vector", "activation_layer", "head"])
    tr_u = unit(np.asarray(art.column("activation_vector").combine_chunks().flatten(), dtype=np.float32)
                .reshape(art.num_rows, -1), s)
    by = defaultdict(list)
    for i, k in enumerate(zip(art.column("activation_layer").to_pylist(), art.column("head").to_pylist(), strict=True)):
        by[k].append(i)
    n_heads = max(h for _, h in by) + 1
    head_mean = np.stack([tr_u[by[(a.layer, h)]].mean(0) if by[(a.layer, h)] else np.zeros(tr_u.shape[1], np.float32)
                          for h in range(n_heads)])

    docs = sorted(d for d in Path(a.viewer_dir).iterdir() if (d / "sites.parquet").exists())
    data = []
    for d in docs:
        S = pq.read_table(d / "sites.parquet", columns=["pos", "head", "rl_text"]).to_pandas()
        if a.max_sites:
            S = S.iloc[: a.max_sites]
        v = np.load(d / "writes.npz")["v"]
        data.append((d, S, v[S.pos.values, S["head"].values].astype(np.float32)))
    print(f"{len(docs)} docs, {sum(len(S) for _, S, _ in data)} sites", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(a.device)
    actor = PeftModel.from_pretrained(model, a.mtry_ckpt).eval()
    vref: list = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1, preserve_magnitude=False)
    torch.manual_seed(0)
    texts = []
    for d, S, vs in data:
        print(f"[gen] {d.name}: {len(S)} sites", flush=True)
        texts.append(av_generate(actor, vref, tok, cfg, [{"prompt": prompt}] * len(S), vs, a.device, a.gen_batch,
                                 a.max_new_tokens, 1.0, None))
    del actor, model
    torch.cuda.empty_cache()

    def first(t: str) -> str:
        return tok.decode(tok.encode(t, add_special_tokens=False)[: a.prefix_tokens]).strip()

    cols = [{"pos": S.pos.values, "head": S["head"].values, "mtry_text": tx} for (_, S, _), tx in zip(data, texts)]
    L = a.prefix_tokens
    for reader_dir, jobs in ((str(Path(a.mtry_ckpt) / "critic"), (("mtry", None), (f"mtry_t{L}", "mtry"))),
                             (a.warm_ar, ((f"rl_t{L}", "rl"),))):
        critic = load_ar_critic(reader_dir, a.base, a.device)
        for (d, S, vs), tx, c in zip(data, texts, cols):
            g = unit(vs, s)
            base_err = ((head_mean[S["head"].values] - g) ** 2).sum(-1)
            for tag, src in jobs:
                desc = tx if src is None else [first(t) for t in (tx if src == "mtry" else S.rl_text.tolist())]
                pred = unit(ar_predict(critic, tok, [critic_explanation(a.layer, h, t) for h, t
                                                     in zip(S["head"].values, desc, strict=True)], s, a.device, 64), s)
                err = ((pred - g) ** 2).sum(-1)
                c[f"{tag}_cos"] = 1 - err / (2 * s ** 2)
                c[f"{tag}_fve"] = 1 - err / base_err
            print(f"[score] {d.name} with {Path(reader_dir).parent.name}: " + ", ".join(
                f"{t} cos {np.mean(c[f'{t}_cos']):.3f}" for t, _ in jobs), flush=True)
        del critic
        torch.cuda.empty_cache()
    for (d, _, _), c in zip(data, cols):
        out = Path(a.out) / d.name if a.out else d
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(c), out / "mtry.parquet")
        print(f"wrote {out / 'mtry.parquet'}", flush=True)


if __name__ == "__main__":
    main()
