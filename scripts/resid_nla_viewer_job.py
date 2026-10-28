"""Standard (residual-stream) NLA explanations for every token of the viewer documents. Runs on a GPU pod.

    python scripts/resid_nla_viewer_job.py --docs <dir>=<hf path prefix> [--docs ...] --out <dir> \
        --av-base <nanonla AV dir> --av-lora <RL LoRA dir> --ar <nanonla AR dir> --rl-data <rl_full.parquet> [--hf-repo ...]

The NLA is the from-scratch nanoNLA on Qwen3-8B layer 24 (syvb/nanonla-qwen3-8b-L24-av + the RL LoRA with no length penalty,
p0.0, read by the frozen syvb/nanonla-qwen3-8b-L24-ar; held-out FVE 0.585 in its own evaluation). It explains the
residual stream: the output of decoder block 24 (HF hidden_states[25]) at a token, raw, the way its training data was
extracted (nla/datagen/extractors.py).

1. Each document's token ids (<doc>/token_ids.npy, the exact ids the viewer shows) go through Qwen3-8B once; the block-24
   output at every token is kept.
2. Check on the NLA's own held-out rows (rl_full.parquet rows 25000+, document-disjoint from its RL training): the
   pipeline must reproduce its reported FVE (about 0.585), else stop before explaining anything.
3. Every token: one explanation (T=1, as in RL), read by the AR. Scores, like the training loss, compare directions:
   prediction and target both scaled to mse_scale; FVE = 1 − err / err of the mean target of the NLA's RL rows 0–24k.
Writes <out>/<doc>/resid.parquet (pos, resid_text, resid_cos, resid_fve, resid_err, resid_base, resid_norm) and uploads
it to <hf path prefix>/<doc>/resid.parquet.
"""
import argparse
import json
import os
import sys
import time
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
from nla.models import NLACriticModel  # noqa: E402
from nla.schema import resolve_target_scale  # noqa: E402
from nla.utils import register_karvonen_hook  # noqa: E402


def log(msg: str) -> None:
    print(f"[resid {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def scaled(x: np.ndarray, s: float) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", action="append", required=True, help="<local dir>=<hf path prefix>; one subdir per document")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--av-base", required=True)
    p.add_argument("--av-lora", required=True)
    p.add_argument("--ar", required=True)
    p.add_argument("--rl-data", required=True, help="the NLA's rl_full.parquet (+ its .nla_meta.yaml sidecar)")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--gen-batch", type=int, default=128)
    p.add_argument("--ar-batch", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="far above the ~127-token mean, so an explanation never ends on the cap")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--check-rows", type=int, default=256)
    p.add_argument("--min-check-fve", type=float, default=0.45)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = load_nla_config(a.rl_data, tok)
    s = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    rl = pq.read_table(a.rl_data, columns=["prompt", "activation_vector"])
    vec = np.asarray(rl.column("activation_vector").combine_chunks().flatten()).reshape(rl.num_rows, -1).astype(np.float32)
    mean_t = scaled(vec[:24000], s).mean(0)                                   # predict-the-mean baseline (not renormalized)
    prompt = rl.column("prompt")[0].as_py()
    assert all(rl.column("prompt")[i].as_py() == prompt for i in (1, 100, 25000)), "the NLA prompt is not constant"
    docs = []
    for spec in a.docs:
        local, prefix = spec.split("=", 1)
        docs += [(d, prefix) for d in sorted(Path(local).iterdir()) if (d / "token_ids.npy").exists()]
    log(f"{len(docs)} documents; mse_scale {s:.1f}; baseline err of the mean on its own rows "
        f"{float(((mean_t - scaled(vec[:24000], s)) ** 2).sum(1).mean()) / s ** 2:.4f} (×s²)")

    # 1. residual stream at every token: output of block --layer (= hidden_states[layer + 1])
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(a.device).eval()
    grab: dict = {}
    hk = base.model.layers[a.layer].register_forward_hook(lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    resid = {}
    with torch.no_grad():
        for d, _ in docs:
            ids = torch.tensor(np.load(d / "token_ids.npy").astype(np.int64))[None].to(a.device)
            base(input_ids=ids)
            resid[d.name] = grab["h"][0].float().cpu().numpy()
            log(f"{d.name}: {ids.shape[1]} tokens; median ‖x‖ {np.median(np.linalg.norm(resid[d.name], axis=1)):.1f}")
    hk.remove()
    del base
    torch.cuda.empty_cache()

    # models
    av = AutoModelForCausalLM.from_pretrained(a.av_base, dtype=torch.bfloat16).to(a.device)
    actor = PeftModel.from_pretrained(av, a.av_lora).eval()
    vref: list = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1, preserve_magnitude=False)
    critic = NLACriticModel.from_pretrained(a.ar, torch_dtype=torch.bfloat16).to(a.device).eval()

    def explain_and_score(vecs: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        rows = [{"prompt": prompt}] * len(vecs)
        with torch.no_grad():
            texts = av_generate(actor, vref, tok, cfg, rows, vecs, a.device, a.gen_batch, a.max_new_tokens, a.temperature)
            pred = ar_predict(critic, tok, texts, s, a.device, a.ar_batch)
        t, ph = scaled(vecs, s), scaled(pred, s)
        return texts, (ph * t).sum(1) / s ** 2, ((ph - t) ** 2).sum(1), ((mean_t - t) ** 2).sum(1)

    # 2. reproduce the NLA's own held-out number before trusting the pipeline
    chk = vec[25000 : 25000 + a.check_rows]
    _, cos, err, base_err = explain_and_score(chk)
    fve = 1 - err.sum() / base_err.sum()
    log(f"held-out check ({len(chk)} rows): FVE {fve:.3f} (reported 0.585 for p0.0), mean cos {cos.mean():.3f}")
    assert fve >= a.min_check_fve, f"held-out FVE {fve:.3f} < {a.min_check_fve}: prompt / injection / scoring mismatch"
    (out).mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps({"rows": len(chk), "fve": float(fve), "mean_cos": float(cos.mean())}))

    # 3. every token
    api = None
    if a.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
    for d, prefix in docs:
        x = resid[d.name]
        texts, cos, err, base_err = explain_and_score(x)
        dd = out / d.name
        dd.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"pos": np.arange(len(x)), "resid_text": texts, "resid_cos": cos.astype(np.float32),
                                 "resid_fve": (1 - err / base_err).astype(np.float32), "resid_err": err.astype(np.float32),
                                 "resid_base": base_err.astype(np.float32), "resid_norm": np.linalg.norm(x, axis=1).astype(np.float32)}),
                       dd / "resid.parquet")
        log(f"{d.name}: pooled FVE {1 - err.sum() / base_err.sum():+.3f} (from token 1: {1 - err[1:].sum() / base_err[1:].sum():+.3f}), mean cos {cos.mean():.3f}")
        if api is not None:
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=str(dd / "resid.parquet"), path_in_repo=f"{prefix}/{d.name}/resid.parquet",
                                    repo_id=a.hf_repo, repo_type="dataset", commit_message=f"standard NLA explanations: {d.name}")
                    log(f"uploaded {prefix}/{d.name}/resid.parquet")
                    break
                except Exception as e:  # noqa: BLE001 — the local copy is fetched anyway
                    log(f"upload attempt {attempt + 1} failed: {e}")
                    time.sleep(30)


if __name__ == "__main__":
    main()
