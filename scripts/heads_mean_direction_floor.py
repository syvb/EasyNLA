"""How far does normalizing the mean move it? The FVE floor of a reader whose output is normalized.

    python scripts/heads_mean_direction_floor.py --out <json>

FVE's 0% is predicting m, the mean of the unit-normalized targets. A reader whose output is normalized can't
output m (|m| = ρ < 1). Its best constant output is the mean direction m/ρ, which scores (ρ−1)/(1+ρ). This
measures ρ and that floor (held out) for:
- head:      our head-NLA targets (each head's normalized write deviation; l24/sft_data, AR-train mean, eval rows);
- head_raw:  the raw head writes they came from (rms·v + the head's mean write; exact except for capped rows);
- head_doc:  every token of the 4 doc-viewer documents (few documents, so writes share a per-document part);
- resid:     the standard NLA's input, Qwen3-8B layer-24 residual (last token, norm none; 21k rows sampled from
             syvb/nanonla-qwen3-8b-L24-av-sft-activations), raw, mean-centered, and without its largest dims.
Reads only public HF data; no GPU.
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "syvb/head-nla-qwen3-8b"
RESID = "syvb/nanonla-qwen3-8b-L24-av-sft-activations"
DOCS = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression",
        "04_law_gainsborough_fraud"]


def get(f: str, repo: str = REPO) -> str:
    return hf_hub_download(repo, f, repo_type="dataset")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def floor(train_u: np.ndarray, test_u: np.ndarray) -> dict:
    """ρ = |mean of unit train targets|; FVE of the mean direction on test, against the mean itself."""
    m = train_u.mean(0)
    mh = m / np.linalg.norm(m)
    return {"rho": float(np.linalg.norm(m)),
            "floor_fve": float(1 - ((test_u - mh) ** 2).sum(1).mean() / ((test_u - m) ** 2).sum(1).mean())}


def per_head(rows: list[dict]) -> dict:
    r, f = np.array([x["rho"] for x in rows]), np.array([x["floor_fve"] for x in rows])
    return {"rho_median": float(np.median(r)), "rho_range": [float(r.min()), float(r.max())],
            "floor_fve_mean": float(f.mean()), "floor_fve_range": [float(f.min()), float(f.max())]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = {}
    st = load_file(get("l24/meta/head_stats.safetensors"))
    rms, mw = st["dev_rms"][0], st["mean_write"][0]

    def load(f):
        t = pq.read_table(get(f), columns=["activation_vector", "head"]).to_pandas()
        return np.stack(t.activation_vector.values).astype(np.float32), t["head"].values
    Vtr, Htr = load("l24/sft_data/ar_sft.parquet")
    Vte, Hte = load("l24/sft_data/eval.parquet")
    head, raw = [], []
    for h in range(32):
        tr, te = Vtr[Htr == h], Vte[Hte == h]
        head.append(floor(unit(tr), unit(te)))
        raw.append(floor(unit(tr * rms[h] + mw[h]), unit(te * rms[h] + mw[h])))
    out["head"], out["head_raw"] = per_head(head), per_head(raw)
    del Vtr, Vte

    V = np.concatenate([np.load(get(f"l24/std_rl/doc_viewer/{d}/writes.npz"))["v"].astype(np.float32) for d in DOCS])
    idx = np.random.default_rng(0).permutation(len(V))
    A, B = idx[: len(V) // 2], idx[len(V) // 2:]
    out["head_doc"] = per_head([floor(unit(V[A, h]), unit(V[B, h])) for h in range(32)])
    del V

    pf = pq.ParquetFile(get("av_sft_full_activations.parquet", RESID))
    rng, X = np.random.default_rng(0), []
    for g in range(0, pf.metadata.num_row_groups, 3):        # rows are length-sorted: spread the sample
        t = pf.read_row_group(g, columns=["activation_vector"]).column(0).to_numpy(zero_copy_only=False)
        X.append(np.stack(t[rng.choice(len(t), size=min(1000, len(t)), replace=False)]).astype(np.float32))
    X = np.concatenate(X)
    rng.shuffle(X)
    tr, te = X[: len(X) // 2], X[len(X) // 2:]
    mu = tr.mean(0)
    top = np.argsort(-np.abs(tr).mean(0))
    out["resid"] = {"n": len(X), "raw": floor(unit(tr), unit(te)), "centered": floor(unit(tr - mu), unit(te - mu)),
                    "top_dims": top[:8].tolist(), "top_dims_mean_abs": np.abs(tr[:, top[:8]]).mean(0).round(1).tolist(),
                    "median_dim_mean_abs": float(np.median(np.abs(tr).mean(0))),
                    "share_sq_norm_top4": float((tr[:, top[:4]] ** 2).sum(1).mean() / (tr ** 2).sum(1).mean())}
    for k in (2, 8):
        keep = np.setdiff1d(np.arange(X.shape[1]), top[:k])
        out["resid"][f"without_top{k}"] = floor(unit(tr[:, keep]), unit(te[:, keep]))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
