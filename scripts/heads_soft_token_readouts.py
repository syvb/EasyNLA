"""Soft-token readouts and probe targets for a head selection (the L24 probe test + SFT run).

    python scripts/heads_soft_token_readouts.py --output-dir <head dataset> --work-dir <dir>

For every row of <output-dir>/warmstart/selection.parquet (picked by write size, never by attention):

  v           export_sft.normalized_writes: the exact vectors the SFT stage trains on
  texts.parquet   example_idx, split, layer, head, avg16, lens  (heads_soft_token_diag.py's readouts;
                  the input to heads_soft_token_explain.py)
  targets.npz     example_idx, y = v @ U_h  (U_h: orthonormal basis of the head's write space, from
                  the SVD of its o_proj slice), the probe target
  hidden.npy      float16 [N, d_model]: the injected prompt's last-position final hidden state, a
                  dense ceiling for what the model's reading of the write keeps

Every output is a function of the write (and its layer) alone.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from heads_soft_token_diag import avg_delta_tokens, final_states, lens_tokens  # noqa: E402

from nla.datagen.heads import soft_token  # noqa: E402
from nla.datagen.heads.export_sft import normalized_writes  # noqa: E402
from nla.datagen.heads.store import atomic_write_table, load_wo, read_meta  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--cap-quantile", type=float, default=0.999, help="as export_sft")
    p.add_argument("--max-norm", type=float, default=4.0, help="as export_sft")
    p.add_argument("--avg-prompts", default="assoc,continue")
    p.add_argument("--avg-carriers", default=" something, it")
    p.add_argument("--hidden-carrier", default=" something")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device-map", default="cuda:0")
    args = p.parse_args()

    out_dir, work = Path(args.output_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out_dir)
    sel = pq.read_table(out_dir / "warmstart" / "selection.parquet").to_pylist()
    print(f"{len(sel)} selected examples over {len({(r['layer'], r['head']) for r in sel})} heads", flush=True)
    vecs = torch.from_numpy(normalized_writes(out_dir, sel, args.cap_quantile, args.max_norm))
    layers = [int(r["layer"]) for r in sel]
    norms = vecs.norm(dim=-1)
    print(f"‖v‖ median {norms.median():.2f} p90 {norms.quantile(0.9):.2f} max {norms.max():.2f}", flush=True)

    d = meta.head_dim
    y = np.zeros((len(sel), d), dtype=np.float32)
    for layer in sorted(set(layers)):
        wo = load_wo(out_dir, layer)
        for head in sorted({int(r["head"]) for r in sel if int(r["layer"]) == layer}):
            idx = [i for i, r in enumerate(sel) if (int(r["layer"]), int(r["head"])) == (layer, head)]
            u = torch.linalg.svd(wo[:, head * d:(head + 1) * d].float(), full_matrices=False).U
            y[idx] = (vecs[idx] @ u).numpy()
    ex = np.array([r["example_idx"] for r in sel], dtype=np.int64)
    np.savez(work / "targets.npz", example_idx=ex, y=y)
    # v lies in the head's write space, so the 128 coordinates keep all of it
    print(f"targets: ‖y‖/‖v‖ = {np.linalg.norm(y) / float(vecs.norm()):.4f} (should be 1)", flush=True)

    sd = soft_token.SoftTokenDescriber(meta.base_model, args.device_map, "bfloat16", args.batch_size)
    avg = [soft_token.Recipe(f"{p_}{c}", soft_token.PROMPTS[p_], beta=1.0, carrier=c, readout="delta")
           for p_ in args.avg_prompts.split(",") for c in args.avg_carriers.split(",")]
    texts = {
        "example_idx": ex.tolist(),
        "split": [r["split"] for r in sel],
        "layer": layers,
        "head": [int(r["head"]) for r in sel],
        "avg16": avg_delta_tokens(sd, avg, vecs, layers),
        "lens": lens_tokens(sd, vecs),
    }
    atomic_write_table(pa.table(texts), work / "texts.parquet")
    print(f"wrote {work / 'texts.parquet'}", flush=True)

    hid_rec = soft_token.Recipe("hidden", soft_token.PROMPTS["assoc"], beta=1.0, carrier=args.hidden_carrier,
                                readout="delta")
    np.save(work / "hidden.npy", final_states(sd, hid_rec, vecs, layers).astype(np.float16))
    print(f"wrote {work / 'hidden.npy'}", flush=True)


if __name__ == "__main__":
    main()
