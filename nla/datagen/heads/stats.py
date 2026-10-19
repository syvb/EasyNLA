"""Head stage `stats`: per-head write statistics → head_stats.safetensors + devnorms/.

For head h of a layer with o_proj slice W_h = W_O[:, h*D:(h+1)*D], the write
into the residual stream is w = W_h z (z = that head's slice of head_out).
Per head we compute, over every sampled record:

  mean_head_out   z̄            → mean_write = W_h z̄  ("the average write")
  dev_norm        ‖W_h (z − z̄)‖ per record — how far this write is from average
  dev_rms         RMS of dev_norm ("typical size": training divides by it)
  dev_quantiles   for choosing an outlier cap at training time

Norms are taken in residual-stream space through the Gram matrix
G_h = W_hᵀ W_h (‖W_h v‖² = vᵀ G_h v), so nothing d_model-wide is materialized
per record. devnorms/L{ll}/part_* files are row-aligned with records/ —
same part names, same row order — so selection can weight by deviation
without re-reading the vectors.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.datagen.heads.store import (
    HeadStats,
    atomic_write_table,
    devnorms_dir,
    devnorms_schema,
    fixed_array,
    fixed_numpy,
    from_bf16_bits,
    load_wo,
    part_files,
    read_meta,
    records_dir,
    save_stats,
)


def _head_out(path: Path, n_heads: int, head_dim: int, device: str) -> tuple[pa.Table, torch.Tensor]:
    t = pq.read_table(path, columns=["doc_id", "pos", "head_out"])
    z = from_bf16_bits(fixed_numpy(t.column("head_out"), np.uint16, n_heads * head_dim))
    return t, z.view(-1, n_heads, head_dim).to(device)


def quad_norm(v: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """‖W_h v_h‖ for v [N, H, D] and gram [H, D, D] → [N, H]."""
    return (torch.einsum("nhd,hde->nhe", v, gram) * v).sum(-1).clamp_min(0).sqrt()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--quantiles", default="0.5,0.9,0.99,0.999")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    meta = read_meta(out_dir)
    h, d = meta.n_heads, meta.head_dim
    q_levels = torch.tensor([float(x) for x in args.quantiles.split(",")])

    mean_head_out, mean_write, dev_rms, dev_q, count = [], [], [], [], []
    for layer in tqdm(meta.layers, desc="layers"):
        files = part_files(records_dir(out_dir, layer))
        assert files, f"layer {layer}: no records — extract stage incomplete?"
        w = load_wo(out_dir, layer).to(args.device).view(meta.d_model, h, d).permute(1, 0, 2)  # [H, d_model, D]
        gram = w.transpose(1, 2) @ w                                                           # [H, D, D]

        total = torch.zeros(h, d, dtype=torch.float64, device=args.device)
        n = 0
        for f in files:
            _, z = _head_out(f, h, d, args.device)
            total += z.double().sum(0)
            n += z.shape[0]
        z_bar = (total / n).float()

        dev_all = []
        for f in files:
            t, z = _head_out(f, h, d, args.device)
            dev = quad_norm(z - z_bar, gram)
            dev_all.append(dev.cpu())
            atomic_write_table(pa.Table.from_arrays([
                t.column("doc_id"), t.column("pos"),
                fixed_array(dev.cpu().numpy().astype(np.float32)),
                fixed_array(quad_norm(z, gram).cpu().numpy().astype(np.float32)),
            ], schema=devnorms_schema(h)), devnorms_dir(out_dir, layer) / f.name)
        dev_cat = torch.cat(dev_all)                                                           # [N, H]

        mean_head_out.append(z_bar.cpu())
        mean_write.append(torch.einsum("hmd,hd->hm", w, z_bar).cpu())
        dev_rms.append(dev_cat.pow(2).mean(0).sqrt())
        dev_q.append(torch.quantile(dev_cat, q_levels, dim=0).T)                             # [H, Q]
        count.append(n)

    stats = HeadStats(
        layers=list(meta.layers),
        mean_head_out=torch.stack(mean_head_out),
        mean_write=torch.stack(mean_write),
        dev_rms=torch.stack(dev_rms),
        dev_quantiles=torch.stack(dev_q),
        quantile_levels=q_levels,
        count=torch.tensor(count),
    )
    save_stats(out_dir, stats)

    rel_max = stats.dev_quantiles[..., -1] / stats.dev_rms.clamp_min(1e-12)
    print(f"wrote head stats for {len(meta.layers)} layers × {h} heads → {out_dir}")
    print(f"  records per layer: min {min(count)}  max {max(count)}")
    print(f"  dev_rms: median {stats.dev_rms.median():.4g}  min {stats.dev_rms.min():.4g}  "
          f"max {stats.dev_rms.max():.4g}")
    print(f"  q{q_levels[-1]:g}/rms: median {rel_max.median():.3g}  max {rel_max.max():.3g}")


if __name__ == "__main__":
    main()
