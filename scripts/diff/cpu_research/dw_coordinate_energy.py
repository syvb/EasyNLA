"""Which residual coordinates does the fitted identity-anchored correction write to?

`h_hat = h_base + dW @ h_base + b`. Row i of dW is everything the map adds to output
coordinate i. This prints the share of ||dW||_F^2 per output coordinate for one or more
Phase 0 directories, to show whether the map is a broad correction or a patch on a few
massive-activation coordinates.

    python dw_coordinate_energy.py --out out.json NAME=PHASE0_DIR [NAME=PHASE0_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+", help="NAME=PHASE0_DIR")
    args = ap.parse_args()
    report: dict = {}
    for spec in args.dirs:
        name, d = spec.split("=", 1)
        cfg = json.loads((Path(d) / "phase0_config.json").read_text())
        art = torch.load(Path(d) / f"align_L{cfg['layer_index']}_id_anchored.pt", map_location="cpu", weights_only=False)
        dw = art["delta_W"].double()
        out_energy = dw.pow(2).sum(1)
        top = torch.topk(out_energy, 5)
        share = out_energy / out_energy.sum()
        report[name] = {
            "alpha": float(art["alpha"]),
            "frobenius_sq": float(out_energy.sum()),
            "top_output_coordinates": [{"coordinate": int(i), "share": float(share[i])} for i in top.indices],
            # Number of output coordinates needed to hold half of the map's energy.
            "coordinates_for_half_energy": int(torch.searchsorted(torch.cumsum(torch.sort(share, descending=True).values, 0),
                                                                  torch.tensor(0.5, dtype=torch.float64)) + 1),
            "largest_bias_coordinates": [{"coordinate": int(i), "bias": float(art["bias"][i])}
                                         for i in torch.topk(art["bias"].abs(), 3).indices],
        }
        print(name, json.dumps(report[name]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
