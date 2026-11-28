"""What the Phase 0 map does in the massive coordinate, with and without masking.

Reports, on the web tokens: each group's share of the alignment objective, the residual in the massive
coordinate by group, the norm and make-up of the map's row for that coordinate, the noise that row adds
on ordinary validation tokens, and how much of the massive rows it predicts.

    python massive_coordinate_detail.py --phase0-dir P0 --robust-dir P0_robust --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from massive_anatomy import load_web_rows, massive_mask

from nla.diff.alignment import apply_alignment, load_alignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--robust-dir", required=True)
    ap.add_argument("--coordinate", type=int, default=2276)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    web = load_web_rows(cfg, AutoTokenizer.from_pretrained(cfg["target_model"]))
    va, tr = web["val"], web["train"]
    c = args.coordinate
    out: dict = {}
    for split, w in (("train", tr), ("val", va)):
        mt, mb = massive_mask(w["ht"]), massive_mask(w["hb"])
        o = ~(mt | mb)
        res = (w["ht"] - w["hb"]).double()
        res = res - res.mean(0)
        tot = (res**2).sum()
        out[split] = {
            "naive_diff_std_coord": float((w["ht"][o][:, c] - w["hb"][o][:, c]).std()),
            "h_target_std_coord": float(w["ht"][o][:, c].std()),
            "mean_sq_norm_share_of_coord_in_ordinary_h_target": float((w["ht"][o][:, c] ** 2).sum() / (w["ht"][o] ** 2).sum()),
            "coord_residual_mean_both_fire": float((w["ht"][mt & mb][:, c] - w["hb"][mt & mb][:, c]).mean()),
            "coord_residual_mean_base_only": float((w["ht"][mb & ~mt][:, c] - w["hb"][mb & ~mt][:, c]).mean()),
            "coord_residual_mean_target_only": float((w["ht"][mt & ~mb][:, c] - w["hb"][mt & ~mb][:, c]).mean()),
            "objective_share": {"one_fires": float((res[mt ^ mb] ** 2).sum() / tot), "both_fire": float((res[mt & mb] ** 2).sum() / tot),
                                "ordinary": float((res[o] ** 2).sum() / tot)},
        }
    one_tr = massive_mask(tr["ht"]) ^ massive_mask(tr["hb"])
    mt, mb = massive_mask(va["ht"]), massive_mask(va["hb"])
    o, one_va = ~(mt | mb), mt ^ mb
    for name, d in (("stock", args.phase0_dir), ("robust", args.robust_dir)):
        art = load_alignment(Path(d) / f"align_L{k}_id_anchored.pt")
        row = art["delta_W"][c].double()
        out[name] = {
            "dW_self_entry": float(row[c]), "row_norm": float(row.norm()), "row_share_on_self": float(row[c] ** 2 / (row**2).sum()),
            "correction_std_on_ordinary_val": float((va["hb"][o].float() @ art["delta_W"][c].float()).std()),
            "bias_coord": float(art["bias"][c]),
            "val_one_fires_median_residual_coord_abs": float(apply_alignment(va["hb"][one_va], va["ht"][one_va], art)[:, c].abs().median()),
            "train_one_fires_median_residual_coord_abs": float(apply_alignment(tr["hb"][one_tr], tr["ht"][one_tr], art)[:, c].abs().median()),
        }
    print(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
