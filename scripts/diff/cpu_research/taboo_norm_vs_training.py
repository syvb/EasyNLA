"""How large are the Taboo `Org - Inst` diffs next to the rows Arm 2 trains on?

Reads the per-organism norms from `taboo_transfer.py` and the Arm 2 training target on the chat
corpus, and reports where the Taboo norms fall in the training norm distribution at the matching
role, with the injection strength ||v|| / sqrt(||v||^2 + tau^2) that Fix 3's soft-norm hook gives each.

    python taboo_norm_vs_training.py --phase0-dir P0 --taboo taboo_transfer.json --rollouts R --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpus_geometry import center_by_role
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment


def strength(x: float) -> float:
    return float(x / np.sqrt(x**2 + 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--taboo", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k, tau, win = int(cfg["layer_index"]), float(cfg["tau_web_median"]), str(cfg["base_wrapper"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    rows = load_corpus_rows(cfg, tok, args.rollouts)
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    art = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    train = center_by_role(apply_alignment(torch.from_numpy(rows.hb[win]), torch.from_numpy(rows.ht), art).numpy(), rows.roles)
    taboo = json.loads(Path(args.taboo).read_text())

    report: dict = {"tau_web_median": tau, "positions": {}}
    for kind, role in (("user_mean", "user_content"), ("asst_last", "asst_last")):
        n = np.linalg.norm(train[(rows.roles == role) & ok], axis=1) / tau
        t = float(np.median(list(taboo["positions"][kind]["median_norm_over_tau"].values())))
        report["positions"][kind] = {
            "training_role": role,
            "n_training_rows": len(n),
            "taboo_norm_over_tau_median": t,
            "train_frac_below_taboo_median": float((n < t).mean()),
            "train_norm_over_tau_q05_q50_q95": [float(q) for q in np.quantile(n, [0.05, 0.5, 0.95])],
            "soft_norm_strength_taboo": strength(t),
            "soft_norm_strength_train_median": strength(float(np.median(n))),
        }
        print(f"[taboo-norm] {kind}: " + json.dumps(report["positions"][kind]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
