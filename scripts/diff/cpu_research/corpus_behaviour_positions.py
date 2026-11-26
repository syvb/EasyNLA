"""Does the diff separate what the model goes on to do, at each role Arm 2 trains on?

Labels come from Qwen3-8B's own rollouts on safety-source prompts: an explicit refusal versus a
plain compliance (hedged answers are left out, see `corpus_geometry.refusal_label`). For every
conversation this takes `asst_last`, the mean over its sampled `user_content` rows and the mean
over its sampled `asst_rollout` rows, and reports Fisher SNR of unit(x - mean) relative to raw
h_target for h_target itself (no Base model) and for the diff under each Base wrapper and
alignment, with a bootstrap over conversations.

`user_content` rows precede the answer, so separation there is prediction of behaviour.
`asst_rollout` rows sit on the answer's own tokens, so both representations also see what was said.

    python corpus_behaviour_positions.py --phase0-dir P0 [--phase0-dir P0_robust] --rollouts R --out out.json
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

from alignment_variants import score
from corpus_geometry import refusal_label
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.wrappers import ACTIVE_ROLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", action="append", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir[0]) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    rows = load_corpus_rows(cfg, tok, args.rollouts, seed=args.seed)
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    labels = np.asarray([refusal_label(t) for t in rows.texts])
    safety = np.asarray([s != "ultrachat" for s in rows.sources])
    keep = np.where(safety & (labels != "ambiguous"))[0]
    y = (labels[keep] == "refuse").astype(int)
    print(f"[behaviour-pos] {len(keep)} safety conversations: {int(y.sum())} refuse, {int((1 - y).sum())} comply", flush=True)

    reps: dict[str, np.ndarray] = {"no_base": rows.ht}
    for w in ("plain", "chatml"):
        reps[f"identity/{w}"] = rows.ht - rows.hb[w]
        for d in args.phase0_dir:
            art = load_alignment(Path(d) / f"align_L{k}_id_anchored.pt")
            reps[f"{Path(d).name}/{w}"] = apply_alignment(torch.from_numpy(rows.hb[w]), torch.from_numpy(rows.ht), art).numpy()

    def pooled(x: np.ndarray, role: str) -> torch.Tensor:
        out = np.zeros((len(keep), x.shape[1]), dtype=np.float64)
        for j, c in enumerate(keep):
            m = (rows.conv == c) & (rows.roles == role) & ok
            assert m.any(), f"conversation {c} has no usable {role} row"
            out[j] = x[m].astype(np.float64).mean(0)
        return torch.from_numpy(out).float()

    report: dict = {"layer": k, "n": len(keep), "n_refuse": int(y.sum()), "roles": {}}
    groups = np.arange(len(keep))
    for role in ACTIVE_ROLES:
        ht = pooled(rows.ht, role)
        r = {name: score(ht, pooled(x, role), y, groups, n_boot=args.n_boot, seed=args.seed) for name, x in reps.items()}
        report["roles"][role] = r
        print(f"[behaviour-pos] {role} (SNR h_target {r['no_base']['snr_h_target']:.3f}): " + " ".join(
            f"{name}={v['ratio']:.2f}[{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]" for name, v in r.items()), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
