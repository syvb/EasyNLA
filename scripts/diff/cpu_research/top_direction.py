"""What is the dominant direction of the Arm 2 target?

On the chat corpus the centred, unit-normalised cleaned diff has an effective rank near 12 to 20
(h_target: about 160), with one direction holding 20 to 30% of its variance. This script takes
that direction per role and reports what its score tracks: position in the sequence, the
conversation it comes from, the source corpus, activation norms, and whether it is a single
residual coordinate. It also repeats the spectrum for the diff under the other Base wrapper and
with identity alignment, to see whether the direction comes from the wrapper or the fitted map.

    python top_direction.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpus_geometry import center_by_role, unit
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.wrappers import ACTIVE_ROLES


def top_pcs(x: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-k variance shares, their directions [k, d], and the participation ratio of centred rows."""
    xt = torch.from_numpy(x - x.mean(0, keepdims=True)).double()
    ev, q = torch.linalg.eigh(xt.T @ xt / len(xt))
    ev = ev.clamp_min(0).flip(0)
    q = q.flip(1)
    return (ev[:k] / ev.sum()).numpy(), q[:, :k].T.numpy(), float(ev.sum() ** 2 / (ev**2).sum())


def between_group_share(score: np.ndarray, groups: np.ndarray) -> float:
    """Share of the score's variance that sits between groups (1 = constant within each group)."""
    means = {g: score[groups == g].mean() for g in np.unique(groups)}
    fitted = np.asarray([means[g] for g in groups.tolist()])
    return float(((fitted - score.mean()) ** 2).sum() / ((score - score.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--state-cache", default="", help="behaviour_link.py state npz, to correlate with KL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    win = str(cfg["base_wrapper"])
    other = "plain" if win == "chatml" else "chatml"
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    rows = load_corpus_rows(cfg, tok, args.rollouts, seed=args.seed)
    art = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    ht = torch.from_numpy(rows.ht)
    diffs = {
        f"idanc_{win}": apply_alignment(torch.from_numpy(rows.hb[win]), ht, art).numpy(),
        f"idanc_{other}": apply_alignment(torch.from_numpy(rows.hb[other]), ht, art).numpy(),
        f"identity_{win}": rows.ht - rows.hb[win],
        f"identity_{other}": rows.ht - rows.hb[other],
    }
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    pos = np.asarray([p for pl in rows.pos["chatml"] for p in pl])
    is_safety = np.asarray([rows.sources[c] != "ultrachat" for c in rows.conv])
    kl = None
    if args.state_cache and Path(args.state_cache).exists():
        z = np.load(args.state_cache)
        kl = {w: z[f"kl_{w}"] for w in ("chatml", "plain")}

    report: dict = {"layer": k, "winning_wrapper": win, "roles": {}}
    for role in ACTIVE_ROLES:
        m = (rows.roles == role) & ok
        if m.sum() < 200:
            continue
        r: dict = {"n": int(m.sum()), "spectrum": {}}
        ref_share, ref_dirs, ref_pr = top_pcs(unit(center_by_role(rows.ht[m], rows.roles[m])))
        r["spectrum"]["h_target"] = {"top3_share": ref_share.tolist(), "participation_ratio": ref_pr}
        dirs: dict[str, np.ndarray] = {}
        for name, d in diffs.items():
            share, v, pr = top_pcs(unit(center_by_role(d[m], rows.roles[m])))
            dirs[name] = v
            r["spectrum"][name] = {"top3_share": share.tolist(), "participation_ratio": pr}
        u = unit(center_by_role(diffs[f"idanc_{win}"][m], rows.roles[m]))
        v1 = dirs[f"idanc_{win}"][0]
        score = (u - u.mean(0)) @ v1
        top_coord = int(np.argmax(np.abs(v1)))
        r["top_direction"] = {
            "largest_coordinate": top_coord,
            "largest_coordinate_share_of_direction": float(v1[top_coord] ** 2),
            "coordinates_for_half_of_direction": int(np.searchsorted(np.cumsum(np.sort(v1**2)[::-1]), 0.5) + 1),
            "abs_cos_with_top_direction_of": {name: float(abs(v[0] @ v1)) for name, v in dirs.items()},
            "abs_cos_with_h_target_top_direction": float(abs(ref_dirs[0] @ v1)),
            "between_conversation_share": between_group_share(score, rows.conv[m]),
            "spearman_with": {
                "token_position": float(spearmanr(score, pos[m]).statistic),
                "h_target_norm": float(spearmanr(score, np.linalg.norm(rows.ht[m], axis=1)).statistic),
                "h_base_norm": float(spearmanr(score, np.linalg.norm(rows.hb[win][m], axis=1)).statistic),
                "cleaned_diff_norm": float(spearmanr(score, np.linalg.norm(center_by_role(diffs[f"idanc_{win}"][m], rows.roles[m]), axis=1)).statistic),
                "safety_source": float(spearmanr(score, is_safety[m]).statistic),
            },
        }
        if kl is not None:
            r["top_direction"]["spearman_with"]["kl_inst_base"] = float(spearmanr(score, kl[win][m]).statistic)
        # Tokens at the two ends of the direction.
        order = np.argsort(score)
        idx = np.where(m)[0]
        r["top_direction"]["tokens_low_end"] = [tok.decode([int(t)]) for t in rows.tokens[idx[order[:15]]]]
        r["top_direction"]["tokens_high_end"] = [tok.decode([int(t)]) for t in rows.tokens[idx[order[-15:]]]]
        report["roles"][role] = r
        print(f"[topdir] {role}: " + json.dumps(r), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
