"""Candidate fixes for the massive-activation problem in the Phase 0 alignment fit, on the same metrics.

Every variant fits the repo's identity-anchored ridge (h_hat = h_base + dW h_base + b) on the same web
tokens with the same alpha, and differs only in how it treats rows:

  none        every row, as Phase 0 does (loaded from the Phase 0 artifact)
  mask_rows   drop rows whose norm is above 5x the median in either model (loaded from the refit)
  clip_norm   keep every row, scale a row pair down so neither norm exceeds 2x the median (bounded leverage)
  huber       no norm information: iteratively reweighted least squares, weight min(1, delta / residual norm)
  trimmed     no norm information: least trimmed squares, drop the 0.5% of rows with the largest residual
  project_1   keep every row, remove the massive direction from both models before fitting and diffing
  project_3   the same with the top three directions of the massive rows

Reported for each: fit quality and noise floor on ordinary validation tokens, where the correction writes,
how much noise it puts in the massive coordinate, what a massive row looks like if it reaches the diff,
and how concentrated the Arm 2 unit target is on the chat corpus.

    python massive_fix_variants.py --phase0-dir P0 --robust-dir P0_robust --rollouts R --out out.json
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

from corpus_geometry import center_by_role, spectrum_stats, unit
from corpus_rows import load_corpus_rows, massive_rows
from massive_anatomy import load_web_rows, massive_mask

from nla.diff.alignment import _ridge_solutions_from_cov, load_alignment

CHUNK = 8192


def project(x: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    """Remove the span of the orthonormal columns of `u` from the rows of `x`."""
    return x if u is None else x - (x @ u) @ u.T


def weighted_fit(hb: torch.Tensor, ht: torch.Tensor, w: torch.Tensor, alpha: float, u: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity-anchored ridge with row weights `w`, in float64, accumulated in chunks. Returns (dW, b) in fp32."""
    d = hb.shape[1]
    sw = float(w.double().sum())
    xm, ym = torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    for s in range(0, len(hb), CHUNK):
        x = project(hb[s : s + CHUNK].double(), u)
        y = project(ht[s : s + CHUNK].double(), u) - x
        ww = w[s : s + CHUNK].double()[:, None]
        xm += (x * ww).sum(0)
        ym += (y * ww).sum(0)
    xm /= sw
    ym /= sw
    cxx, cxy = torch.zeros((d, d), dtype=torch.float64), torch.zeros((d, d), dtype=torch.float64)
    for s in range(0, len(hb), CHUNK):
        x = project(hb[s : s + CHUNK].double(), u)
        y = project(ht[s : s + CHUNK].double(), u) - x
        rw = w[s : s + CHUNK].double().sqrt()[:, None]
        xc, yc = (x - xm) * rw, (y - ym) * rw
        cxx += xc.T @ xc
        cxy += xc.T @ yc
    dw, b = _ridge_solutions_from_cov(xm, ym, cxx / sw, cxy / sw, [alpha])[0]
    return dw.float(), b.float()


def diff(hb: torch.Tensor, ht: torch.Tensor, dw: torch.Tensor, b: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    """The cleaned diff a variant produces: (projected) h_target minus the aligned (projected) h_base."""
    out = torch.empty_like(ht)
    uf = None if u is None else u.float()
    for s in range(0, len(hb), CHUNK):
        x = project(hb[s : s + CHUNK].float(), uf)
        out[s : s + CHUNK] = project(ht[s : s + CHUNK].float(), uf) - (x + x @ dw.T + b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--robust-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="the value Phase 0's CV picks with and without masking")
    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    web = load_web_rows(cfg, tok)
    hb_tr, ht_tr, hb_va, ht_va = web["train"]["hb"], web["train"]["ht"], web["val"]["hb"], web["val"]["ht"]
    del web
    mt_tr, mb_tr = massive_mask(ht_tr), massive_mask(hb_tr)
    mt_va, mb_va = massive_mask(ht_va), massive_mask(hb_va)
    mass_tr, mass_va, one_va = mt_tr | mb_tr, mt_va | mb_va, mt_va ^ mb_va
    n = len(hb_tr)
    ones = torch.ones(n)

    # Massive directions, from the training rows of both models.
    stack = torch.cat([ht_tr[mt_tr], hb_tr[mb_tr]]).double()
    _, sv, vt = torch.linalg.svd(stack, full_matrices=False)
    coord = int(vt[0].abs().argmax())
    print(f"[fix] massive rows: singular values {[round(float(x)) for x in sv[:4]]}; top direction is "
          f"{float(vt[0, coord] ** 2) * 100:.1f}% coordinate {coord}", flush=True)

    variants: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
    notes: dict[str, dict] = {}
    for name, d in (("none", p0), ("mask_rows", Path(args.robust_dir))):
        art = load_alignment(d / f"align_L{k}_id_anchored.pt")
        assert art["mode"] == "id_anchored" and float(art["alpha"]) == args.alpha, (name, art["alpha"])
        variants[name] = (art["delta_W"].float(), art["bias"].float(), None)

    # clip_norm: bounded leverage. Both rows of a pair are scaled by the same factor, so this is a weighted fit.
    s = torch.minimum(2 * ht_tr.norm(dim=1).median() / ht_tr.norm(dim=1), 2 * hb_tr.norm(dim=1).median() / hb_tr.norm(dim=1)).clamp(max=1.0)
    variants["clip_norm"] = (*weighted_fit(hb_tr, ht_tr, s**2, args.alpha, None), None)
    notes["clip_norm"] = {"rows_downweighted": int((s < 1).sum())}
    print("[fix] fitted clip_norm", flush=True)

    # huber and trimmed start from identity plus bias and never look at activation norms.
    for name in ("huber", "trimmed"):
        dw, b = torch.zeros((hb_tr.shape[1],) * 2), (ht_tr - hb_tr).double().mean(0).float()
        for it in range(args.irls_iters):
            e = diff(hb_tr, ht_tr, dw, b, None).norm(dim=1)
            if name == "huber":
                w = (3 * e.median() / e).clamp(max=1.0)
            else:
                w = (e <= torch.quantile(e, 0.995)).float()
            dw, b = weighted_fit(hb_tr, ht_tr, w, args.alpha, None)
            notes[name] = {
                "rows_downweighted": int((w < 1).sum()),
                "massive_rows_downweighted": int(((w < 1) & mass_tr).sum()),
                "share_of_weighted_objective_in_massive_rows": float((w * e**2)[mass_tr].sum() / (w * e**2).sum()),
            }
            print(f"[fix] fitted {name} iteration {it + 1}: {notes[name]}", flush=True)
        variants[name] = (dw, b, None)

    for name, r in (("project_1", 1), ("project_3", 3)):
        u = vt[:r].T.contiguous()
        variants[name] = (*weighted_fit(hb_tr, ht_tr, ones, args.alpha, u), u)
        print(f"[fix] fitted {name}", flush=True)
    del hb_tr, ht_tr

    rows = load_corpus_rows(cfg, tok, args.rollouts)
    win = str(cfg["base_wrapper"])
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    hb_c, ht_c = torch.from_numpy(rows.hb[win]), torch.from_numpy(rows.ht)

    report: dict = {"layer": k, "alpha": args.alpha, "massive_coordinate": coord,
                    "n_val": len(hb_va), "n_val_massive": int(mass_va.sum()), "variants": {}}
    ordinary = ~mass_va
    for name, (dw, b, u) in variants.items():
        uf = None if u is None else u.float()
        dv = diff(hb_va, ht_va, dw, b, u)
        tgt = project(ht_va.float(), uf)
        tau = float(dv[ordinary].norm(dim=1).median())
        energy = (dw.double() ** 2).sum(1)
        r = {
            "fve_ordinary_val": float(1 - (dv[ordinary].double() ** 2).sum() / ((tgt[ordinary] - tgt[ordinary].mean(0)).double() ** 2).sum()),
            "fve_all_val": float(1 - (dv.double() ** 2).sum() / ((tgt - tgt.mean(0)).double() ** 2).sum()),
            "tau_median_ordinary_val": tau,
            "dW_frobenius_sq": float(energy.sum()),
            "dW_share_in_largest_output_coordinate": float(energy.max() / energy.sum()),
            "dW_largest_output_coordinate": int(energy.argmax()),
            # Noise the correction puts in the massive coordinate on ordinary tokens.
            "ordinary_val_diff_std_in_massive_coordinate": float(dv[ordinary][:, coord].std()),
            "ordinary_val_share_of_diff_energy_in_massive_coordinate": float(
                ((dv[ordinary][:, coord] - dv[ordinary][:, coord].mean()) ** 2).sum() / ((dv[ordinary] - dv[ordinary].mean(0)) ** 2).sum()),
            # What a massive row looks like if it reaches the diff anyway.
            "one_model_fires_val_median_diff_norm_over_tau": float(dv[one_va].norm(dim=1).median() / tau),
            "both_fire_val_median_diff_norm_over_tau": float(dv[mt_va & mb_va].norm(dim=1).median() / tau),
            **notes.get(name, {}),
        }
        dc = center_by_role(diff(hb_c, ht_c, dw, b, u).numpy(), rows.roles)
        for role in ("user_content", "asst_rollout"):
            m = (rows.roles == role) & ok
            sp = spectrum_stats(unit(dc[m]).astype(np.float64) - unit(dc[m]).astype(np.float64).mean(0))
            r[f"arm2_target_{role}"] = {"participation_ratio": sp["participation_ratio"], "top_direction_share": sp["top1_share"],
                                        "median_norm_over_tau": float(np.median(np.linalg.norm(dc[m], axis=1)) / tau)}
        report["variants"][name] = r
        print(f"[fix] {name}: " + json.dumps(r), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
