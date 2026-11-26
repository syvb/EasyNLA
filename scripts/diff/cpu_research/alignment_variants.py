"""Can a different linear alignment rescue Gate 0's auditing-SNR criterion?

Phase 0 fits the Base -> Target map on raw web text and applies it at chat positions. This
script fits the same two ridge families on chat activations instead (a Phase 2 corpus of
real conversations with on-policy rollouts), once per Base wrapper, and re-scores the Gate 0
statistic on the benchmark pairs. The benchmark prompts are not in the corpus.

Variants, per Base wrapper:
  no_base          h_target alone, centred and unit-normalised: what the diff has to beat
  web_idanc        Phase 0's identity-anchored artifact (the planned pipeline)
  chat_idanc_all   identity-anchored ridge fitted on all sampled active-role corpus rows
  chat_std_all     standard ridge, same rows
  chat_idanc_last  identity-anchored ridge fitted on `asst_last` rows only (the gate's position)
  chat_std_last    standard ridge, `asst_last` rows only

For each it reports the within-role FVE on held-out conversations (how much of the Target the
map explains at chat positions) and, on the benchmark pairs, the SNR ratio
unit(diff - mean) / raw h_target with a pair-bootstrap CI, split into how much class signal
the diff keeps and how much within-class spread it has, both relative to h_target.

    python alignment_variants.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from corpus_rows import load_corpus_rows, massive_rows
from gate_controls import asst_last_activations, cu, few_shot_acc, labelings, load_tasks

from nla.diff.alignment import apply_alignment, cv_select_alignment, load_alignment
from nla.diff.metrics import compute_fisher_snr
from nla.diff.wrappers import ACTIVE_ROLES

AUDIT = ("refusal", "eval_awareness", "sycophancy")


def class_parts(x: torch.Tensor, y: np.ndarray) -> tuple[float, float]:
    """Squared distance between class means, and mean within-class squared spread."""
    x = x.double()
    m1, m0 = x[y == 1].mean(0), x[y == 0].mean(0)
    within = 0.5 * ((x[y == 1] - m1).pow(2).sum(1).mean() + (x[y == 0] - m0).pow(2).sum(1).mean())
    return float((m1 - m0).pow(2).sum()), float(within)


def score(ht: torch.Tensor, d: torch.Tensor, y: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int) -> dict:
    def ratio(idx: np.ndarray) -> float:
        return compute_fisher_snr(cu(d[idx]), y[idx]) / max(compute_fisher_snr(ht[idx], y[idx]), 1e-12)

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        if min((y[idx] == 0).sum(), (y[idx] == 1).sum()) >= 2:
            boots.append(ratio(idx))
    bt, wt = class_parts(ht, y)
    bd, wd = class_parts(d, y)
    return {
        "snr_h_target": compute_fisher_snr(ht, y),
        "snr_rep": compute_fisher_snr(cu(d), y),
        "ratio": ratio(np.arange(len(y))),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else [float("nan")] * 2,
        "signal_kept": bd / max(bt, 1e-12),
        "within_spread": wd / max(wt, 1e-12),
        "few_shot_k4": few_shot_acc(cu(d), y, groups, seed=seed),
    }


def within_role_fve(ht: np.ndarray, resid: np.ndarray) -> float:
    t = ht.astype(np.float64)
    r = resid.astype(np.float64)
    return float(1.0 - ((r - r.mean(0)) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-convs", type=int, default=0, help="use only the first N conversations (smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="with NLA_RECORD=/dev/null: run end to end on random activations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])

    rows = load_corpus_rows(cfg, tok, args.rollouts, seed=args.seed, max_convs=args.max_convs)
    tasks = load_tasks(args.n_pairs, args.seed)
    if args.smoke:
        for t in tasks.values():  # keep the smoke run small: 12 groups per task
            keep = np.isin(t["groups"], np.unique(t["groups"])[:12])
            t["prompts"] = [p for p, kp in zip(t["prompts"], keep) if kp]
            t["meta"] = [m for m, kp in zip(t["meta"], keep) if kp]
            t["y"], t["groups"] = t["y"][keep], t["groups"][keep]
    g_ht, g_hb = asst_last_activations(tasks, tok, target_model=cfg["target_model"], base_model=cfg["base_model"],
                                       layer=k)
    if cached_io.RECORD_PATH and not args.smoke:
        print("[record] alignment_variants requests recorded")
        return

    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    rng = np.random.default_rng(args.seed)
    convs = np.unique(rows.conv)
    held = np.isin(rows.conv, rng.choice(convs, size=max(1, int(len(convs) * args.holdout_frac)), replace=False))
    train = ok & ~held
    last = rows.roles == "asst_last"
    print(f"[variants] corpus rows: {len(ok)} ({int((~ok).sum())} massive), train {int(train.sum())}, "
          f"held-out {int((ok & held).sum())}; asst_last train {int((train & last).sum())}", flush=True)

    labs = labelings(tasks)
    report: dict = {"layer": k, "n_pairs": args.n_pairs, "corpus_rows": len(ok), "wrappers": {}}
    for w in ("chatml", "plain"):
        hb_t = torch.from_numpy(rows.hb[w])
        ht_t = torch.from_numpy(rows.ht)
        arts: dict[str, dict | None] = {"no_base": None, "web_idanc": load_alignment(p0 / f"align_L{k}_id_anchored.pt")}
        for name, mode, sel in (
            ("chat_idanc_all", "id_anchored", train),
            ("chat_std_all", "standard_ridge", train),
            ("chat_idanc_last", "id_anchored", train & last),
            ("chat_std_last", "standard_ridge", train & last),
        ):
            t0 = time.time()
            idx = torch.from_numpy(sel)
            arts[name] = cv_select_alignment(hb_t[idx], ht_t[idx], rows.conv[sel].tolist(), mode=mode,
                                             n_folds=args.n_folds, seed=args.seed, layer_index=k)
            print(f"[variants] {w}/{name}: alpha {arts[name]['alpha']:g}, CV FVE {arts[name]['best_cv_fve']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        rep_w: dict = {}
        for name, art in arts.items():
            r: dict = {}
            if art is not None:
                r["alpha"] = float(art["alpha"])
                r["cv_fve"] = art.get("best_cv_fve")
                resid = apply_alignment(hb_t, ht_t, art).numpy()
                r["heldout_within_role_fve"] = {
                    role: within_role_fve(rows.ht[m], resid[m])
                    for role in ACTIVE_ROLES if (m := ok & held & (rows.roles == role)).sum() >= 10
                }
            r["tasks"] = {}
            for lname, src, y, kind in labs:
                if min((y == 0).sum(), (y == 1).sum()) < 2:
                    continue
                d = g_ht[src] if art is None else apply_alignment(g_hb[w][src], g_ht[src], art)
                r["tasks"][lname] = {"kind": kind, **score(g_ht[src], d, y, tasks[src]["groups"],
                                                           n_boot=args.n_boot, seed=args.seed)}
            audit = [r["tasks"][n]["ratio"] for n in AUDIT]
            r["mean_audit_ratio"] = float(np.mean(audit))
            r["min_audit_ratio"] = float(np.min(audit))
            r["mean_content_ratio"] = float(np.mean([v["ratio"] for v in r["tasks"].values()
                                                     if v["kind"] == "content_control"]))
            rep_w[name] = r
        report["wrappers"][w] = rep_w

        print(f"\n=== Base wrapper = {w}: SNR ratio unit(diff - mean) / raw h_target [signal kept, within spread] ===")
        print(f"{'variant':16s} " + " ".join(f"{n:>24s}" for n in AUDIT) + f" {'mean':>6s} {'content':>8s}  held-out FVE by role")
        for name, r in rep_w.items():
            cells = [f"{r['tasks'][n]['ratio']:.2f} [{r['tasks'][n]['signal_kept']:.2f}, {r['tasks'][n]['within_spread']:.2f}]"
                     .rjust(24) for n in AUDIT]
            fve = " ".join(f"{role}={v:.2f}" for role, v in r.get("heldout_within_role_fve", {}).items())
            print(f"{name:16s} " + " ".join(cells) + f" {r['mean_audit_ratio']:6.2f} {r['mean_content_ratio']:8.2f}  {fve}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[variants] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
