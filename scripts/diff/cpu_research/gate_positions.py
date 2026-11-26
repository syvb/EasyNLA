"""Gate 0's SNR statistic at the other positions Arm 2 trains on.

Gate 0 scores only `asst_last`. Two thirds of Arm 2's training rows are `user_content` and
`asst_rollout` tokens. This re-scores the same benchmark pairs at the last user-content token
and at the mean over user-content tokens, from activations already in the disk cache.

    python gate_positions.py --phase0-dir P0 [--phase0-dir P0_robust] --out out.json
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

import cached_io
from alignment_variants import score
from gate_controls import cu, labelings, load_tasks

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.metrics import compute_fisher_snr
from nla.diff.wrappers import encode_aligned_pair

AUDIT = ("refusal", "eval_awareness", "sycophancy")


def base_parts(ht: torch.Tensor, hb: torch.Tensor, y: np.ndarray) -> dict:
    """How much of the class signal the Base model carries on its own at this position."""
    def direction(x: torch.Tensor) -> torch.Tensor:
        x = x.double()
        return x[y == 1].mean(0) - x[y == 0].mean(0)

    dt, db = direction(ht), direction(hb)
    return {
        "snr_h_base_over_h_target": compute_fisher_snr(hb, y) / max(compute_fisher_snr(ht, y), 1e-12),
        "between_base_over_target": float(db.pow(2).sum() / dt.pow(2).sum()),
        "cos_class_direction_base_vs_target": float(torch.nn.functional.cosine_similarity(dt, db, dim=0)),
        "snr_naive_diff_over_h_target": compute_fisher_snr(cu(ht - hb), y) / max(compute_fisher_snr(ht, y), 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", action="append", required=True)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir[0]) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    tasks = load_tasks(args.n_pairs, args.seed)
    tgt = cached_io.LazyLM(cfg["target_model"])
    base = cached_io.LazyLM(cfg["base_model"])

    # Per task and position kind: Target rows, and Base rows under each wrapper.
    acts: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for name, t in tasks.items():
        encs = []
        for txt in t["prompts"]:
            ie, _ = encode_aligned_pair(tok, txt, base_wrapper="chatml", enable_thinking=False, max_total_tokens=512)
            _, bp = encode_aligned_pair(tok, txt, base_wrapper="plain", enable_thinking=False, max_total_tokens=512)
            assert len(ie.user_content_positions) == len(bp.user_content_positions) > 0
            encs.append((ie, bp))

        def rows(model: cached_io.LazyLM, which: int) -> dict[str, torch.Tensor]:
            es = [e[which] for e in encs]
            exs = cached_io.cached_extract(model, [e.input_ids for e in es],
                                           [list(e.user_content_positions) + [e.asst_last_idx] for e in es],
                                           layer_index=k, pad_token_id=pad)
            return {
                "asst_last": torch.stack([x.hidden[-1] for x in exs]),
                "user_last": torch.stack([x.hidden[-2] for x in exs]),
                "user_mean": torch.stack([x.hidden[:-1].mean(0) for x in exs]),
            }

        acts[name] = {"target": rows(tgt, 0), "chatml": rows(base, 0), "plain": rows(base, 1)}
    tgt.free()
    base.free()

    report: dict = {"layer": k, "n_pairs": args.n_pairs, "no_base": {}, "alignments": {}}
    labs = labelings(tasks)
    # Control: h_target alone with the same centring and unit-normalisation, no Base model involved.
    for kind in ("asst_last", "user_last", "user_mean"):
        r = {}
        for lname, src, y, lkind in labs:
            ht = acts[src]["target"][kind]
            r[lname] = {"kind": lkind, **score(ht, ht, y, tasks[src]["groups"], n_boot=args.n_boot, seed=args.seed)}
        report["no_base"][kind] = r
        print(f"[positions] no_base {kind}: " + " ".join(
            f"{n}={r[n]['ratio']:.2f}[{r[n]['ci95'][0]:.2f},{r[n]['ci95'][1]:.2f}] (SNR h_t {r[n]['snr_h_target']:.3f})"
            for n in AUDIT), flush=True)
    for d in args.phase0_dir:
        art = load_alignment(Path(d) / f"align_L{k}_id_anchored.pt")
        rep_a: dict = {}
        for w in ("plain", "chatml"):
            for kind in ("asst_last", "user_last", "user_mean"):
                r: dict = {}
                for lname, src, y, lkind in labs:
                    ht = acts[src]["target"][kind]
                    diff = apply_alignment(acts[src][w][kind], ht, art)
                    r[lname] = {"kind": lkind, **score(ht, diff, y, tasks[src]["groups"], n_boot=args.n_boot, seed=args.seed),
                                **base_parts(ht, acts[src][w][kind], y)}
                r["mean_audit_ratio"] = float(np.mean([r[n]["ratio"] for n in AUDIT]))
                r["min_audit_ratio"] = float(np.min([r[n]["ratio"] for n in AUDIT]))
                r["mean_content_ratio"] = float(np.mean([v["ratio"] for v in r.values()
                                                         if isinstance(v, dict) and v["kind"] == "content_control"]))
                rep_a[f"{w}/{kind}"] = r
                print(f"[positions] {Path(d).name} {w}/{kind}: " + " ".join(
                    f"{n}={r[n]['ratio']:.2f}[{r[n]['ci95'][0]:.2f},{r[n]['ci95'][1]:.2f}]" for n in AUDIT)
                    + f" mean={r['mean_audit_ratio']:.2f} content={r['mean_content_ratio']:.2f} | h_base alone: " + " ".join(
                    f"{r[n]['snr_h_base_over_h_target']:.2f}" for n in AUDIT) + " | naive diff: " + " ".join(
                    f"{r[n]['snr_naive_diff_over_h_target']:.2f}" for n in AUDIT) + " | signal kept, spread: " + " ".join(
                    f"{r[n]['signal_kept']:.2f},{r[n]['within_spread']:.2f}" for n in AUDIT), flush=True)
        report["alignments"][Path(d).name] = rep_a

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
