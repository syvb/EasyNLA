"""Do the benchmark prompts contain massive activations, and do they move the user-token-mean results?

Scans the user-content tokens and `asst_last` of every gate prompt in the Target and in Base under both
wrappers, then rescoring every task's user-token mean with those tokens left out of the mean.

    python massive_in_benchmarks.py --phase0-dir P0 [--phase0-dir P0_robust] --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from alignment_variants import score
from gate_controls import load_tasks

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.wrappers import encode_aligned_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", action="append", required=True)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--mult", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir[0]) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    tgt, base = cached_io.LazyLM(cfg["target_model"]), cached_io.LazyLM(cfg["base_model"])
    arts = {Path(d).name: load_alignment(Path(d) / f"align_L{k}_id_anchored.pt") for d in args.phase0_dir}
    report: dict = {"layer": k, "tasks": {}}
    for name, t in load_tasks(args.n_pairs, 0).items():
        encs = []
        for txt in t["prompts"]:
            ie, _ = encode_aligned_pair(tok, txt, base_wrapper="chatml", enable_thinking=False, max_total_tokens=512)
            _, bp = encode_aligned_pair(tok, txt, base_wrapper="plain", enable_thinking=False, max_total_tokens=512)
            encs.append((ie, bp))

        def rows(model: cached_io.LazyLM, which: int) -> list[torch.Tensor]:
            es = [e[which] for e in encs]  # noqa: B023
            return [x.hidden for x in cached_io.cached_extract(
                model, [e.input_ids for e in es], [list(e.user_content_positions) + [e.asst_last_idx] for e in es],
                layer_index=k, pad_token_id=pad)]

        per = {"target": rows(tgt, 0), "chatml": rows(base, 0), "plain": rows(base, 1)}
        med = {key: float(torch.cat(v).norm(dim=1).median()) for key, v in per.items()}
        big = {key: [h.norm(dim=1) > args.mult * med[key] for h in v] for key, v in per.items()}
        hits = {key: [(pi, "asst_last" if j == len(m) - 1 else f"user token {j + 1} of {len(m) - 1}",
                       tok.decode([encs[pi][0 if key != "plain" else 1].input_ids[
                           (list(encs[pi][0 if key != "plain" else 1].user_content_positions)
                            + [encs[pi][0 if key != "plain" else 1].asst_last_idx])[j]]]))
                      for pi, m in enumerate(v) for j in torch.where(m)[0].tolist()] for key, v in big.items()}
        r: dict = {"n_prompts": len(t["prompts"]), "n_user_tokens": sum(len(h) - 1 for h in per["target"]), "massive": hits, "user_mean": {}}
        for masked in (False, True):
            means = {key: [] for key in per}
            for pi in range(len(encs)):
                keep = torch.ones(len(per["target"][pi]) - 1, dtype=torch.bool)
                if masked:
                    for key in per:
                        keep &= ~big[key][pi][:-1]
                for key in per:
                    means[key].append(per[key][pi][:-1][keep].mean(0))
            m = {key: torch.stack(v) for key, v in means.items()}
            tag = "massive_tokens_left_out" if masked else "as_scored"
            r["user_mean"][tag] = {
                f"{a}/{w}": {x: sc[x] for x in ("ratio", "ci95")}
                for a, art in arts.items() for w in ("plain", "chatml")
                for sc in [score(m["target"], apply_alignment(m[w], m["target"], art), t["y"], t["groups"], n_boot=args.n_boot, seed=0)]
            }
        report["tasks"][name] = r
        print(f"[bench] {name}: {r['n_prompts']} prompts, {r['n_user_tokens']} user tokens, massive {hits}", flush=True)
        for tag, v in r["user_mean"].items():
            print(f"[bench]   {tag}: " + " ".join(f"{key}={x['ratio']:.2f}" for key, x in v.items()), flush=True)
    tgt.free()
    base.free()
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
