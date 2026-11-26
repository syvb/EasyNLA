"""Is the Taboo `<think>` mode check informative, and what does Phase 4 do with its answer?

Phase 0 and Phase 4 compare KL(Org || Inst) at `asst_last` under
`enable_thinking=False` and `enable_thinking=True` and take the lower one. This
script reports, for each Taboo LoRA:

  1. the next-token distribution at `asst_last` under both modes (is the
     thinking-mode KL trivially zero because every model emits `<think>`?),
  2. a non-degenerate comparison: KL at the first *content* token of each mode
     (no-think: `asst_last`; think: the position after `<think>\n`),
  3. how far the thinking-mode `asst_last` activation is from the no-think one
     that every arm trains on, in units of tau and as a cosine,
  4. the same for the Org - Inst diff that Benchmark 5 feeds to the verbalizer.

    python taboo_mode_check.py --phase0-dir P0 --out out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.benchmarks import NEUTRAL_QUERIES, PHASE0_TABOO_ORGANISMS, TABOO_ELICITATION_PROMPTS
from nla.diff.metrics import compute_next_token_kl
from nla.diff.wrappers import encode_chatml_dialogue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--n-neutral", type=int, default=20)
    ap.add_argument("--n-taboo-prompts", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tau = float(cfg["tau_web_median"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    think_open = tok.encode("<think>\n", add_special_tokens=False)

    neutral = list(NEUTRAL_QUERIES[: args.n_neutral])
    taboo = list(TABOO_ELICITATION_PROMPTS[: args.n_taboo_prompts])

    # (label, sequences, logit position). "think_content" is the first token the model writes
    # inside its reasoning block, the thinking-mode analogue of the no-think `asst_last`.
    sets: dict[str, tuple[list[list[int]], list[int]]] = {}
    for name, prompts in (("neutral", neutral), ("taboo", taboo)):
        for et in (False, True):
            encs = [encode_chatml_dialogue(tok, q, enable_thinking=et, max_total_tokens=512) for q in prompts]
            sets[f"{name}/{'think' if et else 'nothink'}"] = ([e.input_ids for e in encs], [e.asst_last_idx for e in encs])
            if et:
                seqs = [e.input_ids + think_open for e in encs]
                sets[f"{name}/think_content"] = (seqs, [len(s) - 1 for s in seqs])

    models = {"instruct": cached_io.LazyLM(cfg["target_model"])}
    for org in PHASE0_TABOO_ORGANISMS:
        models[org.name] = cached_io.LazyLM(cfg["target_model"], adapter=org.repo_id)

    logits: dict[str, dict[str, torch.Tensor]] = {}
    hidden: dict[str, dict[str, torch.Tensor]] = {}
    for mname, m in models.items():
        logits[mname], hidden[mname] = {}, {}
        for sname, (seqs, pos) in sets.items():
            exs = cached_io.cached_extract(m, seqs, [[p] for p in pos], layer_index=k, pad_token_id=pad, need_logits=True)
            logits[mname][sname] = torch.cat([e.logits for e in exs])
            hidden[mname][sname] = torch.cat([e.hidden for e in exs])
        m.free()
    if cached_io.RECORD_PATH:
        print("[record] taboo_mode_check requests recorded")
        return

    def top1(lg: torch.Tensor) -> dict:
        p = torch.softmax(lg.float(), dim=-1)
        ids = p.argmax(dim=-1).tolist()
        c = collections.Counter(tok.decode([i]) for i in ids)
        return {"tokens": dict(c.most_common(4)), "mean_top1_prob": float(p.max(dim=-1).values.mean())}

    report: dict = {"layer": k, "tau_web_median": tau, "organisms": {}, "instruct": {}}
    for sname in sets:
        report["instruct"][sname] = top1(logits["instruct"][sname])
    hi = hidden["instruct"]
    d_mode = hi["taboo/think"] - hi["taboo/nothink"]
    report["instruct"]["asst_last_think_vs_nothink"] = {
        "median_distance_over_tau": float(d_mode.norm(dim=1).median() / tau),
        "median_distance_over_h_norm": float((d_mode.norm(dim=1) / hi["taboo/nothink"].norm(dim=1)).median()),
        "mean_cos": float(torch.nn.functional.cosine_similarity(hi["taboo/think"], hi["taboo/nothink"], dim=1).mean()),
        "prompt_variation_think_over_nothink": float(
            (hi["taboo/think"] - hi["taboo/think"].mean(0)).pow(2).sum(1).mean()
            / (hi["taboo/nothink"] - hi["taboo/nothink"].mean(0)).pow(2).sum(1).mean()
        ),
    }

    for org in PHASE0_TABOO_ORGANISMS:
        lo, ho = logits[org.name], hidden[org.name]
        r: dict = {"repo_id": org.repo_id, "secret_word": org.secret_word}
        for sname in ("neutral/nothink", "neutral/think", "neutral/think_content",
                      "taboo/nothink", "taboo/think", "taboo/think_content"):
            r[sname] = {"kl_org_inst_nats": float(compute_next_token_kl(lo[sname], logits["instruct"][sname]).mean()),
                        **top1(lo[sname])}
        # What Phase 4 selects, and what a first-content-token comparison would select.
        r["phase4_rule_picks_thinking"] = bool(r["neutral/think"]["kl_org_inst_nats"] < r["neutral/nothink"]["kl_org_inst_nats"])
        r["first_content_token_rule_picks_thinking"] = bool(
            r["neutral/think_content"]["kl_org_inst_nats"] < r["neutral/nothink"]["kl_org_inst_nats"]
        )
        # Org - Inst diff at asst_last (Benchmark 5 `org_minus_inst`) in each mode.
        diffs = {mode: ho[f"taboo/{mode}"] - hi[f"taboo/{mode}"] for mode in ("nothink", "think")}
        r["org_minus_inst_asst_last"] = {
            mode: {
                "median_norm_over_tau": float(d.norm(dim=1).median() / tau),
                "cos_to_own_mean": float(torch.nn.functional.cosine_similarity(d, d.mean(0, keepdim=True), dim=1).mean()),
            }
            for mode, d in diffs.items()
        }
        r["org_minus_inst_asst_last"]["cos_mean_diff_think_vs_nothink"] = float(
            torch.nn.functional.cosine_similarity(diffs["think"].mean(0), diffs["nothink"].mean(0), dim=0)
        )
        report["organisms"][org.name] = r

    names = [o.name for o in PHASE0_TABOO_ORGANISMS]
    if len(names) >= 2:
        a, b = names[0], names[1]
        report["between_organisms"] = {
            mode: float(torch.nn.functional.cosine_similarity(
                (hidden[a][f"taboo/{mode}"] - hi[f"taboo/{mode}"]).mean(0),
                (hidden[b][f"taboo/{mode}"] - hi[f"taboo/{mode}"]).mean(0), dim=0))
            for mode in ("nothink", "think")
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
