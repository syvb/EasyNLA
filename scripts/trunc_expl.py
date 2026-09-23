"""Does the EXISTING verbalizer express the hidden-prefix memory that trunc_check.py found?

trunc_check.py showed the Qwen3-8B L24 state holds hidden-prefix memory a reader can use:
keywords decoded from the full-prefix state beat keywords from a shuffled-prefix state by
+0.031 nats/token (vocabulary-oracle ceiling +0.197 over window keywords). Here the SFT
activation verbalizer (the RL init) explains the SAME three states for the SAME 1000
held-out documents, and the same reader (last 20 tokens) scores the target's continuations:

    none           window only (no header; descriptive)
    expl_full      explanation of the full-prefix state
    expl_shuf      explanation of the window-behind-another-document's-prefix state
    expl_short     explanation of the window-only state
    expl_full_wrong  expl_full of another held-out document
PRIMARY: expl_full - expl_shuf (header-matched; only the prefix behind the window differs).
Compare its size with kw_full - kw_shuf (+0.031, linear probe) from trunc_check.

States come from trunc_check's states.npz (hidden_states[25] == the stored activation the
verbalizer was trained on; cos 0.9997). Injection is norm-matched, so only direction matters.
Selection and test split are reproduced with trunc_check.select and checked against the file."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trunc_check import BUCKETS, T_CONT, WIN, boot, log, score, select  # noqa: E402

HEADER = "[Notes on the text so far: {}]\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", required=True, help="positions parquet (its .nla_meta.yaml is the sidecar)")
    ap.add_argument("--states", required=True, help="states.npz written by trunc_check.py")
    ap.add_argument("--av", default="syvb/nanonla-qwen3-8b-L24-av", help="merged SFT verbalizer")
    ap.add_argument("--reader", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--target-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=25, help="hidden_states index of the verbalized state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--n-score-docs", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-budget", type=int, default=65536)
    ap.add_argument("--fake-av", action="store_true", help="CPU smoke: canned explanations, no verbalizer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.target_tokenizer)
    rtok = AutoTokenizer.from_pretrained(a.reader)
    probe = ["Hello world, 2026.", " The Hartford Courant reported", "naïve café — ok"]
    if any(tok(s, add_special_tokens=False)["input_ids"] != rtok(s, add_special_tokens=False)["input_ids"]
           for s in probe):
        raise SystemExit("reader and target tokenizers differ; continuation ids cannot be scored as stored")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    T = pq.read_table(a.positions, columns=["doc_id", "n_raw_tokens", "prefix_text", "activation_vector",
                                            "cont_ids", "prompt"]).to_pandas()
    keep, ids, te = select(T, tok, a.n_docs)
    Z = np.load(a.states)
    if not (np.array_equal(Z["keep"], np.array(keep)) and np.array_equal(Z["test"], te)):
        raise SystemExit("selection does not reproduce states.npz")
    ti = np.flatnonzero(te)[:a.n_score_docs]
    ACT = np.stack([np.asarray(T.activation_vector[keep[i]], dtype=np.float32) for i in ti])
    S = {k: Z[f"{k}{a.layer}"][ti].astype(np.float32) for k in ("full", "shuf", "short")}
    cos = float(np.mean(np.sum(S["full"] * ACT, 1) / (np.linalg.norm(S["full"], axis=1) * np.linalg.norm(ACT, axis=1))))
    log(f"{len(ti)} held-out docs; cos(states.npz full{a.layer}, stored activation) = {cos:.4f}")
    if cos < 0.99:
        raise SystemExit(f"layer {a.layer} is not the verbalizer's layer (cos {cos:.4f})")
    prompts = [[dict(m) for m in T.prompt[keep[i]]] for i in ti]    # chat messages, as the gate's loader gives them
    conts = [[[int(t) for t in c] for c in T.cont_ids[keep[i]]] for i in ti]
    win_ids = [ids[i][-WIN:] for i in ti]
    del T

    # ---------------- explanations from the existing SFT verbalizer ----------------
    t0 = time.time()
    EX = {}
    if a.fake_av:
        for k in S:
            EX[f"expl_{k}"] = [f"{k} state of document {r}" for r in range(len(ti))]
    else:
        from nla.config import load_nla_config
        from nla.pred.av import generate_explanations, load_av
        atok = AutoTokenizer.from_pretrained(a.av)
        cfg = load_nla_config(a.positions, atok)
        model, vref = load_av(a.av, None, device=a.device, dtype=torch.bfloat16,
                              inj_ids=(cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                       cfg.injection_right_neighbor_id))
        for k in ("full", "shuf", "short"):
            torch.manual_seed(0)                                # same sampling noise budget for each state
            rows = [{"prompt": p, "activation": v} for p, v in zip(prompts, S[k])]
            EX[f"expl_{k}"] = generate_explanations(model, atok, vref, rows, cfg.injection_char,
                                                    max_new_tokens=a.max_new_tokens, temperature=1.0,
                                                    batch_size=a.gen_batch, device=a.device)
            ok = np.mean([e is not None for e in EX[f"expl_{k}"]])
            log(f"expl_{k}: extraction {ok:.1%} ({(time.time() - t0) / 60:.1f} min)")
        del model, vref
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    order = np.random.default_rng(3).permutation(len(ti)); other = np.empty(len(ti), int)
    other[order] = order[np.r_[1:len(ti), 0]]                   # derangement: another held-out doc
    EX["expl_full_wrong"] = [EX["expl_full"][j] for j in other]
    json.dump({"test_rows": ti.tolist(), **EX}, open(os.path.join(a.out, "explanations.json"), "w"))

    # ---------------- reader scoring ----------------
    COND = ["none", "expl_full", "expl_full_wrong", "expl_shuf", "expl_short"]
    items, where = [], []
    for c in COND:
        for r in range(len(ti)):
            if c == "none":
                ctx = win_ids[r]
            elif EX[c][r] is None:                               # extraction failed: no item, NaN below
                continue
            else:
                ctx = rtok(HEADER.format(" ".join(EX[c][r].split())), add_special_tokens=False)["input_ids"] + win_ids[r]
            for b, cont in enumerate(conts[r]):
                if len(cont) == T_CONT:
                    items.append((ctx, cont)); where.append((c, r, b))
    t0 = time.time()
    if a.fake_av and a.device == "cpu":
        rd = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.float32).eval()
    else:
        rd = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16, device_map=a.device).eval()
    LP = score(rd, items, a.device, a.score_budget, pad_id)
    log(f"reader scored {len(items)} continuations in {(time.time() - t0) / 60:.1f} min")
    L = {c: np.full((len(ti), 4, T_CONT), np.nan, np.float32) for c in COND}
    for (c, r, b), lp in zip(where, LP):
        L[c][r, b] = lp
    np.savez_compressed(os.path.join(a.out, "reader_logprobs.npz"), test_rows=ti, **L)

    def per_doc(c, lo, hi):
        return np.nanmean(L[c][:, :, lo:hi], axis=(1, 2))

    res = {"cos": cos, "extraction": {k: float(np.mean([e is not None for e in v])) for k, v in EX.items()}}
    lines = [f"reader {a.reader}, last {WIN} tokens + '{HEADER.strip()}'; nats/token over {len(ti)} held-out docs",
             f"{'condition':<17}" + "".join(f" {f'tok {lo}-{hi}':>16}" for lo, hi in BUCKETS) + f" {'all 24':>16}"]
    for c in COND[1:]:
        row = f"{c:<17}"
        for lo, hi in BUCKETS + ((0, T_CONT),):
            x = per_doc(c, lo, hi) - per_doc("none", lo, hi); x = x[~np.isnan(x)]
            d, l_, h_ = boot(x); row += f" {d:+.3f}[{l_:+.2f},{h_:+.2f}]"; res[f"{c} - none tok{lo}-{hi}"] = (d, l_, h_)
        lines.append(row)
    lines.append("header-matched contrasts (all 24 tokens | tokens 16-24), docs where both explanations exist:")
    for x, y in (("expl_full", "expl_shuf"), ("expl_full", "expl_short"), ("expl_full", "expl_full_wrong"),
                 ("expl_shuf", "expl_short")):
        out = []
        for lo, hi in ((0, T_CONT), (16, 24)):
            d = per_doc(x, lo, hi) - per_doc(y, lo, hi); d = d[~np.isnan(d)]
            out.append(boot(d)); res[f"{x} - {y} tok{lo}-{hi}"] = out[-1]
        lines.append(f"  {x} - {y}: {out[0][0]:+.4f} [{out[0][1]:+.4f}, {out[0][2]:+.4f}] | "
                     f"{out[1][0]:+.4f} [{out[1][1]:+.4f}, {out[1][2]:+.4f}]  (n={len(d)})")
    lines.append("examples (full | shuffled-prefix | window-only):")
    for r in range(min(3, len(ti))):
        lines.append("  " + "\n    ".join(f"[{(EX[c][r] or '<none>')[:300]}]" for c in ("expl_full", "expl_shuf", "expl_short")))
    print("\n".join(lines), flush=True)
    json.dump({"args": vars(a), **res}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
