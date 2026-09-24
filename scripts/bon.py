"""Best-of-N from the memory-SFT heads: is there headroom for RL near the SFT policy?

memsft.py trained two LoRA heads on the SFT verbalizer to name the hidden prefix's keywords: F from
the full-context state, S from the window-only state. Greedy F beat greedy S by +0.068 nats/token for a
20-token reader (+0.26 weighted toward tokens the hidden prefix matters for). RL would start from the
sampling policy and push toward whatever the reader rewards. This measures what it could find:

  For each held-out test document, N samples (temperature 1) from each head on its own state, plus
  greedy. The reader scores every sample on all 4 stored continuations. Cross-validated selection:
  pick the sample with the best reward on continuations {0,1}, score it on {2,3}, then the reverse,
  and average -- selection noise cannot inflate the estimate.
  Reward = mean continuation log-prob weighted by the per-token hidden-prefix gap
  (full-prefix reader minus window-only reader, clipped at 0); uniform also reported.

Pre-registered (scratchpad/memsft/BON_PLAN.md): RL is worth piloting only if
  headroom      BoN(F) - greedy(F) > 0 (gap-weighted, CI excluding 0), and
  memory        [BoN(F) - BoN(S)] - [greedy(F) - greedy(S)] > 0 (CI excluding 0):
                selection widens the full-vs-window gap, i.e. the gain is hidden-prefix content,
                not generic phrasing that the window-only head can equally find.
Supporting: hidden-prefix-only words and imminent recalled words (hidden-only words in the first 3
tokens of the EVALUATION continuations) named by the selected vs greedy samples."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trunc_check import STOP, T_CONT, WIN, boot, log, score, select  # noqa: E402

HEADER = "[Earlier in this document: {}]\n"
CW = re.compile(r"[a-z][a-z0-9']{2,}")
SPLITS = (([0, 1], [2, 3]), ([2, 3], [0, 1]))           # (select on, evaluate on)


def cw(s):
    return {w for w in CW.findall((s or "").lower()) if w not in STOP}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--states", required=True, help="trunc_check states.npz (window-only states)")
    ap.add_argument("--av", default="syvb/nanonla-qwen3-8b-L24-av")
    ap.add_argument("--f-adapter"); ap.add_argument("--s-adapter")
    ap.add_argument("--reader", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--target-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-score-docs", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--score-budget", type=int, default=65536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake-gen", action="store_true", help="CPU smoke: synthetic samples with known quality")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.target_tokenizer); rtok = AutoTokenizer.from_pretrained(a.reader)
    probe = ["Hello world, 2026.", " The Hartford Courant reported", "naïve café — ok"]
    if any(tok(s, add_special_tokens=False)["input_ids"] != rtok(s, add_special_tokens=False)["input_ids"] for s in probe):
        raise SystemExit("reader and target tokenizers differ")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = pq.read_table(a.positions, columns=["doc_id", "n_raw_tokens", "prefix_text", "activation_vector",
                                            "cont_ids", "prompt"]).to_pandas()
    keep, ids, te = select(T, tok, 5000)
    Z = np.load(a.states)
    if not (np.array_equal(Z["keep"], np.array(keep)) and np.array_equal(Z["test"], te)):
        raise SystemExit("selection does not reproduce states.npz")
    ti = np.flatnonzero(te)[:a.n_score_docs]; nd = len(ti)
    ACT = np.stack([np.asarray(T.activation_vector[keep[i]], dtype=np.float32) for i in ti])
    SH = Z[f"short{a.layer}"][ti].astype(np.float32)
    prompt = [dict(m) for m in T.prompt[keep[ti[0]]]]
    conts = [[[int(t) for t in c] for c in T.cont_ids[keep[i]]] for i in ti]
    hid = [tok.decode(ids[i][:-WIN]) for i in ti]; win = [tok.decode(ids[i][-WIN:]) for i in ti]
    win_ids = [ids[i][-WIN:] for i in ti]
    del T

    # ---------------- samples: TX[head] = list over docs of [greedy, s_1..s_N] ----------------
    TX = {}
    if a.fake_gen:                                          # quality known: sample k keeps k of the true words
        from trunc_check import extract_kw
        rng = np.random.default_rng(0)
        for head, src in (("F", hid), ("S", win)):
            TX[head] = []
            for r in range(nd):
                words = extract_kw(src[r]).split(", ")
                TX[head].append([", ".join(words[:5])] + [", ".join(rng.permutation(words)[:int(rng.integers(1, len(words) + 1))])
                                                          for _ in range(a.n)])
    else:
        from nla.config import load_nla_config
        from nla.pred.av import generate_explanations, load_av
        atok = AutoTokenizer.from_pretrained(a.av)
        cfg = load_nla_config(a.positions, atok)
        inj = (cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
        for head, adapter, X in (("F", a.f_adapter, ACT), ("S", a.s_adapter, SH)):
            t0 = time.time()
            model, vref = load_av(a.av, adapter, device=a.device, dtype=torch.bfloat16, inj_ids=inj)
            rows = [{"prompt": prompt, "activation": v} for v in X]
            greedy = generate_explanations(model, atok, vref, rows, cfg.injection_char, max_new_tokens=a.max_new_tokens,
                                           temperature=0.0, batch_size=a.gen_batch, device=a.device)
            torch.manual_seed(0)
            samp = generate_explanations(model, atok, vref, [r for r in rows for _ in range(a.n)], cfg.injection_char,
                                         max_new_tokens=a.max_new_tokens, temperature=1.0, batch_size=a.gen_batch,
                                         device=a.device)
            TX[head] = [[greedy[r]] + samp[r * a.n:(r + 1) * a.n] for r in range(nd)]
            ok = np.mean([t is not None for t in samp])
            log(f"{head}: greedy + {a.n} samples x {nd} docs, sample extraction {ok:.1%} ({(time.time() - t0) / 60:.1f} min)")
            del model, vref
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
    json.dump({"test_rows": ti.tolist(), **TX}, open(os.path.join(a.out, "samples.json"), "w"))

    # ---------------- reader: every sample x every continuation ----------------
    K = a.n + 1
    items, where = [], []
    for r in range(nd):
        for b, cont in enumerate(conts[r]):
            if len(cont) != T_CONT:
                continue
            items.append((win_ids[r], cont)); where.append(("none", r, 0, b))
            items.append((ids[ti[r]], cont)); where.append(("full", r, 0, b))
            for head in ("F", "S"):
                for k, t in enumerate(TX[head][r]):
                    if t is not None:
                        ctx = rtok(HEADER.format(" ".join(t.split())), add_special_tokens=False)["input_ids"] + win_ids[r]
                        items.append((ctx, cont)); where.append((head, r, k, b))
    dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    rd = AutoModelForCausalLM.from_pretrained(a.reader, dtype=dt, device_map=a.device).eval()
    t0 = time.time(); LP = score(rd, items, a.device, a.score_budget, pad)
    log(f"reader scored {len(items)} continuations in {(time.time() - t0) / 60:.1f} min")
    L = {"none": np.full((nd, 1, 4, T_CONT), np.nan, np.float32), "full": np.full((nd, 1, 4, T_CONT), np.nan, np.float32),
         "F": np.full((nd, K, 4, T_CONT), np.nan, np.float32), "S": np.full((nd, K, 4, T_CONT), np.nan, np.float32)}
    for (c, r, k, b), lp in zip(where, LP):
        L[c][r, k, b] = lp
    np.savez_compressed(os.path.join(a.out, "reader_logprobs.npz"), test_rows=ti, **L)

    gapw = np.clip(L["full"][:, 0] - L["none"][:, 0], 0, None)             # [nd, 4, T]
    W = {"gap-weighted": gapw, "uniform": np.ones_like(gapw)}

    def rew(x, w, br):
        """x: [nd, K, 4, T] -> weighted mean log-prob over branches br: [nd, K] (NaN if nothing scored)."""
        xb = x[:, :, br]; wb = np.broadcast_to(w[:, None, br], xb.shape)
        ok = ~np.isnan(xb); num = np.where(ok, xb * wb, 0).sum((2, 3)); den = np.where(ok, wb, 0).sum((2, 3))
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    only = [cw(h) - cw(w) for h, w in zip(hid, win)]

    def imminent(r, br):
        s = set()
        for b in br:
            s |= cw(tok.decode(conts[r][b][:3]))
        return only[r] & s

    res = {"n_docs": nd, "n": a.n}
    gap3 = np.nanmean(gapw[:, :, :3], (1, 2)); top = gap3 >= np.nanquantile(gap3, 0.8)
    lines = [f"BEST-OF-{a.n} (cross-validated: select on 2 continuations, score on the other 2; both directions averaged)"]
    for wname, w in W.items():
        E = {}
        for head in ("F", "S"):
            g, rnd, bon, best_in = [], [], [], []
            for sel, ev in SPLITS:
                rs, re_ = rew(L[head], w, sel), rew(L[head], w, ev)
                samp_s = rs[:, 1:]; samp_e = re_[:, 1:]
                pick = np.where(np.isnan(samp_s).all(1), -1, np.nanargmax(np.where(np.isnan(samp_s), -np.inf, samp_s), 1))
                bon.append(np.where(pick >= 0, samp_e[np.arange(nd), np.maximum(pick, 0)], np.nan))
                bi = np.nanmax(np.where(np.isnan(samp_e), -np.inf, samp_e), 1)
                g.append(re_[:, 0]); rnd.append(np.nanmean(samp_e, 1)); best_in.append(np.where(np.isinf(bi), np.nan, bi))
                if wname == "gap-weighted":
                    E.setdefault("picks", []).append((pick, ev))
            E[head] = {k: np.nanmean(np.stack(v), 0) for k, v in (("greedy", g), ("random", rnd), ("bon", bon), ("insample_best", best_in))}
            if wname == "gap-weighted":
                E[head]["picks"] = E.pop("picks")
        for lab, m in (("all docs", np.ones(nd, bool)), ("top-20% retrieval docs", top)):
            def c(x):
                x = x[m]; x = x[~np.isnan(x)]; return boot(x)
            F, S = E["F"], E["S"]
            rows = {"BoN(F) - greedy(F)  [headroom]": F["bon"] - F["greedy"],
                    "BoN(F) - random(F)": F["bon"] - F["random"],
                    "BoN(S) - greedy(S)": S["bon"] - S["greedy"],
                    "[BoN(F)-BoN(S)] - [greedy(F)-greedy(S)]  [memory]": (F["bon"] - S["bon"]) - (F["greedy"] - S["greedy"]),
                    "greedy(F) - greedy(S)": F["greedy"] - S["greedy"],
                    "BoN(F) - BoN(S)": F["bon"] - S["bon"],
                    "in-sample best(F) - greedy(F)  [optimistic]": F["insample_best"] - F["greedy"]}
            lines.append(f" {wname}, {lab} (n={int(m.sum())}), nats/token:")
            for k, v in rows.items():
                d = c(v); res[f"{wname} | {lab} | {k}"] = d
                lines.append(f"   {k:<52} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        if wname == "gap-weighted":
            # content of the gap-weighted picks vs greedy (imminent words from the EVALUATION continuations)
            lines.append(" content of gap-weighted picks vs greedy (per doc, averaged over both splits):")
            for head in ("F", "S"):
                hw_b, hw_g, im_b, im_g, ln_b, ln_g = [], [], [], [], [], []
                for pick, ev in E[head]["picks"]:
                    for r in range(nd):
                        if pick[r] < 0 or TX[head][r][0] is None:
                            continue
                        tb, tg = TX[head][r][1 + pick[r]], TX[head][r][0]
                        imm = imminent(r, ev)
                        hw_b.append(len(cw(tb) & only[r])); hw_g.append(len(cw(tg) & only[r]))
                        if imm:
                            im_b.append(bool(cw(tb) & imm)); im_g.append(bool(cw(tg) & imm))
                        ln_b.append(len(tb.split())); ln_g.append(len(tg.split()))
                d = boot(np.array(hw_b, float) - np.array(hw_g, float)); res[f"{head} pick - greedy hidden-only words"] = d
                lines.append(f"   {head}: hidden-only words pick {np.mean(hw_b):.2f} vs greedy {np.mean(hw_g):.2f} "
                             f"({d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]); imminent word named {np.mean(im_b):.1%} vs "
                             f"{np.mean(im_g):.1%} (n={len(im_b)}); words per text {np.mean(ln_b):.1f} vs {np.mean(ln_g):.1f}")
                res[f"{head} imminent pick"] = float(np.mean(im_b)); res[f"{head} imminent greedy"] = float(np.mean(im_g))
        if wname == "gap-weighted":
            EG = E
    lines.append("examples: hidden-prefix keywords | greedy F | gap-weighted pick F (split 1)")
    from trunc_check import extract_kw
    pk = EG["F"]["picks"][0][0]
    for r in range(min(5, nd)):
        lines.append(f"  [{extract_kw(hid[r])}]\n    [{TX['F'][r][0]}]\n    [{TX['F'][r][1 + pk[r]] if pk[r] >= 0 else None}]")
    print("\n".join(lines), flush=True)
    json.dump({"args": vars(a), **res}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
