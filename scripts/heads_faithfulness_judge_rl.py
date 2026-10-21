"""Blinded faithfulness of RL vs SFT head-NLA explanations, two kinds of evidence, vs reconstruction.

    python scripts/heads_faithfulness_judge_rl.py --rl-dir <checks_rl_t1> --sft-dir <checks_sft_t1> \
        --ctx <eval_ctx.parquet> --texts <texts.parquet> --out-dir <dir>

Builds on heads_faithfulness_judge.py (same judge call, cache and 1-5 rubric). Rows: a per-head stratified
sample of the eval rows (the two checks dirs must be row-aligned). Texts judged per row, blinded:
  gold   the Haiku training explanation
  sft    the SFT AV's explanation of this row's write (T=1)
  rl     the RL AV's explanation of this row's write (T=1)
  shuf   the RL AV's explanation of ANOTHER write of the same head (control)
Evidence modes:
  readouts  the write's soft-token / logit-lens readouts only: what the write promotes downstream
  mech      readouts + the current token, the preceding text and the head's top attended tokens (what the
            head read and copied). EVALUATION ONLY: attention never enters warm-start data.
Then relates each text's score to its per-row reconstruction (direction-space cosine; FVE vs the head mean).
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heads_faithfulness_judge as J  # noqa: E402
from heads_soft_token_explain import doc_freq, user_text  # noqa: E402

MECH_SYSTEM = """\
You judge how faithfully a short explanation describes what one attention head wrote at one token of a \
document. The evidence below is the only ground truth.

What the head read: the current token (in [[ ]]) with the text just before it, and the tokens this head \
attended to most, with attention weights. An attention head's write is roughly a weighted copy of \
information from the tokens it attends to, so a strongly attended token's identity or meaning is often \
carried by the write.
What the write does downstream: Readout A lists words the model makes MORE likely when the vector is placed \
into it as a stand-in for a word (distinctive part = rare across vectors, most informative; common filler = \
means little). Readout B lists output tokens the vector points at directly (noisy; strong when coherent).

Score the explanation's FAITHFULNESS as an account of this head's write, 1-5:
5  accurately captures the main thing the write carries or does (e.g. the meaning or identity of a strongly \
attended token, or the coherent theme of the readouts), nothing important invented
4  captures the main signal, with minor omissions or minor unsupported detail
3  partly related: gets a secondary or vague aspect, misses or blurs the main signal
2  mostly unrelated, with at most a superficial overlap
1  unrelated to or contradicted by the evidence

Also flag:
- unsupported_specifics: the explanation names specific things (an event, entity, topic or detail) that \
nothing in the evidence supports
- generic_template: the explanation is generic boilerplate that would fit many writes rather than this one

Judge only against the evidence. Reply with JSON only: \
{"faithful": <1-5>, "unsupported_specifics": <true|false>, "generic_template": <true|false>}"""

CONDS = ("gold", "sft", "rl", "shuf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rl-dir", required=True)
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--ctx", required=True, help="eval_ctx.parquet: tok_cur, ctx_ids, att_tok, att_w, att_pos")
    p.add_argument("--texts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-head", type=int, default=15)
    p.add_argument("--modes", default="readouts,mech")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--effort", default="low")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key-file", default="~/.anthropic_key")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gr = pq.read_table(Path(args.rl_dir) / "generations.parquet").to_pandas()
    gs = pq.read_table(Path(args.sft_dir) / "generations.parquet").to_pandas()
    assert (gr.example_idx.values == gs.example_idx.values).all(), "checks dirs are not row-aligned"
    zr, zs = np.load(Path(args.rl_dir) / "row_errors_direction.npz"), np.load(Path(args.sft_dir) / "row_errors_direction.npz")
    err = {"gold": zr["ar__gold"], "sft": zs["av__true"], "rl": zr["av__true"], "shuf": zr["av__shuf"]}
    cos = {c: 1 - e / 8192.0 for c, e in err.items()}          # both vectors at norm 64 (d = 4096)
    fve = {c: 1 - e / zr["base"] for c, e in err.items()}
    texts = {"gold": gr.gold.tolist(), "sft": gs.av_true.tolist(), "rl": gr.av_true.tolist(), "shuf": gr.av_shuf.tolist()}

    tx = pq.read_table(args.texts).to_pandas().set_index("example_idx")
    df = doc_freq(tx["avg16"].tolist())
    ctx = pq.read_table(args.ctx).to_pandas().set_index("example_idx")
    heads = gr["head"].values
    rng = random.Random(args.seed)
    rows = sorted(i for h in sorted(set(heads)) for i in rng.sample([k for k in range(len(heads)) if heads[k] == h],
                                                                     args.per_head))

    def evidence(i: int, mode: str) -> str:
        e = gr.example_idx.iloc[i]
        ro = user_text(tx.loc[e, "avg16"], tx.loc[e, "lens"], df, 0.08)
        if mode == "readouts":
            return ro
        r = ctx.loc[e]
        w, ap = np.asarray(r.att_w), np.asarray(r.att_pos)
        top = [k for k in np.argsort(-w)[:6] if w[k] >= 0.01]
        att = ", ".join(f"{tok.decode([int(r.att_tok[k])])!r} ({w[k]:.2f}{', the current token' if ap[k] == r.pos else ''})"
                        for k in top)
        before = tok.decode(list(r.ctx_ids[-41:-1]))
        return (f"Text before the token: …{before}[[{tok.decode([int(r.tok_cur)])}]]\n"
                f"Most-attended tokens (weight): {att}\n\n{ro}")

    key = Path(os.path.expanduser(args.key_file)).read_text().strip()
    modes = args.modes.split(",")
    for mode in modes:
        cache = out / f"judgments_{mode}.jsonl"
        items = [(f"{i}:{c}", J.item_text(evidence(i, mode), texts[c][i])) for i in rows for c in CONDS]
        random.Random(args.seed + 1).shuffle(items)     # interleave conditions
        done = set(J.load_cache(cache))
        todo = [it for it in items if it[0] not in done]
        print(f"[{mode}] {len(rows)} rows × {len(CONDS)} texts = {len(items)}; {len(todo)} to send", flush=True)
        if todo and not args.analyze_only:
            if mode == "mech":
                J.SYSTEM = MECH_SYSTEM
            asyncio.run(J.judge_all(todo, cache, args.model, args.effort, args.concurrency, key))

    rep: dict = {"n_rows": len(rows), "per_head": args.per_head}
    for mode in modes:
        jd = J.load_cache(out / f"judgments_{mode}.jsonl")
        ok = [i for i in rows if all(f"{i}:{c}" in jd for c in CONDS)]
        S = {c: np.array([jd[f"{i}:{c}"]["faithful"] for i in ok]) for c in CONDS}
        r = {"n": len(ok)}
        print(f"\n=== evidence: {mode} ({len(ok)} rows) ===")
        print(f"{'text':<6}{'faithful':>9}{'≥4':>6}{'specifics':>10}{'template':>9}{'cos':>7}{'FVE':>7}"
              f"{'ρ(faith,cos)':>14}{'within-head':>13}")
        for c in CONDS:
            ix = np.array(ok)
            sp = np.mean([jd[f"{i}:{c}"]["unsupported_specifics"] for i in ok])
            tp = np.mean([jd[f"{i}:{c}"]["generic_template"] for i in ok])
            rho = spearmanr(S[c], cos[c][ix])[0]
            wh = [spearmanr(S[c][heads[ix] == h], cos[c][ix][heads[ix] == h])[0] for h in sorted(set(heads[ix]))
                  if len(set(S[c][heads[ix] == h])) > 1]
            wh = float(np.nanmean(wh)) if wh else float("nan")
            r[c] = {"faithful": float(S[c].mean()), "ge4": float((S[c] >= 4).mean()), "specifics": float(sp),
                    "template": float(tp), "cos": float(cos[c][ix].mean()), "fve": float(fve[c][ix].mean()),
                    "spearman_cos": float(rho), "within_head_spearman_cos": wh}
            print(f"{c:<6}{S[c].mean():>9.2f}{(S[c] >= 4).mean():>6.0%}{sp:>10.0%}{tp:>9.0%}{cos[c][ix].mean():>7.3f}"
                  f"{fve[c][ix].mean():>7.2f}{rho:>14.3f}{wh:>13.3f}")
        ix = np.array(ok)
        print("RL text: mean cos by faithfulness score:",
              {s: (int((S['rl'] == s).sum()), round(float(cos['rl'][ix][S['rl'] == s].mean()), 3))
               for s in range(1, 6) if (S['rl'] == s).any()})
        hi_cos_lo_faith = float(((cos['rl'][ix] >= 0.6) & (S['rl'] <= 2)).mean())
        print(f"RL rows with cos ≥ 0.6 but faithfulness ≤ 2: {hi_cos_lo_faith:.0%}")
        r["rl_cos_by_score"] = {int(s): float(cos['rl'][ix][S['rl'] == s].mean()) for s in range(1, 6) if (S['rl'] == s).any()}
        r["rl_high_cos_low_faith"] = hi_cos_lo_faith
        rep[mode] = r
    (out / "faithfulness_rl.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {out / 'faithfulness_rl.json'}")


if __name__ == "__main__":
    main()
