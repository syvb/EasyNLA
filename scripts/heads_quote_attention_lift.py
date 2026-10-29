"""How much more does a head attend to the tokens its explanation quotes? For every quote that appears in the text
(strict matching, as in heads_quote_topk_shuffle.py), the attention lift: the head's attention on the quoted
positions as a share of its attention to non-sink tokens, divided by their share of the context (|M| / t). 1 = an
average token; 10 = ten times the attention of an average token.

    python scripts/heads_quote_attention_lift.py --doc hf:l24/std_rl/doc_viewer/01_journalism_ocoee_homicide ... \\
        --mtry-root hf:l24/mtry/doc_viewer --out-dir docs [--src rl_text]

Controls, as in heads_quote_topk_shuffle.py: shuffled explanations (another site of the same head in the same text,
--shuffles derangements; only their quotes that appear in this site's text count), and the other 31 heads' lift on
the same quoted tokens. Attention is evaluation evidence only.

Writes <out-dir>/head_nla_quote_attention_lift_<src>.{json,png}.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_quote_attention import QRE  # noqa: E402
from heads_quote_topk_shuffle import DOC_FILES, Matcher, resolve  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
NAMES = {"rl_text": "Standard NLA", "mtry_text": "Matryoshka NLA"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", action="append", required=True)
    p.add_argument("--mtry-root", default=None)
    p.add_argument("--src", default="rl_text", choices=list(NAMES))
    p.add_argument("--shuffles", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    rng = np.random.default_rng(a.seed)

    true, shuf, oth = [], [], []                                      # lifts
    for spec in a.doc:
        d = resolve(spec, DOC_FILES, ("mtry.parquet",))
        ids = np.load(d / "token_ids.npy").tolist()
        toks = [tok.decode([i]) for i in ids]
        attn = np.load(d / "attn.npz")["attn"].astype(np.float32)    # [H, T, T]
        nonsink = np.stack([attn[:, t, 1:t + 1].sum(1) if t else np.zeros(attn.shape[0]) for t in range(attn.shape[1])], 1)
        s = pq.read_table(d / "sites.parquet", columns=["pos", "head", "rl_text"]).to_pandas()
        if a.src == "mtry_text":
            mp = d / "mtry.parquet"
            if not mp.exists():
                mp = resolve(f"{a.mtry_root.rstrip('/')}/{d.name}", ("mtry.parquet",)) / "mtry.parquet"
            s = s.merge(pq.read_table(mp, columns=["pos", "head", "mtry_text"]).to_pandas(), on=["pos", "head"],
                        validate="1:1")
        s = s.sort_values(["head", "pos"]).reset_index(drop=True)
        quotes = [[m.group(1) for m in QRE.finditer(x or "")] for x in s[a.src]]
        mt = Matcher(toks, "strict")
        pos, head = s["pos"].to_numpy(), s["head"].to_numpy()

        def lift(h, t: int, M: np.ndarray) -> np.ndarray:
            """Attention lift of head(s) h at token t on positions M (non-sink, ≤ t)."""
            return attn[h, t][..., M].sum(-1) / np.maximum(nonsink[h, t], 1e-12) * t / len(M)

        perms = [np.arange(len(s))]
        for _ in range(a.shuffles):
            perm = np.arange(len(s))
            for g in s.groupby("head").indices.values():
                g = np.asarray(g)
                while True:
                    q = rng.permutation(g)
                    if not (q == g).any():
                        break
                perm[g] = q
            perms.append(perm)
        for pi, perm in enumerate(perms):
            for i in range(len(s)):
                t, h = int(pos[i]), int(head[i])
                if t < 2:
                    continue
                for q in quotes[perm[i]]:
                    M = mt.at(q, t)
                    M = M[M > 0]
                    if not len(M):
                        continue
                    if pi == 0:
                        true.append(float(lift(h, t, M)))
                        oth.extend(np.delete(lift(slice(None), t, M), h).tolist())
                    else:
                        shuf.append(float(lift(h, t, M)))
        print(f"{d.name}: {len(true)} true quotes in text so far", flush=True)

    true, shuf, oth = (np.array(x) for x in (true, shuf, oth))
    stats = {}
    for name, x in (("this head, actual quotes", true), ("other 31 heads, same quotes", oth),
                    ("this head, shuffled quotes", shuf)):
        stats[name] = {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()),
                       "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
                       "share_above_10x": float((x > 10).mean()), "share_above_1x": float((x > 1).mean())}
    out = Path(a.out_dir)
    tag = a.src.replace("_text", "")
    (out / f"head_nla_quote_attention_lift_{tag}.json").write_text(json.dumps(
        {"texts": [Path(x).name for x in a.doc], "src": a.src, "matching": "strict", "shuffles": a.shuffles,
         "stats": stats}, indent=1) + "\n")
    for k, v in stats.items():
        print(f"{k:30s} n {v['n']:7d}  median {v['median']:.2f}×  mean {v['mean']:.2f}×  IQR {v['q25']:.2f}–{v['q75']:.2f}  "
              f">10× {v['share_above_10x']:.1%}  >1× {v['share_above_1x']:.1%}")

    # ---- figure: share of quotes whose token gets at least x times an average token's attention ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    xs = np.logspace(-2, 3, 400)
    series = [("this head, shuffled quotes", "Shuffled explanation (same head, same text)", "#1baf7a", "-"),
              ("other 31 heads, same quotes", "Other 31 heads, same quoted tokens", "#eb6834", "-"),
              ("this head, actual quotes", "This head, its explanation's quotes", "#2a78d6", "-")]
    data = {"this head, actual quotes": true, "other 31 heads, same quotes": oth, "this head, shuffled quotes": shuf}
    for key, lab, col, ls in series:
        x = np.sort(data[key])
        ax.plot(xs, 100 * (1 - np.searchsorted(x, xs, side="left") / len(x)), color=col, ls=ls, lw=2, label=lab,
                zorder=3 if key == "this head, actual quotes" else 2)
    for key, *_ in series:                                                     # value labels at 10x
        v = 100 * stats[key]["share_above_10x"]
        main = key == "this head, actual quotes"
        ax.annotate(f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", (10, v), xytext=(6, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5 if main else 8.5, color=INK if main else INK2,
                    fontweight="bold" if main else "normal")
        ax.plot([10], [v], "o", color=dict((k, c) for k, _, c, _ in series)[key], ms=5.5, mec=SURF, mew=1.2, zorder=4)
    ax.axvline(1, color=INK2, lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.set_xlabel("attention on the quoted token, relative to an average token in the text", color=INK2, fontsize=8.8)
    ax.set_ylabel("quotes with at least that much", color=INK2, fontsize=8.8)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="upper right", frameon=False, fontsize=8.2, labelcolor=INK)
    ax.set_title(f"{NAMES[a.src]}: attention on quoted tokens", loc="left", fontsize=11.5, color=INK,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"head_nla_quote_attention_lift_{tag}.png", facecolor=SURF)
    print(f"wrote {out}/head_nla_quote_attention_lift_{tag}.{{json,png}}")


if __name__ == "__main__":
    main()
