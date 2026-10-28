"""Single-file HTML viewer for heads_doc_viewer_job.py output: every token × every head of a layer.

    python scripts/heads_doc_viewer_html.py --docs <job out dir> --out doc_viewer.html [--texts-ref <48k texts>]

Per document: a heads × tokens heatmap (RL FVE by default), the text shaded by the selected head's full attention
row at the selected token, and the selected site's explanations (RL; SFT and gold on request), scores and evidence.
Each document's data is gzip-compressed and base64-embedded, decoded in the browser only when opened.
If a document folder has effects.parquet (scripts/heads_ablation_effects.py), the page also shows each head's
counterfactual relevance at each token (ablate that head at that token: effect on the final residual, on the
next-token logits, on the actual next token, on later tokens), as heatmap colours, rankings and a top-sites list.
If it has mtry.parquet (scripts/heads_doc_viewer_mtry.py), the page also shows the matryoshka AV's explanation of
each site, its FVE whole and from its first 16 tokens (the std RL explanation's first 16 tokens, read by the
warm-up AR, alongside), with the 16-token prefix marked in both texts, and matryoshka colourings.
"""

import argparse
import ast
import base64
import gzip
import html
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_soft_token_explain import doc_freq  # noqa: E402


def toks(s: str) -> list[str]:
    return list(ast.literal_eval("[" + s + "]")) if s else []


def r3(x) -> list:
    return [None if not np.isfinite(v) else round(float(v), 3) for v in x]


def g3(x) -> list:
    """3 significant digits (effects span many orders of magnitude)."""
    return [None if not np.isfinite(v) else float(f"{float(v):.3g}") for v in x]


EFFECTS = ("kl", "resid", "dlogp", "later")


def effect_scales(frames: list) -> dict:
    """Colour ranges shared by all documents: log scale from the 5th to the 99.5th percentile for the magnitudes,
    symmetric linear ±99th percentile of |Δ log p|."""
    out = {}
    for mode in ("mean", "zero"):
        for k in EFFECTS:
            v = np.concatenate([f[f"{k}_{mode}"].to_numpy() for f in frames])
            v = v[np.isfinite(v)]
            if k == "dlogp":
                m = float(np.percentile(np.abs(v), 99))
                out[f"{k}_{mode}"] = {"kind": "div", "lo": -m, "hi": m}
            else:
                pos = v[v > 0]
                out[f"{k}_{mode}"] = {"kind": "mag", "log": True, "lo": float(np.log10(np.percentile(pos, 5))),
                                      "hi": float(np.log10(np.percentile(pos, 99.5)))}
    return out


def chat_roles(tokens: list[str]) -> list[str] | None:
    """Per-token chat role ("user" / "assistant" / "system") for chat-template text, "~"-prefixed for template tokens
    (<|im_start|>, the role name and its newline, <|im_end|> and the newline after it, the <think>…</think> block);
    None for plain text."""
    if "<|im_start|>" not in tokens:
        return None
    roles, role, i, think = [], "", 0, False
    while i < len(tokens):
        tk = tokens[i]
        if tk == "<|im_start|>":
            think = False
            role = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
            n = 3 if i + 2 < len(tokens) and tokens[i + 2] == "\n" else 2
            roles += ["~" + role] * min(n, len(tokens) - i)
            i += n
            continue
        if tk == "<think>":
            think = True
        tpl = think or tk in ("<|im_end|>", "<|endoftext|>") or (tk.strip() == "" and i > 0 and tokens[i - 1] in ("<|im_end|>", "</think>"))
        if tk == "</think>":
            think = False
        roles.append(("~" if tpl else "") + role)
        i += 1
    return roles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--texts-ref", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--common-threshold", type=float, default=0.08)
    p.add_argument("--title", default="Head NLA — every token × every head (Qwen3-8B layer 24)")
    p.add_argument("--n-kv-heads", type=int, default=None, help="GQA KV heads (default: from the base model's config)")
    p.add_argument("--no-intro", action="store_true", help="drop the explanation paragraphs under the title")
    args = p.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    n_kv = args.n_kv_heads or AutoConfig.from_pretrained(args.base_model).num_key_value_heads
    df = doc_freq(pq.read_table(args.texts_ref, columns=["avg16"]).to_pydict()["avg16"]) if args.texts_ref else {}
    docs, eff_frames = [], []
    for dd in sorted(Path(args.docs).iterdir()):
        if not ((dd / "meta.json").exists() and (dd / "sites.parquet").exists() and (dd / "attn.npz").exists()):
            continue
        meta = json.loads((dd / "meta.json").read_text())
        ids = np.load(dd / "token_ids.npy").tolist()
        s = pq.read_table(dd / "sites.parquet").to_pandas().sort_values(["head", "pos"])
        H, T = int(meta["n_heads"]), len(ids)
        assert H % n_kv == 0, f"{H} heads not divisible into {n_kv} KV groups"
        pos = sorted(s.pos.unique())
        assert len(s) == H * len(pos), f"{dd.name}: {len(s)} sites for {H} heads × {len(pos)} positions"
        attn = np.load(dd / "attn.npz")["attn"].astype(np.float32)                    # [H, T, T]
        # full attention rows, causal part only, as uint8 (√w × 255 keeps small weights visible)
        att_q = [[base64.b64encode((np.sqrt(np.clip(attn[h, t, : t + 1], 0, 1)) * 255).round().astype(np.uint8)
                                   .tobytes()).decode() for t in pos] for h in range(H)]
        a16 = [toks(x) for x in s.avg16]
        has_gold = s.gold_text.str.len() > 0          # a failed Haiku call: no gold score, not an empty "explanation"
        thr = args.common_threshold
        d = {"name": meta["name"], "domain": meta.get("domain", ""), "source": meta.get("source", ""),
             "url": meta.get("url", ""), "n_heads": H, "n_kv": n_kv, "layer": meta.get("layer", 24),
             "tokens": [tok.decode([i]) for i in ids], "pos": [int(x) for x in pos],
             # every per-site column is [head][pos-index] (row-major over heads)
             "rl_fve": r3(s.rl_fve), "sft_fve": r3(s.sft_fve), "gold_fve": r3(s.gold_rl_fve.where(has_gold)),
             "rl_cos": r3(s.rl_cos), "sft_cos": r3(s.sft_cos), "gold_cos": r3(s.gold_rl_cos.where(has_gold)),
             "gold_fve_sft": r3(s.gold_sft_fve.where(has_gold)), "first_w": r3(s.first_w), "self_w": r3(s.self_w),
             "entropy": r3(s.entropy), "v_norm": r3(s.v_norm),
             "rl": s.rl_text.tolist(), "sft": s.sft_text.tolist(), "gold": s.gold_text.tolist(),
             "gold_ok": [bool(x) for x in s.gold_ok],
             "distinct": [[t for t in x if df.get(t, 0) < thr] for x in a16],
             "filler": [[t for t in x if df.get(t, 0) >= thr] for x in a16],
             "lens": [toks(x) for x in s.lens], "att": att_q}
        if (dd / "effects.parquet").exists():
            e = pq.read_table(dd / "effects.parquet").to_pandas()
            e = s[["head", "pos"]].merge(e, on=["head", "pos"], how="left", validate="1:1")
            assert len(e) == len(s)
            for k in EFFECTS:
                for mode in ("mean", "zero"):
                    d[f"{k}_{mode}"] = g3(e[f"{k}_{mode}"])
            c0 = e[e["head"] == e["head"].min()].sort_values("pos")
            d["clean_entropy"], d["clean_logp_next"] = r3(c0.clean_entropy), r3(c0.clean_logp_next)
            eff_frames.append(e)
        mtry_mean = None
        if (dd / "mtry.parquet").exists():
            mt = pq.read_table(dd / "mtry.parquet").to_pandas()
            mt = s[["head", "pos"]].merge(mt, on=["head", "pos"], how="left", validate="1:1")
            assert len(mt) == len(s) and mt.mtry_text.notna().all(), f"{dd.name}: mtry.parquet does not cover every site"
            d["mtry"] = mt.mtry_text.tolist()
            for k in ("mtry_fve", "mtry_cos", "mtry_t16_fve", "mtry_t16_cos", "rl_t16_fve", "rl_t16_cos"):
                d[k] = r3(mt[k])

            def p16(t: str) -> int:   # UTF-16 units of t covered by its first 16 tokens (the scored prefix; JS slices in UTF-16)
                n = min(len(t), len(tok.decode(tok.encode(t, add_special_tokens=False)[:16]).rstrip()))
                return len(t[:n].encode("utf-16-le")) // 2
            d["mtry_p16"] = [p16(t) for t in d["mtry"]]
            d["rl_p16"] = [p16(t) for t in d["rl"]]
            mtry_mean = round(float(np.nanmean(mt.mtry_fve)), 3)
        resid_mean = None
        if (dd / "resid.parquet").exists():
            rs = pq.read_table(dd / "resid.parquet").to_pandas().set_index("pos").reindex(pos)
            assert rs.resid_text.notna().all(), f"{dd.name}: resid.parquet does not cover every position"
            d["resid"], d["resid_fve"], d["resid_cos"] = rs.resid_text.tolist(), r3(rs.resid_fve), r3(rs.resid_cos)
            resid_mean = round(float(1 - rs.resid_err.sum() / rs.resid_base.sum()), 3)
        roles = chat_roles(d["tokens"])
        if roles:
            d["roles"] = roles
        blob = base64.b64encode(gzip.compress(json.dumps(d, ensure_ascii=False).encode(), 9)).decode()
        docs.append({"name": meta["name"], "domain": meta.get("domain", ""), "n_tokens": T, "n_pos": len(pos),
                     "rl_fve_mean": round(float(np.nanmean(s.rl_fve)), 3), "mtry_fve_mean": mtry_mean, "chat": bool(roles), "resid_fve_mean": resid_mean,
                     "has_gold": bool(has_gold.any()), "blob": blob})
        print(f"{dd.name}: {T} tokens × {H} heads, {len(blob) / 1e6:.1f} MB embedded")
    assert docs, "no finished documents"
    data = json.dumps(docs).replace("</", "<\\/")
    escales = json.dumps(effect_scales(eff_frames) if eff_frames else {})
    template = TEMPLATE
    if args.no_intro:
        a, b = template.index("</h1>") + len("</h1>"), template.index("</header>")
        template = template[:a] + template[b:]
    Path(args.out).write_text(template.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", data)
                              .replace("__ESCALES__", escales))
    print(f"wrote {args.out}: {len(docs)} documents, {Path(args.out).stat().st_size / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:#fcfcfb;--surface-2:#f3f2ef;--card:#fff;--border:#e3e2dd;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#7d7c77;--accent:#2a78d6;--accent-soft:#e3eefb;
--user:#7a5195;--asst:#2a9d8f;--resid:#5f6b7a;--rl:#eb6834;--sft:#2a78d6;--gold:#8a6d00;--mtry:#1a8a7a;--p16:#dff1ed;--att:42,120,214;--hit:#fff1c2;--hit-ink:#5c4400;--bad:#b3261e;--neg:42,120,214;--pos:235,104,52;--mid:228,227,223}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;
--accent:#3987e5;--accent-soft:#1f3349;--user:#b48ad6;--asst:#4cc3b0;--resid:#9aa7b8;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--mtry:#4cc3b0;--p16:#173a34;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;--neg:70,140,230;--pos:226,114,63;--mid:58,58,55}}
:root[data-theme="dark"]{--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;
--accent:#3987e5;--accent-soft:#1f3349;--user:#b48ad6;--asst:#4cc3b0;--resid:#9aa7b8;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--mtry:#4cc3b0;--p16:#173a34;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;--neg:70,140,230;--pos:226,114,63;--mid:58,58,55}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,main{max-width:1500px;margin:0 auto;padding:0 20px}header{padding-top:16px}h1{font-size:19px;margin:0 0 4px}header p{margin:4px 0;color:var(--ink-2);max-width:1000px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}.bar label,.muted{color:var(--ink-2);font-size:12.5px}
button,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--border);border-radius:8px;padding:5px 10px;cursor:pointer}
button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hm-wrap{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px 10px 6px;overflow-x:auto}
canvas{display:block;image-rendering:pixelated;cursor:crosshair}
.legend{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--ink-2);margin:8px 2px 0;flex-wrap:wrap}.lg{display:inline-flex;flex-direction:column;gap:2px}
.ramp{width:260px;height:10px;border-radius:3px;display:block}.ticks{display:flex;justify-content:space-between;width:260px;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;z-index:5;max-width:320px}
.cols>*{min-width:0}.cols{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px;margin-top:14px}@media (max-width:980px){.cols{grid-template-columns:1fr}}
@media (min-width:981px){#detail{position:sticky;top:10px;align-self:start;max-height:calc(100vh - 20px);overflow-y:auto}}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.ctx{line-height:1.8;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow:auto;font-size:13.5px}
.ctx .t{border-radius:3px;cursor:pointer}.ctx .t.r-user{box-shadow:inset 0 -2px 0 var(--user)}.ctx .t.r-assistant{box-shadow:inset 0 -2px 0 var(--asst)}.ctx .t.r-system{box-shadow:inset 0 -2px 0 var(--ink-3)}
.ctx .t.tpl{color:var(--ink-3);font-family:ui-monospace,Menlo,monospace;font-size:.82em}.ctx .turn{display:inline-block;margin:6px 6px 2px 0;padding:0 7px;border-radius:999px;font:600 10.5px system-ui;letter-spacing:.04em;color:#fff}
.ctx .turn.user{background:var(--user)}.ctx .turn.assistant{background:var(--asst)}.ctx .turn.system{background:var(--ink-3)}ul.chatopt{margin:4px 0 10px;padding-left:20px;color:var(--ink-2);max-width:980px}ul.chatopt li{margin:2px 0}.ctx .cur{outline:2px solid var(--rl);font-weight:600}.ctx .after{color:var(--ink-3)}
h2{font-size:15px;margin:0 0 6px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}.meta{font-weight:400;color:var(--ink-3);font-size:12px}
.fve{font-weight:700;font-size:15px;padding:2px 9px;border-radius:999px;font-variant-numeric:tabular-nums;border:1px solid var(--border);background:var(--surface-2);color:var(--ink-2)}
.fve.p{background:#fde3d6;color:#5a2410;border-color:#f6b99c}.fve.m{background:#f6a77f;color:#3b1406;border-color:#ee8a58}.fve.h{background:#c4471a;color:#fff;border-color:#a8380f}
.ex{margin:8px 0;padding-left:9px;border-left:3px solid var(--border)}.ex.rl{border-color:var(--rl)}.ex.sft{border-color:var(--sft)}.ex.gold{border-color:var(--gold)}.ex.mtry{border-color:var(--mtry)}.ex.resid{border-color:var(--resid)}.p16{background:var(--p16);border-radius:2px}
.ex .lab{font-size:11.5px;color:var(--ink-2);margin-bottom:2px}.ex .lab b{color:var(--ink)}.ex .txt{white-space:pre-wrap;word-break:break-word}
mark.q{background:var(--hit);color:var(--hit-ink);border-radius:3px;padding:0 1px}mark.m{background:none;color:var(--bad);text-decoration:line-through}
.ev{display:grid;grid-template-columns:72px 1fr;gap:4px 8px;margin-top:8px;font-size:12.5px;border-top:1px dashed var(--border);padding-top:8px}.ev .k{color:var(--ink-2)}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chip{border:1px solid var(--border);border-radius:6px;padding:0 5px;background:var(--surface-2);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre}
.chip.f{color:var(--ink-3)}.chip .w{color:var(--ink-3);font-family:system-ui;font-size:11px;margin-left:3px}
table.heads{width:100%;border-collapse:collapse;font-size:12.5px}table.heads td,table.heads th{padding:3px 6px;border-bottom:1px solid var(--border);text-align:left}
table.heads tr{cursor:pointer}table.heads tr.sel{background:var(--accent-soft)}table.heads td.n{font-variant-numeric:tabular-nums;text-align:right}
.loading{color:var(--ink-2);padding:20px}
.ov{margin-top:14px}.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px;margin-top:8px}
.thumb{display:flex;flex-direction:column;justify-content:flex-start;border:1px solid var(--border);border-radius:10px;padding:6px 7px;cursor:pointer;background:var(--card);text-align:left;font:inherit;color:inherit}
.thumb.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}canvas.am{width:100%;aspect-ratio:1;image-rendering:pixelated;display:block;border-radius:4px;background:var(--surface-2)}
.thumb .row1{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px;margin:2px 0}.thumb .row2{font-size:11px;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.35}
.kvgroups{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,600px),1fr));gap:12px;margin-top:8px}
.kvg{border:1px solid var(--border);border-radius:12px;padding:8px;background:var(--surface-2)}.kvg.sel{border-color:var(--accent)}.kvh{font-size:12.5px;color:var(--ink-2);margin:0 2px}
.kvg .thumbs{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:6px}@media (max-width:700px){.kvg .thumbs{grid-template-columns:repeat(2,minmax(0,1fr))}}
.thumb.sib{border-style:dashed;border-color:var(--accent)}.kvt{font-weight:400;color:var(--ink-3);font-size:11px}
.chip.kvb{cursor:pointer;font-family:system-ui;padding:1px 6px;color:var(--ink)}.chip.kvb.on{border-color:var(--accent);background:var(--accent-soft)}
table.heads tr.sib td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
th.srt{cursor:pointer;white-space:nowrap}th.srt:hover{color:var(--accent)}th.srt.on{color:var(--accent)}table.heads td.rk{color:var(--ink-3);font-variant-numeric:tabular-nums;text-align:right}
table.heads tr.top3 td.rk{color:var(--ink);font-weight:700}.toplist{max-height:340px;overflow:auto;margin-top:4px}
.tag{font-size:10.5px;border:1px solid var(--border);border-radius:999px;padding:0 6px;color:var(--ink-2);background:var(--surface-2)}
</style></head><body>
<header><h1>__TITLE__</h1>
<p>Rows are the layer's 32 attention heads, columns are every token of a held-out document. Each cell is one head's write at one token, explained by the <b>AV</b> (it sees only the write vector) and reconstructed from that explanation by the <b>AR</b>. Colour = <b>FVE</b> (1 − error / error of predicting the head's mean write): red above 0, blue below, grey ≈ 0; it is negative below cos ≈ 0.5 by construction (the crossover is 0.41–0.66, depending on the site). Click a cell for its explanation, scores and evidence; the text then shows that head's full attention from the token (evidence only, never used for any explanation).</p>
<p>Every AV explanation (RL, SFT, matryoshka) is <b>one sample at temperature 1</b>, so any single cell is noisy: read patterns across cells, not one cell. Explanations the AV never closed<span class="webopt"> (under 0.1% of RL and SFT texts, about 1.5% of matryoshka texts on these web documents)</span> are scored on what it wrote: RL and SFT texts cut to their first 300 characters, matryoshka texts whole (as the matryoshka trainer reads them). <span class="goldopt"><b>Gold</b> = Claude Haiku's explanation written from the write's vocabulary readouts only (the SFT teacher, not ground truth). </span><b>Promotes</b> = the 16 tokens whose log-probability the write raises most when injected into short prompts (avg16; tokens in at least 8% of all writes' readouts are greyed as filler). <b>Lens</b> = the 8 top tokens of the write's direct unembedding.</p>
<p>Heads come in <b>GQA groups</b>: Qwen3-8B has 8 key/value heads, each shared by 4 query heads (KV0 = H0–3, KV1 = H4–7, …). Heads in a group read the same keys and values and differ only in their query and output projections, so they often share a theme. Groups are banded in the heatmap and boxed in the overview; the selected head's group-mates are marked.</p>
<p id="effintro" class="effopt"><b>Counterfactual relevance</b>: for every token and head, that head's output at that token alone was replaced (by the head's mean output over these 4 documents, or by zero) and the rest of the model re-run. <b>Logit effect</b> = KL between the clean and ablated next-token distributions at that token; <b>Δ log p</b> = how much the head raises the log-probability of the actual next token (positive = it helps); <b>residual effect</b> = relative change of the final-layer residual at that token; <b>later tokens</b> = summed KL over all later tokens (the write is read by later positions). Pick one under <i>Colour</i>, sort the <i>Heads at this token</i> table or the overview by it, or use <i>Most relevant sites</i>.</p>
<p class="chatopt"><b>Chats</b>: Qwen3-8B wrote every assistant reply itself (thinking off; sampled at T 0.7, top-p 0.8, top-k 20 from a fixed seed; kept if it ended on its own at 420–600 tokens). The heads are read on exactly the tokens the model processed, chat-template tokens included (shown muted; user turns underlined purple, assistant turns green). The last reply starts with the template's empty <code>&lt;think&gt;</code>…<code>&lt;/think&gt;</code> block; in a 2-turn chat the earlier reply is shown without it, because the template drops it from the history. Read with care:</p>
<ul class="chatopt"><li><b>Out of distribution.</b> Every AV and AR, the write normalization (each head's mean write and scale) and the FVE baseline come from web text at positions ≥ 50. Chat formatting and assistant prose are new to all of them.</li>
<li><b>Early positions score high regardless.</b> On web text, FVE is far higher near the start (RL 71% at the first token, 46% at positions 1–4, 21% past position 100) because heads sit parked on the first token there. The template tokens and the whole user turn are in that range, so a bright cell there is mostly the parked default, not the NLA reading those tokens. Compare assistant tokens at position ≥ 50 with web text instead.</li>
<li><b>“First token” is <code>&lt;|im_start|&gt;</code></b>: the sink weight, the sink-dominated tag and the overview's sink statistic all mean attention to that template token.</li>
<li><b>Four chats.</b> Per-head FVE varies by 0.1–0.3 between web documents, so per-head differences between chats and web text smaller than that are noise. No gold explanations or ablation relevance were computed for chats.</li></ul>

<p class="mtopt"><b>Matryoshka</b>: the std RL pair (step 400) after 40 AR-only warm-up steps (AV frozen) and 200 GRPO steps with the matryoshka objective. Each training explanation was cut to its first 1–120 tokens before the AR read it, so the AV learned to say the most informative thing first. Its explanation of each site is shown under the RL one. “First 16 tokens” scores only the shaded prefix, read for matryoshka by the matryoshka AR and for RL by the AR at the end of a 40-step AR-only warm-up on RL prefixes (the fair baseline). Colour by them under <i>Matryoshka</i>. Each text is read by its own AR, so “Matryoshka − RL” compares pairs. On 2,000 held-out writes, the fixed step-400 AR reads whole matryoshka texts at cos 0.605 vs 0.594 for std texts, but the matryoshka AR reads them at 0.650: most of the whole-text gap is the reader. The 16-token gap is mostly the AV: the same fixed AR reads matryoshka prefixes at 0.493 vs 0.309.</p></header>
<main>
<div class="bar" id="docs" role="group" aria-label="Document"></div>
<div class="bar"><label for="metric">Colour</label><select id="metric"><optgroup label="Explanations and attention"><option value="rl_fve">RL FVE</option><option value="sft_fve">SFT FVE</option><option value="gold_fve">Gold FVE (RL AR)</option><option value="gap">RL − SFT FVE</option><option value="first_w">attention to first token</option></optgroup><optgroup label="Matryoshka (continued RL that rewards the first tokens)" class="mtopt"><option value="mtry_fve">Matryoshka FVE</option><option value="mgap">Matryoshka − RL FVE</option><option value="mtry_t16_fve">Matryoshka FVE, first 16 tokens</option><option value="rl_t16_fve">RL FVE, first 16 tokens (warm-up AR)</option><option value="mgap16">Matryoshka − RL, first 16 tokens</option></optgroup><optgroup label="Counterfactual relevance (ablate the head at the token)" class="effopt"><option value="kl">effect on next-token logits (KL)</option><option value="dlogp">helps predict the actual next token (Δ log p)</option><option value="resid">effect on final residual</option><option value="later">effect on later tokens (Σ KL)</option></optgroup></select>
<label for="abl" class="effopt">Ablate to</label><select id="abl" class="effopt"><option value="mean">the head's mean output</option><option value="zero">zero</option></select>
<label for="zoom">Cell width</label><select id="zoom"><option value="4">4 px</option><option value="6" selected>6 px</option><option value="9">9 px</option><option value="14">14 px</option></select>
<span style="width:8px"></span><span class="muted">Show</span><button id="t-mtry" aria-pressed="true" class="mtopt">Matryoshka AV</button><button id="t-resid" aria-pressed="true" class="residopt" title="The standard (residual-stream) NLA's explanation of this token">Residual NLA</button><button id="t-sft" aria-pressed="false">SFT AV</button><button id="t-gold" aria-pressed="false" title="Claude Haiku's explanation from the write's vocabulary readouts only (the SFT teacher, not ground truth)">Gold (Haiku)</button><button id="t-hl" aria-pressed="true">Highlight quotes &amp; markup</button><button id="theme" title="Toggle light/dark">◐</button></div>
<div class="hm-wrap"><canvas id="hm" tabindex="0" aria-label="Heads by tokens heatmap; arrow keys move the selection"></canvas>
<div class="legend"><b id="lgname" style="color:var(--ink)"></b><span class="lg"><canvas class="ramp" id="ramp" width="260" height="10"></canvas><span class="ticks"><span id="lo"></span><span id="mid"></span><span id="hi"></span></span></span><span id="lgnote"></span><span class="muted" style="margin-left:12px">← → previous / next token · ↑ ↓ heads (heatmap focused) · rows banded by KV group</span><span id="stat" style="margin-left:auto"></span></div></div>
<div class="panel ov" id="topp"><h2>Most relevant sites in this document <span class="meta">(token, head) pairs ranked by a counterfactual effect; click one to select it. The heatmap marks each token's top head with a dot when a relevance colour is chosen.</span></h2>
<div class="bar"><label for="tmetric">Rank by</label><select id="tmetric"><option value="kl">effect on next-token logits (KL)</option><option value="dlogp">helps predict the actual next token (Δ log p)</option><option value="resid">effect on final residual</option><option value="later">effect on later tokens (Σ KL)</option></select>
<label for="theadf">Head</label><select id="theadf"><option value="-1">all heads</option></select><button id="t-first" aria-pressed="true" title="Ablating the first token (the attention sink) disturbs everything after it">Skip first token</button><span class="muted" id="tnote"></span></div>
<div class="toplist"><table class="heads" id="toplist"></table></div></div>
<div class="panel ov"><h2>Heads overview <span class="meta">each thumbnail is the head's attention over this document (rows = query token, top to bottom; columns = key token, left to right; darker = more attention). Click one to select that head.</span></h2>
<div class="bar"><label for="osort">Sort</label><select id="osort"><option value="h">by head (in KV groups)</option><option value="kv">by KV group, best mean RL FVE first</option><option value="sink">least first-token sink</option><option value="prev">most previous-token</option>
<option value="self">most self</option><option value="ent">most spread out (non-sink)</option><option value="dist">longest range</option><option value="focus">most focused on specific tokens</option><option value="foc">least spread out (non-sink)</option><option value="fve">highest mean RL FVE</option><optgroup label="Counterfactual relevance (mean over tokens)" class="effopt"><option value="e_kl">largest effect on next-token logits</option><option value="e_dlogp">most help predicting next tokens</option><option value="e_resid">largest effect on final residual</option><option value="e_later">largest effect on later tokens</option></optgroup></select>
<button id="t-sink" aria-pressed="true" title="Remove the first-token column and renormalize each row, so the non-sink pattern is visible">Hide first-token sink</button>
<span class="muted">No BOS: Qwen's tokenizer adds none, so the first real token is the attention sink. Thumbnail rows are scaled to their own maximum.</span></div>
<div class="thumbs" id="thumbs"></div></div>
<div class="cols"><div><div class="panel"><h2 id="ctxh">Text</h2><div class="ctx" id="ctx"></div></div>
<div class="panel" style="margin-top:14px"><h2 id="colh">Heads at this token</h2><div style="overflow-x:auto"><table class="heads" id="heads"></table></div></div></div>
<div class="panel" id="detail"></div></div>
</main><div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script id="escales" type="application/json">__ESCALES__</script>
<script>
const DOCS=JSON.parse(document.getElementById('data').textContent),CACHE={};
const S={doc:0,h:0,pi:0,metric:'rl_fve',zoom:6,show:{sft:false,gold:false,hl:true,mtry:true,resid:true},abl:'mean',hsort:'kl',tmetric:'kl',thead:-1,skipFirst:true};let D=null;
const ESC=JSON.parse(document.getElementById('escales').textContent);
const $=id=>document.getElementById(id);
function el(t,a={},...k){const e=document.createElement(t);for(const[x,v]of Object.entries(a)){if(x==='class')e.className=v;else if(x.startsWith('on'))e.addEventListener(x.slice(2),v);else e.setAttribute(x,v)}for(const c of k)if(c!=null)e.append(c);return e}
const pct=x=>x==null?'n/a':(x>=0?'+':'−')+Math.abs(Math.round(100*x))+'%',f2=x=>x==null?'n/a':(x>=0?'':'−')+Math.abs(x).toFixed(2);
const cut=(s,n)=>{const a=[...s];return a.length>n?a.slice(0,n).join('')+'…':s};
const vis=s=>s.replace(/\n/g,'⏎').replace(/ /g,'·').replace(/\t/g,'⇥')||'∅',fveCls=f=>f>=0.5?'fve h':f>=0.25?'fve m':f>0?'fve p':'fve';
async function load(i){if(CACHE[i])return CACHE[i];const b=Uint8Array.from(atob(DOCS[i].blob),c=>c.charCodeAt(0));
const txt=await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).text();return CACHE[i]=JSON.parse(txt)}
const idx=(h,pi)=>h*D.pos.length+pi;
// GQA: query head h reads KV head ⌊h / (n_heads / n_kv)⌋, shared with its group-mates.
const GS=()=>D.n_heads/D.n_kv,kvOf=h=>Math.floor(h/GS()),sibs=h=>[...Array(GS()).keys()].map(i=>kvOf(h)*GS()+i),kvName=q=>`KV${q} (H${q*GS()}–${(q+1)*GS()-1})`;
// Counterfactual relevance (heads_ablation_effects.py): <metric>_<mean|zero> arrays, absent for documents without effects.
const EFF=['kl','dlogp','resid','later'],isEff=m=>EFF.includes(m),hasEff=()=>!!(D&&D.kl_mean);
const EFFNAME={kl:'logit KL',dlogp:'Δ log p',resid:'residual',later:'later Σ KL'};
const EFFNOTE={kl:'KL(clean ‖ ablated) of the next-token distribution at this token, nats; log scale',dlogp:'log p(actual next token), clean − ablated: > 0 = the head helps predict it',resid:'‖Δ final-layer residual‖ / ‖residual‖ at this token; log scale',later:'KL summed over all later tokens, nats; log scale'};
function ev(m,h,pi){const a=D[m+'_'+S.abl];return a?a[idx(h,pi)]:null}
function fmtE(m,v){if(v==null)return 'n/a';if(m==='resid'){const p=100*v;return(p>=10?p.toFixed(0):p>=1?p.toFixed(1):p.toPrecision(1))+'%'}if(m==='dlogp')return(v>=0?'+':'−')+Math.abs(v).toFixed(Math.abs(v)>=0.1?2:3);
return v>=10?v.toFixed(0):v>=1?v.toFixed(1):v>=0.01?v.toFixed(3):v===0?'0':v.toExponential(0).replace('e-','e−')}
function rankAt(m,h,pi){const v=ev(m,h,pi);if(v==null)return null;let r=1;for(let x=0;x<D.n_heads;x++){const w=ev(m,x,pi);if(x!==h&&w!=null&&w>v)r++}return r}
// Matryoshka (heads_doc_viewer_mtry.py): mtry_* and rl_t16_* arrays, absent for documents without mtry.parquet.
const MT=['mtry_fve','mgap','mtry_t16_fve','rl_t16_fve','mgap16'],isMt=m=>MT.includes(m),hasM=()=>!!(D&&D.mtry);
const MGAP={mgap:['mtry_fve','rl_fve'],mgap16:['mtry_t16_fve','rl_t16_fve']};
function val(h,pi){const k=idx(h,pi);if(isEff(S.metric))return ev(S.metric,h,pi);if(S.metric==='gap'){const a=D.rl_fve[k],b=D.sft_fve[k];return a==null||b==null?null:a-b}
if(MGAP[S.metric]){const[x,y]=MGAP[S.metric],a=D[x][k],b=D[y][k];return a==null||b==null?null:a-b}return D[S.metric][k]}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
// Colour scales (dataviz reference palette): FVE metrics and RL−SFT are DIVERGING around 0 (blue below, grey
// midpoint, red above); attention to the first token is SEQUENTIAL blue. Stops are interpolated in OKLab and
// cached as a 256-entry lookup per scale and theme; values beyond a scale's ends are clamped.
const SCALES={fve:{kind:'div',lo:-1,hi:1,ticks:['−100%','0','+100%'],note:'FVE is at most +100% (perfect); values below −100% are shown as −100%'},
 gap:{kind:'div',lo:-2,hi:2,ticks:['−200%','0','+200%'],note:'RL − SFT, in FVE points; clamped at ±200%'},
 mgap:{kind:'div',lo:-2,hi:2,ticks:['−200%','0','+200%'],note:'matryoshka − std RL (each read by its own AR; the 16-token RL prefix by the warm-up AR), in FVE points; clamped at ±200%'},
 seq:{kind:'seq',lo:0,hi:1,ticks:['0','0.5','1'],note:'share of the head’s attention on the first token'}};
// Relevance magnitudes use a separate SEQUENTIAL violet ramp on a log scale (ranges shared by all documents in this
// file, set by the builder); Δ log p uses the diverging ramp (blue = the head hurts the actual next token, red = it helps).
const STOPS={light:{div:['#0d366b','#3987e5','#f0efec','#e66767','#9b1c1c'],seq:['#f0efec','#cde2fb','#86b6ef','#3987e5','#1c5cab','#0d366b'],mag:['#f0efec','#e2d8f6','#c0a6ec','#9673da','#6c46bd','#432485']},
 dark:{div:['#b7d3f6','#3987e5','#383835','#d24d4d','#ffb4ab'],seq:['#383835','#184f95','#256abf','#3987e5','#86b6ef','#cde2fb'],mag:['#383835','#3b2964','#593b96','#7b5ac2','#a688df','#d6c6f5']}};
function scaleOf(m){if(isEff(m)){const e=ESC[m+'_'+S.abl];if(!e)return SCALES.fve;
 if(e.log)return{kind:'mag',log:true,lo:e.lo,hi:e.hi,ticks:[fmtE(m,10**e.lo),fmtE(m,10**((e.lo+e.hi)/2)),fmtE(m,10**e.hi)],note:EFFNOTE[m]+'; ablated to '+(S.abl==='mean'?'the mean output':'zero')};
 return{kind:'div',lo:e.lo,hi:e.hi,ticks:[fmtE(m,e.lo),'0',fmtE(m,e.hi)],note:EFFNOTE[m]+'; clamped'}}
 return m==='gap'?SCALES.gap:MGAP[m]?SCALES.mgap:m==='first_w'?SCALES.seq:SCALES.fve}
function toLin(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4)}function toSrgb(c){c=c<=0.0031308?12.92*c:1.055*Math.pow(c,1/2.4)-0.055;return Math.round(255*Math.max(0,Math.min(1,c)))}
function hex2lab(h){const n=parseInt(h.slice(1),16),r=toLin(n>>16&255),g=toLin(n>>8&255),b=toLin(n&255);
const l=Math.cbrt(0.4122214708*r+0.5363325363*g+0.0514459929*b),m=Math.cbrt(0.2119034982*r+0.6806995451*g+0.1073969566*b),s=Math.cbrt(0.0883024619*r+0.2817188376*g+0.6299787005*b);
return[0.2104542553*l+0.7936177850*m-0.0040720468*s,1.9779984951*l-2.4285922050*m+0.4505937099*s,0.0259040371*l+0.7827717662*m-0.8086757660*s]}
function lab2rgb([L,A,B]){const l=(L+0.3963377774*A+0.2158037573*B)**3,m=(L-0.1055613458*A-0.0638541728*B)**3,s=(L-0.0894841775*A-1.2914855480*B)**3;
return[toSrgb(4.0767416621*l-3.3077115913*m+0.2309699292*s),toSrgb(-1.2684380046*l+2.6097574011*m-0.3413193965*s),toSrgb(-0.0041960863*l-0.7034186147*m+1.7076147010*s)]}
let LUT={};function isDark(){const n=parseInt(css('--surface').slice(1),16);return((n>>16&255)+(n>>8&255)+(n&255))/3<128}
function lut(kind){const key=kind+(isDark()?'d':'l');if(LUT[key])return LUT[key];const st=STOPS[isDark()?'dark':'light'][kind].map(hex2lab),out=[];
for(let i=0;i<256;i++){const x=i/255*(st.length-1),k=Math.min(st.length-2,Math.floor(x)),f=x-k,a=st[k],b=st[k+1];const [r,g,bl]=lab2rgb([0,1,2].map(j=>a[j]+(b[j]-a[j])*f));out.push(`rgb(${r},${g},${bl})`)}return LUT[key]=out}
function color(v,m=S.metric){if(v==null)return null;const sc=scaleOf(m),x=sc.log?(v>0?Math.log10(v):sc.lo):v,t=Math.max(0,Math.min(1,(x-sc.lo)/(sc.hi-sc.lo)));return lut(sc.kind)[Math.round(t*255)]}
const LAB=46,TOP=4;
function drawHM(){LUT={};const c=$('hm'),H=D.n_heads,P=D.pos.length,cw=S.zoom,ch=11,dpr=window.devicePixelRatio||1;
c.width=(LAB+P*cw)*dpr;c.height=(TOP+H*ch)*dpr;c.style.width=(LAB+P*cw)+'px';c.style.height=(TOP+H*ch)+'px';const g=c.getContext('2d');g.scale(dpr,dpr);
g.fillStyle=css('--card');g.fillRect(0,0,LAB+P*cw,TOP+H*ch);g.font='10px system-ui';g.fillStyle=css('--ink-2');g.textBaseline='middle';
const G=GS(),sib=new Set(sibs(S.h));for(let q=0;q<D.n_kv;q++){const y=TOP+q*G*ch;g.fillStyle=css(q%2?'--surface-2':'--card');g.fillRect(0,y,LAB-4,G*ch);g.fillStyle=css('--ink-3');g.font='9px system-ui';g.fillText('kv'+q,1,y+G*ch/2)}
g.font='10px system-ui';for(let h=0;h<H;h++){if(sib.has(h)){g.fillStyle=css('--accent');g.fillRect(LAB-6,TOP+h*ch,2,ch-1)}g.fillStyle=css(h===S.h?'--ink':'--ink-2');g.fillText('H'+h,18,TOP+h*ch+ch/2);for(let pi=0;pi<P;pi++){const col=color(val(h,pi)),x=LAB+pi*cw,y=TOP+h*ch;if(col){g.fillStyle=col;g.fillRect(x,y,cw-(cw>4?1:0),ch-1)}else{g.fillStyle=css('--ink-3');g.fillRect(x+cw/2-1,y+ch/2-1,2,2)}}g.fillStyle=css('--ink-2')}
g.fillStyle=css('--ink-2');for(let q=1;q<D.n_kv;q++)g.fillRect(0,TOP+q*G*ch-1,LAB+P*cw,1);   // group separators sit in the row gap
g.strokeStyle=css('--ink');g.lineWidth=2;g.strokeRect(LAB+S.pi*cw-1,TOP+S.h*ch-1,cw+1,ch+1);
g.strokeStyle=css('--accent');g.lineWidth=1;g.strokeRect(LAB+S.pi*cw-.5,TOP-2,cw,H*ch+3);
if(isEff(S.metric)&&hasEff()){g.fillStyle=css('--ink');for(let pi=0;pi<P;pi++){let b=-1,bv=-Infinity;for(let h=0;h<H;h++){const v=val(h,pi);if(v!=null&&v>bv){bv=v;b=h}}if(b>=0)g.fillRect(LAB+pi*cw+cw/2-1,TOP+b*ch+ch/2-1.5,2,2)}}
const sc=scaleOf(S.metric),rc=$('ramp'),r=rc.getContext('2d'),RW=rc.width,LT=lut(sc.kind);for(let x=0;x<RW;x++){r.fillStyle=LT[Math.round(255*x/(RW-1))];r.fillRect(x,0,1,rc.height)}
$('lo').textContent=sc.ticks[0];$('mid').textContent=sc.ticks[1];$('hi').textContent=sc.ticks[2];$('lgname').textContent=$('metric').selectedOptions[0].text;$('lgnote').textContent=sc.note;
const vs=[];for(let h=0;h<H;h++)for(let pi=0;pi<P;pi++){const v=D.rl_fve[idx(h,pi)];if(v!=null)vs.push(v)}const m=vs.reduce((a,b)=>a+b,0)/vs.length;
$('stat').textContent=`${D.pos.length} tokens × ${H} heads · RL FVE mean ${pct(m)} · ${Math.round(100*vs.filter(v=>v>0).length/vs.length)}% of cells > 0`;
if(hasM()){const mv=D.mtry_fve.filter(v=>v!=null);$('stat').textContent+=` · matryoshka FVE mean ${pct(mv.reduce((a,b)=>a+b,0)/mv.length)}`}
if(isEff(S.metric)&&hasEff()){const cnt=new Array(H).fill(0);for(let pi=0;pi<P;pi++){let b=-1,bv=-Infinity;for(let h=0;h<H;h++){const v=val(h,pi);if(v!=null&&v>bv){bv=v;b=h}}if(b>=0)cnt[b]++}
const tops=[...cnt.keys()].sort((a,b)=>cnt[b]-cnt[a]).slice(0,4).filter(h=>cnt[h]);$('stat').textContent=`top head per token (dots): `+tops.map(h=>`H${h} ×${cnt[h]}`).join(', ')}}
function cellAt(e){const r=$('hm').getBoundingClientRect(),x=e.clientX-r.left-LAB,y=e.clientY-r.top-TOP,pi=Math.floor(x/S.zoom),h=Math.floor(y/11);
return(pi>=0&&pi<D.pos.length&&h>=0&&h<D.n_heads)?{h,pi}:null}
function attRow(h,pi){const s=atob(D.att[h][pi]);const a=new Float32Array(s.length);for(let i=0;i<s.length;i++){const q=s.charCodeAt(i)/255;a[i]=q*q}return a}
function renderCtx(){const P=D.pos[S.pi],a=attRow(S.h,S.pi),box=$('ctx');box.replaceChildren();const end=Math.min(D.tokens.length,P+30);
let mx=0;for(let i=0;i<a.length;i++)if(i!==0&&a[i]>mx)mx=a[i];
for(let i=0;i<end;i++){const s=el('span',{class:'t'});s.textContent=D.tokens[i];if(i===P)s.classList.add('cur');else if(i>P)s.classList.add('after');
if(D.roles){const r=D.roles[i],tpl=r.startsWith('~'),role=tpl?r.slice(1):r;if(tpl)s.classList.add('tpl');else if(role&&!D.tokens[i].includes('\n'))s.classList.add('r-'+role);
if(D.tokens[i]==='<|im_start|>'&&role)box.append(el('span',{class:'turn '+role},role))}
if(i<=P){const w=a[i];if(w>0.003){s.style.background=`rgba(var(--att),${(0.12+0.88*Math.min(1,w/Math.max(mx,0.05))).toFixed(2)})`}s.title=`attention ${w.toFixed(3)}`}
const pi=D.pos.indexOf(i);if(pi>=0)s.onclick=()=>{S.pi=pi;update()};box.append(s)}
if(end<D.tokens.length)box.append(el('span',{class:'after'},' …'));
$('ctxh').replaceChildren('Text',el('span',{class:'meta'},`token #${P} “${vis(D.tokens[P])}” · shading = H${S.h}'s attention from it (first token ${D.first_w[idx(S.h,S.pi)]?.toFixed(2)})`));
const cur=box.querySelector('.cur');if(cur){const top=cur.offsetTop-box.offsetTop;if(top<box.scrollTop||top>box.scrollTop+box.clientHeight-30)box.scrollTop=Math.max(0,top-box.clientHeight/2)}}
function hl(text){const f=document.createDocumentFragment();if(!S.show.hl){f.append(text);return f}const re=/(<\/?[A-Za-z][\w-]*>|\[end\])|([“"][^”"\n]{1,80}[”"])/g;let l=0,m;
while((m=re.exec(text))){f.append(text.slice(l,m.index));f.append(el('mark',{class:m[1]?'m':'q'},m[0]));l=re.lastIndex}f.append(text.slice(l));return f}
function pre(text,n){if(n==null||!n)return hl(text);return[el('span',{class:'p16',title:'first 16 tokens: the prefix scored as “first 16 tokens”'},hl(text.slice(0,n))),hl(text.slice(n))]}
function renderDetail(){const k=idx(S.h,S.pi),P=D.pos[S.pi],a=attRow(S.h,S.pi),top=[...a.keys()].sort((x,y)=>a[y]-a[x]).slice(0,10).filter(i=>a[i]>=0.005);
$('detail').replaceChildren(...[el('h2',{},`L${D.layer}H${S.h} at #${P} “${vis(D.tokens[P])}”`,hasM()?el('span',{class:fveCls(D.mtry_fve[k]??-9)},'Matryoshka FVE '+pct(D.mtry_fve[k])):null,el('span',{class:fveCls(D.rl_fve[k]??-9)},'RL FVE '+pct(D.rl_fve[k])),
el('span',{class:'meta'},`${kvName(kvOf(S.h))} · cos ${f2(D.rl_cos[k])} · ‖v‖ ${D.v_norm[k]?.toFixed(2)} · first-token attn ${D.first_w[k]?.toFixed(2)} · entropy ${D.entropy[k]?.toFixed(2)}`)),
hasM()&&S.show.mtry?el('div',{class:'ex mtry'},el('div',{class:'lab'},el('b',{},'Matryoshka AV'),`  FVE ${pct(D.mtry_fve[k])} · cos ${f2(D.mtry_cos[k])} · first 16 tokens ${pct(D.mtry_t16_fve[k])}`),el('div',{class:'txt'},...[].concat(pre(D.mtry[k],D.mtry_p16[k])))):null,
el('div',{class:'ex rl'},el('div',{class:'lab'},el('b',{},'RL AV'),`  FVE ${pct(D.rl_fve[k])} · cos ${f2(D.rl_cos[k])}`+(hasM()?` · first 16 tokens ${pct(D.rl_t16_fve[k])} (warm-up AR)`:'')),el('div',{class:'txt'},...[].concat(hasM()&&S.show.mtry?pre(D.rl[k],D.rl_p16[k]):hl(D.rl[k])))),
S.show.sft?el('div',{class:'ex sft'},el('div',{class:'lab'},el('b',{},'SFT AV'),`  FVE ${pct(D.sft_fve[k])} · cos ${f2(D.sft_cos[k])}`),el('div',{class:'txt'},hl(D.sft[k]))):null,
D.resid&&S.show.resid?el('div',{class:'ex resid'},el('div',{class:'lab'},el('b',{},'Residual stream at this token'),`  standard NLA (nanoNLA L24, every head the same) · FVE ${pct(D.resid_fve[S.pi])} · cos ${f2(D.resid_cos[S.pi])}`),el('div',{class:'txt'},hl(D.resid[S.pi]))):null,
S.show.gold&&D.gold[k]?el('div',{class:'ex gold'},el('div',{class:'lab'},el('b',{},'Gold (Haiku)'),`  FVE ${pct(D.gold_fve[k])} via RL AR · ${pct(D.gold_fve_sft[k])} via SFT AR${D.gold_ok[k]?'':' · breaks rules'}`),el('div',{class:'txt'},hl(D.gold[k]))):null,
el('div',{class:'ev'},el('div',{class:'k'},'KV group'),el('div',{class:'chips'},el('span',{class:'w',style:'align-self:center;color:var(--ink-3);font-size:11.5px'},kvName(kvOf(S.h))+' · same keys & values:'),...sibs(S.h).map(s=>{const q=idx(s,S.pi);
return el('button',{class:'chip kvb'+(s===S.h?' on':''),title:`H${s} at this token: RL FVE · first-token attention`,onclick:()=>{S.h=s;update()}},'H'+s,el('span',{class:'w'},pct(D.rl_fve[q])+' · sink '+(D.first_w[q]??0).toFixed(2)))})),
hasEff()?el('div',{class:'k'},'Relevance'):null,hasEff()?el('div',{class:'chips'},...EFF.map(m=>el('span',{class:'chip',style:'font-family:system-ui',title:EFFNOTE[m]+' (rank among the '+D.n_heads+' heads at this token)'},EFFNAME[m]+' '+fmtE(m,ev(m,S.h,S.pi)),el('span',{class:'w'},'#'+rankAt(m,S.h,S.pi)))),
el('span',{class:'w',style:'align-self:center;color:var(--ink-3);font-size:11.5px'},`ablated to ${S.abl==='mean'?'mean output':'zero'} · clean next-token entropy ${D.clean_entropy[D.pos[S.pi]]?.toFixed(2)} nats`)):null,
el('div',{class:'k'},'Reads'),el('div',{class:'chips'},...top.map(i=>el('span',{class:'chip'},vis(D.tokens[i]),el('span',{class:'w'},(i===P?'self ':i===0?'first ':'')+a[i].toFixed(2))))),
el('div',{class:'k',title:'avg16: the 16 tokens whose log-probability the write raises most when injected into short prompts; greyed = common filler (in ≥ 8% of all writes’ readouts)'},'Promotes'),el('div',{class:'chips'},...D.distinct[k].map(t=>el('span',{class:'chip'},vis(t))),...D.filler[k].map(t=>el('span',{class:'chip f',title:'common filler'},vis(t)))),
el('div',{class:'k',title:'top 8 tokens of the write’s direct unembedding'},'Lens'),el('div',{class:'chips'},...D.lens[k].map(t=>el('span',{class:'chip'},vis(t)))))].filter(Boolean))}
function renderHeads(){const E=hasEff(),key=E?S.hsort:'rl_fve',get=(m,h)=>m==='rl_fve'?D.rl_fve[idx(h,S.pi)]:ev(m,h,S.pi);
const rows=[...Array(D.n_heads).keys()].map(h=>({h,v:get(key,h)})).sort((a,b)=>(b.v??-1e9)-(a.v??-1e9));
$('colh').replaceChildren('Heads at this token',el('span',{class:'meta'},(E?'click a column to rank by it; ':'')+'sorted by '+(key==='rl_fve'?'RL FVE':EFFNAME[key])+'; click a row to select; blue edge = the selected head\'s KV group'));
const th=(m,lab,tip)=>E?el('th',{class:'srt'+(key===m?' on':''),title:tip||('rank by '+lab),onclick:()=>{S.hsort=m;renderHeads()}},lab+(key===m?' ▾':'')):el('th',{},lab);
$('heads').replaceChildren(el('tr',{},el('th',{},'#'),el('th',{},'head'),el('th',{},'KV'),...(E?EFF.map(m=>th(m,EFFNAME[m],EFFNOTE[m])):[]),th('rl_fve','RL FVE'),el('th',{},'sink'),el('th',{},'RL explanation')),
...rows.map(({h},i)=>{const k=idx(h,S.pi);return el('tr',{class:(h===S.h?'sel':kvOf(h)===kvOf(S.h)?'sib':'')+(i<3&&E?' top3':''),onclick:()=>{S.h=h;update()}},el('td',{class:'rk'},String(i+1)),el('td',{},'H'+h),el('td',{class:'n'},'kv'+kvOf(h)),
...(E?EFF.map(m=>el('td',{class:'n'},fmtE(m,ev(m,h,S.pi)))):[]),el('td',{class:'n'},pct(D.rl_fve[k])),
el('td',{class:'n'},(D.first_w[k]??0).toFixed(2)),el('td',{},cut(D.rl[k].split('\n')[0],80)))}))}
// Most relevant (token, head) sites of the document by one effect.
function renderTop(){const box=$('topp');box.style.display=hasEff()?'':'none';if(!hasEff())return;const m=S.tmetric,P=D.pos.length,c=[];
for(let h=0;h<D.n_heads;h++){if(S.thead>=0&&h!==S.thead)continue;for(let pi=0;pi<P;pi++){if(S.skipFirst&&D.pos[pi]===0)continue;const v=ev(m,h,pi);if(v!=null)c.push([v,h,pi])}}
c.sort((a,b)=>b[0]-a[0]);const top=c.slice(0,60);$('tnote').textContent=`${c.length.toLocaleString()} sites · ablated to ${S.abl==='mean'?'mean output':'zero'}`;
$('toplist').replaceChildren(el('tr',{},el('th',{},'#'),el('th',{},'token'),el('th',{},'head'),...EFF.map(x=>el('th',{class:x===m?'srt on':''},EFFNAME[x])),el('th',{},'rank at token'),el('th',{},'RL FVE'),el('th',{},'RL explanation')),
...top.map(([v,h,pi],i)=>{const k=idx(h,pi),P0=D.pos[pi];return el('tr',{class:h===S.h&&pi===S.pi?'sel':'','data-h':String(h),'data-pi':String(pi),onclick:()=>{S.h=h;S.pi=pi;update();keepVisible()}},el('td',{class:'rk'},String(i+1)),
el('td',{},`#${P0} “${vis(D.tokens[P0])}” → “${vis(D.tokens[P0+1]??'')}”`),el('td',{},'H'+h),...EFF.map(x=>el('td',{class:'n'},fmtE(x,ev(x,h,pi)))),el('td',{class:'n'},'#'+rankAt(m,h,pi)),el('td',{class:'n'},pct(D.rl_fve[k])),
el('td',{},cut(D.rl[k].split('\n')[0],70)))}))}
let LUTN={};function lutNum(kind){const key=kind+(isDark()?'d':'l');if(LUTN[key])return LUTN[key];LUTN[key]=lut(kind).map(c=>c.match(/\d+/g).map(Number));return LUTN[key]}
let HS=null;S.osort='h';S.dropSink=true;
// Per-head attention summary over the document (query tokens t ≥ 1). "Non-sink" = attention renormalized over
// keys 1..t. Key focus = share of all non-sink attention landing on the head's 5 most-attended key tokens.
function headStats(){const H=D.n_heads,P=D.pos.length,T=D.tokens.length,out=[];for(let h=0;h<H;h++){let sink=0,self=0,prev=0,ent=0,dist=0,n=0;const rows=[],col=new Float64Array(T);
for(let pi=0;pi<P;pi++){const t=D.pos[pi],a=attRow(h,pi);rows.push(a);if(t<1)continue;n++;sink+=a[0];self+=a[t];prev+=a[t-1];let sm=0;for(let j=1;j<=t;j++)sm+=a[j];
if(sm>1e-6){let e=0,dd=0;for(let j=1;j<=t;j++){const q=a[j]/sm;col[j]+=q;if(q>1e-9){e-=q*Math.log(q);dd+=q*(t-j)}}ent+=e;dist+=dd}}
const f=D.rl_fve.slice(h*P,(h+1)*P).filter(x=>x!=null),tot=col.reduce((x,y)=>x+y,0)||1,top=[...col.keys()].sort((x,y)=>col[y]-col[x]).slice(0,5);
const r={h,sink:sink/n,self:self/n,prev:prev/n,ent:ent/n,dist:dist/n,focus:top.reduce((x,j)=>x+col[j],0)/tot,topKeys:top.filter(j=>col[j]>0),fve:f.reduce((x,y)=>x+y,0)/f.length,rows,eff:{}};
if(D.kl_mean)for(const m of EFF)for(const ab of['mean','zero']){const a=D[m+'_'+ab].slice(h*P,(h+1)*P).filter((x,pi)=>x!=null&&D.pos[pi]>0);r.eff[m+'_'+ab]=a.reduce((x,y)=>x+y,0)/a.length}
const ns=Math.max(1e-6,1-r.sink),sn=r.self/ns,pv=r.prev/ns;
r.tag=r.sink>=0.85?'sink-dominated':sn>=0.3?'self / current token':pv>=0.4?'previous token':r.dist<=20?'local':r.focus>=0.25?'focused on specific tokens':r.ent<=2.5?'focused':'diffuse / long-range';out.push(r)}return out}
function drawThumb(cv,st){const P=D.pos.length,img=new ImageData(P,P),L=lutNum('seq');for(let pi=0;pi<P;pi++){const a=st.rows[pi],t=D.pos[pi];
let mx=0;for(let j=S.dropSink?1:0;j<a.length;j++)mx=Math.max(mx,a[j]);mx=Math.max(mx,1e-6);   // each row scaled to its own max
for(let j=0;j<P;j++){const o=(pi*P+j)*4;let v=0;if(j<=t&&!(S.dropSink&&j===0))v=a[j]/mx;const c=L[Math.round(255*Math.sqrt(Math.min(1,v)))];img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=j<=t?255:0}}
cv.width=P;cv.height=P;cv.getContext('2d').putImageData(img,0,0)}
function thumbBtn(r,kv){const cv=el('canvas',{class:'am','aria-hidden':'true'});drawThumb(cv,r);return el('button',{class:'thumb'+(r.h===S.h?' sel':''),'data-h':String(r.h),'aria-label':`Head ${r.h}: ${r.tag}`,onclick:()=>{S.h=r.h;update()}},cv,
el('div',{class:'row1'},el('b',{},'H'+r.h,kv?el('span',{class:'kvt'},' · kv'+kvOf(r.h)):null),el('span',{class:'tag'},r.tag)),
el('div',{class:'row2',title:'sink = attention on the first token; prev/self = on the previous/current token; spread = entropy of non-sink attention; dist = mean tokens back; focus = share of non-sink attention on the 5 most-attended tokens'},
`sink ${(100*r.sink).toFixed(0)}% · prev ${(100*r.prev).toFixed(0)}% · self ${(100*r.self).toFixed(0)}%`,el('br'),`spread ${r.ent.toFixed(2)} · dist ${r.dist.toFixed(0)} · focus ${(100*r.focus).toFixed(0)}%`,el('br'),
`top: ${r.topKeys.slice(0,3).map(j=>vis(D.tokens[j]).trim()).join(', ')}`,el('br'),`mean RL FVE ${pct(r.fve)}`,
...(r.eff.kl_mean!=null?[el('br'),`KL ${fmtE('kl',r.eff['kl_'+S.abl])} · Δlogp ${fmtE('dlogp',r.eff['dlogp_'+S.abl])} · resid ${fmtE('resid',r.eff['resid_'+S.abl])}`]:[])))}
function renderThumbs(){const box=$('thumbs');box.replaceChildren();
if(S.osort==='h'||S.osort==='kv'){box.className='kvgroups';const G=GS(),gs=[...Array(D.n_kv).keys()].map(q=>({q,hs:HS.slice(q*G,(q+1)*G)}));
for(const x of gs){x.f=x.hs.reduce((a,r)=>a+r.fve,0)/G;x.sink=x.hs.reduce((a,r)=>a+r.sink,0)/G}if(S.osort==='kv')gs.sort((a,b)=>b.f-a.f);
for(const x of gs)box.append(el('div',{class:'kvg','data-kv':String(x.q)},el('div',{class:'kvh'},el('b',{style:'color:var(--ink)'},kvName(x.q)),` · shared keys & values · mean RL FVE ${pct(x.f)} · mean sink ${(100*x.sink).toFixed(0)}%`),
el('div',{class:'thumbs'},...x.hs.map(r=>thumbBtn(r,false)))));selThumbs();return}
box.className='thumbs';const key=S.osort.startsWith('e_')?(r=>-(r.eff[S.osort.slice(2)+'_'+S.abl]??-1e9)):{h:r=>r.h,sink:r=>r.sink,prev:r=>-r.prev,self:r=>-r.self,ent:r=>-r.ent,foc:r=>r.ent,dist:r=>-r.dist,focus:r=>-r.focus,fve:r=>-r.fve}[S.osort];
for(const r of [...HS].sort((a,b)=>key(a)-key(b)))box.append(thumbBtn(r,true));selThumbs()}
function selThumbs(){const q=kvOf(S.h);document.querySelectorAll('.thumb').forEach(b=>{const h=+b.dataset.h;b.classList.toggle('sel',h===S.h);b.classList.toggle('sib',h!==S.h&&kvOf(h)===q)});
document.querySelectorAll('.kvg').forEach(b=>b.classList.toggle('sel',+b.dataset.kv===q))}
function update(){drawHM();renderCtx();renderDetail();renderHeads();selThumbs();if(hasEff())document.querySelectorAll('#toplist tr[data-h]').forEach(r=>r.classList.toggle('sel',+r.dataset.h===S.h&&+r.dataset.pi===S.pi))}
function renderDocs(){$('docs').replaceChildren(el('label',{},'Document'),...DOCS.map((d,i)=>el('button',{'aria-pressed':String(i===S.doc),onclick:()=>openDoc(i)},`${d.name.replace(/_/g,' ')} · ${d.n_pos} tok · RL FVE ${pct(d.rl_fve_mean)}`+(d.mtry_fve_mean!=null?` · matryoshka ${pct(d.mtry_fve_mean)}`:'')+(d.resid_fve_mean!=null?` · residual NLA ${pct(d.resid_fve_mean)}`:''))))}
async function openDoc(i){S.doc=i;renderDocs();$('detail').replaceChildren(el('div',{class:'loading'},'Loading…'));D=await load(i);
let best=[0,0,-9];for(let h=0;h<D.n_heads;h++)for(let pi=0;pi<D.pos.length;pi++){const v=D.rl_fve[idx(h,pi)];if(v!=null&&v>best[2]&&D.first_w[idx(h,pi)]<0.9)best=[h,pi,v]}
S.h=best[0];S.pi=best[1];HS=headStats();document.querySelectorAll('.effopt').forEach(e=>e.style.display=hasEff()?'':'none');
if(!hasEff()&&isEff(S.metric)){S.metric='rl_fve';$('metric').value='rl_fve'}
document.querySelectorAll('.mtopt').forEach(e=>e.style.display=hasM()?'':'none');document.querySelectorAll('.residopt').forEach(e=>e.style.display=D.resid?'':'none');
const hasG=D.gold.some(Boolean);$('t-gold').style.display=hasG?'':'none';$('metric').querySelector('option[value=gold_fve]').style.display=hasG?'':'none';if(!hasG&&S.metric==='gold_fve'){S.metric='rl_fve';$('metric').value='rl_fve'}
document.querySelectorAll('.chatopt').forEach(e=>e.style.display=DOCS.some(d=>d.chat)?'':'none');document.querySelectorAll('.webopt').forEach(e=>e.style.display=DOCS.some(d=>d.chat)?'none':'');
document.querySelectorAll('.goldopt').forEach(e=>e.style.display=DOCS.some(d=>d.has_gold)?'':'none');if(!hasM()&&isMt(S.metric)){S.metric='rl_fve';$('metric').value='rl_fve'}$('theadf').replaceChildren(el('option',{value:'-1'},'all heads'),...[...Array(D.n_heads).keys()].map(h=>el('option',{value:String(h)},'H'+h)));$('theadf').value=String(S.thead);
renderThumbs();update();renderTop()}
const tip=$('tip');$('hm').addEventListener('mousemove',e=>{const c=cellAt(e);if(!c){tip.style.display='none';return}const k=idx(c.h,c.pi);
tip.replaceChildren(el('b',{},`H${c.h} · kv${kvOf(c.h)} · #${D.pos[c.pi]} “${vis(D.tokens[D.pos[c.pi]])}”`),el('br'),`RL FVE ${pct(D.rl_fve[k])} · SFT ${pct(D.sft_fve[k])}`+(hasM()?` · matryoshka ${pct(D.mtry_fve[k])} (16 tok ${pct(D.mtry_t16_fve[k])} vs RL ${pct(D.rl_t16_fve[k])})`:'')+` · first-token ${(D.first_w[k]??0).toFixed(2)}`,el('br'),
...(hasEff()?[EFF.map(m=>`${EFFNAME[m]} ${fmtE(m,ev(m,c.h,c.pi))}`).join(' · '),el('br')]:[]),cut(D.rl[k].split('\n')[0],140));
tip.style.display='block';tip.style.left=Math.min(innerWidth-330,e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'});
$('hm').addEventListener('mouseleave',()=>tip.style.display='none');
$('hm').addEventListener('click',e=>{$('hm').focus({preventScroll:true});const c=cellAt(e);if(c){S.h=c.h;S.pi=c.pi;update()}});
// ← / → anywhere on the page: previous / next token for the current head. ↑ / ↓ switch heads only while the
// heatmap has focus, so they don't take over page scrolling. Ignored while typing or with modifier keys.
function keepVisible(){const w=$('hm').parentElement,x=LAB+S.pi*S.zoom;if(x<w.scrollLeft+LAB||x>w.scrollLeft+w.clientWidth-30)w.scrollLeft=Math.max(0,x-w.clientWidth/2)}
document.addEventListener('keydown',e=>{if(!D||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;const t=e.target.tagName;if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA')return;
const lr={ArrowLeft:-1,ArrowRight:1}[e.key],ud=e.target===$('hm')?{ArrowUp:-1,ArrowDown:1}[e.key]:undefined;if(lr==null&&ud==null)return;e.preventDefault();
if(lr!=null)S.pi=Math.max(0,Math.min(D.pos.length-1,S.pi+lr));if(ud!=null)S.h=Math.max(0,Math.min(D.n_heads-1,S.h+ud));update();keepVisible()});
$('metric').onchange=e=>{S.metric=e.target.value;drawHM()};$('abl').onchange=e=>{S.abl=e.target.value;renderThumbs();update();renderTop()};
$('tmetric').onchange=e=>{S.tmetric=e.target.value;renderTop()};$('theadf').onchange=e=>{S.thead=+e.target.value;renderTop()};$('t-first').onclick=e=>{S.skipFirst=!S.skipFirst;e.currentTarget.setAttribute('aria-pressed',String(S.skipFirst));renderTop()};$('osort').onchange=e=>{S.osort=e.target.value;renderThumbs()};$('t-sink').onclick=e=>{S.dropSink=!S.dropSink;e.currentTarget.setAttribute('aria-pressed',String(S.dropSink));renderThumbs()};$('zoom').onchange=e=>{S.zoom=+e.target.value;drawHM()};
for(const k of['sft','gold','hl','mtry','resid'])$('t-'+k).onclick=e=>{S.show[k]=!S.show[k];e.currentTarget.setAttribute('aria-pressed',String(S.show[k]));renderDetail()};
$('theme').onclick=()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';LUTN={};drawHM();renderThumbs()};
openDoc(0);
</script></body></html>"""

if __name__ == "__main__":
    main()
