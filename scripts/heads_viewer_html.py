"""Single-file HTML viewer: every head of a layer at a few positions of held-out texts, with explanations
(RL AV, SFT AV, Haiku gold), reconstruction scores, and mechanistic evidence (attention, readouts).

    python scripts/heads_viewer_html.py --results <dir> --out viewer.html [--texts-ref <big texts.parquet>]

<dir> is the viewer job's results: viewer_selection.parquet, viewer_docs.parquet, viewer_attn.parquet,
viewer_readouts/{texts,explain}.parquet, checks_rl_t1/ and checks_sft_t1/ (heads_sft_checks.py, T=1).
--texts-ref sets the avg16 common-filler table (as the explainer and judge use); default: the viewer's own texts.
"""

import argparse
import ast
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--texts-ref", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--common-threshold", type=float, default=0.08)
    p.add_argument("--title", default="Head NLA viewer — Qwen3-8B layer 24")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    R = Path(args.results)

    sel = pq.read_table(R / "viewer_selection.parquet").to_pandas().set_index("example_idx")
    texts = pq.read_table(R / "viewer_readouts" / "texts.parquet").to_pandas().set_index("example_idx")
    ref = pq.read_table(args.texts_ref).to_pandas() if args.texts_ref else texts.reset_index()
    df = doc_freq(ref["avg16"].tolist())
    gold = {}
    if (R / "viewer_readouts" / "explain.parquet").exists():
        e = pq.read_table(R / "viewer_readouts" / "explain.parquet").to_pydict()
        gold = dict(zip(e["example_idx"], e["explain"], strict=True))
    att = pq.read_table(R / "viewer_attn.parquet").to_pandas().set_index(["doc_id", "pos", "head"])
    docs = pq.read_table(R / "viewer_docs.parquet").to_pandas()

    def checks(name: str):
        d = R / name
        g = pq.read_table(d / "generations.parquet").to_pandas()
        z = np.load(d / "row_errors_direction.npz")
        cos = {k: 1 - z[k] / 8192.0 for k in ("av__true", "ar__gold")}
        fve = {k: 1 - z[k] / z["base"] for k in ("av__true", "ar__gold")}
        return {int(ex): {"text": g.av_true.iloc[i], "cos": float(cos["av__true"][i]), "fve": float(fve["av__true"][i]),
                          "gold_cos": float(cos["ar__gold"][i]), "gold_fve": float(fve["ar__gold"][i])}
                for i, ex in enumerate(g.example_idx)}
    rl, sft = checks("checks_rl_t1"), checks("checks_sft_t1")

    D = {"docs": [], "layer": args.layer}
    for _, d in docs.iterrows():
        ids = list(d.token_ids)
        rows = sel[sel.doc_id == d.doc_id]
        positions = []
        for pos in sorted(rows.pos.unique()):
            heads = []
            for ex, r in rows[rows.pos == pos].sort_values("head").iterrows():
                t = texts.loc[ex]
                a16 = toks(t["avg16"])
                a = att.loc[(d.doc_id, int(pos), int(r["head"]))]
                aw, ap = np.asarray(a.att_w), np.asarray(a.att_pos)
                keep = [k for k in np.argsort(-aw) if aw[k] >= 0.005][:10]
                heads.append({
                    "h": int(r["head"]), "ex": int(ex),
                    "rl": rl.get(int(ex)), "sft": sft.get(int(ex)), "gold": gold.get(int(ex), ""),
                    "distinct": [x for x in a16 if df.get(x, 0) < args.common_threshold],
                    "filler": [x for x in a16 if df.get(x, 0) >= args.common_threshold],
                    "lens": toks(t["lens"]),
                    "att": [[int(ap[k]), round(float(aw[k]), 4)] for k in keep],
                    "first_w": round(float(a.first_w), 4), "self_w": round(float(a.self_w), 4),
                    "entropy": round(float(a.entropy), 3),
                })
            positions.append({"pos": int(pos), "heads": heads})
        last = max(pp["pos"] for pp in positions)
        show = ids[: min(len(ids), last + 25)]
        D["docs"].append({"doc_id": d.doc_id, "domain": d.get("domain", "") or "", "url": d.get("url", "") or "",
                          "tokens": [tok.decode([i]) for i in show], "positions": positions})
    data = json.dumps(D, ensure_ascii=False).replace("</", "<\\/")
    Path(args.out).write_text(TEMPLATE.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", data))
    n = sum(len(pp["heads"]) for d in D["docs"] for pp in d["positions"])
    print(f"wrote {args.out}: {len(D['docs'])} texts, {sum(len(d['positions']) for d in D['docs'])} positions, {n} head-sites, "
          f"{Path(args.out).stat().st_size / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:#fcfcfb;--surface-2:#f3f2ef;--card:#ffffff;--border:#e3e2dd;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#7d7c77;
--accent:#2a78d6;--accent-soft:#e3eefb;--rl:#eb6834;--sft:#2a78d6;--gold:#8a6d00;--att:42,120,214;--hit:#fff1c2;--hit-ink:#5c4400;--bad:#b3261e;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;
--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;--accent:#3987e5;--accent-soft:#1f3349;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;}}
:root[data-theme="dark"]{--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;
--accent:#3987e5;--accent-soft:#1f3349;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 20px 8px;max-width:1400px;margin:0 auto}h1{font-size:19px;margin:0 0 4px}header p{margin:4px 0;color:var(--ink-2);max-width:980px}
main{max-width:1400px;margin:0 auto;padding:0 20px 40px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
.bar label{color:var(--ink-2);font-size:12.5px}
button,select,input{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--border);border-radius:8px;padding:5px 10px}
button{cursor:pointer}button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.docs button{max-width:260px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left}
.ctx{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:8px 0 14px;line-height:1.75;
white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto;font-size:13.5px}
.ctx .t{border-radius:3px}.ctx .cur{outline:2px solid var(--rl);background:color-mix(in srgb,var(--rl) 22%,transparent);font-weight:600}
.ctx .after{color:var(--ink-3)}.ctx .pos{border-bottom:2px dotted var(--ink-3);cursor:pointer}
.legend{font-size:12px;color:var(--ink-2);margin-top:-6px;margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.card.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.card h3{margin:0 0 6px;font-size:14.5px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.card h3 .meta{font-weight:400;color:var(--ink-3);font-size:12px}
.ex{margin:7px 0;padding-left:9px;border-left:3px solid var(--border)}
.ex.rl{border-color:var(--rl)}.ex.sft{border-color:var(--sft)}.ex.gold{border-color:var(--gold)}
.ex .lab{font-size:11.5px;color:var(--ink-2);display:flex;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.ex .lab b{color:var(--ink)}.ex .txt{white-space:pre-wrap;word-break:break-word}
.ex .txt.clamp{display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden}
.more{border:none;background:none;color:var(--accent);padding:0;font-size:12px}
mark.q{background:var(--hit);color:var(--hit-ink);border-radius:3px;padding:0 1px}
mark.m{background:none;color:var(--bad);text-decoration:line-through}
.ev{display:grid;grid-template-columns:70px 1fr;gap:3px 8px;margin-top:8px;font-size:12.5px;border-top:1px dashed var(--border);padding-top:8px}
.ev .k{color:var(--ink-2)}.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{border:1px solid var(--border);border-radius:6px;padding:0 5px;background:var(--surface-2);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre}
.chip.f{color:var(--ink-3)}.chip .w{color:var(--ink-3);font-family:system-ui;font-size:11px;margin-left:3px}
.score{font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:baseline}
.fve{font-weight:700;font-size:15px;padding:2px 9px;border-radius:999px;font-variant-numeric:tabular-nums;border:1px solid var(--border);background:var(--surface-2);color:var(--ink-2)}
.fve.p{background:#fde3d6;color:#5a2410;border-color:#f6b99c}.fve.m{background:#f6a77f;color:#3b1406;border-color:#ee8a58}.fve.h{background:#c4471a;color:#fff;border-color:#a8380f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .fve.p{background:#4a2617;color:#ffd9c7;border-color:#6b3620}:root:not([data-theme="light"]) .fve.m{background:#9a4220;color:#fff;border-color:#b5532b}:root:not([data-theme="light"]) .fve.h{background:#e2723f;color:#1a0a03;border-color:#f08a5b}}
:root[data-theme="dark"] .fve.p{background:#4a2617;color:#ffd9c7;border-color:#6b3620}:root[data-theme="dark"] .fve.m{background:#9a4220;color:#fff;border-color:#b5532b}:root[data-theme="dark"] .fve.h{background:#e2723f;color:#1a0a03;border-color:#f08a5b}
.card.lowf{opacity:.72}.sum{color:var(--ink-2);font-size:12.5px;margin:2px 0 10px}.sum b{color:var(--ink)}
.notes{color:var(--ink-2);font-size:12.5px;max-width:980px;margin-top:22px}
.notes h2{font-size:14px;color:var(--ink)}
@media (max-width:520px){.grid{grid-template-columns:1fr}header,main{padding-left:16px;padding-right:16px}}
</style></head><body>
<header>
<h1>__TITLE__</h1>
<p>Each card is one attention head's write at the selected token of a held-out document (never used in training). The <b>AV</b> sees only that write vector and explains it; the <b>AR</b> reads the explanation (plus the head's name) and reconstructs the write's direction. <b>cos</b> = cosine between the reconstruction and the true write (0 ≈ no information); <b>FVE</b> = 1 − error / error of predicting the head's mean write (negative below cos ≈ 0.5 by construction).</p>
<p>Evidence (shown, not used to make any explanation): <b>Reads</b> = the head's most-attended tokens and weights (hover a card to shade them in the text); <b>Promotes</b> = what the write makes more likely when injected as a word (distinctive, then common filler) and the tokens it points at directly (lens).</p>
</header>
<main>
<div class="bar docs" id="docs" role="group" aria-label="Held-out text"></div>
<div class="bar" id="positions" role="group" aria-label="Token position"></div>
<div class="ctx" id="ctx" aria-live="polite"></div>
<div class="legend"><span class="sw" style="background:var(--rl)"></span>current token ·
<span class="sw" style="background:rgba(var(--att),.45)"></span>attended by the hovered/selected head (darker = more weight) · dotted = other positions you can select · grey = text after the token (the model hasn't seen it)</div>
<div class="bar">
<label for="sort">Sort heads</label><select id="sort"><option value="fve" selected>by RL FVE (high first)</option><option value="h">by head</option><option value="rl">by RL cos</option>
<option value="gap">by RL − SFT cos</option><option value="norm">by attention entropy</option><option value="first">by attention to first token</option></select>
<label for="minf">Min RL FVE</label><select id="minf"><option value="-1e9">all</option><option value="0">&gt; 0%</option><option value="0.25">&gt; 25%</option><option value="0.5">&gt; 50%</option></select>
<span style="width:10px"></span><span class="lab" style="color:var(--ink-2);font-size:12.5px">Show</span>
<button id="t-rl" aria-pressed="true">RL AV</button><button id="t-sft" aria-pressed="false">SFT AV</button><button id="t-gold" aria-pressed="false">Gold (Haiku)</button>
<button id="t-hl" aria-pressed="true">Highlight quotes &amp; markup</button>
<span style="width:10px"></span><input id="q" type="search" placeholder="Filter heads by text…" aria-label="Filter heads by text" size="22">
<button id="theme" title="Toggle light/dark">◐</button>
</div>
<div class="sum" id="sum"></div><div class="grid" id="grid"></div>
<div class="notes"><h2>About this page</h2>
<p><b>RL AV</b>: verbalizer after 400 GRPO steps, scored by its co-trained AR. <b>SFT AV</b>: the warm-start verbalizer, scored by the SFT AR. Both sampled at T=1 (as in RL). <b>Gold</b>: Claude Haiku 5.5's explanation written from the write's readouts only (the SFT target); its scores are through the RL AR and the SFT AR.</p>
<p>Caveat: RL explanations often quote tokens (highlighted) and add markup (struck through); a blinded judge found them no more faithful to the evidence than SFT's, and much of their reconstruction comes from an AV↔AR code. Data: <code>syvb/head-nla-qwen3-8b</code> (Hugging Face), code: <code>sv/head-nla</code> branch of EasyNLA.</p></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const S={doc:0,pos:0,sel:null,hover:null,show:{rl:true,sft:false,gold:false,hl:true},sort:'fve',q:'',minf:-1e9};
const fveCls=f=>f>=0.5?'fve h':f>=0.25?'fve m':f>0?'fve p':'fve';
const $=id=>document.getElementById(id);
function el(t,a={},...kids){const e=document.createElement(t);for(const[k,v]of Object.entries(a)){if(k==='class')e.className=v;else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v)}for(const c of kids)if(c!=null)e.append(c);return e}
const f2=x=>(x>=0?'':'−')+Math.abs(x).toFixed(2), pct=x=>(x>=0?'+':'−')+Math.abs(Math.round(100*x))+'%';
function vis(s){return s.replace(/\n/g,'⏎').replace(/ /g,'·').replace(/\t/g,'⇥')||'∅'}
function curDoc(){return D.docs[S.doc]}function curPos(){return curDoc().positions[S.pos]}
function renderDocs(){const box=$('docs');box.replaceChildren(el('label',{},'Text'));D.docs.forEach((d,i)=>{const words=d.tokens.join('').trim().slice(0,60);
box.append(el('button',{'aria-pressed':String(i===S.doc),title:(d.domain?d.domain+' · ':'')+words,onclick:()=>{S.doc=i;S.pos=0;S.sel=null;renderAll(true)}},`${i+1}. ${d.domain?d.domain.replace(/_/g,' ')+': ':''}${words}…`))})}
function renderPositions(){const box=$('positions');box.replaceChildren(el('label',{},'Token'));curDoc().positions.forEach((p,i)=>{const t=curDoc().tokens[p.pos];
const npos=p.heads.filter(H=>(H.rl?.fve??-9)>0).length;
box.append(el('button',{'aria-pressed':String(i===S.pos),title:`${npos} of ${p.heads.length} heads with RL FVE > 0`,onclick:()=>{S.pos=i;S.sel=null;renderAll(true)}},`#${p.pos} “${t.trim()||vis(t)}” · ${npos} > 0`))})}
function attOf(h){if(h==null)return null;const H=curPos().heads.find(x=>x.h===h);return H?new Map(H.att.map(([p,w])=>[p,w])):null}
function renderCtx(scroll){const d=curDoc(),P=curPos().pos,m=attOf(S.hover??S.sel),others=new Set(d.positions.map(p=>p.pos));
const start=Math.max(0,P-160),box=$('ctx');box.replaceChildren();if(start>0)box.append(el('span',{class:'after'},'… '));
const end=Math.min(d.tokens.length,P+26);
for(let i=start;i<end;i++){const t=d.tokens[i];const s=el('span',{class:'t'});s.textContent=t;
if(i===P)s.classList.add('cur');else{if(i>P)s.classList.add('after');if(others.has(i)){s.classList.add('pos');s.title='select this position';s.onclick=()=>{S.pos=d.positions.findIndex(p=>p.pos===i);S.sel=null;renderAll(true)}}}
if(m&&m.has(i)){const w=m.get(i);s.style.background=`rgba(var(--att),${(0.15+0.85*Math.min(1,w/0.5)).toFixed(2)})`;s.title=`attention ${w.toFixed(3)}`}
box.append(s)}if(end<d.tokens.length)box.append(el('span',{class:'after'},' …'));const cur=box.querySelector('.cur');if(scroll&&cur)box.scrollTop=Math.max(0,cur.offsetTop-box.offsetTop-box.clientHeight/2)}
function hl(text){const frag=document.createDocumentFragment();if(!S.show.hl){frag.append(text);return frag}
const re=/(<\/?[A-Za-z][\w-]*>|\[end\])|([“"][^”"\n]{1,80}[”"])/g;let last=0,m;while((m=re.exec(text))){frag.append(text.slice(last,m.index));frag.append(el('mark',{class:m[1]?'m':'q'},m[0]));last=re.lastIndex}frag.append(text.slice(last));return frag}
function exBlock(kind,lab,obj,extra){if(!obj)return null;const txt=el('div',{class:'txt clamp'},hl(obj.text));const more=el('button',{class:'more',onclick:e=>{e.stopPropagation();txt.classList.toggle('clamp');more.textContent=txt.classList.contains('clamp')?'more':'less'}},'more');
return el('div',{class:'ex '+kind},el('div',{class:'lab'},el('b',{},lab),el('span',{class:'score'},extra)),txt,obj.text.length>330?more:null)}
function renderGrid(){const g=$('grid');g.replaceChildren();let hs=[...curPos().heads];const q=S.q.toLowerCase();
if(q)hs=hs.filter(H=>[H.rl?.text,H.sft?.text,H.gold,H.distinct.join(' '),H.lens.join(' ')].join(' ').toLowerCase().includes(q));
hs=hs.filter(H=>(H.rl?.fve??-9)>S.minf);
const all=curPos().heads,pos=all.filter(H=>(H.rl?.fve??-9)>0).length,mean=all.reduce((a,H)=>a+(H.rl?.fve??0),0)/all.length;
$('sum').replaceChildren('Showing ',el('b',{},String(hs.length)),` of ${all.length} heads · `,el('b',{},String(pos)),' with RL FVE > 0 at this token · mean RL FVE ',el('b',{},pct(mean)));
const key={fve:H=>-(H.rl?.fve??-9),h:H=>H.h,rl:H=>-(H.rl?.cos??-9),gap:H=>-((H.rl?.cos??0)-(H.sft?.cos??0)),norm:H=>H.entropy,first:H=>-H.first_w}[S.sort];hs.sort((a,b)=>key(a)-key(b));
const d=curDoc(),P=curPos().pos;
for(const H of hs){const atts=H.att.map(([p,w])=>el('span',{class:'chip'},vis(d.tokens[p]??'?'),el('span',{class:'w'},(p===P?'self ':p===0?'first ':'')+w.toFixed(2))));
const card=el('article',{class:'card'+(S.sel===H.h?' sel':'')+((H.rl?.fve??-9)<=0?' lowf':''),tabindex:'0','data-h':String(H.h),'aria-label':`Head ${H.h}, RL FVE ${pct(H.rl?.fve??0)}`,
onmouseenter:()=>{S.hover=H.h;renderCtx(false)},onmouseleave:()=>{S.hover=null;renderCtx(false)},
onclick:()=>selectHead(H.h),onkeydown:e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectHead(H.h)}}},
el('h3',{},`L${D.layer}H${H.h}`,H.rl?el('span',{class:fveCls(H.rl.fve),title:'RL pair FVE at this token'},'FVE '+pct(H.rl.fve)):null,el('span',{class:'meta'},`attn to first token ${H.first_w.toFixed(2)} · self ${H.self_w.toFixed(2)} · entropy ${H.entropy.toFixed(2)}`)),
S.show.rl?exBlock('rl','RL AV',H.rl,`FVE ${pct(H.rl?.fve??0)} · cos ${f2(H.rl?.cos??0)}`):null,
S.show.sft?exBlock('sft','SFT AV',H.sft,`FVE ${pct(H.sft?.fve??0)} · cos ${f2(H.sft?.cos??0)}`):null,
S.show.gold&&H.gold?exBlock('gold','Gold (Haiku)',{text:H.gold},`FVE ${pct(H.rl?.gold_fve??0)} via RL AR · ${pct(H.sft?.gold_fve??0)} via SFT AR`):null,
el('div',{class:'ev'},el('div',{class:'k'},'Reads'),el('div',{class:'chips'},...atts),
el('div',{class:'k'},'Promotes'),el('div',{class:'chips'},...H.distinct.map(t=>el('span',{class:'chip'},vis(t))),...H.filler.map(t=>el('span',{class:'chip f',title:'common filler'},vis(t)))),
el('div',{class:'k'},'Lens'),el('div',{class:'chips'},...H.lens.map(t=>el('span',{class:'chip'},vis(t))))));
g.append(card)}if(!hs.length)g.append(el('p',{},'No heads match the filter.'))}
function selectHead(h){S.sel=S.sel===h?null:h;document.querySelectorAll('.card').forEach(c=>c.classList.toggle('sel',+c.dataset.h===S.sel));renderCtx(false)}
function renderAll(scroll){renderDocs();renderPositions();renderCtx(scroll);renderGrid()}
for(const k of['rl','sft','gold','hl'])$('t-'+k).onclick=e=>{S.show[k]=!S.show[k];e.currentTarget.setAttribute('aria-pressed',String(S.show[k]));renderGrid()};
$('sort').onchange=e=>{S.sort=e.target.value;renderGrid()};$('minf').onchange=e=>{S.minf=+e.target.value;renderGrid()};$('q').oninput=e=>{S.q=e.target.value;renderGrid()};
$('theme').onclick=()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark'};
renderAll(true);
</script></body></html>"""

if __name__ == "__main__":
    main()
