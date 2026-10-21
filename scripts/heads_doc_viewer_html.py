"""Single-file HTML viewer for heads_doc_viewer_job.py output: every token × every head of a layer.

    python scripts/heads_doc_viewer_html.py --docs <job out dir> --out doc_viewer.html [--texts-ref <48k texts>]

Per document: a heads × tokens heatmap (RL FVE by default), the text shaded by the selected head's full attention
row at the selected token, and the selected site's explanations (RL; SFT and gold on request), scores and evidence.
Each document's data is gzip-compressed and base64-embedded, decoded in the browser only when opened.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--texts-ref", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--common-threshold", type=float, default=0.08)
    p.add_argument("--title", default="Head NLA — every token × every head (Qwen3-8B layer 24)")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    df = doc_freq(pq.read_table(args.texts_ref, columns=["avg16"]).to_pydict()["avg16"]) if args.texts_ref else {}
    docs = []
    for dd in sorted(Path(args.docs).iterdir()):
        if not (dd / "DONE").exists():
            continue
        meta = json.loads((dd / "meta.json").read_text())
        ids = np.load(dd / "token_ids.npy").tolist()
        s = pq.read_table(dd / "sites.parquet").to_pandas().sort_values(["head", "pos"])
        H, T = int(meta["n_heads"]), len(ids)
        pos = sorted(s.pos.unique())
        assert len(s) == H * len(pos), f"{dd.name}: {len(s)} sites for {H} heads × {len(pos)} positions"
        attn = np.load(dd / "attn.npz")["attn"].astype(np.float32)                    # [H, T, T]
        # full attention rows, causal part only, as uint8 (√w × 255 keeps small weights visible)
        att_q = [[base64.b64encode((np.sqrt(np.clip(attn[h, t, : t + 1], 0, 1)) * 255).round().astype(np.uint8)
                                   .tobytes()).decode() for t in pos] for h in range(H)]
        a16 = [toks(x) for x in s.avg16]
        thr = args.common_threshold
        d = {"name": meta["name"], "domain": meta.get("domain", ""), "source": meta.get("source", ""),
             "url": meta.get("url", ""), "n_heads": H, "layer": meta.get("layer", 24),
             "tokens": [tok.decode([i]) for i in ids], "pos": [int(x) for x in pos],
             # every per-site column is [head][pos-index] (row-major over heads)
             "rl_fve": r3(s.rl_fve), "sft_fve": r3(s.sft_fve), "gold_fve": r3(s.gold_rl_fve),
             "rl_cos": r3(s.rl_cos), "sft_cos": r3(s.sft_cos), "gold_cos": r3(s.gold_rl_cos),
             "gold_fve_sft": r3(s.gold_sft_fve), "first_w": r3(s.first_w), "self_w": r3(s.self_w),
             "entropy": r3(s.entropy), "v_norm": r3(s.v_norm),
             "rl": s.rl_text.tolist(), "sft": s.sft_text.tolist(), "gold": s.gold_text.tolist(),
             "gold_ok": [bool(x) for x in s.gold_ok],
             "distinct": [[t for t in x if df.get(t, 0) < thr] for x in a16],
             "filler": [[t for t in x if df.get(t, 0) >= thr] for x in a16],
             "lens": [toks(x) for x in s.lens], "att": att_q}
        blob = base64.b64encode(gzip.compress(json.dumps(d, ensure_ascii=False).encode(), 9)).decode()
        docs.append({"name": meta["name"], "domain": meta.get("domain", ""), "n_tokens": T, "n_pos": len(pos),
                     "rl_fve_mean": round(float(np.nanmean(s.rl_fve)), 3), "blob": blob})
        print(f"{dd.name}: {T} tokens × {H} heads, {len(blob) / 1e6:.1f} MB embedded")
    assert docs, "no finished documents"
    data = json.dumps(docs).replace("</", "<\\/")
    Path(args.out).write_text(TEMPLATE.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", data))
    print(f"wrote {args.out}: {len(docs)} documents, {Path(args.out).stat().st_size / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:#fcfcfb;--surface-2:#f3f2ef;--card:#fff;--border:#e3e2dd;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#7d7c77;--accent:#2a78d6;--accent-soft:#e3eefb;
--rl:#eb6834;--sft:#2a78d6;--gold:#8a6d00;--att:42,120,214;--hit:#fff1c2;--hit-ink:#5c4400;--bad:#b3261e;--neg:42,120,214;--pos:235,104,52;--mid:228,227,223}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;
--accent:#3987e5;--accent-soft:#1f3349;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;--neg:70,140,230;--pos:226,114,63;--mid:58,58,55}}
:root[data-theme="dark"]{--surface:#1a1a19;--surface-2:#232321;--card:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#97968e;
--accent:#3987e5;--accent-soft:#1f3349;--rl:#e2723f;--sft:#5b9ae8;--gold:#d8b54a;--att:90,160,240;--hit:#4a3b0a;--hit-ink:#f3d98a;--bad:#ff8a80;--neg:70,140,230;--pos:226,114,63;--mid:58,58,55}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,main{max-width:1500px;margin:0 auto;padding:0 20px}header{padding-top:16px}h1{font-size:19px;margin:0 0 4px}header p{margin:4px 0;color:var(--ink-2);max-width:1000px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}.bar label,.muted{color:var(--ink-2);font-size:12.5px}
button,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--border);border-radius:8px;padding:5px 10px;cursor:pointer}
button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hm-wrap{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px 10px 6px;overflow-x:auto}
canvas{display:block;image-rendering:pixelated;cursor:crosshair}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2);margin:6px 2px 0}.ramp{width:220px;height:10px;border-radius:3px}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12.5px;box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;z-index:5;max-width:320px}
.cols{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px;margin-top:14px}@media (max-width:980px){.cols{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.ctx{line-height:1.8;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow:auto;font-size:13.5px}
.ctx .t{border-radius:3px;cursor:pointer}.ctx .cur{outline:2px solid var(--rl);font-weight:600}.ctx .after{color:var(--ink-3)}
h2{font-size:15px;margin:0 0 6px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}.meta{font-weight:400;color:var(--ink-3);font-size:12px}
.fve{font-weight:700;font-size:15px;padding:2px 9px;border-radius:999px;font-variant-numeric:tabular-nums;border:1px solid var(--border);background:var(--surface-2);color:var(--ink-2)}
.fve.p{background:#fde3d6;color:#5a2410;border-color:#f6b99c}.fve.m{background:#f6a77f;color:#3b1406;border-color:#ee8a58}.fve.h{background:#c4471a;color:#fff;border-color:#a8380f}
.ex{margin:8px 0;padding-left:9px;border-left:3px solid var(--border)}.ex.rl{border-color:var(--rl)}.ex.sft{border-color:var(--sft)}.ex.gold{border-color:var(--gold)}
.ex .lab{font-size:11.5px;color:var(--ink-2);margin-bottom:2px}.ex .lab b{color:var(--ink)}.ex .txt{white-space:pre-wrap;word-break:break-word}
mark.q{background:var(--hit);color:var(--hit-ink);border-radius:3px;padding:0 1px}mark.m{background:none;color:var(--bad);text-decoration:line-through}
.ev{display:grid;grid-template-columns:72px 1fr;gap:4px 8px;margin-top:8px;font-size:12.5px;border-top:1px dashed var(--border);padding-top:8px}.ev .k{color:var(--ink-2)}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chip{border:1px solid var(--border);border-radius:6px;padding:0 5px;background:var(--surface-2);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre}
.chip.f{color:var(--ink-3)}.chip .w{color:var(--ink-3);font-family:system-ui;font-size:11px;margin-left:3px}
table.heads{width:100%;border-collapse:collapse;font-size:12.5px}table.heads td,table.heads th{padding:3px 6px;border-bottom:1px solid var(--border);text-align:left}
table.heads tr{cursor:pointer}table.heads tr.sel{background:var(--accent-soft)}table.heads td.n{font-variant-numeric:tabular-nums;text-align:right}
.loading{color:var(--ink-2);padding:20px}
</style></head><body>
<header><h1>__TITLE__</h1>
<p>Rows are the layer's 32 attention heads, columns are every token of a held-out document. Each cell is one head's write at one token, explained by the <b>AV</b> (it sees only the write vector) and reconstructed from that explanation by the <b>AR</b>. Colour = <b>FVE</b> (1 − error / error of predicting the head's mean write): orange above 0, blue below, grey ≈ 0; it is negative below cos ≈ 0.5 by construction. Click a cell for its explanation, scores and evidence; the text then shows that head's full attention from the token (evidence only, never used for any explanation).</p></header>
<main>
<div class="bar" id="docs" role="group" aria-label="Document"></div>
<div class="bar"><label for="metric">Colour</label><select id="metric"><option value="rl_fve">RL FVE</option><option value="sft_fve">SFT FVE</option><option value="gold_fve">Gold FVE (RL AR)</option><option value="gap">RL − SFT FVE</option><option value="first_w">attention to first token</option></select>
<label for="zoom">Cell width</label><select id="zoom"><option value="4">4 px</option><option value="6" selected>6 px</option><option value="9">9 px</option><option value="14">14 px</option></select>
<span style="width:8px"></span><span class="muted">Show</span><button id="t-sft" aria-pressed="false">SFT AV</button><button id="t-gold" aria-pressed="false">Gold (Haiku)</button><button id="t-hl" aria-pressed="true">Highlight quotes &amp; markup</button><button id="theme" title="Toggle light/dark">◐</button></div>
<div class="hm-wrap"><canvas id="hm" tabindex="0" aria-label="Heads by tokens heatmap; arrow keys move the selection"></canvas>
<div class="legend"><span id="lo">−100%</span><canvas class="ramp" id="ramp" width="220" height="10"></canvas><span id="hi">+100%</span><span id="lgname"></span><span id="stat" style="margin-left:auto"></span></div></div>
<div class="cols"><div><div class="panel"><h2 id="ctxh">Text</h2><div class="ctx" id="ctx"></div></div>
<div class="panel" style="margin-top:14px"><h2 id="colh">Heads at this token</h2><table class="heads" id="heads"></table></div></div>
<div class="panel" id="detail"></div></div>
</main><div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DOCS=JSON.parse(document.getElementById('data').textContent),CACHE={};
const S={doc:0,h:0,pi:0,metric:'rl_fve',zoom:6,show:{sft:false,gold:false,hl:true}};let D=null;
const $=id=>document.getElementById(id);
function el(t,a={},...k){const e=document.createElement(t);for(const[x,v]of Object.entries(a)){if(x==='class')e.className=v;else if(x.startsWith('on'))e.addEventListener(x.slice(2),v);else e.setAttribute(x,v)}for(const c of k)if(c!=null)e.append(c);return e}
const pct=x=>x==null?'n/a':(x>=0?'+':'−')+Math.abs(Math.round(100*x))+'%',f2=x=>x==null?'n/a':(x>=0?'':'−')+Math.abs(x).toFixed(2);
const vis=s=>s.replace(/\n/g,'⏎').replace(/ /g,'·').replace(/\t/g,'⇥')||'∅',fveCls=f=>f>=0.5?'fve h':f>=0.25?'fve m':f>0?'fve p':'fve';
async function load(i){if(CACHE[i])return CACHE[i];const b=Uint8Array.from(atob(DOCS[i].blob),c=>c.charCodeAt(0));
const txt=await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).text();return CACHE[i]=JSON.parse(txt)}
const idx=(h,pi)=>h*D.pos.length+pi;
function val(h,pi){const k=idx(h,pi);if(S.metric==='gap'){const a=D.rl_fve[k],b=D.sft_fve[k];return a==null||b==null?null:a-b}return D[S.metric][k]}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function color(v){if(v==null)return'rgb(128,128,128)';const [nr,ng,nb]=css('--neg').split(',').map(Number),[pr,pg,pb]=css('--pos').split(',').map(Number),[mr,mg,mb]=css('--mid').split(',').map(Number);
let t=S.metric==='first_w'?v*2-1:Math.max(-1,Math.min(1,v));const [r,g,b]=t>=0?[pr,pg,pb]:[nr,ng,nb];t=Math.abs(t);return `rgb(${Math.round(mr+(r-mr)*t)},${Math.round(mg+(g-mg)*t)},${Math.round(mb+(b-mb)*t)})`}
const LAB=34,TOP=4;
function drawHM(){const c=$('hm'),H=D.n_heads,P=D.pos.length,cw=S.zoom,ch=11,dpr=window.devicePixelRatio||1;
c.width=(LAB+P*cw)*dpr;c.height=(TOP+H*ch)*dpr;c.style.width=(LAB+P*cw)+'px';c.style.height=(TOP+H*ch)+'px';const g=c.getContext('2d');g.scale(dpr,dpr);
g.fillStyle=css('--card');g.fillRect(0,0,LAB+P*cw,TOP+H*ch);g.font='10px system-ui';g.fillStyle=css('--ink-2');g.textBaseline='middle';
for(let h=0;h<H;h++){if(h%2===0||ch>=11)g.fillText('H'+h,2,TOP+h*ch+ch/2);for(let pi=0;pi<P;pi++){g.fillStyle=color(val(h,pi));g.fillRect(LAB+pi*cw,TOP+h*ch,cw-(cw>4?1:0),ch-1)}g.fillStyle=css('--ink-2')}
g.strokeStyle=css('--ink');g.lineWidth=2;g.strokeRect(LAB+S.pi*cw-1,TOP+S.h*ch-1,cw+1,ch+1);
g.strokeStyle=css('--accent');g.lineWidth=1;g.strokeRect(LAB+S.pi*cw-.5,TOP-2,cw,H*ch+3);
const r=$('ramp').getContext('2d');for(let x=0;x<220;x++){r.fillStyle=color(S.metric==='first_w'?x/219:(x/219)*2-1);r.fillRect(x,0,1,10)}
$('lo').textContent=S.metric==='first_w'?'0':'−100%';$('hi').textContent=S.metric==='first_w'?'1':'+100%';$('lgname').textContent=$('metric').selectedOptions[0].text;
const vs=[];for(let h=0;h<H;h++)for(let pi=0;pi<P;pi++){const v=D.rl_fve[idx(h,pi)];if(v!=null)vs.push(v)}const m=vs.reduce((a,b)=>a+b,0)/vs.length;
$('stat').textContent=`${D.pos.length} tokens × ${H} heads · RL FVE mean ${pct(m)} · ${Math.round(100*vs.filter(v=>v>0).length/vs.length)}% of cells > 0`}
function cellAt(e){const r=$('hm').getBoundingClientRect(),x=e.clientX-r.left-LAB,y=e.clientY-r.top-TOP,pi=Math.floor(x/S.zoom),h=Math.floor(y/11);
return(pi>=0&&pi<D.pos.length&&h>=0&&h<D.n_heads)?{h,pi}:null}
function attRow(h,pi){const s=atob(D.att[h][pi]);const a=new Float32Array(s.length);for(let i=0;i<s.length;i++){const q=s.charCodeAt(i)/255;a[i]=q*q}return a}
function renderCtx(){const P=D.pos[S.pi],a=attRow(S.h,S.pi),box=$('ctx');box.replaceChildren();const end=Math.min(D.tokens.length,P+30);
let mx=0;for(let i=0;i<a.length;i++)if(i!==0&&a[i]>mx)mx=a[i];
for(let i=0;i<end;i++){const s=el('span',{class:'t'});s.textContent=D.tokens[i];if(i===P)s.classList.add('cur');else if(i>P)s.classList.add('after');
if(i<=P){const w=a[i];if(w>0.003){s.style.background=`rgba(var(--att),${(0.12+0.88*Math.min(1,w/Math.max(mx,0.05))).toFixed(2)})`}s.title=`attention ${w.toFixed(3)}`}
const pi=D.pos.indexOf(i);if(pi>=0)s.onclick=()=>{S.pi=pi;update()};box.append(s)}
if(end<D.tokens.length)box.append(el('span',{class:'after'},' …'));
$('ctxh').replaceChildren('Text',el('span',{class:'meta'},`token #${P} “${vis(D.tokens[P])}” · shading = H${S.h}'s attention from it (first token ${D.first_w[idx(S.h,S.pi)]?.toFixed(2)})`));
const cur=box.querySelector('.cur');if(cur){const top=cur.offsetTop-box.offsetTop;if(top<box.scrollTop||top>box.scrollTop+box.clientHeight-30)box.scrollTop=Math.max(0,top-box.clientHeight/2)}}
function hl(text){const f=document.createDocumentFragment();if(!S.show.hl){f.append(text);return f}const re=/(<\/?[A-Za-z][\w-]*>|\[end\])|([“"][^”"\n]{1,80}[”"])/g;let l=0,m;
while((m=re.exec(text))){f.append(text.slice(l,m.index));f.append(el('mark',{class:m[1]?'m':'q'},m[0]));l=re.lastIndex}f.append(text.slice(l));return f}
function renderDetail(){const k=idx(S.h,S.pi),P=D.pos[S.pi],a=attRow(S.h,S.pi),top=[...a.keys()].sort((x,y)=>a[y]-a[x]).slice(0,10).filter(i=>a[i]>=0.005);
$('detail').replaceChildren(...[el('h2',{},`L${D.layer}H${S.h} at #${P} “${vis(D.tokens[P])}”`,el('span',{class:fveCls(D.rl_fve[k]??-9)},'RL FVE '+pct(D.rl_fve[k])),
el('span',{class:'meta'},`cos ${f2(D.rl_cos[k])} · ‖v‖ ${D.v_norm[k]?.toFixed(2)} · first-token attn ${D.first_w[k]?.toFixed(2)} · entropy ${D.entropy[k]?.toFixed(2)}`)),
el('div',{class:'ex rl'},el('div',{class:'lab'},el('b',{},'RL AV'),`  FVE ${pct(D.rl_fve[k])} · cos ${f2(D.rl_cos[k])}`),el('div',{class:'txt'},hl(D.rl[k]))),
S.show.sft?el('div',{class:'ex sft'},el('div',{class:'lab'},el('b',{},'SFT AV'),`  FVE ${pct(D.sft_fve[k])} · cos ${f2(D.sft_cos[k])}`),el('div',{class:'txt'},hl(D.sft[k]))):null,
S.show.gold&&D.gold[k]?el('div',{class:'ex gold'},el('div',{class:'lab'},el('b',{},'Gold (Haiku)'),`  FVE ${pct(D.gold_fve[k])} via RL AR · ${pct(D.gold_fve_sft[k])} via SFT AR${D.gold_ok[k]?'':' · breaks rules'}`),el('div',{class:'txt'},hl(D.gold[k]))):null,
el('div',{class:'ev'},el('div',{class:'k'},'Reads'),el('div',{class:'chips'},...top.map(i=>el('span',{class:'chip'},vis(D.tokens[i]),el('span',{class:'w'},(i===P?'self ':i===0?'first ':'')+a[i].toFixed(2))))),
el('div',{class:'k'},'Promotes'),el('div',{class:'chips'},...D.distinct[k].map(t=>el('span',{class:'chip'},vis(t))),...D.filler[k].map(t=>el('span',{class:'chip f',title:'common filler'},vis(t)))),
el('div',{class:'k'},'Lens'),el('div',{class:'chips'},...D.lens[k].map(t=>el('span',{class:'chip'},vis(t)))))].filter(Boolean))}
function renderHeads(){const rows=[...Array(D.n_heads).keys()].map(h=>({h,f:D.rl_fve[idx(h,S.pi)]})).sort((a,b)=>(b.f??-9)-(a.f??-9));
$('colh').replaceChildren('Heads at this token',el('span',{class:'meta'},'sorted by RL FVE; click to select'));
$('heads').replaceChildren(el('tr',{},el('th',{},'head'),el('th',{},'RL FVE'),el('th',{},'SFT FVE'),el('th',{},'first-token attn'),el('th',{},'RL explanation')),
...rows.map(({h,f})=>{const k=idx(h,S.pi);return el('tr',{class:h===S.h?'sel':'',onclick:()=>{S.h=h;update()}},el('td',{},'H'+h),el('td',{class:'n'},pct(f)),el('td',{class:'n'},pct(D.sft_fve[k])),
el('td',{class:'n'},(D.first_w[k]??0).toFixed(2)),el('td',{},D.rl[k].split('\n')[0].slice(0,90)+(D.rl[k].length>90?'…':'')))}))}
function update(){drawHM();renderCtx();renderDetail();renderHeads()}
function renderDocs(){$('docs').replaceChildren(el('label',{},'Document'),...DOCS.map((d,i)=>el('button',{'aria-pressed':String(i===S.doc),onclick:()=>openDoc(i)},`${d.name.replace(/_/g,' ')} · ${d.n_pos} tok · RL FVE ${pct(d.rl_fve_mean)}`)))}
async function openDoc(i){S.doc=i;renderDocs();$('detail').replaceChildren(el('div',{class:'loading'},'Loading…'));D=await load(i);
let best=[0,0,-9];for(let h=0;h<D.n_heads;h++)for(let pi=0;pi<D.pos.length;pi++){const v=D.rl_fve[idx(h,pi)];if(v!=null&&v>best[2]&&D.first_w[idx(h,pi)]<0.9)best=[h,pi,v]}
S.h=best[0];S.pi=best[1];update()}
const tip=$('tip');$('hm').addEventListener('mousemove',e=>{const c=cellAt(e);if(!c){tip.style.display='none';return}const k=idx(c.h,c.pi);
tip.replaceChildren(el('b',{},`H${c.h} · #${D.pos[c.pi]} “${vis(D.tokens[D.pos[c.pi]])}”`),el('br'),`RL FVE ${pct(D.rl_fve[k])} · SFT ${pct(D.sft_fve[k])} · first-token ${(D.first_w[k]??0).toFixed(2)}`,el('br'),D.rl[k].split('\n')[0].slice(0,140));
tip.style.display='block';tip.style.left=Math.min(innerWidth-330,e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'});
$('hm').addEventListener('mouseleave',()=>tip.style.display='none');
$('hm').addEventListener('click',e=>{const c=cellAt(e);if(c){S.h=c.h;S.pi=c.pi;update()}});
$('hm').addEventListener('keydown',e=>{const m={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[e.key];if(!m)return;e.preventDefault();
S.h=Math.max(0,Math.min(D.n_heads-1,S.h+m[0]));S.pi=Math.max(0,Math.min(D.pos.length-1,S.pi+m[1]));update()});
$('metric').onchange=e=>{S.metric=e.target.value;drawHM()};$('zoom').onchange=e=>{S.zoom=+e.target.value;drawHM()};
for(const k of['sft','gold','hl'])$('t-'+k).onclick=e=>{S.show[k]=!S.show[k];e.currentTarget.setAttribute('aria-pressed',String(S.show[k]));renderDetail()};
$('theme').onclick=()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';drawHM()};
openDoc(0);
</script></body></html>"""

if __name__ == "__main__":
    main()
