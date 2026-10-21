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
        if not ((dd / "meta.json").exists() and (dd / "sites.parquet").exists() and (dd / "attn.npz").exists()):
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
        has_gold = s.gold_text.str.len() > 0          # a failed Haiku call: no gold score, not an empty "explanation"
        thr = args.common_threshold
        d = {"name": meta["name"], "domain": meta.get("domain", ""), "source": meta.get("source", ""),
             "url": meta.get("url", ""), "n_heads": H, "layer": meta.get("layer", 24),
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
.legend{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--ink-2);margin:8px 2px 0;flex-wrap:wrap}.lg{display:inline-flex;flex-direction:column;gap:2px}
.ramp{width:260px;height:10px;border-radius:3px;display:block}.ticks{display:flex;justify-content:space-between;width:260px;font-variant-numeric:tabular-nums}
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
.ov{margin-top:14px}.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px;margin-top:8px}
.thumb{border:1px solid var(--border);border-radius:10px;padding:6px 7px;cursor:pointer;background:var(--card);text-align:left;font:inherit;color:inherit}
.thumb.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}canvas.am{width:100%;aspect-ratio:1;image-rendering:pixelated;display:block;border-radius:4px;background:var(--surface-2)}
.thumb .row1{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px;margin:2px 0}.thumb .row2{font-size:11px;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.35}
.tag{font-size:10.5px;border:1px solid var(--border);border-radius:999px;padding:0 6px;color:var(--ink-2);background:var(--surface-2)}
</style></head><body>
<header><h1>__TITLE__</h1>
<p>Rows are the layer's 32 attention heads, columns are every token of a held-out document. Each cell is one head's write at one token, explained by the <b>AV</b> (it sees only the write vector) and reconstructed from that explanation by the <b>AR</b>. Colour = <b>FVE</b> (1 − error / error of predicting the head's mean write): orange above 0, blue below, grey ≈ 0; it is negative below cos ≈ 0.5 by construction. Click a cell for its explanation, scores and evidence; the text then shows that head's full attention from the token (evidence only, never used for any explanation).</p></header>
<main>
<div class="bar" id="docs" role="group" aria-label="Document"></div>
<div class="bar"><label for="metric">Colour</label><select id="metric"><option value="rl_fve">RL FVE</option><option value="sft_fve">SFT FVE</option><option value="gold_fve">Gold FVE (RL AR)</option><option value="gap">RL − SFT FVE</option><option value="first_w">attention to first token</option></select>
<label for="zoom">Cell width</label><select id="zoom"><option value="4">4 px</option><option value="6" selected>6 px</option><option value="9">9 px</option><option value="14">14 px</option></select>
<span style="width:8px"></span><span class="muted">Show</span><button id="t-sft" aria-pressed="false">SFT AV</button><button id="t-gold" aria-pressed="false">Gold (Haiku)</button><button id="t-hl" aria-pressed="true">Highlight quotes &amp; markup</button><button id="theme" title="Toggle light/dark">◐</button></div>
<div class="hm-wrap"><canvas id="hm" tabindex="0" aria-label="Heads by tokens heatmap; arrow keys move the selection"></canvas>
<div class="legend"><b id="lgname" style="color:var(--ink)"></b><span class="lg"><canvas class="ramp" id="ramp" width="260" height="10"></canvas><span class="ticks"><span id="lo"></span><span id="mid"></span><span id="hi"></span></span></span><span id="lgnote"></span><span class="muted" style="margin-left:12px">← → previous / next token · ↑ ↓ heads (heatmap focused)</span><span id="stat" style="margin-left:auto"></span></div></div>
<div class="panel ov"><h2>Heads overview <span class="meta">each thumbnail is the head's attention over this document (rows = query token, top to bottom; columns = key token, left to right; darker = more attention). Click one to select that head.</span></h2>
<div class="bar"><label for="osort">Sort</label><select id="osort"><option value="h">by head</option><option value="sink">least first-token sink</option><option value="prev">most previous-token</option>
<option value="self">most self</option><option value="ent">most spread out (non-sink)</option><option value="dist">longest range</option><option value="focus">most focused on specific tokens</option><option value="foc">least spread out (non-sink)</option><option value="fve">highest mean RL FVE</option></select>
<button id="t-sink" aria-pressed="true" title="Remove the first-token column and renormalize each row, so the non-sink pattern is visible">Hide first-token sink</button>
<span class="muted">No BOS: Qwen's tokenizer adds none, so the first real token is the attention sink. Thumbnail rows are scaled to their own maximum.</span></div>
<div class="thumbs" id="thumbs"></div></div>
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
// Colour scales (dataviz reference palette): FVE metrics and RL−SFT are DIVERGING around 0 (blue below, grey
// midpoint, red above); attention to the first token is SEQUENTIAL blue. Stops are interpolated in OKLab and
// cached as a 256-entry lookup per scale and theme; values beyond a scale's ends are clamped.
const SCALES={fve:{kind:'div',lo:-1,hi:1,ticks:['−100%','0','+100%'],note:'FVE is at most +100% (perfect); values below −100% are shown as −100%'},
 gap:{kind:'div',lo:-2,hi:2,ticks:['−200%','0','+200%'],note:'RL − SFT, in FVE points; clamped at ±200%'},
 seq:{kind:'seq',lo:0,hi:1,ticks:['0','0.5','1'],note:'share of the head’s attention on the first token'}};
const STOPS={light:{div:['#0d366b','#3987e5','#f0efec','#e66767','#9b1c1c'],seq:['#f0efec','#cde2fb','#86b6ef','#3987e5','#1c5cab','#0d366b']},
 dark:{div:['#b7d3f6','#3987e5','#383835','#d24d4d','#ffb4ab'],seq:['#383835','#184f95','#256abf','#3987e5','#86b6ef','#cde2fb']}};
const scaleOf=m=>m==='gap'?SCALES.gap:m==='first_w'?SCALES.seq:SCALES.fve;
function toLin(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4)}function toSrgb(c){c=c<=0.0031308?12.92*c:1.055*Math.pow(c,1/2.4)-0.055;return Math.round(255*Math.max(0,Math.min(1,c)))}
function hex2lab(h){const n=parseInt(h.slice(1),16),r=toLin(n>>16&255),g=toLin(n>>8&255),b=toLin(n&255);
const l=Math.cbrt(0.4122214708*r+0.5363325363*g+0.0514459929*b),m=Math.cbrt(0.2119034982*r+0.6806995451*g+0.1073969566*b),s=Math.cbrt(0.0883024619*r+0.2817188376*g+0.6299787005*b);
return[0.2104542553*l+0.7936177850*m-0.0040720468*s,1.9779984951*l-2.4285922050*m+0.4505937099*s,0.0259040371*l+0.7827717662*m-0.8086757660*s]}
function lab2rgb([L,A,B]){const l=(L+0.3963377774*A+0.2158037573*B)**3,m=(L-0.1055613458*A-0.0638541728*B)**3,s=(L-0.0894841775*A-1.2914855480*B)**3;
return[toSrgb(4.0767416621*l-3.3077115913*m+0.2309699292*s),toSrgb(-1.2684380046*l+2.6097574011*m-0.3413193965*s),toSrgb(-0.0041960863*l-0.7034186147*m+1.7076147010*s)]}
let LUT={};function isDark(){const n=parseInt(css('--surface').slice(1),16);return((n>>16&255)+(n>>8&255)+(n&255))/3<128}
function lut(kind){const key=kind+(isDark()?'d':'l');if(LUT[key])return LUT[key];const st=STOPS[isDark()?'dark':'light'][kind].map(hex2lab),out=[];
for(let i=0;i<256;i++){const x=i/255*(st.length-1),k=Math.min(st.length-2,Math.floor(x)),f=x-k,a=st[k],b=st[k+1];const [r,g,bl]=lab2rgb([0,1,2].map(j=>a[j]+(b[j]-a[j])*f));out.push(`rgb(${r},${g},${bl})`)}return LUT[key]=out}
function color(v,m=S.metric){if(v==null)return null;const sc=scaleOf(m),t=Math.max(0,Math.min(1,(v-sc.lo)/(sc.hi-sc.lo)));return lut(sc.kind)[Math.round(t*255)]}
const LAB=34,TOP=4;
function drawHM(){LUT={};const c=$('hm'),H=D.n_heads,P=D.pos.length,cw=S.zoom,ch=11,dpr=window.devicePixelRatio||1;
c.width=(LAB+P*cw)*dpr;c.height=(TOP+H*ch)*dpr;c.style.width=(LAB+P*cw)+'px';c.style.height=(TOP+H*ch)+'px';const g=c.getContext('2d');g.scale(dpr,dpr);
g.fillStyle=css('--card');g.fillRect(0,0,LAB+P*cw,TOP+H*ch);g.font='10px system-ui';g.fillStyle=css('--ink-2');g.textBaseline='middle';
for(let h=0;h<H;h++){if(h%2===0||ch>=11)g.fillText('H'+h,2,TOP+h*ch+ch/2);for(let pi=0;pi<P;pi++){const col=color(val(h,pi)),x=LAB+pi*cw,y=TOP+h*ch;if(col){g.fillStyle=col;g.fillRect(x,y,cw-(cw>4?1:0),ch-1)}else{g.fillStyle=css('--ink-3');g.fillRect(x+cw/2-1,y+ch/2-1,2,2)}}g.fillStyle=css('--ink-2')}
g.strokeStyle=css('--ink');g.lineWidth=2;g.strokeRect(LAB+S.pi*cw-1,TOP+S.h*ch-1,cw+1,ch+1);
g.strokeStyle=css('--accent');g.lineWidth=1;g.strokeRect(LAB+S.pi*cw-.5,TOP-2,cw,H*ch+3);
const sc=scaleOf(S.metric),rc=$('ramp'),r=rc.getContext('2d'),RW=rc.width;for(let x=0;x<RW;x++){r.fillStyle=color(sc.lo+(sc.hi-sc.lo)*x/(RW-1));r.fillRect(x,0,1,rc.height)}
$('lo').textContent=sc.ticks[0];$('mid').textContent=sc.ticks[1];$('hi').textContent=sc.ticks[2];$('lgname').textContent=$('metric').selectedOptions[0].text;$('lgnote').textContent=sc.note;
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
let LUTN={};function lutNum(kind){const key=kind+(isDark()?'d':'l');if(LUTN[key])return LUTN[key];LUTN[key]=lut(kind).map(c=>c.match(/\d+/g).map(Number));return LUTN[key]}
let HS=null;S.osort='h';S.dropSink=true;
// Per-head attention summary over the document (query tokens t ≥ 1). "Non-sink" = attention renormalized over
// keys 1..t. Key focus = share of all non-sink attention landing on the head's 5 most-attended key tokens.
function headStats(){const H=D.n_heads,P=D.pos.length,T=D.tokens.length,out=[];for(let h=0;h<H;h++){let sink=0,self=0,prev=0,ent=0,dist=0,n=0;const rows=[],col=new Float64Array(T);
for(let pi=0;pi<P;pi++){const t=D.pos[pi],a=attRow(h,pi);rows.push(a);if(t<1)continue;n++;sink+=a[0];self+=a[t];prev+=a[t-1];let sm=0;for(let j=1;j<=t;j++)sm+=a[j];
if(sm>1e-6){let e=0,dd=0;for(let j=1;j<=t;j++){const q=a[j]/sm;col[j]+=q;if(q>1e-9){e-=q*Math.log(q);dd+=q*(t-j)}}ent+=e;dist+=dd}}
const f=D.rl_fve.slice(h*P,(h+1)*P).filter(x=>x!=null),tot=col.reduce((x,y)=>x+y,0)||1,top=[...col.keys()].sort((x,y)=>col[y]-col[x]).slice(0,5);
const r={h,sink:sink/n,self:self/n,prev:prev/n,ent:ent/n,dist:dist/n,focus:top.reduce((x,j)=>x+col[j],0)/tot,topKeys:top.filter(j=>col[j]>0),fve:f.reduce((x,y)=>x+y,0)/f.length,rows};
const ns=Math.max(1e-6,1-r.sink),sn=r.self/ns,pv=r.prev/ns;
r.tag=r.sink>=0.85?'sink-dominated':sn>=0.3?'self / current token':pv>=0.4?'previous token':r.dist<=20?'local':r.focus>=0.25?'focused on specific tokens':r.ent<=2.5?'focused':'diffuse / long-range';out.push(r)}return out}
function drawThumb(cv,st){const P=D.pos.length,img=new ImageData(P,P),L=lutNum('seq');for(let pi=0;pi<P;pi++){const a=st.rows[pi],t=D.pos[pi];
let mx=0;for(let j=S.dropSink?1:0;j<a.length;j++)mx=Math.max(mx,a[j]);mx=Math.max(mx,1e-6);   // each row scaled to its own max
for(let j=0;j<P;j++){const o=(pi*P+j)*4;let v=0;if(j<=t&&!(S.dropSink&&j===0))v=a[j]/mx;const c=L[Math.round(255*Math.sqrt(Math.min(1,v)))];img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=j<=t?255:0}}
cv.width=P;cv.height=P;cv.getContext('2d').putImageData(img,0,0)}
function renderThumbs(){const key={h:r=>r.h,sink:r=>r.sink,prev:r=>-r.prev,self:r=>-r.self,ent:r=>-r.ent,foc:r=>r.ent,dist:r=>-r.dist,focus:r=>-r.focus,fve:r=>-r.fve}[S.osort];
const rows=[...HS].sort((a,b)=>key(a)-key(b)),box=$('thumbs');box.replaceChildren();
for(const r of rows){const cv=el('canvas',{class:'am','aria-hidden':'true'});drawThumb(cv,r);
box.append(el('button',{class:'thumb'+(r.h===S.h?' sel':''),'data-h':String(r.h),'aria-label':`Head ${r.h}: ${r.tag}`,onclick:()=>{S.h=r.h;update()}},cv,
el('div',{class:'row1'},el('b',{},'H'+r.h),el('span',{class:'tag'},r.tag)),
el('div',{class:'row2',title:'sink = attention on the first token; prev/self = on the previous/current token; spread = entropy of non-sink attention; dist = mean tokens back; focus = share of non-sink attention on the 5 most-attended tokens'},
`sink ${(100*r.sink).toFixed(0)}% · prev ${(100*r.prev).toFixed(0)}% · self ${(100*r.self).toFixed(0)}%`,el('br'),`spread ${r.ent.toFixed(2)} · dist ${r.dist.toFixed(0)} · focus ${(100*r.focus).toFixed(0)}%`,el('br'),
`top: ${r.topKeys.slice(0,3).map(j=>vis(D.tokens[j]).trim()).join(', ')}`,el('br'),`mean RL FVE ${pct(r.fve)}`)))}}
function selThumbs(){document.querySelectorAll('.thumb').forEach(b=>b.classList.toggle('sel',+b.dataset.h===S.h))}
function update(){drawHM();renderCtx();renderDetail();renderHeads();selThumbs()}
function renderDocs(){$('docs').replaceChildren(el('label',{},'Document'),...DOCS.map((d,i)=>el('button',{'aria-pressed':String(i===S.doc),onclick:()=>openDoc(i)},`${d.name.replace(/_/g,' ')} · ${d.n_pos} tok · RL FVE ${pct(d.rl_fve_mean)}`)))}
async function openDoc(i){S.doc=i;renderDocs();$('detail').replaceChildren(el('div',{class:'loading'},'Loading…'));D=await load(i);
let best=[0,0,-9];for(let h=0;h<D.n_heads;h++)for(let pi=0;pi<D.pos.length;pi++){const v=D.rl_fve[idx(h,pi)];if(v!=null&&v>best[2]&&D.first_w[idx(h,pi)]<0.9)best=[h,pi,v]}
S.h=best[0];S.pi=best[1];HS=headStats();renderThumbs();update()}
const tip=$('tip');$('hm').addEventListener('mousemove',e=>{const c=cellAt(e);if(!c){tip.style.display='none';return}const k=idx(c.h,c.pi);
tip.replaceChildren(el('b',{},`H${c.h} · #${D.pos[c.pi]} “${vis(D.tokens[D.pos[c.pi]])}”`),el('br'),`RL FVE ${pct(D.rl_fve[k])} · SFT ${pct(D.sft_fve[k])} · first-token ${(D.first_w[k]??0).toFixed(2)}`,el('br'),D.rl[k].split('\n')[0].slice(0,140));
tip.style.display='block';tip.style.left=Math.min(innerWidth-330,e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'});
$('hm').addEventListener('mouseleave',()=>tip.style.display='none');
$('hm').addEventListener('click',e=>{$('hm').focus({preventScroll:true});const c=cellAt(e);if(c){S.h=c.h;S.pi=c.pi;update()}});
// ← / → anywhere on the page: previous / next token for the current head. ↑ / ↓ switch heads only while the
// heatmap has focus, so they don't take over page scrolling. Ignored while typing or with modifier keys.
function keepVisible(){const w=$('hm').parentElement,x=LAB+S.pi*S.zoom;if(x<w.scrollLeft+LAB||x>w.scrollLeft+w.clientWidth-30)w.scrollLeft=Math.max(0,x-w.clientWidth/2)}
document.addEventListener('keydown',e=>{if(!D||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;const t=e.target.tagName;if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA')return;
const lr={ArrowLeft:-1,ArrowRight:1}[e.key],ud=e.target===$('hm')?{ArrowUp:-1,ArrowDown:1}[e.key]:undefined;if(lr==null&&ud==null)return;e.preventDefault();
if(lr!=null)S.pi=Math.max(0,Math.min(D.pos.length-1,S.pi+lr));if(ud!=null)S.h=Math.max(0,Math.min(D.n_heads-1,S.h+ud));update();keepVisible()});
$('metric').onchange=e=>{S.metric=e.target.value;drawHM()};$('osort').onchange=e=>{S.osort=e.target.value;renderThumbs()};$('t-sink').onclick=e=>{S.dropSink=!S.dropSink;e.currentTarget.setAttribute('aria-pressed',String(S.dropSink));renderThumbs()};$('zoom').onchange=e=>{S.zoom=+e.target.value;drawHM()};
for(const k of['sft','gold','hl'])$('t-'+k).onclick=e=>{S.show[k]=!S.show[k];e.currentTarget.setAttribute('aria-pressed',String(S.show[k]));renderDetail()};
$('theme').onclick=()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';LUTN={};drawHM();renderThumbs()};
openDoc(0);
</script></body></html>"""

if __name__ == "__main__":
    main()
