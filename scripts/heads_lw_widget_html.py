"""Compact head-NLA viewer sized to be iframed into a LessWrong post widget (340–700 px wide): one page, several texts.

    python scripts/heads_lw_widget_html.py --doc <folder> --label <menu name> [--start HEAD:POS] [--doc ... --label ...] \\
        --out-dir <dir> [--default <folder name>]

Each folder is one document of the doc viewer (token_ids.npy, meta.json, sites.parquet, attn.npz, mtry.parquet: e.g.
l24/mtry/doc_viewer/<doc>/ + l24/std_rl/doc_viewer/<doc>/ of the HF dataset) or one chat, l24/chat_viewer/<chat>/
(chat-template tokens muted, each turn labelled User / Assistant, opened scrolled to the assistant's reply). A menu picks
the text. For each, the page shows every token × every head of the head NLA's explanations, matryoshka (step 640) or
standard RL (step 400) by a toggle:
  - the text, shaded by the selected head's attention from the selected token, or by that head's FVE at each token;
  - the 32 heads at the selected token, coloured by FVE;
  - the selected site's explanation.
The page fills its iframe's height (the panes scroll), so the embedding sets the height.

Output: <out-dir>/widget.html (small; the menu and data versions inline) and, per text, data/<doc>.js (tokens,
attention, matryoshka explanations) and data/<doc>.std.js (standard explanations, fetched on the first switch). Data
files are gzip+base64 JSON loaded by <script> tag, so they need no CORS headers even from a sandboxed iframe; their URLs
carry a content hash, so a new build is never served from a stale cache. URL options: ?theme=dark|auto, ?chat=<doc>.
"""

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_doc_viewer_html import chat_roles  # noqa: E402


ATT_LEVELS = 31


def display_tokens(tok, ids: list[int]) -> list[str]:
    """Each token's text as it adds to the decoded document, so a character split over two tokens shows whole on the
    second instead of as two replacement characters."""
    out, prev = [], ""
    for k in range(len(ids)):
        cur = tok.decode(ids[: k + 1])
        out.append(cur[len(prev):] if cur.startswith(prev) else tok.decode([ids[k]]))
        prev = cur
    return out


def b64gz(b: bytes) -> str:
    return base64.b64encode(gzip.compress(b, 9, mtime=0)).decode()     # mtime=0: same data, same bytes, same hash


def doc_parts(tok, d: Path, label: str, start: str | None) -> tuple[dict, dict, str]:
    """One document → (data-file parts {base, att, mtry}, {rl}, log line). Each part is gzip+base64 JSON (att: raw bytes)."""
    meta = json.loads((d / "meta.json").read_text())
    ids = np.load(d / "token_ids.npy").tolist()
    T, H = len(ids), int(meta["n_heads"])
    s = pq.read_table(d / "sites.parquet", columns=["pos", "head", "rl_text", "rl_cos", "rl_fve"]).to_pandas()
    mt = pq.read_table(d / "mtry.parquet", columns=["pos", "head", "mtry_text", "mtry_cos", "mtry_fve"]).to_pandas()
    s = s.merge(mt, on=["pos", "head"], validate="one_to_one").sort_values(["head", "pos"]).reset_index(drop=True)
    assert len(s) == H * T and (s.pos.to_numpy() == np.tile(np.arange(T), H)).all(), "expected every token × every head"
    # per-site FVE denominator (the head's mean training write as the guess), recovered from the stored RL scores
    base = 2 * (1 - s.rl_cos) / np.maximum(1 - s.rl_fve, 1e-6)
    assert np.allclose(1 - 2 * (1 - s.mtry_cos) / base, s.mtry_fve, atol=2e-3), "mtry_fve is not on the RL site baseline"
    models = {}
    for key in ("mtry", "rl"):
        err = 2 * (1 - s[f"{key}_cos"])
        models[key] = {"fve": [round(float(x), 3) for x in s[f"{key}_fve"]], "text": s[f"{key}_text"].tolist(),
                       "head_fve": [round(float(1 - err[s["head"] == h].sum() / base[s["head"] == h].sum()), 3) for h in range(H)],
                       "doc_fve": round(float(1 - err.sum() / base.sum()), 3)}
    attn = np.load(d / "attn.npz")["attn"].astype(np.float32)                          # [H, T, T]
    if start:
        h0, t0 = map(int, start.split(":"))
    else:
        ok = (s.pos >= 60) & (attn[:, :, 0][s["head"], s.pos] < 0.3)
        r = s[ok].sort_values("mtry_fve", ascending=False).iloc[0]
        h0, t0 = int(r["head"]), int(r.pos)
    # causal attention rows for shading: √w (small weights stay visible) in ATT_LEVELS steps, concatenated per head then
    # per query token; the top keys and their exact weights are listed separately
    att = np.concatenate([np.sqrt(np.clip(attn[h, t, : t + 1], 0, 1)) for h in range(H) for t in range(T)])
    reads = []
    for h in range(H):
        for t in range(T):
            w = attn[h, t, : t + 1]
            top = np.argsort(-w, kind="stable")[:3]
            reads.append([[int(j), round(float(w[j]), 3)] for j in top if w[j] >= 0.03])
    tokens = display_tokens(tok, ids)
    base_d = {"layer": int(meta.get("layer", 24)), "n_heads": H, "T": T, "label": label, "tokens": tokens,
              "roles": chat_roles(tokens), "reads": reads, "att_levels": ATT_LEVELS, "start": [h0, t0]}
    js = lambda o: b64gz(json.dumps(o, ensure_ascii=False).encode())                    # noqa: E731
    parts = {"base": js(base_d), "att": b64gz((att * ATT_LEVELS).round().astype(np.uint8).tobytes()), "mtry": js(models["mtry"])}
    line = (f"{d.name}: {T} tokens × {H} heads, start H{h0} #{t0}; pooled FVE matryoshka {models['mtry']['doc_fve']:+.3f}, "
            f"standard {models['rl']['doc_fve']:+.3f}")
    return parts, {"rl": js(models["rl"])}, line


def write_data(path: Path, name: str, parts: dict) -> str:
    """data/<name>.js: one call to the page's loader. Loaded by a <script> tag, so no CORS is needed even when the page
    runs with an opaque origin (a sandboxed LessWrong widget). Returns a content hash for cache-busting."""
    body = f"HEADNLA_LOAD({json.dumps(name)},{json.dumps(parts)});\n"
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", action="append", required=True, help="a doc-viewer or chat-viewer folder (repeatable, in menu order)")
    p.add_argument("--label", action="append", required=True, help="menu name of each --doc, in the same order")
    p.add_argument("--start", action="append", default=None, help="HEAD:POS selected when each --doc opens, or 'auto'")
    p.add_argument("--default", default=None, help="folder name of the --doc shown first (default: the first)")
    p.add_argument("--out-dir", required=True, help="writes widget.html and data/<doc>.js, data/<doc>.std.js")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    a = p.parse_args()
    starts = a.start or ["auto"] * len(a.doc)
    assert len(a.label) == len(a.doc) == len(starts), "give one --label (and --start, if any) per --doc"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    out = Path(a.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    chats = []
    for d, label, st in zip(map(Path, a.doc), a.label, starts):
        main_parts, std_parts, line = doc_parts(tok, d, label, None if st == "auto" else st)
        v = write_data(out / "data" / f"{d.name}.js", d.name, main_parts)
        vs = write_data(out / "data" / f"{d.name}.std.js", f"{d.name}.std", std_parts)
        chats.append({"key": d.name, "label": label, "v": v, "vs": vs})
        mb = lambda f: (out / "data" / f).stat().st_size / 1e6                          # noqa: E731
        print(f"{line}; data {mb(d.name + '.js'):.1f} MB + standard {mb(d.name + '.std.js'):.1f} MB", flush=True)
    keys = [c["key"] for c in chats]
    manifest = {"chats": chats, "default": a.default or keys[0]}
    assert manifest["default"] in keys, f"--default {a.default} is not one of {keys}"
    (out / "widget.html").write_text(TEMPLATE.replace("__MANIFEST__", json.dumps(manifest).replace("</", "<\\/")))
    print(f"wrote {out / 'widget.html'} ({(out / 'widget.html').stat().st_size / 1e3:.0f} kB) and {len(chats) * 2} data files")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head NLA explorer</title>
<script>try{const t=new URLSearchParams(location.search).get('theme');if(t==='dark'||t==='auto')document.documentElement.dataset.theme=t}catch(e){}</script>
<!-- ?theme=dark|auto (default light); ?chat=<key> opens that chat instead of the default -->
<style>
:root{--bg:#ffffff;--panel:#faf9f7;--border:#e3e2dd;--ink:#1a1a19;--ink-2:#52514e;--ink-3:#8a8984;--accent:#2a78d6;--sel:#1a1a19;--pill:#f3f2ef;--user:#7a5195;--asst:#1f7f73}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#8f8e87;--accent:#5b9ae8;--sel:#f0efec;--pill:#2a2a28;--user:#8d62ad;--asst:#2a8d80}
@media (prefers-color-scheme:dark){:root[data-theme="auto"]{--bg:#1a1a19;--panel:#20201e;--border:#3a3a37;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#8f8e87;--accent:#5b9ae8;--sel:#f0efec;--pill:#2a2a28;--user:#8d62ad;--asst:#2a8d80}}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font:13px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-text-size-adjust:100%}
#app{height:100vh;min-height:360px;max-width:760px;margin:0 auto;display:grid;grid-template-rows:auto minmax(70px,1fr) auto minmax(90px,1.2fr) auto;gap:6px;padding:8px 10px 6px;position:relative}
/* title (with the chat menu) left, controls right; when both don't fit, the controls wrap as one group onto a second row */
.top{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:6px 8px}
.ttl{font-weight:650;white-space:nowrap;display:flex;align-items:center;gap:5px;flex:none}.ttl .sub{font-weight:400;color:var(--ink-3)}
.ctl{display:flex;align-items:center;gap:8px;flex:none}@media (max-width:420px){.mdl{display:none}}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:7px;overflow:hidden;flex:none}
.seg button{border:0;border-radius:0;background:var(--bg);padding:3px 8px}.seg button+button{border-left:1px solid var(--border)}
button{font:inherit;font-size:12px;color:var(--ink);background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:3px 8px;cursor:pointer;line-height:1.3}
button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.shl{color:var(--ink-3);font-size:12px;flex:none}
#chat{font:inherit;font-size:12.5px;font-weight:400;color:var(--ink);background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:2px 4px;cursor:pointer;flex:none}
#chat:focus-visible{outline:2px solid var(--accent);outline-offset:1px}@media (max-width:420px){.shl{display:none}}
.pane{border:1px solid var(--border);border-radius:9px;background:var(--panel);overflow-y:auto;min-height:0;overscroll-behavior:contain}
#ctx{padding:6px 9px;white-space:pre-wrap;word-break:break-word;line-height:1.75;font-size:13px}
#ctx .t{border-radius:3px;cursor:pointer;padding:1px 0}#ctx .t.fut{color:var(--ink-3)}
#ctx .t.cur{outline:2px solid var(--sel);outline-offset:0;font-weight:650}
#ctx .t.tpl{color:var(--ink-3);font-family:ui-monospace,Menlo,monospace;font-size:.82em}
#ctx .turn{display:inline-block;margin:4px 6px 2px 0;padding:0 7px;border-radius:999px;font:600 10.5px/1.6 system-ui,sans-serif;letter-spacing:.03em;color:#fff;background:var(--ink-3)}
#ctx .turn.user{background:var(--user)}#ctx .turn.assistant{background:var(--asst)}
.hh{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ink-2);margin-bottom:4px;min-width:0}
.hh .q{color:var(--ink);font-weight:650;white-space:pre;overflow:hidden;text-overflow:ellipsis;max-width:40%}.hh .sp{flex:1}
.hh button{padding:0 7px;font-size:13px}
#heads{display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:2px}@media (min-width:600px){#heads{grid-template-columns:repeat(32,minmax(0,1fr))}}
#heads button{height:20px;padding:0;border:0;border-radius:3px;font-size:10px;font-variant-numeric:tabular-nums;line-height:20px;text-align:center}
#heads button.on{outline:2px solid var(--sel);outline-offset:1px;font-weight:700}
#detail{padding:7px 10px}
.dh{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 8px}.dh b{font-size:14px}
.pill{font-weight:650;font-variant-numeric:tabular-nums;border-radius:999px;padding:0 8px;font-size:12.5px}
.mut{color:var(--ink-3);font-size:12px}.rd{font-size:12px;color:var(--ink-2);margin:3px 0 0}.rd .tk{white-space:pre;background:var(--pill);border-radius:4px;padding:0 3px;color:var(--ink)}
.ex{margin-top:6px;padding-left:9px;border-left:3px solid var(--ink-3);white-space:pre-wrap;word-break:break-word;font-size:13px}
.lg{display:flex;flex-wrap:wrap;align-items:center;gap:2px 12px;font-size:11px;color:var(--ink-3)}.lg span{display:inline-flex;align-items:center;gap:4px}
.lg canvas{width:70px;height:8px;border-radius:2px}.lg .k{font-variant-numeric:tabular-nums}.lg .hint{margin-left:auto}@media (max-width:520px){.lg .hint{display:none}}
#load{color:var(--ink-3);padding:6px 2px}
</style></head><body>
<div id="app">
<div class="top"><span class="ttl">Head NLA <span class="sub">·</span><select id="chat" aria-label="Chat"></select></span><span class="ctl"><span class="shl">Shade text by</span>
<span class="seg" role="group" aria-label="Shade text by"><button id="m-att" aria-pressed="true" title="the selected head's attention from the selected token">Attention</button><button id="m-fve" aria-pressed="false" title="the selected head's FVE at each token">FVE</button></span>
<span class="seg" role="group" aria-label="Head NLA model"><button id="x-mtry" aria-pressed="true" title="matryoshka-trained (RL step 640): puts the most informative content first">Matryoshka</button><button id="x-rl" aria-pressed="false" title="standard RL training (step 400)">Standard</button></span></span></div>
<div class="pane" id="ctx" aria-label="Text; click a token"><div id="load">Loading…</div></div>
<div><div class="hh"><span><span class="mdl">Qwen3-8B </span>L24 heads at</span><span class="q" id="qtok"></span><span class="mut" id="qpos"></span><span class="sp"></span><button id="prev" aria-label="Previous token">‹</button><button id="next" aria-label="Next token">›</button></div>
<div id="heads" role="group" aria-label="Attention heads, coloured by FVE at this token"></div></div>
<div class="pane" id="detail" aria-live="polite"></div>
<div class="lg"><span>FVE <span class="k">−100%</span><canvas id="r-fve" width="140" height="8"></canvas><span class="k">+100%</span></span>
<span id="lg-att">attention <span class="k">0</span><canvas id="r-att" width="140" height="8"></canvas><span class="k">100%</span></span><span class="hint">← → tokens · ↑ ↓ heads</span></div>
</div>
<script id="manifest" type="application/json">__MANIFEST__</script>
<script>
const $=id=>document.getElementById(id);
const MAN=JSON.parse($('manifest').textContent);
const S={h:0,t:0,mode:'att',model:'mtry',chat:null};let D=null,SPANS=[],CELLS=[],GEN=0;
const LABEL={mtry:'Matryoshka NLA explanation',rl:'Standard NLA explanation (RL step 400)'};
const M=()=>D.models[S.model];
async function gunzip(b64){const b=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 return new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())}
// Data files (data/<name>.js) call HEADNLA_LOAD. They load by <script> tag rather than fetch(): inside a sandboxed
// widget iframe this page has an opaque origin, and a script tag needs no CORS headers where fetch would.
const PENDING={},FILES={},CHATS={};
window.HEADNLA_LOAD=(name,parts)=>{const r=PENDING[name];delete PENDING[name];if(r)r(parts)};
function file(name,v){
 if(!FILES[name])FILES[name]=new Promise((res,rej)=>{PENDING[name]=res;const sc=document.createElement('script');
  sc.src='data/'+name+'.js?v='+v;sc.onerror=()=>{delete FILES[name];delete PENDING[name];rej(new Error('could not load data/'+name+'.js'))};document.head.append(sc)})
  .then(async p=>{const o={};for(const[k,b]of Object.entries(p))o[k]=await gunzip(b);return o});
 return FILES[name]}
const json=u=>JSON.parse(new TextDecoder().decode(u));
async function chatData(key){
 if(CHATS[key])return CHATS[key];const c=MAN.chats.find(x=>x.key===key),p=await file(key,c.v),d=json(p.base);
 d.att=p.att;d.tri=d.T*(d.T+1)/2;d.models={mtry:json(p.mtry)};return CHATS[key]=d}
async function stdData(d,key){if(!d.models.rl){const c=MAN.chats.find(x=>x.key===key);d.models.rl=json((await file(key+'.std',c.vs)).rl)}}
const pct=x=>x==null?'n/a':(x>=0?'+':'−')+Math.abs(Math.round(100*x))+'%';
const vis=s=>s.replace(/\n/g,'⏎').replace(/\t/g,'⇥')||'∅';
const quote=s=>'“'+(s.trim()||vis(s))+'”';
// Colour scales (dataviz reference palette, interpolated in OKLab): FVE is DIVERGING around 0 (blue below, grey
// midpoint, red above); attention is SEQUENTIAL violet, by √weight so small weights stay visible.
const STOPS={light:{div:['#0d366b','#3987e5','#f0efec','#e66767','#9b1c1c'],seq:['#faf9f7','#e2d8f6','#c0a6ec','#9673da','#6c46bd','#432485']},
 dark:{div:['#b7d3f6','#3987e5','#383835','#d24d4d','#ffb4ab'],seq:['#20201e','#3b2964','#593b96','#7b5ac2','#a688df','#d6c6f5']}};
function toLin(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4)}function toSrgb(c){c=c<=0.0031308?12.92*c:1.055*Math.pow(c,1/2.4)-0.055;return Math.round(255*Math.max(0,Math.min(1,c)))}
function hex2lab(h){const n=parseInt(h.slice(1),16),r=toLin(n>>16&255),g=toLin(n>>8&255),b=toLin(n&255);
const l=Math.cbrt(0.4122214708*r+0.5363325363*g+0.0514459929*b),m=Math.cbrt(0.2119034982*r+0.6806995451*g+0.1073969566*b),s=Math.cbrt(0.0883024619*r+0.2817188376*g+0.6299787005*b);
return[0.2104542553*l+0.7936177850*m-0.0040720468*s,1.9779984951*l-2.4285922050*m+0.4505937099*s,0.0259040371*l+0.7827717662*m-0.8086757660*s]}
function lab2rgb([L,A,B]){const l=(L+0.3963377774*A+0.2158037573*B)**3,m=(L-0.1055613458*A-0.0638541728*B)**3,s=(L-0.0894841775*A-1.2914855480*B)**3;
return[toSrgb(4.0767416621*l-3.3077115913*m+0.2309699292*s),toSrgb(-1.2684380046*l+2.6097574011*m-0.3413193965*s),toSrgb(-0.0041960863*l-0.7034186147*m+1.7076147010*s)]}
const isDark=()=>{const t=document.documentElement.dataset.theme;return t==='dark'||(t==='auto'&&!!window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches)};
let LUT={};
function lut(kind){const key=kind+(isDark()?'d':'l');if(LUT[key])return LUT[key];const st=STOPS[isDark()?'dark':'light'][kind].map(hex2lab),out=[];
 for(let i=0;i<256;i++){const x=i/255*(st.length-1),k=Math.min(st.length-2,Math.floor(x)),f=x-k,a=st[k],b=st[k+1],lab=[0,1,2].map(j=>a[j]+(b[j]-a[j])*f);
  const [r,g,bl]=lab2rgb(lab);out.push({bg:`rgb(${r},${g},${bl})`,fg:lab[0]<0.62?'#fff':'#1a1a19'})}return LUT[key]=out}
// FVE in [−1, 1] → diverging; values below −100% are shown as −100%
const fveC=v=>lut('div')[Math.round(Math.max(0,Math.min(1,(v+1)/2))*255)];
const attC=q=>lut('seq')[q];
function ramp(id,kind){const c=$(id),x=c.getContext('2d'),L=lut(kind);for(let i=0;i<c.width;i++){x.fillStyle=L[Math.round(i/(c.width-1)*255)].bg;x.fillRect(i,0,1,c.height)}}
const k=(h,t)=>h*D.T+t,off=(h,t)=>h*D.tri+t*(t+1)/2;
function attRow(h,t){return D.att.subarray(off(h,t),off(h,t)+t+1)}     // √w × att_levels for key positions 0..t
function el(t,a={},...c){const e=document.createElement(t);for(const[x,v]of Object.entries(a)){if(x==='class')e.className=v;else if(x==='style')e.style.cssText=v;else e.setAttribute(x,v)}for(const y of c)if(y!=null)e.append(y);return e}
function build(){
 // chats: template tokens muted; a User / Assistant label where each turn opens (role names come from chat_roles)
 const ctx=$('ctx'),R=D.roles;ctx.replaceChildren();SPANS=D.tokens.map((s,t)=>{
  if(R&&s==='<|im_start|>'&&t+1<D.T){const r=R[t+1].replace('~','');ctx.append(el('span',{class:'turn '+r},r==='user'?'User':r==='assistant'?'Assistant':r))}
  const e=el('span',{class:'t'+(R&&R[t].startsWith('~')?' tpl':'')},s);e.dataset.t=t;ctx.append(e);return e});
 const hd=$('heads');hd.replaceChildren();CELLS=[];
 for(let h=0;h<D.n_heads;h++){const b=el('button',{'aria-label':'Head '+h},String(h));b.onclick=()=>sel(h,S.t);hd.append(b);CELLS.push(b)}
}
$('ctx').addEventListener('click',e=>{const t=e.target.closest('.t');if(t&&D)sel(S.h,+t.dataset.t)});
// chats open scrolled to the assistant's reply: its first non-template token at the top of the pane (scroll up for the
// user turn and the template)
function scrollToReply(){const R=D.roles,a=R?R.findIndex(r=>r==='assistant'):-1;if(a<0)return;const box=$('ctx');
 box.scrollTop+=SPANS[a].getBoundingClientRect().top-box.getBoundingClientRect().top-6}
function keepVisible(box,e){const b=box.getBoundingClientRect(),r=e.getBoundingClientRect();if(r.top<b.top+4)box.scrollTop-=b.top+4-r.top;else if(r.bottom>b.bottom-4)box.scrollTop+=r.bottom-b.bottom+4}
function sel(h,t,scroll=true){
 S.h=Math.max(0,Math.min(D.n_heads-1,h));S.t=Math.max(0,Math.min(D.T-1,t));h=S.h;t=S.t;
 // text shading
 const row=S.mode==='att'?attRow(h,t):null;
 SPANS.forEach((e,j)=>{let c=null;if(S.mode==='att'){if(j<=t)c=attC(Math.round(row[j]/D.att_levels*255))}else{const v=M().fve[k(h,j)];if(v!=null)c=fveC(v)}
  e.style.background=c?c.bg:'';e.style.color=c?c.fg:'';e.classList.toggle('fut',S.mode==='att'&&j>t);e.classList.toggle('cur',j===t)});
 if(scroll)keepVisible($('ctx'),SPANS[t]);
 // heads at this token
 CELLS.forEach((b,x)=>{const v=M().fve[k(x,t)],c=fveC(v);b.style.background=c.bg;b.style.color=c.fg;b.classList.toggle('on',x===h);
  b.title=`H${x}: FVE ${pct(v)} at this token (${pct(M().head_fve[x])} over the whole text)`;b.setAttribute('aria-pressed',x===h)});
 $('qtok').textContent=quote(D.tokens[t]);$('qpos').textContent=`token ${t+1} of ${D.T}`;
 // detail
 const v=M().fve[k(h,t)],c=fveC(v),reads=D.reads[k(h,t)];
 const rd=el('div',{class:'rd'},'Reads from: ');
 reads.forEach(([j,w],i)=>{if(i)rd.append(' · ');rd.append(el('span',{class:'tk'},j===t?'itself':j===0?'first token':D.tokens[j].trim()||vis(D.tokens[j])),' '+Math.round(100*w)+'%')});
 if(!reads.length)rd.append('spread thin (no token above 3%)');
 $('detail').replaceChildren(
  el('div',{class:'dh'},el('b',{},`Head ${h}`),el('span',{class:'pill',style:`background:${c.bg};color:${c.fg}`},'FVE '+pct(v)),el('span',{class:'mut'},`this head over the whole text: ${pct(M().head_fve[h])}`)),
  rd,el('div',{class:'ex'},M().text[k(h,t)]),
  el('div',{class:'mut',style:'margin-top:5px'},LABEL[S.model]+' · one sample at temperature 1'));
 $('detail').scrollTop=0;
}
function setMode(m){S.mode=m;$('m-att').setAttribute('aria-pressed',m==='att');$('m-fve').setAttribute('aria-pressed',m==='fve');$('lg-att').style.visibility=m==='att'?'visible':'hidden';if(D)sel(S.h,S.t,false)}
$('m-att').onclick=()=>setMode('att');$('m-fve').onclick=()=>setMode('fve');
const loading=msg=>$('detail').replaceChildren(el('div',{class:'mut'},msg));
const failed=e=>loading('Could not load the data: '+e.message);
async function setModel(m){
 $('x-mtry').setAttribute('aria-pressed',m==='mtry');$('x-rl').setAttribute('aria-pressed',m==='rl');
 const g=++GEN;if(D&&m==='rl'&&!D.models.rl){loading('Loading the standard explanations…');try{await stdData(D,S.chat)}catch(e){if(g===GEN)failed(e);return}}
 if(g!==GEN)return;S.model=m;if(D)sel(S.h,S.t,false)}
// open a chat at its default site, scrolled to the assistant's reply; keeps the shading and model choices
async function openChat(key){
 const g=++GEN;S.chat=key;$('chat').value=key;D=null;$('ctx').replaceChildren(el('div',{id:'load'},'Loading…'));loading('');
 let d;try{d=await chatData(key);if(S.model==='rl')await stdData(d,key)}catch(e){if(g===GEN){$('ctx').textContent='';failed(e)}return}
 if(g!==GEN)return;D=d;build();sel(D.start[0],D.start[1],!D.roles);if(D.roles)scrollToReply()}
for(const c of MAN.chats)$('chat').append(el('option',{value:c.key},c.label));
$('chat').onchange=e=>openChat(e.target.value);
$('x-mtry').onclick=()=>setModel('mtry');$('x-rl').onclick=()=>setModel('rl');
$('prev').onclick=()=>sel(S.h,S.t-1);$('next').onclick=()=>sel(S.h,S.t+1);
document.addEventListener('keydown',e=>{if(!D||e.metaKey||e.ctrlKey||e.altKey)return;const m={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[e.key];
 if(!m)return;e.preventDefault();sel(S.h+m[0],S.t+m[1])});
ramp('r-fve','div');ramp('r-att','seq');
{let q=null;try{q=new URLSearchParams(location.search).get('chat')}catch(e){}
 openChat(MAN.chats.some(c=>c.key===q)?q:MAN.default)}
if(window.matchMedia)matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change',()=>{if(D){LUT={};ramp('r-fve','div');ramp('r-att','seq');sel(S.h,S.t,false)}});
</script></body></html>
"""

if __name__ == "__main__":
    main()
