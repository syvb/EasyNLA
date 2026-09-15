"""Future Oracle explorer — click a token, read what the model is planning to say next.

ZeroGPU Space. Base models + every oracle adapter are loaded at module scope (ZeroGPU manages the
CUDA transfer); the GPU functions only run forwards. `Refresh oracles` rescans the HF dataset repo
and attaches newly trained oracles without a restart.
"""
from __future__ import annotations

import html as html_lib
import os
import random
import traceback

import gradio as gr
import torch

try:
    import spaces
except ImportError:                       # local CPU dev
    class _S:                             # noqa: D401
        @staticmethod
        def GPU(*a, **k):
            return (lambda f: f) if not (a and callable(a[0])) else a[0]
    spaces = _S()

import oracle as O

DEVICE = "cuda" if (torch.cuda.is_available() or os.environ.get("SPACE_ID")) else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_TOKENS = 512
K_MAX = 9

REGISTRY: dict[str, O.Oracle] = {}
BANK = O.Bank(device=DEVICE, dtype=DTYPE)


def refresh_registry() -> str:
    """Rescan the repo; attach new oracles (runs in the main process, so ZeroGPU workers see them)."""
    new = []
    only = {x.strip() for x in os.environ.get("FO_SIZES", "").split(",") if x.strip()}   # dev: e.g. FO_SIZES=0.6B
    for o in O.scan_registry():
        if only and o.size not in only:
            continue
        if o.run not in REGISTRY:
            try:
                BANK.attach(o)
            except Exception as e:           # keep serving the others
                print(f"[registry] failed to attach {o.run}: {e}")
                continue
            REGISTRY[o.run] = o
            new.append(o.name)
    if new:
        try:
            BANK.to_device()
        except Exception as e:
            return f"{len(REGISTRY)} oracles loaded; new adapters attached but could not be moved to the GPU ({e}); restart the Space"
    return (f"{len(REGISTRY)} oracles loaded" + (f"; new: {', '.join(new)}" if new else "; nothing new"))


STATUS = refresh_registry()
BANK.to_device()          # one CUDA move per base model (ZeroGPU packs the tensors here)
print("[startup]", STATUS)


def oracle_choices() -> list[str]:
    return [o.name for o in REGISTRY.values()]


def default_choices() -> list[str]:
    scan = [o.name for o in REGISTRY.values() if len(o.layers) == 1]   # the size-scan oracles
    return scan or oracle_choices()[:1]


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

CSS = """
.gradio-container{max-width:1320px !important; margin:0 auto !important;}
#fo-click-idx{display:none !important;}
.tokpanel{border:1px solid rgba(128,128,128,.35); border-radius:12px; overflow:hidden;}
.tokhead{display:flex; justify-content:space-between; padding:8px 14px; font-size:10.5px; letter-spacing:.05em;
  text-transform:uppercase; opacity:.7; border-bottom:1px solid rgba(128,128,128,.25);}
.tokscroll{padding:12px 14px 16px; max-height:50vh; overflow-y:auto; white-space:pre-wrap; overflow-wrap:anywhere;
  font-size:14px; line-height:2.0;}
.fo-tok{cursor:pointer; border-radius:4px; padding:2.5px 0;}
.fo-tok:nth-child(2n){background:rgba(128,128,128,.10);}
.fo-tok:hover{background:rgba(42,120,214,.25);}
.fo-tok.sel{background:#2a78d6; color:#fff;}
.fo-tok .nl{opacity:.6; font-size:10px;}
.res{border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:14px 16px; font-size:14px;}
.res table{border-collapse:collapse; width:100%;}
.res th{text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.05em; opacity:.7;
  padding:6px 8px; border-bottom:1px solid rgba(128,128,128,.3);}
.res td{padding:7px 8px; vertical-align:top; border-bottom:1px solid rgba(128,128,128,.15);}
.res .ctx{opacity:.75; font-family:ui-monospace,monospace; font-size:12.5px; margin-bottom:10px;}
.t{display:inline-block; border-radius:4px; padding:1px 4px; margin:1px 1px; font-family:ui-monospace,monospace;
  font-size:12.5px; background:rgba(128,128,128,.12); white-space:pre;}
.t.hit{background:rgba(46,160,67,.28);}
.t.miss{background:rgba(227,73,72,.22);}
.badge{display:inline-block; font-size:10.5px; padding:1px 6px; border-radius:8px; background:rgba(128,128,128,.18); margin-left:6px;}
.badge.ok{background:rgba(46,160,67,.28);} .badge.no{background:rgba(227,73,72,.22);}
.score{font-weight:600;}
"""

CLICK_JS = """
() => {
  const send = (t) => {
    document.querySelectorAll('.fo-tok.sel').forEach((x) => x.classList.remove('sel'));
    t.classList.add('sel');
    const box = document.querySelector('#fo-click-idx textarea, #fo-click-idx input');
    if (!box) return;
    box.value = t.dataset.i;
    box.dispatchEvent(new Event('input', { bubbles: true }));
  };
  document.addEventListener('click', (e) => { const t = e.target.closest('.fo-tok'); if (t) send(t); });
}
"""

EMPTY = '<div class="res">Tokenize a text, then click any token. The oracles read the residual-stream vector at that token and say what the model will produce next.</div>'


def render_tokens(pieces: list[str], n_total: int) -> str:
    spans = []
    for i, p in enumerate(pieces):
        body = html_lib.escape(p).replace("\n", '<span class="nl">⏎</span><br>')
        spans.append(f'<span class="fo-tok" data-i="{i}" title="#{i}">{body}</span>')
    trunc = f" (truncated to the first {len(pieces)})" if n_total > len(pieces) else ""
    head = f'<div class="tokhead"><span>{len(pieces)} tokens{trunc}</span><span>click a token to read its future</span></div>'
    return f'<div class="tokpanel">{head}<div class="tokscroll">{"".join(spans)}</div></div>'


def tok_spans(tok, ids: list[int], hits: list[bool] | None = None) -> str:
    out = []
    for i, t in enumerate(ids):
        cls = "t" + ("" if hits is None else (" hit" if hits[i] else " miss"))
        out.append(f'<span class="{cls}">{html_lib.escape(tok.decode([t]))}</span>')
    return "".join(out) or '<span class="t">∅</span>'


def render_results(tok, ids: list[int], t: int, rows: list[dict]) -> str:
    ctx_l = tok.decode(ids[max(0, t - 12): t])
    ctx_c = tok.decode([ids[t]])
    ctx_r = tok.decode(ids[t + 1: t + 9])
    head = (f'<div class="ctx">…{html_lib.escape(ctx_l)}<b style="background:#2a78d6;color:#fff;border-radius:3px;padding:0 3px">'
            f'{html_lib.escape(ctx_c)}</b>{html_lib.escape(ctx_r)}…</div>')
    trs = []
    for r in rows:
        if "error" in r:
            trs.append(f'<tr><td colspan="4">{html_lib.escape(r["oracle"])}: <i>{html_lib.escape(r["error"])}</i></td></tr>')
            continue
        n_hit = sum(r["hits"])
        ctrl = "" if r["control"] == "real" else f'<span class="badge">{r["control"]}</span>'
        top1 = ('<span class="badge ok" title="the model predicts the text\'s next token correctly here (the oracles were trained on such positions)">top-1 ✓</span>'
                if r["top1_correct"] else '<span class="badge no" title="the model\'s next-token prediction is wrong here: out of the oracle\'s training distribution">top-1 ✗</span>')
        who = html_lib.escape(f'Qwen3-{r["size"]} · L{r["layer"]}') + ctrl
        trs.append("<tr>"
                   f'<td><b>{who}</b><br><span style="opacity:.6;font-size:11px">{html_lib.escape(r["run"])}</span></td>'
                   f'<td>{tok_spans(tok, r["readout"], r["hits"])}<br><span class="score">{n_hit}/{r["k"]}</span> tokens match</td>'
                   f'<td>{tok_spans(tok, r["greedy"])} {top1 if r["control"] == "real" else ""}</td>'
                   f'<td>{tok_spans(tok, r["actual"])}</td></tr>')
    table = ('<table><tr><th>oracle</th><th>oracle readout (from the vector)</th>'
             '<th>model\'s own continuation (ground truth)</th><th>text continuation</th></tr>' + "".join(trs) + "</table>")
    return f'<div class="res">{head}{table}</div>'


# ----------------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------------

def tokenize(text: str):
    tok = BANK.tok
    ids = tok.encode(text or "", add_special_tokens=False)
    n_total = len(ids)
    ids = ids[:MAX_TOKENS]
    pieces = [tok.decode([i]) for i in ids]
    return render_tokens(pieces, n_total), ids, EMPTY


@spaces.GPU(duration=120)
def analyze(ids: list[int], idx: str, names: list[str], k: int, ctrls: list[str], layer8: int):
    if not ids or idx in (None, ""):
        return EMPTY
    t = int(idx)
    if t < 0 or t >= len(ids):
        return EMPTY
    tok = BANK.tok
    rows = []
    rng = random.Random(t)
    for name in names:
        o = next((x for x in REGISTRY.values() if x.name == name), None)
        if o is None:
            continue
        layer = o.layers[0] if len(o.layers) == 1 else (int(layer8) if int(layer8) in o.layers else o.layers[len(o.layers) // 2])
        conds = ["real"] + [c for c in ("none", "shuffled") if c in ctrls]
        for c in conds:
            try:
                st = rng.choice([i for i in range(len(ids)) if i != t]) if (c == "shuffled" and len(ids) > 1) else None
                rows.append(O.read_future(BANK, o, layer, ids, t, int(k), control=c, shuffle_t=st))
            except Exception as e:
                traceback.print_exc()
                rows.append({"oracle": name, "error": f"{type(e).__name__}: {e}"})
    return render_results(tok, ids, t, rows)


def do_refresh():
    msg = refresh_registry()
    return gr.update(choices=oracle_choices()), msg


DEFAULT_TEXTS = [
    "The Eiffel Tower is located in the capital of France, which is Paris. It was completed in 1889 and remains one of the most visited monuments in the world.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
    "Dear hiring manager,\n\nI am writing to apply for the position of software engineer at your company. I have five years of experience building distributed systems and",
]

with gr.Blocks(css=CSS, js=CLICK_JS, title="Future Oracle explorer") as demo:
    gr.Markdown(
        "# 🔮 Future Oracles — read what a model is about to say from one activation vector\n"
        "A **Future Oracle** is a LoRA decoder trained on a Qwen3 base model that takes a single residual-stream vector "
        "(the output of one transformer block at one token) and verbalises the tokens the *model itself* will produce next. "
        "Click any token: each oracle reads that token's vector at its layer and writes the next *k* tokens. "
        "Ground truth is the model's own greedy continuation from that token (green = the oracle's token matches it). "
        "Controls: **none** leaves the marker embedding untouched, **shuffled** injects another token's vector. "
        f"Adapters and evals: [`{O.REPO}`](https://huggingface.co/datasets/{O.REPO}).",
    )
    ids_state = gr.State([])
    click_idx = gr.Textbox(elem_id="fo-click-idx", visible=True)
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text_in = gr.Textbox(label="Text", lines=5, max_lines=12, value=DEFAULT_TEXTS[0])
            tokenize_btn = gr.Button("Tokenize", variant="primary")
            tokens_out = gr.HTML()
            gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in], label="Or try one of these")
        with gr.Column(scale=5):
            oracles_dd = gr.Dropdown(oracle_choices(), value=default_choices(), multiselect=True, label="Oracles to run")
            with gr.Row():
                k_in = gr.Slider(1, K_MAX, value=5, step=1, label="k — tokens to read")
                layer8_in = gr.Dropdown([8, 12, 16, 20, 24, 28, 32], value=24, label="layer for multi-layer oracles")
            ctrl_in = gr.CheckboxGroup(["none", "shuffled"], value=[], label="also run controls")
            with gr.Row():
                refresh_btn = gr.Button("Refresh oracles (pick up newly trained ones)")
                status = gr.Markdown(STATUS)
            results = gr.HTML(EMPTY)

    tokenize_btn.click(tokenize, [text_in], [tokens_out, ids_state, results])
    text_in.submit(tokenize, [text_in], [tokens_out, ids_state, results])
    click_idx.input(analyze, [ids_state, click_idx, oracles_dd, k_in, ctrl_in, layer8_in], [results])
    refresh_btn.click(do_refresh, [], [oracles_dd, status])
    demo.load(tokenize, [text_in], [tokens_out, ids_state, results])

demo.queue(default_concurrency_limit=2)
if __name__ == "__main__":
    demo.launch()
