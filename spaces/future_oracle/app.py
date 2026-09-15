"""Future Oracle explorer — click a token, read what the model is planning to say next.

ZeroGPU Space. Base models + every oracle adapter are assembled on CPU at module scope and moved to
CUDA once (ZeroGPU packs the tensors); the GPU functions only run forwards. `Refresh oracles`
rescans the HF dataset repo and attaches newly trained oracles without a restart.
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
LAYER_CHOICES = [8, 12, 16, 20, 24, 28, 32]

REGISTRY: dict[str, O.Oracle] = {}     # run -> oracle
LABELS: dict[str, str] = {}            # short label -> run
BANK = O.Bank(device=DEVICE, dtype=DTYPE)


def short_label(o: O.Oracle) -> str:
    lay = f"L{o.layers[0]}" if len(o.layers) == 1 else f"L{o.layers[0]}–{o.layers[-1]} (multi-layer)"
    tag = " · RL" if o.run.startswith("rl_") else ""
    if len(o.layers) > 1:      # the 8B profile runs differ only in the objective
        tag += " · distilled" if o.extra.get("distill") else " · hard labels"
    return f"Qwen3-{o.size} · {lay}{tag}"


def refresh_registry() -> str:
    """Rescan the repo; attach new oracles (runs in the main process, so ZeroGPU workers see them)."""
    new = []
    only = {x.strip() for x in os.environ.get("FO_SIZES", "").split(",") if x.strip()}   # dev: FO_SIZES=0.6B
    for o in O.scan_registry():
        if only and o.size not in only:
            continue
        if o.run not in REGISTRY:
            try:
                BANK.attach(o)
            except Exception as e:           # keep serving the others
                print(f"[registry] failed to attach {o.run}: {e}")
                traceback.print_exc()
                continue
            REGISTRY[o.run] = o
            lab = short_label(o)
            while lab in LABELS:
                lab += " ′"
            LABELS[lab] = o.run
            new.append(lab)
    if new:
        try:
            BANK.to_device()
        except Exception as e:
            return f"{len(REGISTRY)} oracles loaded; new adapters attached but could not be moved to the GPU ({e}); restart the Space"
    return f"{len(REGISTRY)} oracles loaded" + (f" · new: {', '.join(new)}" if new else "")


STATUS = refresh_registry()
BANK.to_device()          # one CUDA move per base model (ZeroGPU packs the tensors here)
print("[startup]", STATUS)


def oracle_choices() -> list[str]:
    return list(LABELS)


def default_choices() -> list[str]:
    single = [lab for lab, run in LABELS.items() if len(REGISTRY[run].layers) == 1]   # the size-scan oracles
    return single or oracle_choices()[:1]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

CSS = """
.gradio-container{max-width:1180px !important; margin:0 auto !important;}
#fo-click-idx{display:none !important;}
#fo-title h1{font-size:26px; margin:0 0 2px 0;}
#fo-title p{margin:4px 0 0 0; opacity:.8;}

/* token panel */
.tokpanel{border:1px solid rgba(128,128,128,.35); border-radius:12px; overflow:hidden;}
.tokhead{display:flex; justify-content:space-between; padding:7px 14px; font-size:10.5px; letter-spacing:.06em;
  text-transform:uppercase; opacity:.65; border-bottom:1px solid rgba(128,128,128,.25);}
.tokscroll{padding:12px 14px 14px; max-height:40vh; overflow-y:auto; white-space:pre-wrap; overflow-wrap:anywhere;
  font-size:15px; line-height:2.05;}
.fo-tok{cursor:pointer; border-radius:4px; padding:3px 0;}
.fo-tok:nth-child(2n){background:rgba(128,128,128,.10);}
.fo-tok:hover{background:rgba(42,120,214,.28);}
.fo-tok.sel{background:#2a78d6; color:#fff;}
.fo-tok .nl{opacity:.6; font-size:10px;}

/* results */
.res-empty{padding:18px; border:1px dashed rgba(128,128,128,.4); border-radius:12px; opacity:.75;}
.ctx{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; opacity:.85; margin:2px 0 12px 0;
  white-space:pre-wrap; overflow-wrap:anywhere;}
.ctx .here{background:#2a78d6; color:#fff; border-radius:3px; padding:0 3px;}
.legend{font-size:12px; opacity:.7; margin:0 0 10px 0;}
.cards{display:grid; grid-template-columns:repeat(auto-fit, minmax(440px, 1fr)); gap:12px;}
.card{border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:10px 14px 8px;}
.card-h{display:flex; align-items:baseline; gap:8px; margin-bottom:6px;}
.card-h .who{font-weight:650; font-size:15px;}
.card-h .dim{opacity:.6; font-size:12px;}
.card-h .score{margin-left:auto; font-weight:650; font-size:14px;}
.card-h .score.good{color:#2ea043;} .card-h .score.meh{color:#c98500;} .card-h .score.bad{color:#e34948;}
.strip{display:flex; align-items:center; gap:6px; padding:5px 0; border-top:1px solid rgba(128,128,128,.15);
  flex-wrap:wrap;}
.strip .lab{flex:0 0 118px; font-size:10.5px; letter-spacing:.05em; text-transform:uppercase; opacity:.6;}
.t{display:inline-block; border-radius:5px; padding:2px 6px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:13px; background:rgba(128,128,128,.13); white-space:pre; line-height:1.5;}
.t.hit{background:rgba(46,160,67,.30);}
.t.miss{background:rgba(227,73,72,.24);}
.t.empty{opacity:.5;}
.badge{display:inline-block; font-size:10.5px; padding:1px 7px; border-radius:9px; background:rgba(128,128,128,.18);}
.badge.ok{background:rgba(46,160,67,.28);} .badge.no{background:rgba(227,73,72,.22);}
.badge.ctrl{background:rgba(201,133,0,.25);}
.card.ctrl{border-style:dashed;}
.err{opacity:.8; font-size:12px;}
#fo-foot{opacity:.7; font-size:12px;}
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
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const t = e.target.closest && e.target.closest('.fo-tok');
    if (t) { e.preventDefault(); send(t); }
  });
}
"""

EMPTY = ('<div class="res-empty">Click a token above. Each oracle reads the residual-stream vector at that token '
         'from its own model and writes the tokens the model is about to produce.</div>')


def render_tokens(pieces: list[str], n_total: int) -> str:
    spans = []
    for i, p in enumerate(pieces):
        body = html_lib.escape(p).replace("\n", '<span class="nl">⏎</span><br>')
        spans.append(f'<span class="fo-tok" data-i="{i}" title="token #{i}" role="button" tabindex="0">{body}</span>')
    trunc = f" · truncated to the first {len(pieces)}" if n_total > len(pieces) else ""
    head = f'<div class="tokhead"><span>{len(pieces)} tokens{trunc}</span><span>click a token to read its future</span></div>'
    return f'<div class="tokpanel">{head}<div class="tokscroll">{"".join(spans)}</div></div>'


def tok_spans(tok, ids: list[int], hits: list[bool] | None = None) -> str:
    if not ids:
        return '<span class="t empty">(end of text)</span>'
    out = []
    for i, t in enumerate(ids):
        cls = "t" + ("" if hits is None else (" hit" if hits[i] else " miss"))
        out.append(f'<span class="{cls}">{html_lib.escape(tok.decode([t]))}</span>')
    return "".join(out)


def render_results(tok, ids: list[int], t: int, rows: list[dict]) -> str:
    ctx_l = html_lib.escape(tok.decode(ids[max(0, t - 16): t]))
    ctx_c = html_lib.escape(tok.decode([ids[t]]))
    ctx_r = html_lib.escape(tok.decode(ids[t + 1: t + 10]))
    head = f'<div class="ctx">…{ctx_l}<span class="here">{ctx_c}</span>{ctx_r}…</div>'
    legend = ('<div class="legend"><span class="t hit">green</span> = the oracle\'s token equals what the model itself '
              'would say at that offset · <span class="t miss">red</span> = differs · the text\'s real continuation is '
              'shown for reference only</div>')
    cards = []
    for r in rows:
        if "error" in r:
            cards.append(f'<div class="card"><div class="card-h"><span class="who">{html_lib.escape(r["oracle"])}</span></div>'
                         f'<div class="err">{html_lib.escape(r["error"])}</div></div>')
            continue
        n_hit, k = sum(r["hits"]), r["k"]
        cls = "good" if n_hit >= 0.6 * k else ("meh" if n_hit >= 0.3 * k else "bad")
        ctrl = "" if r["control"] == "real" else f'<span class="badge ctrl">control: {r["control"]}</span>'
        top1 = ('<span class="badge ok" title="the model predicts the text\'s next token correctly here; the oracles were trained on such positions">in-distribution</span>'
                if r["top1_correct"] else
                '<span class="badge no" title="the model\'s own next-token guess is wrong here; the oracles never trained on such positions">off-distribution</span>')
        cards.append(
            f'<div class="card{" ctrl" if ctrl else ""}">'
            f'<div class="card-h"><span class="who">Qwen3-{html_lib.escape(r["size"])}</span>'
            f'<span class="dim">layer {r["layer"]}</span>{ctrl}<span class="score {cls}">{n_hit}/{k}</span></div>'
            f'<div class="strip"><span class="lab">oracle reads</span>{tok_spans(tok, r["readout"], r["hits"])}</div>'
            f'<div class="strip"><span class="lab">model will say</span>{tok_spans(tok, r["greedy"])}'
            f'{" " + top1 if r["control"] == "real" else ""}</div>'
            f'<div class="strip"><span class="lab">text continues</span>{tok_spans(tok, r["actual"])}</div>'
            '</div>')
    return f'{head}{legend}<div class="cards">{"".join(cards)}</div>'


# ----------------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------------

def tokenize(text: str):
    ids = BANK.tok.encode(text or "", add_special_tokens=False)
    n_total = len(ids)
    ids = ids[:MAX_TOKENS]
    return render_tokens([BANK.tok.decode([i]) for i in ids], n_total), ids, EMPTY, ""


@spaces.GPU(duration=120)
def analyze(ids: list[int], idx: str, labels: list[str], k: int, ctrls: list[str], layer8: int):
    if not ids or idx in (None, ""):
        return EMPTY
    t = int(idx)
    if t < 0 or t >= len(ids):
        return EMPTY
    rows, rng = [], random.Random(t)
    for lab in labels or []:
        o = REGISTRY.get(LABELS.get(lab, ""))
        if o is None:
            continue
        layer = o.layers[0] if len(o.layers) == 1 else (int(layer8) if int(layer8) in o.layers else o.layers[len(o.layers) // 2])
        for c in ["real"] + [c for c in ("none", "shuffled") if c in ctrls]:
            try:
                st = rng.choice([i for i in range(len(ids)) if i != t]) if (c == "shuffled" and len(ids) > 1) else None
                rows.append(O.read_future(BANK, o, layer, ids, t, int(k), control=c, shuffle_t=st))
            except Exception as e:
                traceback.print_exc()
                rows.append({"oracle": lab, "error": f"{type(e).__name__}: {e}"})
    if not rows:
        return '<div class="res-empty">Select at least one oracle.</div>'
    return render_results(BANK.tok, ids, t, rows)


@spaces.GPU(duration=120)
def probe(text: str, token_index: int, oracles: list[str] | None = None, k: int = 5,
          controls: list[str] | None = None, layer_for_multilayer: int = 24) -> str:
    """API: tokenize `text`, read the future at `token_index` with the given oracles (short labels;
    default: all single-layer oracles). Returns the results HTML."""
    _, ids, _, _ = tokenize(text)
    return analyze(ids, str(int(token_index)), oracles or default_choices(), int(k), controls or [], int(layer_for_multilayer))


def do_refresh():
    msg = refresh_registry()
    return gr.update(choices=oracle_choices()), msg


DEFAULT_TEXTS = [
    "The Eiffel Tower is located in the capital of France, which is Paris. It was completed in 1889 and remains one of the most visited monuments in the world.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
    "Dear hiring manager,\n\nI am writing to apply for the position of software engineer at your company. I have five years of experience building distributed systems and",
    "Once upon a time, in a small village at the edge of a great forest, there lived a girl named",
]

with gr.Blocks(css=CSS, js=CLICK_JS, title="Future Oracles", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔮 Future Oracles\n"
        "Read what a language model is *about to say* from a single activation vector. Click a token: each oracle "
        "takes the residual-stream vector at that token (one layer, one position) from its own Qwen3 model and writes "
        "the next *k* tokens. It is scored against what the model itself would go on to say from there.",
        elem_id="fo-title",
    )
    ids_state = gr.State([])
    click_idx = gr.Textbox(elem_id="fo-click-idx", visible=True)

    with gr.Row():
        text_in = gr.Textbox(label="Text", lines=3, max_lines=10, value=DEFAULT_TEXTS[0], scale=8, show_label=False,
                             placeholder="Paste any text…")
        tokenize_btn = gr.Button("Tokenize", variant="primary", scale=1, min_width=120)
    gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in], label="Examples", examples_per_page=5)

    with gr.Row(equal_height=True):
        oracles_in = gr.CheckboxGroup(oracle_choices(), value=default_choices(), label="Oracles", scale=5)
        k_in = gr.Slider(1, K_MAX, value=5, step=1, label="k · tokens to read", scale=2)
        ctrl_in = gr.CheckboxGroup(["none", "shuffled"], value=[], label="Controls", scale=2,
                                   info="none: marker embedding untouched · shuffled: another token's vector")
        layer8_in = gr.Dropdown(LAYER_CHOICES, value=24, label="Layer (multi-layer oracles)", scale=1, min_width=150)

    tokens_out = gr.HTML()
    results = gr.HTML(EMPTY)

    with gr.Row(elem_id="fo-foot"):
        refresh_btn = gr.Button("Refresh oracles", size="sm", scale=0, min_width=140)
        status = gr.Markdown(f"{STATUS} · adapters, data and evals: "
                             f"[{O.REPO}](https://huggingface.co/datasets/{O.REPO}) · API endpoint `/probe`")

    inputs = [ids_state, click_idx, oracles_in, k_in, ctrl_in, layer8_in]
    tokenize_btn.click(tokenize, [text_in], [tokens_out, ids_state, results, click_idx])
    text_in.submit(tokenize, [text_in], [tokens_out, ids_state, results, click_idx])
    click_idx.input(analyze, inputs, [results])
    for comp in (oracles_in, k_in, ctrl_in, layer8_in):     # re-read the selected token when settings change
        comp.change(analyze, inputs, [results])
    refresh_btn.click(do_refresh, [], [oracles_in, status])
    demo.load(tokenize, [text_in], [tokens_out, ids_state, results, click_idx])
    gr.api(probe, api_name="probe")

demo.queue(default_concurrency_limit=2)
if __name__ == "__main__":
    demo.launch()
