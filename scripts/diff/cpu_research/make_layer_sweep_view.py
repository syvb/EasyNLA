"""Write the layer-sweep visual (one self-contained HTML page) from the sweep result JSONs.

    python make_layer_sweep_view.py <results_dir> <out.html>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
AUDIT = ("refusal", "eval_awareness", "sycophancy")


def series(path: Path) -> dict | None:
    if not path.exists():
        return None
    sw = json.loads(path.read_text())
    out: dict = {"n_pairs": sw["n_pairs"], "layers": sw["n_layers"]}
    for w in ("chatml", "plain"):
        rows = [sw["layers"][str(li)][w] for li in range(sw["n_layers"])]
        out[w] = {
            "audit": [[round(r[n]["ratio_naive_diff_cu"], 3) for n in AUDIT] for r in rows],
            "base": [[round(r[n]["ratio_h_base"], 3) for n in AUDIT] for r in rows],
            "content": [round(r["mean_content_ratio"], 3) for r in rows],
        }
    return out


data = {"asst_last": series(R / "layer_sweep_8b.json"), "user_last": series(R / "layer_sweep_8b_user_last.json")}
data = {k: v for k, v in data.items() if v}

HTML = r"""<div class="ui-header">
	<p class="ui-title" id="title"></p>
	<p class="ui-sub" id="sub"></p>
</div>
__CONTROLS__<div id="chart"></div>
<p class="ui-caption" id="caption" style="margin-top: 8px"></p>
<script>
	{
		const data = __DATA__
		const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length
		const times = (v) => `${v.toFixed(2)}×`
		const positions = Object.keys(data)
		const state = { position: positions[0] }
		const labels = { asst_last: 'Assistant boundary', user_last: 'Last user token' }
		let chart = null

		const view = () => {
			const d = data[state.position]
			const layers = Array.from({ length: d.layers }, (_, i) => i)
			const chatml = d.chatml.audit.map(mean)
			const plain = d.plain.audit.map(mean)
			const base = d.plain.base.map(mean)
			const best = chatml.indexOf(Math.max(...chatml))
			const passes = (w) =>
				layers.filter((i) => mean(d[w].audit[i]) >= 1.25 && Math.min(...d[w].audit[i]) >= 0.85)
			const span = (a) => {
				if (!a.length) return 'none'
				const runs = []
				for (const i of a) {
					const last = runs[runs.length - 1]
					if (last && i === last[1] + 1) last[1] = i
					else runs.push([i, i])
				}
				return runs.map(([lo, hi]) => (lo === hi ? `${lo}` : `${lo} to ${hi}`)).join(', ')
			}
			const pc = passes('chatml')
			const pp = passes('plain')
			const title =
				pc.length === 0
					? `With Base on the same tokens, the diff peaks at ${times(chatml[best])} raw h_target's audit SNR; Gate 0 needs 1.25×`
					: `With Base on the same tokens, the diff clears Gate 0's 1.25× at layers ${span(pc)}`
			const top = Math.max(1.5, ...chatml, ...plain, ...base)
			return {
				title,
				sub: `Fisher SNR of the unit-normalised naive diff over raw h_target, ${labels[state.position].toLowerCase()}, mean of refusal, eval-awareness and sycophancy (${d.n_pairs} pairs each), Qwen3-8B vs Qwen3-8B-Base`,
				caption: `Layers that clear both SNR legs (mean at least 1.25×, every task at least 0.85×): ChatML ${span(pc)}; Plain ${span(pp)}. The gray line is the Base model alone under Plain: where it is above 1×, Base separates the classes better than Instruct does. At layer 24 the ChatML diff is ${times(chatml[24])}, the Plain diff ${times(plain[24])} and Base alone ${times(base[24])}.`,
				options: {
					x: layers.map(String),
					series: [
						{ name: 'Diff, Base reads ChatML', values: chatml, highlight: true },
						{ name: 'Diff, Base reads Plain', values: plain, highlight: true },
						{ name: 'Base alone, Plain', values: base },
					],
					format: times,
					min: 0,
					max: Math.ceil(top * 2) / 2,
					height: 240,
					xTicks: [0, 6, 12, 18, 24, 30, d.layers - 1],
					tooltipTitle: (i) => `Layer ${i}`,
					annotations: [
						{ type: 'rule', y: 1.25, label: 'Gate 0: 1.25×' },
						{ type: 'marker', at: 24, label: 'Planned layer' },
						{ type: 'point', series: 'Diff, Base reads ChatML', at: best, label: times(chatml[best]) },
					],
				},
			}
		}

		const render = () => {
			const v = view()
			document.getElementById('title').textContent = v.title
			document.getElementById('sub').textContent = v.sub
			document.getElementById('caption').textContent = v.caption
			const el = document.getElementById('chart')
			el.replaceChildren()
			chart = ui.line(el, v.options)
		}

		const seg = document.getElementById('position')
		if (seg) {
			seg.addEventListener('change', (e) => {
				state.position = e.detail.value
				render()
			})
		}
		render()
	}
</script>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
LABELS = {"asst_last": "Assistant boundary", "user_last": "Last user token"}
controls = ""
if len(data) > 1:
    buttons = "".join(
        f'<button data-value="{k}" aria-selected="{"true" if i == 0 else "false"}">{LABELS[k]}</button>' for i, k in enumerate(data)
    )
    controls = f'<div class="ui-row" style="margin-bottom: 8px"><div class="ui-seg" id="position">{buttons}</div></div>\n'
OUT.write_text(HTML.replace("__CONTROLS__", controls).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes), positions: {list(data)}")
