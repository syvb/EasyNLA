// Runs a view's inline script against a stub DOM and view kit, and prints what it would show.
const fs = require('fs')
const html = fs.readFileSync(process.argv[2], 'utf8')
const script = html.match(/<script>([\s\S]*)<\/script>/)[1]
const els = {}
const el = (id) => (els[id] ||= { id, textContent: '', children: [], replaceChildren() { this.children = [] }, append(...c) { this.children.push(...c) }, remove() { this.removed = true }, addEventListener(t, f) { this['on' + t] = f }, setAttribute() {}, dataset: {} })
global.document = { getElementById: el, createElement: () => el('new' + Math.random()) }
global.ui = { line: (e, o) => { console.log('ui.line series:', o.series.map((s) => `${s.name} [${s.values.length}] max ${Math.max(...s.values).toFixed(2)}`).join(' | '), '| y max', o.max, '| annotations', JSON.stringify(o.annotations)); return {} } }
eval(script)
for (const id of ['title', 'sub', 'caption']) console.log(id + ':', els[id].textContent)
const seg = Object.values(els).find((e) => e.onchange)
if (seg) { seg.onchange({ detail: { value: 'user_last' } }); for (const id of ['title', 'caption']) console.log('[user_last] ' + id + ':', els[id].textContent) }
