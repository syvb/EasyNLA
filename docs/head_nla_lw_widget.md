# Head NLA explorer widget (for the LessWrong post)

An interactive page for browsing the head NLA's explanations of every token × every head in four of Qwen3-8B's own
chats.
- **Live:** https://i.smitop.com/head-nla/widget.html
- **Copy:** HF dataset `syvb/head-nla-qwen3-8b`, `l24/lw_widget/`
- **Generator:** `scripts/heads_lw_widget_html.py`
- **Exact build:** `scripts/heads_lw_widget_build.sh`

## What it shows

- **A menu of four chats:**
  - IPv4 checker (code);
  - lighthouse story (the default);
  - train problem (maths);
  - Ada Lovelace (2 turns).
- **A Matryoshka / Standard toggle:** the matryoshka AV and AR at RL step 640, or the standard pair at RL step 400.
- **The text:** shaded by the selected head's attention from the selected token, or by that head's FVE at each token.
  Chat-template tokens are muted and each turn is labelled User or Assistant. A chat opens scrolled to the
  assistant's reply.
- **The 32 heads at the selected token:** coloured by FVE.
- **The selected site's explanation:** with its FVE, the head's FVE over the whole chat, and its top attended tokens.
  Attention is shown for the reader only; neither model sees it.
- **Default site per chat:** a head that reads one clear token, in the first lines of the reply, so it is visible on
  load at both 340 px and LessWrong's 690 px.
  - Lighthouse: H7 at "wounded" (reads "wind", 35%, after the first token at 38%).
  - IPv4: H5 at "```" (reads "Python" in the question, 55%).
  - Train: H5 at "train" (reads "catch", 48%).
  - Ada: H13 at "the" (reads "ace" of Lovelace in the question, 35%).

## Inputs (all public)

For each chat, five files from `l24/chat_viewer/<chat>/` of the HF dataset: `token_ids.npy`, `meta.json`,
`sites.parquet` (the RL and SFT explanations and scores), `attn.npz` and `mtry.parquet` (the matryoshka explanations).
The tokenizer is `Qwen/Qwen3-8B`.

Those folders come from the chat viewer pipeline:
- `scripts/heads_chat_gen.py` with `scripts/heads_chat_prompts.json`;
- `scripts/runpod/heads_l24_chatviewer_remote.sh`;
- results doc §11.

## Build

```bash
scripts/heads_lw_widget_build.sh <out dir>     # CPU only, a few minutes; downloads the inputs anonymously
```

The output is deterministic (gzip with `mtime=0`). A build in an empty HF cache, without a token, matched the published
files byte for byte on 2026-10-29: `widget.html` and all 8 data files, both on HF and on i.smitop.com.

| File | Size | Content |
|---|---|---|
| `widget.html` | 17 kB | the page, with the menu and each data file's content hash inline |
| `data/<chat>.js` | 6.0–7.4 MB; about 5.4 MB with brotli | tokens, roles, attention, matryoshka explanations |
| `data/<chat>.std.js` | 4.4–5.3 MB | standard explanations, loaded on the first switch |

## Publishing

Upload the data files first and `widget.html` last, under `head-nla/` in the public R2 bucket behind i.smitop.com. We
used the `r2up` helper, which is not part of this repo and refuses to overwrite without `-f`.

```bash
for f in data/*.js; do r2up -f "$f" "head-nla/$f"; done; r2up -f widget.html head-nla/widget.html
```

Before an upload, scan the output for secrets: `scripts/hf_secret_scan.py <out dir>` with the usual `--secret-file`
flags. The data files are gzip and base64, so also decode them and check for local paths. The 2026-10-29 build had
none.

Embed in the post's widget:

```html
<style>html,body{margin:0}iframe{display:block;width:100%;height:560px;border:0}@media (max-width:500px){iframe{height:640px}}</style>
<iframe src="https://i.smitop.com/head-nla/widget.html" title="Head NLA explorer"></iframe>
```

URL options: `?chat=01_ipv4_code | 02_lighthouse_story | 03_train_problem | 04_ada_lovelace` and `?theme=dark | auto`.
The default theme is light.

## Design notes

- **No CORS needed.** A LessWrong widget is a sandboxed iframe (`allow-scripts`, no `allow-same-origin`), and the page
  nested in it inherits the sandbox. Its origin is `null`, so `fetch()` of its own data files fails without CORS
  headers; this was confirmed in Chromium. Data files are loaded by `<script>` tag instead. Each one calls
  `HEADNLA_LOAD(name, parts)` with gzip+base64 JSON parts, which `DecompressionStream` decodes.
- **Caching.** Cloudflare doesn't cache `/head-nla/*` (`cf-cache-status: DYNAMIC`) and serves brotli. There is no
  `Cache-Control` header, so browsers may reuse a cached `widget.html` for a while after an update. The data URLs carry
  content hashes (`?v=`), so a page never pairs with stale data.
- **Size.** The page fills its iframe's height and the panes scroll, so the embed sets the height. It is laid out for
  340–700 px. The header's controls wrap onto a second row when they don't fit beside the title and menu.
