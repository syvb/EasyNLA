#!/usr/bin/env bash
# Build the LessWrong widget exactly as published (https://i.smitop.com/head-nla/widget.html, HF l24/lw_widget/) from
# public data only: the 4 chats of the HF dataset syvb/head-nla-qwen3-8b, downloaded anonymously, and the Qwen3-8B
# tokenizer. Output is deterministic, so it matches the published files byte for byte. CPU only, a few minutes.
#   scripts/heads_lw_widget_build.sh <out dir>
# Publishing (docs/head_nla_lw_widget.md): upload <out dir>/data/*.js, then widget.html, under head-nla/.
set -euo pipefail
out=${1:?usage: $0 <out dir>}
cd "$(dirname "$0")/.."
python scripts/heads_lw_widget_html.py \
  --doc hf:l24/chat_viewer/01_ipv4_code        --label "IPv4 checker (code)"    --start 0:115 \
  --doc hf:l24/chat_viewer/02_lighthouse_story --label "Lighthouse story"       --start 7:200 \
  --doc hf:l24/chat_viewer/03_train_problem    --label "Train problem (maths)"  --start 5:92 \
  --doc hf:l24/chat_viewer/04_ada_lovelace     --label "Ada Lovelace (2 turns)" --start 26:73 \
  --default 02_lighthouse_story --out-dir "$out"
