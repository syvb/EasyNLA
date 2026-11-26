"""Fill the activation cache for every request recorded by a NLA_RECORD=... dry run.

    python warm_cache.py <manifest.jsonl> [tokenizer-name]
"""

from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

tok = AutoTokenizer.from_pretrained(sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B")
pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
cached_io.warm_from_manifest(sys.argv[1], pad)
print("WARM_DONE", flush=True)
