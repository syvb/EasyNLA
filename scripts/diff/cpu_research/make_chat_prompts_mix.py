"""Write a small chat-prompt mix mirroring Phase 2's sources, as JSONL for
`phase2_build_corpus.py --custom-prompts-jsonl`.

    python make_chat_prompts_mix.py <out.jsonl> <n_single> <n_multi> <n_safety>

Sources follow `_sample_chat_prompts`: UltraChat (single-turn, and third-turn
prompts with two turns of history) and PKU-SafeRLHF (falling back to hh-rlhf).
WildChat-1M is gated and is replaced here by more UltraChat.
"""

from __future__ import annotations

import json
import sys

from datasets import load_dataset

out = sys.argv[1]
n_single, n_multi, n_safety = (int(x) for x in sys.argv[2:5])
rows: list[dict] = []

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
got_s = got_m = 0
for row in ds:
    msgs = [m for m in (row.get("messages") or []) if m.get("role") in ("user", "assistant")]
    if not msgs or msgs[0]["role"] != "user":
        continue
    first = str(msgs[0]["content"]).strip()
    if got_m < n_multi and len(msgs) >= 3 and msgs[2]["role"] == "user":
        a1 = str(msgs[1]["content"]).strip()
        u2 = str(msgs[2]["content"]).strip()
        if 20 <= len(first) <= 400 and 20 <= len(a1) <= 700 and 20 <= len(u2) <= 400:
            rows.append({"source_corpus": "ultrachat", "user_text": u2,
                         "history": [{"role": "user", "content": first},
                                     {"role": "assistant", "content": a1}]})
            got_m += 1
            continue
    if got_s < n_single and 20 <= len(first) <= 600:
        rows.append({"source_corpus": "ultrachat", "user_text": first, "history": []})
        got_s += 1
    if got_s >= n_single and got_m >= n_multi:
        break

got_x = 0
try:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True)
    seen: set[str] = set()
    for row in ds:
        ut = str(row.get("prompt", "")).strip()
        if 20 <= len(ut) <= 600 and ut not in seen:
            seen.add(ut)
            rows.append({"source_corpus": "pku_saferlhf", "user_text": ut, "history": []})
            got_x += 1
        if got_x >= n_safety:
            break
except Exception as exc:  # noqa: BLE001
    print(f"PKU-SafeRLHF unavailable ({exc}); falling back to Anthropic/hh-rlhf")
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    for row in ds:
        chosen = str(row.get("chosen", ""))
        if "Human:" in chosen and "Assistant:" in chosen:
            ut = chosen.split("Human:", 1)[1].split("Assistant:", 1)[0].strip()
            if 20 <= len(ut) <= 600:
                rows.append({"source_corpus": "hh_rlhf", "user_text": ut, "history": []})
                got_x += 1
        if got_x >= n_safety:
            break

with open(out, "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print(f"wrote {len(rows)} prompts to {out}: single={got_s} multi={got_m} safety={got_x}")
