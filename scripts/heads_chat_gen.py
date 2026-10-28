"""Chat examples for the head-NLA viewer: Qwen3-8B answers fixed user prompts; writes one doc-viewer queue spec per chat.

    python scripts/heads_chat_gen.py --prompts <chats.json> --queue <dir> [--base Qwen/Qwen3-8B]

<chats.json>: [{"name": "...", "turns": ["user message", "follow-up", ...]}, ...]. For each chat, every user turn gets
a reply from the model in non-thinking mode (the chat template's enable_thinking=False: an empty <think></think>
block, then the answer), sampled with Qwen3's recommended non-thinking settings (T 0.7, top-p 0.8, top-k 20) from a
fixed seed. A chat is kept only if every reply ends on its own (end-of-turn token, not the length cap) and the whole
chat is --min-tokens..--max-tokens long; otherwise the next seed is tried (--tries), and failing that the attempt
closest to the target that ended on its own is kept; if none ended on its own, the closest attempt (logged).

The spec's "text" is exactly the token sequence the model processed for its last reply plus that reply, decoded
with special tokens kept. Re-tokenizing it must give the same ids (checked), so the viewer job reads the heads on
the model's own chat, chat-template tokens included. Also writes <queue>/../chats.json (turns, replies, seeds).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--min-tokens", type=int, default=420)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--tries", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", default="", help="comma-separated chat names to generate (default: all)")
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(a.device).eval()
    end_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    queue = Path(a.queue)
    queue.mkdir(parents=True, exist_ok=True)
    record = []
    target = (a.min_tokens + a.max_tokens) / 2
    only = {x for x in a.only.split(",") if x}
    for ci, chat in enumerate(json.loads(Path(a.prompts).read_text())):
        if only and chat["name"] not in only:
            continue
        best, closest = None, None
        for attempt in range(a.tries):
            seed = 1000 * ci + attempt
            torch.manual_seed(seed)
            msgs, replies, finished = [], [], True
            for turn in chat["turns"]:
                msgs.append({"role": "user", "content": turn})
                prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
                ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device)
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8,
                                         top_k=20, eos_token_id=list(end_ids), pad_token_id=tok.eos_token_id)
                gen = out[0, ids.shape[1]:].tolist()
                cut = next((i for i, t in enumerate(gen) if t in end_ids), None)
                finished &= cut is not None
                gen = gen[: cut + 1] if cut is not None else gen
                reply = tok.decode([t for t in gen if t not in end_ids], skip_special_tokens=False)
                replies.append(reply)
                msgs.append({"role": "assistant", "content": reply})
                full = ids[0].tolist() + gen                      # what the model processed + its reply
            n = len(full)
            ok = finished and a.min_tokens <= n <= a.max_tokens and all(r.strip() for r in replies)
            print(f"[chat] {chat['name']} seed {seed}: {n} tokens, finished={finished}, kept={ok}", flush=True)
            cand = {"seed": seed, "ids": full, "replies": replies, "n": n, "finished": finished}
            if finished and (best is None or abs(n - target) < abs(best["n"] - target)):
                best = cand
            if closest is None or abs(n - target) < abs(closest["n"] - target):
                closest = cand
            if ok:
                best = cand
                break
        if best is None:      # never ended on its own: keep the closest attempt rather than lose the chat
            best = closest
            print(f"[chat] !!! {chat['name']}: no attempt ended on its own within {a.max_new_tokens} tokens; keeping a cut-off reply", flush=True)
        text = tok.decode(best["ids"], skip_special_tokens=False)
        assert tok(text, add_special_tokens=True).input_ids == best["ids"], f"{chat['name']}: decode/encode round trip changed the ids"
        name = f"{ci + 1:02d}_{chat['name']}"
        (queue / f"{name}.json").write_text(json.dumps({"name": name, "text": text, "source": "chat (Qwen3-8B's own replies)",
                                                        "domain": "chat"}, ensure_ascii=False))
        record.append({"name": name, "turns": chat["turns"], "replies": best["replies"], "seed": best["seed"], "n_tokens": best["n"],
                       "finished": best["finished"]})
        print(f"[chat] {name}: kept seed {best['seed']}, {best['n']} tokens; reply words "
              f"{[len(r.split()) for r in best['replies']]}", flush=True)
    (queue.parent / "chats.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
