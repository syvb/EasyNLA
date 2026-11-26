"""Crash-resumable Anthropic Message Batches provider.

Prevents double-billing if a cloud VM, SSH session, or spot pod disconnects
while polling an already-submitted Anthropic Batch API job (15-60 min).

How it works:
  1. For each sub-batch of prompts, compute a deterministic SHA-256 fingerprint
     over `(model, max_tokens, temperature, prompts)`.
  2. Look up `<state_dir>/batch_<fp>.json`:
     - If present, inspect the existing `batch_id` via `messages.batches.retrieve`:
       - `in_progress` / `canceling`: resume polling the existing batch (ZERO new create!).
       - `ended` with `succeeded > 0` (or `errored + canceled + expired < len`):
         skip polling and immediately stream results from the existing batch!
       - `ended` with 0 succeeded and all canceled/expired: stale -> resubmit fresh.
     - If absent, call `messages.batches.create(...)` and ATOMICALLY persist
       `{"batch_id": bid, ...}` BEFORE entering the poll loop.
  3. After streaming results cleanly, remove the state file so completed jobs
     do not accumulate clutter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from nla.datagen.providers import BatchAnthropicProvider


def _subbatch_fingerprint(
    model: str,
    max_tokens: int,
    temperature: float | None,
    prompts: list[str],
) -> str:
    h = hashlib.sha256()
    h.update(f"{model}\0{max_tokens}\0{temperature}\0{len(prompts)}\0".encode())
    for p in prompts:
        pb = p.encode("utf-8")
        h.update(f"{len(pb)}:".encode())
        h.update(pb)
    return h.hexdigest()[:24]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class ResumableBatchAnthropicProvider(BatchAnthropicProvider):
    """Drop-in `BatchAnthropicProvider` that persists batch IDs to disk for crash recovery."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float | None = 1.0,
        max_batch_size: int = 50_000,
        poll_interval_s: float = 30.0,
        state_dir: str | Path | None = None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_batch_size=max_batch_size,
            poll_interval_s=poll_interval_s,
        )
        self.state_dir = Path(state_dir) if state_dir is not None else None

    async def _submit_subbatch(self, prompts: list[str], offset: int) -> list[str | None]:
        if self.state_dir is None:
            return await super()._submit_subbatch(prompts, offset)

        fp = _subbatch_fingerprint(self.model, self.max_tokens, self.temperature, prompts)
        state_path = self.state_dir / f"batch_{fp}.json"
        bid: str | None = None
        already_ended = False

        if state_path.exists():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                saved_bid = saved.get("batch_id")
                if isinstance(saved_bid, str) and saved_bid:
                    b = await self.client.messages.batches.retrieve(saved_bid)
                    status = getattr(b, "processing_status", None)
                    if status == "ended":
                        rc = b.request_counts
                        if rc.succeeded > 0 or (rc.canceled + rc.expired) < len(prompts):
                            bid = saved_bid
                            already_ended = True
                            print(
                                f"  [batch {bid}] RESUMED already-ended batch "
                                f"(succeeded={rc.succeeded}, errored={rc.errored})",
                                flush=True,
                            )
                        else:
                            print(
                                f"  [batch {saved_bid}] previous batch had 0 succeeded "
                                f"(canceled={rc.canceled}, expired={rc.expired}); resubmitting.",
                                flush=True,
                            )
                    else:
                        bid = saved_bid
                        print(
                            f"  [batch {bid}] RESUMED in-progress batch (status={status}, offset={offset})",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [batch] could not resume state {state_path.name} ({exc}); submitting fresh.",
                    flush=True,
                )

        if bid is None:
            requests = [
                {
                    "custom_id": f"r{i:07d}",
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        **({} if self.temperature is None else {"temperature": self.temperature}),
                        "messages": [{"role": "user", "content": p}],
                    },
                }
                for i, p in enumerate(prompts)
            ]
            batch = await self.client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
            bid = batch.id
            _atomic_write_json(
                state_path,
                {
                    "batch_id": bid,
                    "model": self.model,
                    "n_prompts": len(prompts),
                    "offset": offset,
                    "submitted_at_unix": round(time.time(), 2),
                },
            )
            print(
                f"  [batch {bid}] submitted {len(prompts)} prompts "
                f"(sub-batch offset={offset}; saved -> {state_path.name})",
                flush=True,
            )

        if not already_ended:
            t0 = time.monotonic()
            while True:
                await asyncio.sleep(self.poll_interval_s)
                b = await self.client.messages.batches.retrieve(bid)
                if b.processing_status == "ended":
                    rc = b.request_counts
                    elapsed = time.monotonic() - t0
                    print(
                        f"  [batch {bid}] ended in {elapsed / 60:.1f}min — "
                        f"succeeded={rc.succeeded} errored={rc.errored} "
                        f"canceled={rc.canceled} expired={rc.expired}",
                        flush=True,
                    )
                    break

        out: list[str | None] = [None] * len(prompts)
        first_err = None
        results_stream = await self.client.messages.batches.results(bid)
        async for entry in results_stream:
            idx = int(entry.custom_id[1:])
            r = entry.result
            if r.type != "succeeded":
                if first_err is None:
                    first_err = f"{r.type}: {getattr(r, 'error', '')!s:.300}"
                continue
            msg = r.message
            if msg.stop_reason == "refusal":
                continue
            if not msg.content or msg.content[0].type != "text":
                continue
            text = msg.content[0].text.strip()
            if text:
                out[idx] = text

        if prompts and all(o is None for o in out) and first_err is not None:
            raise RuntimeError(
                f"[batch {bid}] 0/{len(prompts)} requests succeeded — systemic "
                f"failure (auth/params), refusing to emit an empty chunk. "
                f"First error: {first_err}"
            )

        # Clean up persisted batch state only after results are fully retrieved!
        if state_path.exists():
            state_path.unlink()
        return out
