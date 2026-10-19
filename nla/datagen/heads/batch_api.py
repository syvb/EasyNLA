"""Head stages `prepare` / `submit` / `status` / `collect` / `wait` / `retry` — warm-start descriptions.

    prepare   selection.parquet → requests/shard_*.jsonl (Message Batches request
              bodies), requests/index.parquet, rule_descriptions.parquet. Local
              and free; prints an estimated cost.
    submit    PAID. Submits every shard that has no batch yet. Without
              --confirm-spend it only prints what it would submit.
              --max-shards N submits at most N (e.g. 1 for a pilot).
    status    refresh + print each batch's processing status.
    collect   download results of ended batches → results/*.jsonl, then rebuild
              descriptions.parquet (all results + rule rows, parsed + flagged)
              and print the actual spend from returned usage.
    wait      status + collect every --poll-seconds until every batch ended.
    retry     new shards from requests whose result failed (errored / expired /
              canceled / truncated / unparseable); `submit` picks them up.
    run-sync  PAID, like submit, but through the standard Messages API with bounded
              concurrency instead of a batch: 2× the price, minutes instead of up to
              24 h (for quick tests where an idle GPU waits on the answers). Writes
              the same results files; resumes mid-shard.

State lives in warmstart/batches/state.json, rewritten atomically. A shard is
marked "submitting" BEFORE the create call and gets its batch id right after,
so a crash in between is detected (not silently re-submitted — that would pay
twice): the shard then needs a manual check of the batch list.

Records the model would answer "little or nothing" by rule (≥ little_threshold
of the head's attention on the first token, which the instructions already
map to that answer) are written straight to rule_descriptions.parquet and
never sent.

Credentials: ANTHROPIC_API_KEY_BATCH if set (as BatchAnthropicProvider),
else the SDK's default resolution.
"""

import argparse
import datetime
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from nla.datagen._common import load_tokenizer
from nla.datagen.heads.prompts import INSTRUCTIONS, LITTLE, PromptConfig, parse_response, render_user_prompt
from nla.datagen.heads.store import atomic_write_table, fixed_numpy, read_docs, read_meta, read_records

# Anthropic batch limits are 100k requests / 256 MB; stay well inside both.
_DEFAULT_MAX_REQUESTS = 25_000
_DEFAULT_MAX_BYTES = 200 * 1024 * 1024


def _ws(out_dir: Path) -> Path:
    return Path(out_dir) / "warmstart"


def _now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _state_path(out_dir: Path) -> Path:
    return _ws(out_dir) / "batches" / "state.json"


def load_state(out_dir: Path) -> dict[str, Any]:
    p = _state_path(out_dir)
    return json.loads(p.read_text()) if p.exists() else {"shards": {}}


def save_state(out_dir: Path, state: dict[str, Any]) -> None:
    _write_json(_state_path(out_dir), state)


def make_client() -> Any:
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY_BATCH") or None, max_retries=10)


def custom_id(example_idx: int) -> str:
    return f"ex{example_idx:07d}"


def request_params(user_text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    system: dict[str, Any] = {"type": "text", "text": INSTRUCTIONS}
    if cfg["cache_system"]:
        system["cache_control"] = {"type": "ephemeral"}
    params: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "system": [system],
        "messages": [{"role": "user", "content": user_text}],
        "output_config": {"effort": cfg["effort"]},
    }
    # No temperature/top_p/top_k: Haiku 5.5 rejects non-default sampling params.
    if cfg["thinking"] == "disabled":
        params["thinking"] = {"type": "disabled"}
    return params


# ── prepare ──────────────────────────────────────────────────────────────

class _ShardWriter:
    """Streams request lines into shard files, cutting a new shard before either
    batch limit is crossed. Request bodies never all sit in memory (460k × ~8 KB)."""

    def __init__(self, out_dir: Path, prefix: str, max_requests: int, max_bytes: int, state: dict[str, Any]):
        self.req_dir = _ws(out_dir) / "requests"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.max_requests, self.max_bytes, self.state = prefix, max_requests, max_bytes, state
        self.k = 0
        self.f: Any = None
        self.assigned: list[tuple[str, str]] = []

    def _open(self) -> None:
        while (f"{self.prefix}{self.k:04d}" in self.state["shards"]
               or (self.req_dir / f"{self.prefix}{self.k:04d}.jsonl").exists()):
            self.k += 1
        self.name = f"{self.prefix}{self.k:04d}"
        self.tmp = self.req_dir / f".{self.name}.jsonl.tmp"
        self.f = self.tmp.open("w")
        self.n, self.size = 0, 0
        self.k += 1

    def _close(self) -> None:
        self.f.close()
        os.replace(self.tmp, self.req_dir / f"{self.name}.jsonl")
        self.state["shards"][self.name] = {"status": "prepared", "n_requests": self.n, "batch_id": None}
        self.f = None

    def add(self, cid: str, line: str) -> None:
        nbytes = len(line.encode()) + 1
        if self.f is not None and (self.n >= self.max_requests or self.size + nbytes > self.max_bytes):
            self._close()
        if self.f is None:
            self._open()
        self.f.write(line + "\n")
        self.n += 1
        self.size += nbytes
        self.assigned.append((cid, self.name))

    def close(self) -> list[tuple[str, str]]:
        """(custom_id, shard) for every line written."""
        if self.f is not None:
            self._close()
        return self.assigned


def cmd_prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    sent = [n for n, s in state["shards"].items() if s.get("batch_id") or s["status"] == "submitting"]
    if sent:
        raise SystemExit(f"{len(sent)} shards were already submitted ({sent[:3]}…) — re-preparing would "
                         f"change which example each custom_id names and orphan their results. Move "
                         f"{_ws(out_dir)} aside to start a separate request set.")
    for name in list(state["shards"]):   # unsent shards from an earlier prepare: rebuild from scratch
        (_ws(out_dir) / "requests" / f"{name}.jsonl").unlink(missing_ok=True)
        del state["shards"][name]
    meta = read_meta(out_dir)
    sel = pq.read_table(_ws(out_dir) / "selection.parquet")
    tok = load_tokenizer(meta.base_model)
    def decode(ids):
        return tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pcfg = PromptConfig(context_tokens=args.context_tokens, window_tokens=args.window_tokens,
                        max_sources=args.max_sources, min_source_weight=args.min_source_weight)
    req_cfg = {"model": args.model, "max_tokens": args.max_tokens, "effort": args.effort,
               "thinking": args.thinking, "cache_system": args.cache_system}

    docs = read_docs(out_dir, columns=["doc_id", "token_ids"])
    doc_row = {d: i for i, d in enumerate(docs.column("doc_id").to_pylist())}
    token_col = docs.column("token_ids")

    h, k = meta.n_heads, meta.top_k
    writer = _ShardWriter(out_dir, "shard_", args.max_requests_per_batch, args.max_batch_bytes, state)
    rules: dict[str, list] = {"example_idx": [], "first_w": []}
    est_chars = 0
    for layer in sorted(set(sel.column("layer").to_pylist())):
        rows = sel.filter(pc.equal(sel.column("layer"), layer))
        rec = read_records(out_dir, layer, columns=["doc_id", "pos", "attn_top_idx", "attn_top_w",
                                                     "attn_first_w", "attn_self_w"])
        rec_row = {(d, p): i for i, (d, p) in enumerate(zip(rec.column("doc_id").to_pylist(),
                                                            rec.column("pos").to_pylist(), strict=True))}
        top_i = fixed_numpy(rec.column("attn_top_idx"), np.int32, h * k).reshape(-1, h, k)
        top_w = fixed_numpy(rec.column("attn_top_w"), np.float32, h * k).reshape(-1, h, k)
        first = fixed_numpy(rec.column("attn_first_w"), np.float32, h)
        selfw = fixed_numpy(rec.column("attn_self_w"), np.float32, h)
        for ex, doc_id, pos, head in tqdm(zip(rows.column("example_idx").to_pylist(),
                                              rows.column("doc_id").to_pylist(),
                                              rows.column("pos").to_pylist(),
                                              rows.column("head").to_pylist(), strict=True),
                                          total=rows.num_rows, desc=f"L{layer} prompts", leave=False):
            r = rec_row[(doc_id, pos)]
            if first[r, head] >= args.little_threshold:
                rules["example_idx"].append(ex)
                rules["first_w"].append(float(first[r, head]))
                continue
            ids = token_col[doc_row[doc_id]].values.to_numpy()
            user = render_user_prompt(decode, ids, pos, layer, head, meta.n_layers, h,
                                      top_i[r, head], top_w[r, head], float(first[r, head]),
                                      float(selfw[r, head]), pcfg)
            est_chars += len(user)
            body = {"custom_id": custom_id(ex), "params": request_params(user, req_cfg)}
            writer.add(body["custom_id"], json.dumps(body, ensure_ascii=False))

    assigned = writer.close()
    atomic_write_table(pa.table({
        "custom_id": [c for c, _ in assigned],
        "example_idx": [int(c[2:]) for c, _ in assigned],
        "shard": [s for _, s in assigned],
    }), _ws(out_dir) / "requests" / "index.parquet")
    atomic_write_table(pa.table({
        "example_idx": pa.array(rules["example_idx"], type=pa.int64()),
        "first_w": pa.array(rules["first_w"], type=pa.float32()),
    }), _ws(out_dir) / "rule_descriptions.parquet")
    save_state(out_dir, state)

    # ~3 chars/token for the Claude tokenizer on English web text (Haiku 5.5's
    # tokenizer counts ~30% more tokens than older ones) — deliberately high.
    n = len(assigned)
    in_tok = est_chars / 3.0 + n * len(INSTRUCTIONS) / 3.0
    out_tok = n * args.est_output_tokens
    cost = in_tok / 1e6 * args.price_input + out_tok / 1e6 * args.price_output
    summary = {
        "created_at": _now(), "n_requests": n, "n_rule_little": len(rules["example_idx"]),
        "n_shards": len({s for _, s in assigned}), "request_config": req_cfg, "prompt_config": pcfg.__dict__,
        "little_threshold": args.little_threshold,
        "estimate": {"input_tokens": int(in_tok), "output_tokens": int(out_tok), "usd": round(cost, 2),
                     "price_per_mtok": {"input": args.price_input, "output": args.price_output}},
    }
    _write_json(_ws(out_dir) / "requests" / "prepare.json", summary)
    print(f"prepared {n} requests in {summary['n_shards']} shards (+{len(rules['example_idx'])} rule "
          f"'{LITTLE}') → {_ws(out_dir) / 'requests'}")
    print(f"  estimated: {in_tok / 1e6:.0f}M input + {out_tok / 1e6:.1f}M output tokens ≈ ${cost:,.0f} "
          f"(batch prices; system-prompt cache hits would lower this)")


# ── submit / status ──────────────────────────────────────────────────────

def _pending(state: dict[str, Any]) -> list[str]:
    return sorted(n for n, s in state["shards"].items() if s["status"] == "prepared")


def cmd_submit(args: argparse.Namespace, client_factory: Callable[[], Any] = make_client) -> None:
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    stuck = [n for n, s in state["shards"].items() if s["status"] == "submitting"]
    if stuck:
        raise SystemExit(
            f"shards {stuck} were mid-submission when a previous run died — a batch may or may not exist "
            f"for them. Check `client.messages.batches.list()` (or the Console); then set their entry in "
            f"{_state_path(out_dir)} to the batch id with status 'submitted', or back to 'prepared'.")
    pending = _pending(state)
    if args.max_shards is not None:
        pending = pending[: args.max_shards]
    n_req = sum(state["shards"][n]["n_requests"] for n in pending)
    if not pending:
        print("nothing to submit — every shard already has a batch")
        return
    est = json.loads((_ws(out_dir) / "requests" / "prepare.json").read_text())["estimate"]
    total_req = sum(s["n_requests"] for s in state["shards"].values())
    usd = est["usd"] * n_req / max(total_req, 1)
    print(f"{len(pending)} shards / {n_req} requests to submit (≈ ${usd:,.0f} by the prepare estimate)")
    if not args.confirm_spend:
        print("dry run — re-run with --confirm-spend to submit (this spends money)")
        return
    client = client_factory()
    req_dir = _ws(out_dir) / "requests"
    for name in pending:
        requests = [json.loads(line) for line in (req_dir / f"{name}.jsonl").read_text().splitlines() if line]
        state["shards"][name]["status"] = "submitting"
        save_state(out_dir, state)
        # One attempt only: an SDK retry after a timeout could create (and bill)
        # a second batch for the same shard. A 4xx means nothing was created, so
        # the shard goes back to "prepared"; anything else stays "submitting"
        # for the manual check above.
        try:
            batch = client.with_options(max_retries=0).messages.batches.create(requests=requests)
        except Exception as e:
            if 400 <= getattr(e, "status_code", 0) < 500:
                state["shards"][name]["status"] = "prepared"
                save_state(out_dir, state)
            raise
        state["shards"][name].update(batch_id=batch.id, status="submitted", submitted_at=_now(),
                                     processing_status=batch.processing_status)
        save_state(out_dir, state)
        print(f"  {name}: {len(requests)} requests → batch {batch.id}")


def _counts(batch: Any) -> dict[str, int]:
    rc = batch.request_counts
    return {k: int(getattr(rc, k)) for k in ("processing", "succeeded", "errored", "canceled", "expired")}


def refresh_status(out_dir: Path, client: Any) -> dict[str, Any]:
    state = load_state(out_dir)
    for name, s in sorted(state["shards"].items()):
        if s.get("batch_id") and s.get("processing_status") != "ended":
            b = client.messages.batches.retrieve(s["batch_id"])
            s["processing_status"] = b.processing_status
            s["request_counts"] = _counts(b)
            if b.processing_status == "ended":
                s["status"] = "ended"
                s["ended_at"] = _now()
    save_state(out_dir, state)
    return state


def cmd_status(args: argparse.Namespace, client_factory: Callable[[], Any] = make_client) -> None:
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    if any(s.get("batch_id") for s in state["shards"].values()):
        state = refresh_status(out_dir, client_factory())
    for name, s in sorted(state["shards"].items()):
        rc = s.get("request_counts") or {}
        print(f"  {name}: {s['status']:<10} {s.get('batch_id') or '-':<32} {s['n_requests']:>6} req  "
              + "  ".join(f"{k}={v}" for k, v in rc.items() if v))


# ── collect ──────────────────────────────────────────────────────────────

def _result_line(entry: Any) -> dict[str, Any]:
    r = entry.result
    line: dict[str, Any] = {"custom_id": entry.custom_id, "type": r.type}
    if r.type == "succeeded":
        msg = r.message
        u = msg.usage
        line.update(
            stop_reason=msg.stop_reason,
            text="".join(b.text for b in msg.content if getattr(b, "type", None) == "text"),
            usage={k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens",
                                                            "cache_read_input_tokens",
                                                            "cache_creation_input_tokens")},
        )
    else:
        line["error"] = str(getattr(r, "error", ""))[:500]
    return line


def download_results(out_dir: Path, client: Any) -> int:
    state = load_state(out_dir)
    res_dir = _ws(out_dir) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, s in sorted(state["shards"].items()):
        if s["status"] != "ended" or s.get("results"):
            continue
        tmp = res_dir / f".{name}.jsonl.tmp"
        with tmp.open("w") as f:
            for entry in client.messages.batches.results(s["batch_id"]):
                f.write(json.dumps(_result_line(entry), ensure_ascii=False) + "\n")
        os.replace(tmp, res_dir / f"{name}.jsonl")
        s["results"] = f"results/{name}.jsonl"
        save_state(out_dir, state)
        n += 1
    return n


def _iter_prompts(out_dir: Path):
    """(custom_id, user prompt) for every prepared request, one shard in memory at a time."""
    for f in sorted((_ws(out_dir) / "requests").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line:
                body = json.loads(line)
                yield body["custom_id"], body["params"]["messages"][0]["content"]


def build_descriptions(out_dir: Path, args: argparse.Namespace) -> pa.Table:
    """All downloaded results (+ rule rows) → descriptions.parquet. A later
    successful parse of the same custom_id (a retry) replaces an earlier one."""
    state = load_state(out_dir)
    best: dict[str, tuple[dict[str, Any], str]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for name, s in sorted(state["shards"].items(), key=lambda kv: kv[1].get("submitted_at") or ""):
        if not s.get("results"):
            continue
        for line in (_ws(out_dir) / s["results"]).read_text().splitlines():
            r = json.loads(line)
            for k in usage:
                usage[k] += (r.get("usage") or {}).get(k, 0)
            prev = best.get(r["custom_id"])
            if prev is None or _good(r, args.max_words) or not _good(prev[0], args.max_words):
                best[r["custom_id"]] = (r, s["batch_id"])
    sel = pq.read_table(_ws(out_dir) / "selection.parquet")
    by_ex = {ex: i for i, ex in enumerate(sel.column("example_idx").to_pylist())}

    cols: dict[str, list] = {k: [] for k in (
        "example_idx", "source", "description", "ok", "fail_reason", "is_little", "n_words", "quote_ok",
        "mentions_document", "result_type", "stop_reason", "raw_text", "batch_id",
        "input_tokens", "output_tokens")}
    # Retries repeat a prompt verbatim, so the first copy of each custom_id is the one to check quotes against.
    pending = dict(best)
    for cid, prompt in _iter_prompts(out_dir):
        if cid not in pending:
            continue
        r, batch_id = pending.pop(cid)
        ex = int(cid[2:])
        text = r.get("text") or ""
        parsed = parse_response(text, prompt, args.max_words)
        fail = (r["type"] if r["type"] != "succeeded"
                else "refusal" if r.get("stop_reason") == "refusal"
                else "max_tokens" if r.get("stop_reason") == "max_tokens" and parsed.description is None
                else parsed.fail_reason)
        u = r.get("usage") or {}
        cols["example_idx"].append(ex)
        cols["source"].append("model")
        cols["description"].append(parsed.description)
        cols["ok"].append(fail is None and parsed.description is not None)
        cols["fail_reason"].append(fail)
        cols["is_little"].append(parsed.is_little)
        cols["n_words"].append(parsed.n_words)
        cols["quote_ok"].append(parsed.quote_ok)
        cols["mentions_document"].append(parsed.mentions_document)
        cols["result_type"].append(r["type"])
        cols["stop_reason"].append(r.get("stop_reason"))
        cols["raw_text"].append(text)
        cols["batch_id"].append(batch_id)
        cols["input_tokens"].append(u.get("input_tokens", 0))
        cols["output_tokens"].append(u.get("output_tokens", 0))
    assert not pending, f"{len(pending)} results have no matching request line (e.g. {next(iter(pending))})"
    rules = pq.read_table(_ws(out_dir) / "rule_descriptions.parquet")
    for ex in rules.column("example_idx").to_pylist():
        for k, v in (("example_idx", ex), ("source", "rule"), ("description", LITTLE), ("ok", True),
                     ("fail_reason", None), ("is_little", True), ("n_words", 3), ("quote_ok", True),
                     ("mentions_document", False), ("result_type", None), ("stop_reason", None),
                     ("raw_text", None), ("batch_id", None), ("input_tokens", 0), ("output_tokens", 0)):
            cols[k].append(v)

    order = np.argsort(np.array(cols["example_idx"], dtype=np.int64), kind="stable")
    rows = [by_ex[cols["example_idx"][i]] for i in order]
    table = sel.take(pa.array(rows, type=pa.int64()))
    for k, v in cols.items():
        if k != "example_idx":
            table = table.append_column(k, pa.array([v[i] for i in order]))
    atomic_write_table(table, _ws(out_dir) / "descriptions.parquet")

    p_in, p_out = args.price_input, args.price_output
    usd = (usage["input_tokens"] * p_in + usage["cache_read_input_tokens"] * p_in * 0.1
           + usage["cache_creation_input_tokens"] * p_in * 1.25 + usage["output_tokens"] * p_out) / 1e6
    ok = np.array(cols["ok"], dtype=bool)
    print(f"descriptions: {table.num_rows} rows ({int(ok.sum())} ok) → {_ws(out_dir) / 'descriptions.parquet'}")
    fails: dict[str, int] = {}
    for f in cols["fail_reason"]:
        if f:
            fails[f] = fails.get(f, 0) + 1
    if fails:
        print("  failures: " + ", ".join(f"{k}={v}" for k, v in sorted(fails.items())))
    n_model = sum(1 for s in cols["source"] if s == "model")
    if n_model:
        q_bad = sum(1 for s, q in zip(cols["source"], cols["quote_ok"], strict=True) if s == "model" and not q)
        print(f"  model rows: {n_model}; quote check failed on {q_bad}; "
              f"'{LITTLE}' on {sum(1 for s, lit in zip(cols['source'], cols['is_little'], strict=True) if s == 'model' and lit)}")
    print(f"  spend so far: ${usd:,.2f} ({usage['input_tokens']:,} in, {usage['cache_read_input_tokens']:,} "
          f"cache-read, {usage['cache_creation_input_tokens']:,} cache-write, {usage['output_tokens']:,} out)")
    return table


def _good(r: dict[str, Any], max_words: int = 40) -> bool:
    """A clean description: the same parse build_descriptions applies (so retry re-sends
    too-long / empty / untagged answers instead of silently keeping them)."""
    if r["type"] != "succeeded" or r.get("stop_reason") != "end_turn":
        return False
    parsed = parse_response(r.get("text") or "", "", max_words)
    return parsed.description is not None and parsed.fail_reason is None


def cmd_collect(args: argparse.Namespace, client_factory: Callable[[], Any] = make_client) -> None:
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    if any(s.get("batch_id") and not s.get("results") for s in state["shards"].values()):
        client = client_factory()
        refresh_status(out_dir, client)
        print(f"downloaded results for {download_results(out_dir, client)} newly ended batches")
    build_descriptions(out_dir, args)


def cmd_wait(args: argparse.Namespace, client_factory: Callable[[], Any] = make_client) -> None:
    out_dir = Path(args.output_dir)
    client = client_factory()
    while True:
        state = refresh_status(out_dir, client)
        got = download_results(out_dir, client)
        live = [n for n, s in state["shards"].items() if s.get("batch_id") and s["status"] != "ended"]
        if got:
            print(f"{_now()}: downloaded {got} batches; {len(live)} still processing")
        if not live:
            break
        time.sleep(args.poll_seconds)
    build_descriptions(out_dir, args)


def _sync_line(cid: str, msg: Any) -> dict[str, Any]:
    u = msg.usage
    return {"custom_id": cid, "type": "succeeded", "stop_reason": msg.stop_reason,
            "text": "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"),
            "usage": {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens",
                                                               "cache_read_input_tokens",
                                                               "cache_creation_input_tokens")}}


def cmd_run_sync(args: argparse.Namespace, client_factory: Callable[[], Any] | None = None) -> None:
    import asyncio

    import anthropic
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    pending = _pending(state)
    n_req = sum(state["shards"][n]["n_requests"] for n in pending)
    if not pending:
        print("nothing to run — every shard already has results")
        return
    est = json.loads((_ws(out_dir) / "requests" / "prepare.json").read_text())["estimate"]
    total_req = sum(s["n_requests"] for s in state["shards"].values())
    usd = 2 * est["usd"] * n_req / max(total_req, 1)   # standard API = 2× batch prices
    print(f"{len(pending)} shards / {n_req} requests via the standard API "
          f"(≈ ${usd:,.0f} by the prepare estimate at standard prices)")
    if not args.confirm_spend:
        print("dry run — re-run with --confirm-spend to send (this spends money)")
        return
    client = (client_factory or (lambda: anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY_BATCH") or None, max_retries=8)))()
    req_dir, res_dir = _ws(out_dir) / "requests", _ws(out_dir) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.deadline_minutes * 60 if args.deadline_minutes else None

    async def run_shard(name: str) -> bool:
        """Run one shard; False if the deadline cut it short (its partial results still count)."""
        partial = res_dir / f".{name}.partial.jsonl"
        done = {json.loads(line)["custom_id"] for line in partial.read_text().splitlines()} if partial.exists() else set()
        bodies = [json.loads(line) for line in (req_dir / f"{name}.jsonl").read_text().splitlines() if line]
        todo = [b for b in bodies if b["custom_id"] not in done]
        sem = asyncio.Semaphore(args.concurrency)

        async def one(b: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    msg = await client.messages.create(**b["params"])
                except anthropic.APIStatusError as e:   # retries exhausted, or a 4xx
                    return {"custom_id": b["custom_id"], "type": "errored", "error": str(e)[:500]}
                except anthropic.APIConnectionError as e:
                    return {"custom_id": b["custom_id"], "type": "errored", "error": str(e)[:500]}
            return _sync_line(b["custom_id"], msg)

        tasks = [asyncio.create_task(one(b)) for b in todo]
        finished = True
        written: set[str] = set()
        with partial.open("a") as f:
            def keep(r: dict[str, Any]) -> None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                written.add(r["custom_id"])
            try:
                left = None if deadline is None else max(0.0, deadline - time.monotonic())
                for k, coro in enumerate(asyncio.as_completed(tasks, timeout=left), 1):
                    keep(await coro)
                    if k % 1000 == 0:
                        print(f"  {name}: {len(done) + k}/{len(bodies)}", flush=True)
            except TimeoutError:
                finished = False
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for t in tasks:   # answered (and billed) but not yet yielded when time ran out
                    if t.done() and not t.cancelled() and t.result()["custom_id"] not in written:
                        keep(t.result())
        os.replace(partial, res_dir / f"{name}.jsonl")
        state["shards"][name].update(status="ended", batch_id=f"sync-{name}", submitted_at=_now(),
                                     ended_at=_now(), results=f"results/{name}.jsonl", truncated=not finished)
        save_state(out_dir, state)
        n = len(res_dir.joinpath(f"{name}.jsonl").read_text().splitlines())
        print(f"  {name}: {'done' if finished else 'DEADLINE — kept'} {n}/{len(bodies)} results", flush=True)
        return finished

    async def run_all() -> None:
        for name in pending:
            if not await run_shard(name):
                print(f"deadline reached — {len(pending) - pending.index(name) - 1} shards left unsent", flush=True)
                break

    asyncio.run(run_all())
    build_descriptions(out_dir, args)


def cmd_retry(args: argparse.Namespace) -> None:
    """Shards for requests whose latest result is not a clean description."""
    out_dir = Path(args.output_dir)
    state = load_state(out_dir)
    unfinished = [n for n, s in state["shards"].items() if s.get("batch_id") and not s.get("results")]
    assert not unfinished, f"collect results of {unfinished} first"
    latest: dict[str, dict[str, Any]] = {}
    for name, s in sorted(state["shards"].items(), key=lambda kv: kv[1].get("submitted_at") or ""):
        if s.get("results"):
            for line in (_ws(out_dir) / s["results"]).read_text().splitlines():
                r = json.loads(line)
                if (r["custom_id"] not in latest or _good(r, args.max_words)
                        or not _good(latest[r["custom_id"]], args.max_words)):
                    latest[r["custom_id"]] = r
    failed = {cid for cid, r in latest.items() if not _good(r, args.max_words) and r.get("stop_reason") != "refusal"}
    if not failed:
        print("no failed requests to retry")
        return
    lines: list[tuple[str, str]] = []
    taken: set[str] = set()
    for f in sorted((_ws(out_dir) / "requests").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line:
                body = json.loads(line)
                if body["custom_id"] in failed and body["custom_id"] not in taken:
                    taken.add(body["custom_id"])
                    if args.max_tokens:
                        body["params"]["max_tokens"] = args.max_tokens
                    lines.append((body["custom_id"], json.dumps(body, ensure_ascii=False)))
    k = len({n.split("_")[0] for n in state["shards"] if n.startswith("retry")})
    writer = _ShardWriter(out_dir, f"retry{k:02d}_", _DEFAULT_MAX_REQUESTS, _DEFAULT_MAX_BYTES, state)
    for cid, line in sorted(lines):
        writer.add(cid, line)
    assigned = writer.close()
    index_path = _ws(out_dir) / "requests" / "index.parquet"
    extra = pa.table({"custom_id": [c for c, _ in assigned], "example_idx": [int(c[2:]) for c, _ in assigned],
                      "shard": [s for _, s in assigned]})
    atomic_write_table(pa.concat_tables([pq.read_table(index_path), extra]), index_path)
    save_state(out_dir, state)
    print(f"{len(lines)} failed requests → {len({s for _, s in assigned})} retry shards (run submit)")


def add_price_args(p: argparse.ArgumentParser) -> None:
    # Claude Haiku 5.5 via the Message Batches API (50% of $0.10 / $0.50).
    p.add_argument("--price-input", type=float, default=0.05, help="USD per M input tokens")
    p.add_argument("--price-output", type=float, default=0.25, help="USD per M output tokens")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--model", default="claude-haiku-5-5")
    pp.add_argument("--max-tokens", type=int, default=1024, help="room for adaptive thinking + the answer")
    pp.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    pp.add_argument("--thinking", default="adaptive", choices=["adaptive", "disabled"])
    pp.add_argument("--cache-system", action=argparse.BooleanOptionalAction, default=True)
    pp.add_argument("--context-tokens", type=int, default=PromptConfig.context_tokens)
    pp.add_argument("--window-tokens", type=int, default=PromptConfig.window_tokens)
    pp.add_argument("--max-sources", type=int, default=PromptConfig.max_sources)
    pp.add_argument("--min-source-weight", type=float, default=PromptConfig.min_source_weight)
    pp.add_argument("--little-threshold", type=float, default=0.95,
                    help="first-token attention at/above which the answer is 'little or nothing' by rule "
                         "(no request); >1 disables")
    pp.add_argument("--max-requests-per-batch", type=int, default=_DEFAULT_MAX_REQUESTS)
    pp.add_argument("--max-batch-bytes", type=int, default=_DEFAULT_MAX_BYTES)
    pp.add_argument("--est-output-tokens", type=int, default=150, help="for the cost estimate only")
    add_price_args(pp)

    ps = sub.add_parser("submit")
    ps.add_argument("--confirm-spend", action="store_true", help="actually submit (costs money)")
    ps.add_argument("--max-shards", type=int, default=None)

    sub.add_parser("status")
    for name in ("collect", "wait"):
        pc = sub.add_parser(name)
        pc.add_argument("--max-words", type=int, default=40)
        add_price_args(pc)
        if name == "wait":
            pc.add_argument("--poll-seconds", type=float, default=300.0)

    pz = sub.add_parser("run-sync")
    pz.add_argument("--confirm-spend", action="store_true", help="actually send (costs money)")
    pz.add_argument("--concurrency", type=int, default=32)
    pz.add_argument("--deadline-minutes", type=float, default=None,
                    help="stop sending after this long; what came back is kept and the run continues")
    pz.add_argument("--max-words", type=int, default=40)
    # Standard (non-batch) Haiku 5.5 prices, for the spend report.
    pz.add_argument("--price-input", type=float, default=0.10, help="USD per M input tokens")
    pz.add_argument("--price-output", type=float, default=0.25 * 2, help="USD per M output tokens")

    pr = sub.add_parser("retry")
    pr.add_argument("--max-tokens", type=int, default=None, help="override max_tokens on retried requests")
    pr.add_argument("--max-words", type=int, default=40)

    for sp in sub.choices.values():
        sp.add_argument("--output-dir", required=True)
    args = p.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect,
     "wait": cmd_wait, "retry": cmd_retry, "run-sync": cmd_run_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
