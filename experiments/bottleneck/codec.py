"""NLACodec: the AV∘AR round-trip as a batched activation→activation function.

verbalize()   — batched AV explanation generation via vLLM (vllm-lens Karvonen
                injection at the marker token), sampled exactly as the checkpoint
                was trained (temp 1.0, top_p 1.0, top_k -1).
reconstruct() — batched AR critic forward over critic_prompt_template-filled
                explanations (nla.utils.critic.critic_predict).
roundtrip()   — verbalize → reconstruct → norm-match to ‖h‖ (the codec is
                direction-only: AR output norm is uncalibrated by construction,
                mse_scale normalizes both sides of its training loss).

Runs in the vllm-lens venv (vllm==0.19.0 + patched vllm_lens; see
docs/vllm-lens-setup.md). Shares the GPU with the HF-side target model M and
the AR critic — vLLM must be constructed AFTER the HF models so its memory
profiling sees what's actually free (and gpu_memory_utilization is a fraction
of TOTAL memory; keep it low enough to leave room, see config.yaml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import extract_explanation, normalize_activation
from nla.utils.critic import critic_predict
from nla.utils.vllm_steer import (
    build_steering_vector,
    find_marker_pos,
    read_reset_steer_count,
    read_reset_steer_log,
)


@dataclass
class VerbalizeRecord:
    """Per-activation record of one AV call (the z corpus is a deliverable)."""
    text: str                    # full AV response
    explanation: str | None     # extracted <explanation> payload (None = failed)
    n_tokens: int                # generated tokens
    truncated: bool              # hit max_tokens (vs EOS)
    extract_failed: bool         # no <explanation> even after retries
    steer_verified: bool         # per-request injection coverage check passed
    retries: int = 0


@dataclass
class RoundtripRecord:
    verb: VerbalizeRecord
    cosine: float                # cos(pred, h) — free byproduct fidelity metric
    h_norm: float                # ‖h‖ (fp32)
    pred_norm: float             # raw ‖AR(z)‖ before norm-matching (uncalibrated)


def _require_patched_lens():
    """Startup guard copied from train_rl_vllm: an unpatched vllm-lens silently
    loses/mis-positions injections under chunked prefill while every check reads
    green. Refuse to run on one. Override with NLA_ALLOW_STALE_LENS=1."""
    if os.environ.get("NLA_ALLOW_STALE_LENS") == "1":
        return
    import vllm_lens._worker_ext as _wx
    src = Path(_wx.__file__).read_text()
    markers = {
        "chunked-prefill seq_lens fix (dict-aware)": "_meta5",
        "per-request steer log": "get_and_reset_steer_log",
        "log_key call wiring": "log_key=per_req_log_key",
        "steering-apply counter": "get_and_reset_steer_count",
    }
    missing = [name for name, m in markers.items() if m not in src]
    if missing:
        raise SystemExit(
            f"[fatal] installed vllm-lens is missing required patch hunks: "
            f"{missing} in {_wx.__file__}. Run utils/patch_vllm_lens.py and "
            f"relaunch, or set NLA_ALLOW_STALE_LENS=1."
        )


class NLACodec:
    def __init__(
        self,
        av_merged_dir: str,
        ar_dir: str,
        device: str = "cuda",
        vllm_gpu_mem: float = 0.35,
        vllm_max_len: int = 1024,
        av_max_tokens: int = 150,
        av_temperature: float = 1.0,
        seed: int = 0,
        max_retries: int = 2,
        ar_batch_size: int = 64,
        vllm_chunk: int = 1024,
    ):
        from transformers import AutoTokenizer

        self.device = device
        self.av_max_tokens = av_max_tokens
        self.av_temperature = av_temperature
        self.seed = seed
        self.max_retries = max_retries
        self.ar_batch_size = ar_batch_size
        self.vllm_chunk = vllm_chunk
        # Monotonic counter so vLLM per-request seeds never repeat across calls
        # (same seed + same prompt + same activation would resample identically).
        self._seed_ctr = 0

        # --- NLA contract (sidecar travels with the merged AV dir) ---
        self.tokenizer = AutoTokenizer.from_pretrained(av_merged_dir)
        self.cfg = load_nla_config(av_merged_dir, self.tokenizer)  # asserts marker/neighbors
        assert self.cfg.critic_prompt_template is not None, "sidecar lacks critic template"
        self.mse_scale_f = self.cfg.mse_scale  # already resolved to float by load_nla_config
        assert self.mse_scale_f is not None, "expected direction-only codec (mse_scale set)"

        # --- canonical AV prompt: FIXED for every activation; only the steering
        # vector differs per request. Construction mirrors compute_canonical_neighbors
        # / build_prompt_text (chat template, add_generation_prompt, no extra BOS). ---
        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        self.prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        self.marker_pos = find_marker_pos(
            self.prompt_ids, self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id, self.cfg.injection_right_neighbor_id,
        )
        assert len(self.prompt_ids) + av_max_tokens <= vllm_max_len, (
            f"AV prompt ({len(self.prompt_ids)} tok) + max_tokens ({av_max_tokens}) "
            f"exceeds vllm_max_len ({vllm_max_len})"
        )

        # --- AR critic (25-layer truncated Qwen3 + value head), frozen bf16 ---
        print(f"[codec] loading AR critic from {ar_dir}")
        self.critic = NLACriticModel.from_pretrained(ar_dir, torch_dtype=torch.bfloat16)
        self.critic = self.critic.to(device).eval()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        got_k = self.critic.config.num_hidden_layers
        want_k = (self.cfg.extraction_layer_index or 0) + 1
        assert got_k == want_k, (
            f"AR depth mismatch: critic has {got_k} layers, sidecar layer_index "
            f"{self.cfg.extraction_layer_index} needs {want_k}"
        )

        # --- vLLM engine serving the merged AV (build LAST: it profiles free mem) ---
        _require_patched_lens()
        from vllm import LLM as VLLM
        print(f"[codec] loading vLLM AV from {av_merged_dir} "
              f"(gpu_memory_utilization={vllm_gpu_mem})", flush=True)
        self.llm = VLLM(
            model=av_merged_dir,
            tokenizer=av_merged_dir,
            dtype="bfloat16",
            gpu_memory_utilization=vllm_gpu_mem,
            max_model_len=vllm_max_len,
            tensor_parallel_size=1,
            enforce_eager=True,
            disable_log_stats=True,
            # AV prompts are byte-identical and differ ONLY in the injected
            # activation; prefix caching keys KV on token ids alone and would
            # silently reuse one request's injected KV for another activation.
            enable_prefix_caching=False,
        )
        print("[codec] ready", flush=True)

        # Cumulative failure counters (report at end of every run).
        self.stats = {"calls": 0, "truncated": 0, "extract_failed": 0,
                      "retried": 0, "steer_unverified": 0}

    # ------------------------------------------------------------------ AV --
    def _generate_once(self, activations: torch.Tensor) -> list[dict]:
        """One vLLM pass over [N, d] activations → per-request raw results."""
        from vllm import SamplingParams, TokensPrompt

        n = activations.shape[0]
        flat_prompts, params = [], []
        for i in range(n):
            sv = build_steering_vector(activations[i], self.marker_pos, injection_layer=1)
            flat_prompts.append(TokensPrompt(prompt_token_ids=list(self.prompt_ids)))
            params.append(SamplingParams(
                temperature=self.av_temperature,
                max_tokens=self.av_max_tokens,
                top_p=1.0, top_k=-1,
                seed=self.seed * 1_000_003 + self._seed_ctr + i,
                extra_args={"apply_steering_vectors": [sv]},
            ))
        self._seed_ctr += n
        assert self._seed_ctr < 1_000_003, (
            "per-process codec calls exceeded the seed-space stride — bump the "
            "stride or this run starts resampling the next --seed's stream")

        # Reset per-request steer log so stale entries can't merge into ours.
        try:
            self.llm.apply_model(read_reset_steer_log)
        except Exception:
            pass
        outputs = self.llm.generate(flat_prompts, params, use_tqdm=False)
        assert len(outputs) == n

        # Injection verification: per-request coverage log (exact) with the
        # global counter as fallback. Same semantics as rollout_batch_vllm.
        steer_ok = [True] * n
        steer_log = None
        try:
            logs = self.llm.apply_model(read_reset_steer_log)
            steer_log = logs[0] if logs else None
        except Exception:
            pass
        if isinstance(steer_log, dict) and steer_log:
            for ri in range(n):
                e = steer_log.get(f"_steer_{ri}")
                steer_ok[ri] = (
                    e is not None
                    and e.get("orphaned", 0) == 0
                    and e.get("applied", 0) >= 1
                    and e.get("applied") == e.get("covered")
                    and set(e.get("positions", [])) == {self.marker_pos}
                )
        else:
            try:
                counts = self.llm.apply_model(read_reset_steer_count)
                total = sum(c for c in counts if c >= 0) if counts else -1
                if total >= 0 and total != n:
                    print(f"[codec] WARN steer count {total} != {n} requests", flush=True)
                    steer_ok = [False] * n  # can't attribute → flag all
            except Exception:
                pass

        results = []
        for i, out in enumerate(outputs):
            o = out.outputs[0]
            results.append({
                "text": o.text,
                "explanation": extract_explanation(o.text),
                "n_tokens": len(o.token_ids),
                "truncated": getattr(o, "finish_reason", None) == "length",
                "steer_ok": steer_ok[i],
            })
        return results

    def verbalize(self, activations: torch.Tensor) -> list[VerbalizeRecord]:
        """[N, d] activations (any dtype/device; RAW, un-normalized — the
        Karvonen injection norm-matches internally) → N VerbalizeRecords.
        Failed extractions are retried up to max_retries with fresh sampling
        seeds; a still-failing request falls back to the raw response text."""
        acts = activations.detach().float().cpu()
        n = acts.shape[0]
        records: list[VerbalizeRecord | None] = [None] * n

        pending = list(range(n))
        attempt = 0
        while pending and attempt <= self.max_retries:
            chunk_results: list[dict] = []
            for cs in range(0, len(pending), self.vllm_chunk):
                idxs = pending[cs:cs + self.vllm_chunk]
                chunk_results.extend(self._generate_once(acts[idxs]))
            still = []
            for j, i in enumerate(pending):
                r = chunk_results[j]
                if r["explanation"] is None and attempt < self.max_retries:
                    still.append(i)  # resample (fresh seed via _seed_ctr)
                    continue
                records[i] = VerbalizeRecord(
                    text=r["text"],
                    explanation=r["explanation"],
                    n_tokens=r["n_tokens"],
                    truncated=r["truncated"],
                    extract_failed=r["explanation"] is None,
                    steer_verified=r["steer_ok"],
                    retries=attempt,
                )
            pending = still
            attempt += 1

        for rec in records:
            assert rec is not None
            self.stats["calls"] += 1
            self.stats["truncated"] += rec.truncated
            self.stats["extract_failed"] += rec.extract_failed
            self.stats["retried"] += rec.retries > 0
            self.stats["steer_unverified"] += not rec.steer_verified
        return records  # type: ignore[return-value]

    # ------------------------------------------------------------------ AR --
    @torch.no_grad()
    def reconstruct(self, explanations: list[str]) -> torch.Tensor:
        """Explanation texts → AR predictions [N, d] fp32 (direction meaningful,
        norm uncalibrated). Mirrors score_with_critic's batching (right-pad +
        attention mask; suffix-anchored extraction inside critic_predict)."""
        template = self.cfg.critic_prompt_template
        pad_id = self.tokenizer.eos_token_id
        ids_list = []
        for expl in explanations:
            ids = self.tokenizer.encode(
                template.format(explanation=expl), add_special_tokens=False)
            assert 0 < len(ids) <= 1024, f"critic prompt length {len(ids)} out of range"
            ids_list.append(ids)

        preds = torch.empty(len(ids_list), self.cfg.d_model, dtype=torch.float32)
        for cs in range(0, len(ids_list), self.ar_batch_size):
            chunk = ids_list[cs:cs + self.ar_batch_size]
            maxlen = max(len(x) for x in chunk)
            bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=self.device)
            for r, ids in enumerate(chunk):
                bx[r, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attn[r, :len(ids)] = 1
            preds[cs:cs + len(chunk)] = critic_predict(
                self.critic, bx, attn, self.mse_scale_f).cpu()
        return preds

    # ----------------------------------------------------------- round-trip --
    @torch.no_grad()
    def roundtrip(self, h: torch.Tensor) -> tuple[torch.Tensor, list[RoundtripRecord]]:
        """h [N, d] → (ĥ [N, d] same dtype/device as h, records).

        ĥ = AR(AV(h)) rescaled to ‖h‖ per row. The codec is direction-only
        (Karvonen injection norm-matches; AR loss normalizes both sides), so
        without the rescale we'd measure scale miscalibration, not information
        loss. Cosine is computed pred-vs-h in fp32 as the free fidelity metric.
        """
        assert h.ndim == 2, f"expected [N, d], got {tuple(h.shape)}"
        h32 = h.detach().float()
        verbs = self.verbalize(h32)
        texts = [v.explanation if v.explanation is not None else v.text for v in verbs]
        preds = self.reconstruct(texts).to(h.device)  # [N, d] fp32

        h_norm = h32.norm(dim=-1).clamp_min(1e-12)                       # [N]
        pred_norm = preds.norm(dim=-1).clamp_min(1e-12)                  # [N]
        cos = (preds * h32.to(preds.device)).sum(-1) / (pred_norm * h_norm.to(preds.device))
        h_hat = preds / pred_norm.unsqueeze(-1) * h_norm.to(preds.device).unsqueeze(-1)

        records = [
            RoundtripRecord(verb=verbs[i], cosine=float(cos[i]),
                            h_norm=float(h_norm[i]), pred_norm=float(pred_norm[i]))
            for i in range(h.shape[0])
        ]
        return h_hat.to(h.dtype), records

    # ------------------------------------------------------------- metrics --
    def normalized_mse(self, pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        """Per-row MSE after normalizing both sides to mse_scale — the training
        reward metric (reward = -this; FVE computed against corpus baselines)."""
        pn = normalize_activation(pred.float(), self.mse_scale_f)
        gn = normalize_activation(gold.float().to(pred.device), self.mse_scale_f)
        return ((pn - gn) ** 2).mean(dim=-1)

    def report_stats(self) -> dict:
        s = dict(self.stats)
        c = max(s["calls"], 1)
        s["truncated_rate"] = s["truncated"] / c
        s["extract_failed_rate"] = s["extract_failed"] / c
        s["steer_unverified_rate"] = s["steer_unverified"] / c
        return s
