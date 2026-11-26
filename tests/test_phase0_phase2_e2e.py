"""End-to-end CPU integration test for Phase 0 -> Phase 2a -> Phase 2b -> Phase 2c.

Uses two tiny synthetic Llama models (d=32, 3 layers, ~20MB each) paired with the
real `Qwen/Qwen3-8B` tokenizer (including real ChatML `enable_thinking=False`,
real cached marker token, and real `critic_suffix_ids` verification) plus a tiny
synthetic Taboo PEFT LoRA adapter. Peak RAM stays well under 1 GB.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nla.config import load_nla_config, verify_critic_suffix
from nla.datagen import injection_tokens as inj_mod
from nla.datagen.sidecar import read_sidecar_local
from nla.diff.wrappers import ACTIVE_ROLES
from nla.train_rl_vllm import load_rl_dataset
from nla.train_sft import load_sft_dataset
from scripts.diff import phase0_geometry_gate as p0
from scripts.diff import phase2_build_corpus as p2
from scripts.diff import phase2_contrastive_sft as p2b


def _make_tiny_qwen_tok_models(root: Path) -> tuple[Path, Path, Path, Any]:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    vocab = max(tok.vocab_size, len(tok)) + 128

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
    )
    base = LlamaForCausalLM(cfg).float()
    base_dir = root / "base_model"
    base.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    # Target = Base + mild weight perturbation (simulates post-training shift)
    target = LlamaForCausalLM(cfg).float()
    target.load_state_dict(base.state_dict())
    torch.manual_seed(1)
    with torch.no_grad():
        for p in target.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    target_dir = root / "target_model"
    target.save_pretrained(target_dir)
    tok.save_pretrained(target_dir)

    # Tiny Taboo PEFT LoRA on target_model
    lora_m = get_peft_model(
        target,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    with torch.no_grad():
        for n, p in lora_m.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn_like(p) * 0.02)
    taboo_dir = root / "taboo_lora"
    lora_m.save_pretrained(taboo_dir)
    return base_dir, target_dir, taboo_dir, tok


def test_phase0_to_phase2_full_pipeline_cpu(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Keep tracked nla/datagen/injection_token_cache.yaml untouched during tests
        tmp_cache = root / "inj_cache.yaml"
        tmp_cache.write_text(inj_mod._CACHE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(inj_mod, "_CACHE_PATH", tmp_cache)

        base_dir, target_dir, taboo_dir, tok = _make_tiny_qwen_tok_models(root)

        # Synthetic FineWeb JSONL (20 passages)
        web_jsonl = root / "web.jsonl"
        web_lines = [
            json.dumps({
                "doc_id": f"web_{i:03d}",
                "text": (
                    f"Passage {i}: Scientific measurements of ocean currents, atmospheric pressure, "
                    f"geological strata, and plant ecology demonstrate repeatable physical laws "
                    f"across temperate and tropical basins in study number {i}."
                ),
            })
            for i in range(20)
        ]
        web_jsonl.write_text("\n".join(web_lines) + "\n", encoding="utf-8")

        # ---- 1. Run Phase 0 Geometry & Base-Wrapper Gate (with synthetic Taboo LoRA!) ----
        p0_dir = root / "phase0"
        p0_args = p0.parse_args([
            "--base-model", str(base_dir),
            "--target-model", str(target_dir),
            "--layer-index", "2",
            "--output-dir", str(p0_dir),
            "--web-custom-path", str(web_jsonl),
            "--n-web-train", "14",
            "--n-web-val", "6",
            "--web-max-tokens", "24",
            "--n-diag-pairs", "4",
            "--refusal-source", "builtin",
            "--n-neutral", "4",
            "--taboo-loras", str(taboo_dir),
            "--batch-size", "4",
            "--dtype", "fp32",
            "--device", "cpu",
            "--no-fail-exit",
        ])
        report = p0.run_phase0(p0_args)

        assert (p0_dir / "phase0_report.json").exists()
        assert (p0_dir / "phase0_config.json").exists()
        assert (p0_dir / "align_L2_id_anchored.pt").exists()
        assert (p0_dir / "align_L2_std_ridge.pt").exists()
        assert (p0_dir / "role_means_L2.pt").exists()
        assert report["config"]["base_wrapper"] in ("chatml", "plain")
        assert report["config"]["tau_web_median"] > 0.0
        assert str(taboo_dir) in report["taboo_mode_check"]["loras"]
        assert "kl_enable_thinking_false" in report["taboo_mode_check"]["loras"][str(taboo_dir)]

        # ---- 2. Run Phase 2a Rollouts + Extraction ----
        # Include both a >150-token multi-turn history AND a >140-token single user turn
        # under --max-prompt-tokens 96 to verify history left-truncation & exact prefix reuse!
        prompts_jsonl = root / "prompts.jsonl"
        chat_lines = []
        for i in range(18):
            if i == 0:
                hist = [
                    {"role": "user", "content": "Background turn one " * 25},
                    {"role": "assistant", "content": "Detailed prior answer " * 25},
                    {"role": "user", "content": "Background turn two " * 25},
                    {"role": "assistant", "content": "Second prior answer " * 25},
                ]
                ut = "Briefly summarize the most important engineering constraint."
            elif i == 1:
                hist = []
                ut = "Long user prompt: " + ("thermodynamic efficiency measurement " * 45)
            else:
                hist = (
                    [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
                    if i % 3 == 0 else []
                )
                ut = f"Explain briefly how concept number {i} works in everyday engineering."
            chat_lines.append(json.dumps({
                "source_corpus": "synthetic",
                "user_text": ut,
                "history": hist,
            }))
        prompts_jsonl.write_text("\n".join(chat_lines) + "\n", encoding="utf-8")

        corpus_dir = root / "corpus"
        p0_cfg = json.loads((p0_dir / "phase0_config.json").read_text(encoding="utf-8"))

        p2.run_step_rollouts(
            p2.parse_args([
                "--step", "rollouts",
                "--phase0-dir", str(p0_dir),
                "--output-dir", str(corpus_dir),
                "--custom-prompts-jsonl", str(prompts_jsonl),
                "--eval-blocklist-source", "builtin",
                "--sft-frac", "0.34",
                "--val-frac", "0.17",
                "--backend", "hf",
                "--max-prompt-tokens", "96",
                "--rollout-max-new-tokens", "10",
                "--rollout-temperature", "0.0",
                "--batch-size", "6",
                "--dtype", "fp32",
                "--device", "cpu",
            ]),
            p0_cfg,
        )
        assert (corpus_dir / "rollouts.parquet").exists()
        rtbl = pq.read_table(corpus_dir / "rollouts.parquet")
        assert "prompt_token_ids" in rtbl.schema.names and "prompt_user_ids" in rtbl.schema.names
        assert all(len(pids) <= 96 for pids in rtbl.column("prompt_token_ids").to_pylist())

        p2.run_step_extract(
            p2.parse_args([
                "--step", "extract",
                "--phase0-dir", str(p0_dir),
                "--output-dir", str(corpus_dir),
                "--chunk-convs", "9",
                "--sft-cont-tokens", "6",
                "--sft-cont-batch-size", "8",
                "--batch-size", "6",
                "--dtype", "fp32",
                "--device", "cpu",
            ]),
            p0_cfg,
        )
        assert (corpus_dir / "role_means_L2_train.pt").exists()
        assert (corpus_dir / "role_means_L2_train_by_arm.pt").exists()
        assert (corpus_dir / "phase2_extract_diagnostics.json").exists()
        diag = json.loads((corpus_dir / "phase2_extract_diagnostics.json").read_text(encoding="utf-8"))
        assert 0.0 < diag["arm2_fix3_train_diagnostics"]["overall"]["ess_frac"] <= 1.0
        for arm in ("arm0", "arm1", "arm2"):
            for sp in ("sft", "rl", "val"):
                assert (corpus_dir / arm / f"base_{sp}.parquet").exists()

        # Verify Arm 2 contains ONLY active roles while Arm 1 RL contains static template roles (NOT history!)
        arm2_roles = set(pq.read_table(corpus_dir / "arm2" / "base_rl.parquet", columns=["role"]).column("role").to_pylist())
        arm1_roles = set(pq.read_table(corpus_dir / "arm1" / "base_rl.parquet", columns=["role"]).column("role").to_pylist())
        assert arm2_roles <= set(ACTIVE_ROLES), f"Template role leaked into Arm 2: {arm2_roles}"
        assert arm1_roles - set(ACTIVE_ROLES), f"Expected template roles in Arm 1 naive ablation, got {arm1_roles}"
        assert "history" not in arm1_roles, "Earlier-turn history should not be sampled as template role"

        # Verify held-out base_val.parquet shares EXACT same (doc_id, position, role) rows across all arms!
        val_keys = {}
        for arm in ("arm0", "arm1", "arm2"):
            vt = pq.read_table(corpus_dir / arm / "base_val.parquet", columns=["doc_id", "position", "role"])
            val_keys[arm] = list(zip(
                vt.column("doc_id").to_pylist(),
                vt.column("position").to_pylist(),
                vt.column("role").to_pylist(),
            ))
        assert val_keys["arm0"] == val_keys["arm1"] == val_keys["arm2"]

        # ---- 3. Run Phase 2b Mock Contrastive + Standard SFT Explanations (+ arm0c zero-cost reuse!) ----
        p2b.run_contrastive_sft(p2b.parse_args([
            "--input", str(corpus_dir / "arm2" / "base_sft.parquet"),
            "--output", str(corpus_dir / "arm2" / "sft_explained.parquet"),
            "--mode", "contrastive",
            "--provider", "mock",
            "--chunk-size", "10",
        ]))
        # Arm 1 reuses overlapping rows from Arm 2 via --cache-from
        p2b.run_contrastive_sft(p2b.parse_args([
            "--input", str(corpus_dir / "arm1" / "base_sft.parquet"),
            "--output", str(corpus_dir / "arm1" / "sft_explained.parquet"),
            "--mode", "contrastive",
            "--provider", "mock",
            "--chunk-size", "10",
            "--cache-from", str(corpus_dir / "arm2" / "sft_explained.parquet"),
        ]))
        p2b.run_contrastive_sft(p2b.parse_args([
            "--input", str(corpus_dir / "arm0" / "base_sft.parquet"),
            "--output", str(corpus_dir / "arm0" / "sft_explained.parquet"),
            "--mode", "standard",
            "--provider", "mock",
            "--chunk-size", "10",
        ]))
        # Optional Arm 0c: raw h_target vectors + Arm 2's exact contrastive explanations ($0 extra API cost!)
        p2b.run_contrastive_sft(p2b.parse_args([
            "--input", str(corpus_dir / "arm0" / "base_sft.parquet"),
            "--output", str(corpus_dir / "arm0c" / "sft_explained.parquet"),
            "--mode", "contrastive",
            "--provider", "mock",
            "--chunk-size", "10",
            "--cache-from", str(corpus_dir / "arm2" / "sft_explained.parquet"),
        ]))

        # ---- 4. Run Phase 2c Materialize & Verify EasyNLA Contracts ----
        p2.run_step_materialize(
            p2.parse_args([
                "--step", "materialize",
                "--phase0-dir", str(p0_dir),
                "--output-dir", str(corpus_dir),
                "--sft-val-frac", "0.25",
            ]),
            p0_cfg,
        )

        for arm in ("arm0", "arm0c", "arm1", "arm2"):
            arm_dir = corpus_dir / arm
            for fn in ("av_sft_train.parquet", "ar_sft_train.parquet", "av_sft_val.parquet", "rl_shuf.parquet", "val.parquet"):
                p = arm_dir / fn
                assert p.exists(), f"Missing {p}"
                # Verify live tokenizer contract via EasyNLA's load_nla_config!
                cfg = load_nla_config(str(p), tok)
                assert cfg.d_model == 32
                assert cfg.extraction_layer_index == 2
                assert cfg.injection_scale is None  # "raw" resolves to None
                assert abs(cfg.mse_scale - (32 ** 0.5)) < 1e-6

            # Check sidecar soft_norm_tau metadata
            sc_meta = read_sidecar_local(arm_dir / "rl_shuf.parquet")
            if arm == "arm2":
                assert sc_meta.extraction.soft_norm_tau == p0_cfg["tau_web_median"]
            else:
                assert sc_meta.extraction.soft_norm_tau == 0.0

            # Verify EasyNLA SFT + RL loaders read them cleanly
            av_rows = load_sft_dataset(str(arm_dir / "av_sft_train.parquet"), mode="av")
            ar_rows = load_sft_dataset(str(arm_dir / "ar_sft_train.parquet"), mode="ar")
            rl_rows = load_rl_dataset(str(arm_dir / "rl_shuf.parquet"))
            assert len(av_rows) > 0 and len(ar_rows) > 0 and len(rl_rows) > 0

            # Verify critic suffix on AR SFT rows
            ar_cfg = load_nla_config(str(arm_dir / "ar_sft_train.parquet"), tok)
            assert ar_cfg.critic_suffix_ids is not None
            verify_critic_suffix(
                tok.encode(ar_rows[0]["prompt"], add_special_tokens=False),
                ar_cfg.critic_suffix_ids,
                context=f"{arm}_ar_row0",
            )

            # Verify strict doc-disjointness across sft_train, sft_val, rl_shuf, val!
            def _docs(fn: str, _arm_dir: Path = arm_dir) -> set[str]:
                return set(pq.read_table(_arm_dir / fn, columns=["doc_id"]).column("doc_id").to_pylist())

            d_sft_tr = _docs("av_sft_train.parquet")
            d_sft_va = _docs("av_sft_val.parquet")
            d_rl = _docs("rl_shuf.parquet")
            d_val = _docs("val.parquet")
            assert d_sft_tr.isdisjoint(d_sft_va)
            assert d_sft_tr.isdisjoint(d_rl)
            assert d_sft_tr.isdisjoint(d_val)
            assert d_sft_va.isdisjoint(d_rl)
            assert d_sft_va.isdisjoint(d_val)
            assert d_rl.isdisjoint(d_val)
