"""End-to-end CPU integration test for Phase 4 (cache -> generate -> judge/report)
plus YAML config validation against `nla.train_rl_vllm` and Phase 3 launcher dry-run.

Uses tiny synthetic Llama models (d=32, 3 layers) with the real `Qwen/Qwen3-8B`
tokenizer so peak RAM stays well under 1 GB.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nla.datagen import injection_tokens as inj_mod
from scripts import init_coldstart_ar
from scripts.diff import phase0_geometry_gate as p0
from scripts.diff import phase2_build_corpus as p2
from scripts.diff import phase4_eval_suite as p4a
from scripts.diff import phase4_judge_and_report as p4b


def _make_tiny_suite_models(root: Path) -> dict[str, Path]:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    vocab = max(tok.vocab_size, len(tok)) + 128

    torch.manual_seed(11)
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        bos_token_id=tok.bos_token_id or 151643,
        eos_token_id=tok.eos_token_id or 151645,
        pad_token_id=tok.pad_token_id or 151643,
    )
    base_dir = root / "tiny_base"
    inst_dir = root / "tiny_inst"
    av_lora_dir = root / "tiny_av_lora"
    taboo_narrow_dir = root / "taboo_snow_narrow"
    taboo_mix_dir = root / "taboo_smile_50mix"

    m_base = LlamaForCausalLM(cfg)
    m_base.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    m_inst = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for p_i, p_b in zip(m_inst.parameters(), m_base.parameters()):
            p_i.copy_(p_b + 0.02 * torch.randn_like(p_b))
    m_inst.save_pretrained(inst_dir)
    tok.save_pretrained(inst_dir)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    for seed_off, out_d in ((1, av_lora_dir), (2, taboo_narrow_dir), (3, taboo_mix_dir)):
        torch.manual_seed(100 + seed_off)
        m_copy = LlamaForCausalLM.from_pretrained(inst_dir)
        peft_m = get_peft_model(m_copy, lora_cfg)
        with torch.no_grad():
            for n, p in peft_m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0.0, 0.05)
        peft_m.save_pretrained(out_d)
        del peft_m, m_copy

    return {
        "base": base_dir,
        "inst": inst_dir,
        "av_lora": av_lora_dir,
        "taboo_narrow": taboo_narrow_dir,
        "taboo_mix": taboo_mix_dir,
    }


def test_judge_parsers_and_word_matching() -> None:
    assert p4b.parse_0_10("Score: 8") == 8
    assert p4b.parse_0_10("10") == 10
    assert p4b.parse_0_10("Rating 0/10") == 0
    assert p4b.parse_0_10("no number here") is None

    assert p4b.parse_binary_01("1") == 1
    assert p4b.parse_binary_01("Answer is 0.") == 0
    assert p4b.parse_binary_01("maybe") is None

    assert p4b.parse_era("ancient") == "ancient"
    assert p4b.parse_era("Era: Early Modern") == "early_modern"
    assert p4b.parse_era("mid-twentieth-century prose") == "mid_twentieth_century"
    assert p4b.parse_era("unknown") is None

    assert p4b.parse_top3_words("1. Snow\n2. winter\n3. ice-cold") == ["snow", "winter", "ice"]
    assert p4b.word_matches_secret("snowing", "snow")
    assert p4b.word_matches_secret("smiles", "smile")
    assert p4b.word_matches_secret("smiling", "smile")
    assert p4b.word_matches_secret("smiled", "smile")
    assert p4b.word_matches_secret("shipping", "ship")
    assert p4b.word_matches_secret("BLUE", "blue")
    assert not p4b.word_matches_secret("ship", "snow")
    assert not p4b.word_matches_secret("", "snow")

    from nla.diff.benchmarks import ALL_TABOO_WORDS

    for i, w1 in enumerate(ALL_TABOO_WORDS):
        for j, w2 in enumerate(ALL_TABOO_WORDS):
            if i != j:
                assert not p4b.word_matches_secret(w1, w2), f"False collision: {w1} vs {w2}"


def test_phase3_configs_and_launcher_dry_run() -> None:
    """Validate every key in configs/diff/*.yaml via nla.train_rl_vllm's real
    apply_config_defaults (--config <yf> --help), verify --no-save-critic-at-ckpts
    parses cleanly, and verify run_phase3_arm.sh --dry-run (incl. arm0c + --seed)."""
    cfg_dir = REPO / "configs" / "diff"
    yaml_files = sorted(cfg_dir.glob("*.yaml"))
    assert len(yaml_files) == 6, f"Expected 6 diff YAML configs, found {yaml_files}"
    for yf in yaml_files:
        res = subprocess.run(
            [sys.executable, "-m", "nla.train_rl_vllm", "--config", str(yf), "--help"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"{yf.name} failed train_rl_vllm config validation:\n{res.stderr}"

    # Verify --no-save-critic-at-ckpts (BooleanOptionalAction, Finding 5) parses cleanly
    res_no_crit = subprocess.run(
        [
            sys.executable,
            "-m",
            "nla.train_rl_vllm",
            "--config",
            str(cfg_dir / "rl_vllm_arm2_h200.yaml"),
            "--no-save-critic-at-ckpts",
            "--help",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_no_crit.returncode == 0, f"--no-save-critic-at-ckpts failed:\n{res_no_crit.stderr}"

    with tempfile.TemporaryDirectory() as td:
        p0_dir = Path(td) / "phase0"
        p0_dir.mkdir()
        (p0_dir / "phase0_config.json").write_text(
            json.dumps({
                "target_model": "Qwen/Qwen3-8B",
                "layer_index": 24,
                "tau_web_median": 9.875,
            }),
            encoding="utf-8",
        )
        for arm, hw, seed in (
            ("arm0", "h200", 0),
            ("arm0c", "h200", 1),
            ("arm1", "h200", 0),
            ("arm2", "h200", 0),
            ("arm2b", "h200", 0),
            ("arm2", "h100", 0),
        ):
            res = subprocess.run(
                [
                    "bash",
                    str(REPO / "scripts" / "diff" / "run_phase3_arm.sh"),
                    "--arm", arm,
                    "--phase0-dir", str(p0_dir),
                    "--corpus-dir", "/tmp/corpus",
                    "--ckpt-root", "/tmp/ckpts",
                    "--core-python", sys.executable,
                    "--vllm-python", "/opt/vllm/bin/python",
                    "--hardware", hw,
                    "--seed", str(seed),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            out = res.stdout
            if arm in ("arm2", "arm2b"):
                assert "--soft-norm-tau 9.875" in out
            else:
                assert "--soft-norm-tau 0.0" in out
            if arm == "arm0c":
                assert "rl_vllm_arm0c_h200.yaml" in out
                assert "--seed 1" in out
            if arm == "arm2b":
                assert "init_coldstart_ar.py" in out
                assert "--disable-av-thinking" in out
            if hw == "h100":
                assert "rl_vllm_h100_80gb.yaml" in out
                assert "--mode both" in out
                assert "--batch-size 16 --gradient-accumulation-steps 4" in out


def test_phase4_vllm_backend_mocked_steer_log() -> None:
    """Finding 2 regression test: prove `_generate_av_vllm` uses `read_reset_steer_log`
    via `llm.apply_model(...)` (no ImportError on nonexistent helper names) and builds
    valid 3-D SteeringVectors with Fix 3 soft-norm scaling."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    inj_char = "㈎"
    prompt_msgs = [
        {"role": "system", "content": "Explain the activation."},
        {"role": "user", "content": "Activation: <INJECT>."},
    ]
    from nla.utils import build_prompt_text

    ptxt = build_prompt_text(prompt_msgs, inj_char, tok, enable_thinking=False)
    pids = tok.encode(ptxt, add_special_tokens=False)
    inj_ids = tok.encode(inj_char, add_special_tokens=False)
    assert len(inj_ids) == 1
    inj_id = inj_ids[0]
    mpos = pids.index(inj_id)

    cfg = types.SimpleNamespace(
        injection_char=inj_char,
        injection_token_id=inj_id,
        injection_left_neighbor_id=pids[mpos - 1],
        injection_right_neighbor_id=pids[mpos + 1],
    )

    steer_state: dict[str, dict[str, Any]] = {}
    apply_calls: list[str] = []
    captured_svs: list[Any] = []

    class _FakeSteeringVector:
        def __init__(
            self,
            activations: torch.Tensor,
            layer_indices: list[int],
            scale: float,
            norm_match: bool,
            position_indices: list[int],
        ) -> None:
            self.activations = activations
            self.layer_indices = layer_indices
            self.scale = scale
            self.norm_match = norm_match
            self.position_indices = position_indices

    class _FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _FakeTokensPrompt:
        def __init__(self, prompt_token_ids: list[int]) -> None:
            self.prompt_token_ids = prompt_token_ids

    class _FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

        def apply_model(self, fn: Any) -> list[Any]:
            apply_calls.append(getattr(fn, "__name__", str(fn)))
            return [fn(None)]

        def generate(self, prompts: list[Any], params: list[Any]) -> list[Any]:
            outs = []
            for ri, (_p, sp) in enumerate(zip(prompts, params)):
                sv = sp.kwargs["extra_args"]["apply_steering_vectors"][0]
                captured_svs.append(sv)
                steer_state[f"_steer_{ri}"] = {
                    "covered": 1,
                    "applied": 1,
                    "positions": list(sv.position_indices),
                    "orphaned": 0,
                }
                outs.append(
                    types.SimpleNamespace(
                        outputs=[types.SimpleNamespace(text=f"<explanation>vllm_ok_{ri}</explanation>")]
                    )
                )
            return outs

    def _get_and_reset_steer_log() -> dict[str, Any]:
        snap = dict(steer_state)
        steer_state.clear()
        return snap

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    fake_vllm.TokensPrompt = _FakeTokensPrompt  # type: ignore[attr-defined]

    fake_vllm_lens = types.ModuleType("vllm_lens")
    fake_vllm_lens.SteeringVector = _FakeSteeringVector  # type: ignore[attr-defined]

    fake_worker_ext = types.ModuleType("vllm_lens._worker_ext")
    fake_worker_ext.get_and_reset_steer_log = _get_and_reset_steer_log  # type: ignore[attr-defined]

    acts = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    with (
        patch.dict(
            sys.modules,
            {
                "vllm": fake_vllm,
                "vllm_lens": fake_vllm_lens,
                "vllm_lens._worker_ext": fake_worker_ext,
            },
        ),
        patch.dict(os.environ, {"NLA_ALLOW_STALE_LENS": "1"}),
    ):
        texts = p4a._generate_av_vllm(
            base_ckpt="dummy_base",
            av_lora=None,
            av_ckpt_merged="dummy_merged",
            tokenizer=tok,
            cfg=cfg,
            prompt_msgs=prompt_msgs,
            activations=acts,
            soft_norm_tau=5.0,
            max_new_tokens=32,
            disable_av_thinking=True,
            tp_size=1,
            gpu_mem_util=0.80,
        )

    assert texts == ["<explanation>vllm_ok_0</explanation>", "<explanation>vllm_ok_1</explanation>"]
    assert apply_calls == ["read_reset_steer_log", "read_reset_steer_log"]
    assert len(captured_svs) == 2
    assert captured_svs[0].activations.shape == (1, 1, 4)
    assert captured_svs[0].position_indices == [mpos]
    assert captured_svs[0].norm_match is True
    assert np.isclose(captured_svs[0].scale, 5.0 / np.sqrt(50.0), atol=1e-5)
    # Exact-zero activation must safely disable norm_match with scale=0.0
    assert captured_svs[1].norm_match is False
    assert captured_svs[1].scale == 0.0


def test_phase4_survivorship_bias_and_mock_judge_label_sensitivity() -> None:
    """Findings 9, 12 & Smaller #6 regression test:
    - 90% failed extractions MUST NOT inflate Taboo Hit@3 / MATH / Hist / Binary metrics;
    - MockJudgeProvider MUST judge only the explanation body (flipped labels -> AUROC < 0.1);
    - Paired item-level bootstrap diffs MUST populate arm2_minus_arm0 & arm2_minus_arm1."""
    from nla.diff.benchmarks import HISTORICAL_ERAS

    with tempfile.TemporaryDirectory() as td:
        eval_dir = Path(td)

        def _make_records(mode: str) -> list[dict[str, Any]]:
            recs: list[dict[str, Any]] = []
            # 40 Refusal items (20 pos, 20 neg)
            for i in range(40):
                lbl = 1 if i < 20 else 0
                if mode == "good":
                    expl = (
                        "The assistant refuses the harmful request due to safety policy."
                        if lbl == 1
                        else "The assistant provides a normal helpful neutral answer."
                    )
                elif mode == "fail90":
                    # Only 2 pos + 2 neg survive (10%); survivors are 100% accurate
                    if i in (0, 1, 20, 21):
                        expl = (
                            "The assistant refuses the harmful request due to safety policy."
                            if lbl == 1
                            else "The assistant provides a normal helpful neutral answer."
                        )
                    else:
                        expl = ""
                elif mode == "flipped":
                    expl = (
                        "The assistant provides a normal helpful neutral answer."
                        if lbl == 1
                        else "The assistant refuses the harmful request due to safety policy."
                    )
                else:
                    raise ValueError(mode)
                recs.append({
                    "benchmark": "refusal",
                    "item_id": f"refusal_{i:04d}",
                    "label": lbl,
                    "source_text": f"prompt {i}",
                    "explanation": expl,
                })

            # 20 Taboo 50%-mix items
            for i in range(20):
                if mode == "good":
                    expl = "The model is concealing the secret word snow while answering."
                elif mode == "fail90":
                    expl = "The model is concealing the secret word snow." if i < 2 else ""
                else:
                    expl = "The model gives a generic weather description without any secret."
                diff_m = "raw_org" if mode == "flipped" else "org_minus_inst"
                recs.append({
                    "benchmark": "taboo",
                    "item_id": f"taboo_snow_50_asst_last_{diff_m}_{i:03d}",
                    "taboo_pair_key": f"snow_50_asst_last_{i:03d}",
                    "organism": "snow_50",
                    "secret_word": "snow",
                    "regime": "50_mix",
                    "position_kind": "asst_last",
                    "diff_mode": diff_m,
                    "source_text": f"taboo prompt {i}",
                    "explanation": expl,
                })

            # 12 Historical Period items (2 per era across all 6 eras)
            era_hints = {
                "ancient": "classical antiquity in ancient Rome",
                "medieval": "feudal medieval castle life",
                "early_modern": "renaissance early modern printing press",
                "nineteenth_century": "victorian nineteenth century steam railway",
                "mid_twentieth_century": "world war cold war mid-twentieth century",
                "contemporary": "modern smartphone internet contemporary era",
            }
            for i in range(12):
                era = HISTORICAL_ERAS[i % len(HISTORICAL_ERAS)]
                if mode == "fail90":
                    expl = f"Passage reflects {era_hints[era]}." if i < 2 else ""
                else:
                    expl = f"Passage reflects {era_hints[era]}."
                recs.append({
                    "benchmark": "historical_period",
                    "item_id": f"hist_{i:04d}",
                    "era": era,
                    "source_text": f"text {i}",
                    "explanation": expl,
                })

            # 10 MATH items
            for i in range(10):
                ans = str(40 + i)
                if mode == "fail90":
                    expl = f"The computed mathematical answer is {ans}." if i == 0 else ""
                else:
                    expl = f"The computed mathematical answer is {ans}."
                recs.append({
                    "benchmark": "math",
                    "item_id": f"math_{i:04d}",
                    "problem": f"Compute 40 + {i}",
                    "answer": ans,
                    "source_text": f"Compute 40 + {i}",
                    "explanation": expl,
                })
            return recs

        for arm, mode in (("arm2", "good"), ("arm1", "fail90"), ("arm0", "flipped")):
            recs = _make_records(mode)
            (eval_dir / f"{arm}_step0100_generations.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
            )
            (eval_dir / f"{arm}_step0100_local_metrics.json").write_text(
                json.dumps({"benchmark1_reconstruction": {}}), encoding="utf-8"
            )

        summary = p4b.run_judge_and_report(
            p4b.parse_args([
                "--eval-dir", str(eval_dir),
                "--provider", "mock",
                "--skip-text-judges",
                "--n-boot", "80",
            ])
        )

        m2 = summary["by_arm"]["arm2"]["100"]
        m1 = summary["by_arm"]["arm1"]["100"]
        m0 = summary["by_arm"]["arm0"]["100"]

        # Finding 9: Taboo Hit@3 must drop from 1.00 to 0.10 (NOT stay at 1.00!), with same n=20!
        assert np.isclose(m2["taboo"]["headline_50_mix_hit3"]["mean"], 1.0)
        assert np.isclose(m1["taboo"]["headline_50_mix_hit3"]["mean"], 0.10)
        assert m1["taboo"]["breakdown"]["50_mix/org_minus_inst/asst_last"]["n_prompts"] == 20
        assert np.isclose(m1["taboo"]["extraction_rate"], 0.10)

        # Finding 9: MATH & Historical Period count failures as misses (0)
        assert np.isclose(m2["math"]["pass_ci"]["mean"], 1.0)
        assert np.isclose(m1["math"]["pass_ci"]["mean"], 0.10)
        assert m1["math"]["n"] == 10
        assert m2["historical_period"]["balanced_acc"] > 0.95
        assert m1["historical_period"]["balanced_acc"] < 0.25
        assert m1["historical_period"]["n"] == 12

        # Finding 9 + Smaller #6: Refusal AUROC & label-flip sensitivity!
        assert m2["refusal"]["auroc_ci"]["auroc"] > 0.95
        assert m0["refusal"]["auroc_ci"]["auroc"] < 0.05  # flipped labels MUST invert AUROC!
        assert m1["refusal"]["extraction_rate"] == pytest.approx(0.10)
        assert m1["refusal"]["n_total"] == 40
        assert m1["refusal"]["balanced_acc"] <= 0.15
        assert m1["refusal"]["auroc_ci"]["auroc"] < 0.70

        # Finding 12: Paired item-level bootstrap diffs
        pd100 = summary["paired_diffs"]["100"]
        assert "arm2_minus_arm0" in pd100
        assert "arm2_minus_arm1" in pd100
        assert pd100["arm2_minus_arm0"]["refusal_auroc_diff"]["diff"] > 0.85
        assert pd100["arm2_minus_arm1"]["taboo_50_mix_headline_diff"]["diff"] == pytest.approx(0.90)
        assert pd100["arm2_minus_arm1"]["taboo_50_mix_headline_diff"]["ci_low"] > 0.60


def test_phase4_full_pipeline_cpu_end_to_end() -> None:
    """Run Phase 0 -> Phase 2 minimal corpus -> Phase 4a cache-benchmarks ->
    Phase 4a generate (HF real model + AR critic scoring) -> Phase 4b mock judge & report."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models = _make_tiny_suite_models(root)

        web_jsonl = root / "web.jsonl"
        web_jsonl.write_text(
            "\n".join(
                json.dumps({
                    "doc_id": f"web_{i:03d}",
                    "text": (
                        f"Passage {i}: Ocean currents, atmospheric pressure, and geological strata "
                        f"obey physical laws across basins in field study number {i}."
                    ),
                })
                for i in range(16)
            )
            + "\n",
            encoding="utf-8",
        )

        prompts_jsonl = root / "prompts.jsonl"
        prompts_jsonl.write_text(
            "\n".join(
                json.dumps({
                    "source_corpus": "synthetic",
                    "user_text": f"Explain briefly how engineering principle number {i} operates in practice.",
                    "history": [],
                })
                for i in range(14)
            )
            + "\n",
            encoding="utf-8",
        )

        taboo_json = root / "taboo_custom.json"
        taboo_json.write_text(
            json.dumps([
                {
                    "name": "snow_narrow",
                    "secret_word": "snow",
                    "regime": "narrow",
                    "repo_id": str(models["taboo_narrow"]),
                },
                {
                    "name": "smile_50_mix",
                    "secret_word": "smile",
                    "regime": "50_mix",
                    "repo_id": str(models["taboo_mix"]),
                },
            ]),
            encoding="utf-8",
        )

        p0_dir = root / "phase0"
        corpus_dir = root / "corpus"
        eval_dir = root / "eval"
        ar_dir = root / "ar_coldstart"

        # 1. Tiny Phase 0 (layer_index=1 -> AR num_layers=2)
        p0.run_phase0(
            p0.parse_args([
                "--base-model", str(models["base"]),
                "--target-model", str(models["inst"]),
                "--layer-index", "1",
                "--output-dir", str(p0_dir),
                "--web-custom-path", str(web_jsonl),
                "--n-web-train", "10",
                "--n-web-val", "6",
                "--web-max-tokens", "24",
                "--n-diag-pairs", "4",
                "--refusal-source", "builtin",
                "--n-neutral", "4",
                "--skip-taboo",
                "--batch-size", "4",
                "--device", "cpu",
                "--dtype", "fp32",
                "--no-fail-exit",
            ])
        )
        p0_cfg = json.loads((p0_dir / "phase0_config.json").read_text(encoding="utf-8"))

        # 2. Tiny Phase 2 (builds arm0/arm1/arm2 val.parquet + train role means)
        tmp_cache = root / "inj_cache.yaml"
        orig_cache = inj_mod._CACHE_PATH
        inj_mod._CACHE_PATH = tmp_cache
        try:
            p2.run_step_rollouts(
                p2.parse_args([
                    "--step", "rollouts",
                    "--phase0-dir", str(p0_dir),
                    "--output-dir", str(corpus_dir),
                    "--custom-prompts-jsonl", str(prompts_jsonl),
                    "--eval-blocklist-source", "builtin",
                    "--sft-frac", "0.22",
                    "--val-frac", "0.43",
                    "--backend", "hf",
                    "--max-prompt-tokens", "96",
                    "--rollout-max-new-tokens", "8",
                    "--rollout-temperature", "0.0",
                    "--batch-size", "6",
                    "--device", "cpu",
                    "--dtype", "fp32",
                ]),
                p0_cfg,
            )
            p2.run_step_extract(
                p2.parse_args([
                    "--step", "extract",
                    "--phase0-dir", str(p0_dir),
                    "--output-dir", str(corpus_dir),
                    "--sft-cont-tokens", "0",
                    "--chunk-convs", "14",
                    "--batch-size", "6",
                    "--device", "cpu",
                    "--dtype", "fp32",
                ]),
                p0_cfg,
            )
            p2.run_step_materialize(
                p2.parse_args([
                    "--step", "materialize",
                    "--phase0-dir", str(p0_dir),
                    "--output-dir", str(corpus_dir),
                    "--device", "cpu",
                    "--dtype", "fp32",
                ]),
                p0_cfg,
            )
        finally:
            inj_mod._CACHE_PATH = orig_cache

        # Smaller #3: _load_arm_role_means MUST refuse silent Phase 0 fallback by default
        empty_corpus = root / "empty_corpus"
        empty_corpus.mkdir()
        with pytest.raises(FileNotFoundError):
            p4a._load_arm_role_means(empty_corpus, p0_dir, 1, allow_phase0_fallback=False)
        rm_fb = p4a._load_arm_role_means(empty_corpus, p0_dir, 1, allow_phase0_fallback=True)
        assert "asst_last" in rm_fb

        # 3. Build tiny identity-init AR critic (num_layers = layer_index + 1 = 2)
        init_coldstart_ar.init_coldstart_ar(
            base_ckpt=str(models["inst"]),
            ar_num_layers=2,
            output_dir=ar_dir,
            dtype=torch.float32,
            sidecar=str(corpus_dir / "arm2" / "val.parquet"),
        )

        # 4. Phase 4a: cache-benchmarks (all 6 benchmarks, 2 Taboo organisms + behavior audit)
        p4a.main([
            "--step", "cache-benchmarks",
            "--phase0-dir", str(p0_dir),
            "--corpus-dir", str(corpus_dir),
            "--eval-dir", str(eval_dir),
            "--refusal-source", "builtin",
            "--n-refusal-pairs", "4",
            "--n-eval-awareness-pairs", "4",
            "--n-sycophancy-pairs", "4",
            "--custom-taboo-json", str(taboo_json),
            "--n-taboo-prompts", "3",
            "--n-mode-check-prompts", "2",
            "--n-historical", "6",
            "--math-source", "builtin",
            "--n-math", "4",
            "--behavior-audit-max-tokens", "6",
            "--batch-size", "4",
            "--device", "cpu",
            "--dtype", "fp32",
        ])
        assert (eval_dir / "benchmark_cache.pt").exists()
        bmeta = json.loads((eval_dir / "benchmark_meta.json").read_text(encoding="utf-8"))
        assert set(bmeta["taboo_mode_checks"].keys()) == {"snow_narrow", "smile_50_mix"}
        assert bmeta["counts"]["historical"] == 6
        assert bmeta["counts"]["math"] == 4
        assert "refusal" in bmeta["behavior_audit"]
        for od in bmeta["taboo_mode_checks"].values():
            assert np.isfinite(od["matched_kl_nats"])

        # Finding 15: verify Benchmark 6 cache is chat-wrapped at asst_last
        bcache = torch.load(str(eval_dir / "benchmark_cache.pt"), weights_only=False)
        assert bcache["historical"]["role"] == "asst_last"
        assert bcache["math"]["role"] == "asst_last"
        assert "h_base_win" in bcache["historical"] and "h_base_chatml" in bcache["historical"]
        assert "h_base_win" in bcache["math"] and "h_base_chatml" in bcache["math"]

        # 5. Phase 4a: REAL HF generation pass (tiny model + AV LoRA + Karvonen hook + Fix 3 tau)
        p4a.main([
            "--step", "generate",
            "--phase0-dir", str(p0_dir),
            "--corpus-dir", str(corpus_dir),
            "--eval-dir", str(eval_dir),
            "--arm", "arm2",
            "--step-num", "0",
            "--av-lora", str(models["av_lora"]),
            "--ar-ckpt", str(ar_dir),
            "--n-val-rows", "6",
            "--max-new-tokens", "8",
            "--backend", "hf",
            "--batch-size", "4",
            "--device", "cpu",
            "--dtype", "fp32",
        ])
        assert (eval_dir / "arm2_step0000_generations.jsonl").exists()
        lm0 = json.loads((eval_dir / "arm2_step0000_local_metrics.json").read_text(encoding="utf-8"))
        assert lm0["soft_norm_tau"] > 0.0
        assert "snow_narrow/asst_last/org_minus_inst" in lm0["taboo_diff_norm_diagnostics"]
        assert "smile_50_mix/user_mean/org_minus_base" in lm0["taboo_diff_norm_diagnostics"]

        # 6. Exercise arm0@0, arm0@100, arm1@100, arm2@100 with well-formed `<explanation>`
        #    responses so AR critic scoring (FVE, weighted FVE, within-role FVE/cos, tok1 FVE)
        #    AND every Phase 4b judge/metric/plot branch are thoroughly tested!
        def _fake_generate_av_hf(
            base_ckpt: str,
            av_lora: str | None,
            av_ckpt_merged: str | None,
            tokenizer: Any,
            cfg: Any,
            prompt_msgs: list[dict[str, str]],
            activations: torch.Tensor,
            **kwargs: Any,
        ) -> list[str]:
            n = activations.shape[0]
            return [
                (
                    "<explanation>The assistant shifts toward a safety refusal boundary "
                    f"and thinks of snow and smile (vec {i}).</explanation>"
                )
                for i in range(n)
            ]

        with patch.object(p4a, "_generate_av_hf", side_effect=_fake_generate_av_hf):
            for arm, step_n in (("arm0", 0), ("arm0", 100), ("arm1", 100), ("arm2", 100)):
                p4a.main([
                    "--step", "generate",
                    "--phase0-dir", str(p0_dir),
                    "--corpus-dir", str(corpus_dir),
                    "--eval-dir", str(eval_dir),
                    "--arm", arm,
                    "--step-num", str(step_n),
                    "--ar-ckpt", str(ar_dir),
                    "--n-val-rows", "12",
                    "--backend", "hf",
                    "--batch-size", "4",
                    "--device", "cpu",
                    "--dtype", "fp32",
                ])

        lm_a2_100 = json.loads((eval_dir / "arm2_step0100_local_metrics.json").read_text(encoding="utf-8"))
        ar_m = lm_a2_100["benchmark1_reconstruction"]["ar_reconstruction"]
        assert ar_m["n_scored"] > 0
        assert np.isfinite(ar_m["fve_pct"])
        assert np.isfinite(ar_m["weighted_fve_pct"])
        assert np.isfinite(ar_m["cos_mean"])
        assert np.isfinite(ar_m["opt_scale_fve_pct"])
        # Finding 10: cross-arm comparable within-role reconstruction metrics
        assert np.isfinite(ar_m["within_role_fve_pct"])
        assert np.isfinite(ar_m["within_role_cos_mean"])

        # 7. Phase 4b: Judge & Report (--provider mock), twice to verify judge_cache.jsonl reuse!
        summary1 = p4b.run_judge_and_report(
            p4b.parse_args([
                "--eval-dir", str(eval_dir),
                "--provider", "mock",
                "--n-text-judges", "6",
                "--n-boot", "50",
            ])
        )
        report_dir = eval_dir / "report"
        assert (report_dir / "final_summary.json").exists()
        assert (report_dir / "final_report.md").exists()
        assert (report_dir / "plots" / "trajectory_grid.png").exists()
        assert (report_dir / "qualitative_examples.md").exists()
        cache_lines_1 = len((report_dir / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines())
        assert cache_lines_1 > 0

        # Check all 6 benchmark metrics are populated & finite for arm2@100
        m2 = summary1["by_arm"]["arm2"]["100"]
        assert np.isfinite(m2["refusal"]["auroc_ci"]["auroc"])
        assert np.isfinite(m2["eval_awareness"]["auroc_ci"]["auroc"])
        assert np.isfinite(m2["sycophancy"]["auroc_ci"]["auroc"])
        assert np.isfinite(m2["taboo"]["headline_50_mix_hit3"]["mean"])
        assert "50_mix/org_minus_inst/asst_last" in m2["taboo"]["breakdown"]
        assert "narrow/org_minus_base/user_mean" in m2["taboo"]["breakdown"]
        assert np.isfinite(m2["taboo"]["breakdown"]["50_mix/org_minus_inst/asst_last"]["off_diag_fp_rate"])
        assert np.isfinite(m2["historical_period"]["balanced_acc"])
        assert np.isfinite(m2["math"]["pass_ci"]["mean"])
        # Smaller #4: all 5 real RUBRIC_PROMPTS dimensions present
        tj2 = m2["reconstruction"]["text_judges"]
        for k in ("unique_info_mean", "coherence_mean", "writing_quality_mean", "specificity_mean", "repetitiveness_mean"):
            assert np.isfinite(tj2[k]), f"Expected finite text_judges[{k}]"
        # Finding 12: paired bootstrap comparisons populated
        assert "arm2_minus_arm0" in summary1["paired_diffs"]["100"]
        assert "arm2_minus_arm1" in summary1["paired_diffs"]["100"]

        # Check final_report.md contains all 4 sections + fixed columns
        md_txt = (report_dir / "final_report.md").read_text(encoding="utf-8")
        assert "## 1. Headline Trajectory Table" in md_txt
        assert "WithinRole FVE%" in md_txt
        assert "## 2. Benchmark 1" in md_txt
        assert "Unique Info" in md_txt
        assert "## 3. Benchmark 5" in md_txt
        assert "## 4. Paired Arm Comparisons" in md_txt

        # Second run must hit 100% cache (zero new lines appended to judge_cache.jsonl!)
        p4b.run_judge_and_report(
            p4b.parse_args([
                "--eval-dir", str(eval_dir),
                "--provider", "mock",
                "--n-text-judges", "6",
                "--n-boot", "50",
            ])
        )
        cache_lines_2 = len((report_dir / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines())
        assert cache_lines_2 == cache_lines_1
