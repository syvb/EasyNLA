"""Run an unmodified repo script with model IO swapped for the cached split loader.

    python run_patched.py scripts/diff/phase0_geometry_gate.py --output-dir ... [script args]

Only `load_causal_lm`, `extract_at_positions` and `unload_model` are replaced in
the script's namespace; every metric, gate and threshold is the script's own.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    argv = sys.argv[2:]
    spec = importlib.util.spec_from_file_location(f"patched_{script.stem}", str(script))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if os.environ.get("NLA_PATCH_MODE", "cached") == "jit_generate":
        # Whole-model generation (Phase 2 rollouts): bf16-resident weights, fp32 arithmetic.
        from jit_full import jit_load_causal_lm, length_sorted_generate

        mod.load_causal_lm = jit_load_causal_lm
        if hasattr(mod, "_generate_rollouts_hf"):
            mod._generate_rollouts_hf = length_sorted_generate(mod._generate_rollouts_hf)
    else:
        for name, repl in (
            ("load_causal_lm", cached_io.lazy_load_causal_lm),
            ("extract_at_positions", cached_io.cached_extract),
            ("unload_model", cached_io.lazy_unload_model),
        ):
            if hasattr(mod, name):
                setattr(mod, name, repl)
        # Phase 0 attaches Taboo LoRAs with PeftModel.from_pretrained(target_model, lora_id).
        import peft

        peft.PeftModel.from_pretrained = staticmethod(cached_io.lazy_peft_from_pretrained)  # type: ignore[method-assign]
        # Record mode returns random activations, so stop before the script analyses them.
        stop_at = os.environ.get("NLA_RECORD_STOP_AT", "")
        if cached_io.RECORD_PATH and stop_at:
            assert hasattr(mod, stop_at), f"{script.name} has no attribute {stop_at}"

            def _stop(*_a: object, **_k: object) -> None:
                print(f"[record] reached {stop_at}; every extraction request is recorded", flush=True)
                raise SystemExit(0)

            setattr(mod, stop_at, _stop)
        if os.environ.get("NLA_LEAN_ALIGN", "1") == "1":
            # Chunked covariance accumulation (same maths) so the d=4096 CV fit fits in RAM.
            from nla.diff import alignment

            alignment._centered_covariances = cached_io.lean_centered_covariances
    mod.main(argv)


if __name__ == "__main__":
    main()
