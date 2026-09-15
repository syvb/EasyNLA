"""GRPO for the future-lens decoder (single GPU, HF generate, no reconstructor).

Fork of nla/train_rl_self_contained.py with the AR critic removed and the reward
replaced by a future-token reward:

  --reward exact_match  r = (1/K) sum_j 1[y_j == x_{t+1+j}]  (length violation -0.1)
  --reward target_logp  r = [sum_j log p_target(y_j | true prefix) - c * (#missing slots)] / K
                        (frozen target = same weights, adapters disabled, no injection;
                        prefix from docs.parquet; c = --missing-token-penalty so that
                        stopping early is never rewarded)

Kept from EasyNLA: the policy LoRA is the SFT adapter continued in place, with a
frozen copy loaded as adapter "reference" for the KL term (the --av-adapter
warm-start; a fresh adapter costs ~12pp in the README's measurements); k3/dist KL
estimators; the micro-batched on-policy update (`grpo_update_microbatched`);
marker well-formedness masking; sidecar snapshot + overwrite guard; resume with
Adam state.

Changed: all B x G rollouts run in ONE batched generate() (readouts are <= 12
tokens, so rollout time is dominated by per-call overhead); advantages default to
Dr. GRPO: group-mean only (--adv-norm none) AND a constant per-sequence normaliser
(--seq-agg const scales each sample's token-mean surrogate by n_resp / max_new_tokens,
i.e. sum over tokens / constant; the KL term stays a per-token mean); per-step logs
include the within-group reward std (must stay > 0 or there is no gradient); the
periodic eval is the controlled future-lens eval (real vs shuffled precision@1 per
layer/N) on a seeded random subsample of eval rows per layer.

    python -m nla.future_lens.train_rl --config configs/future_lens/rl.yaml \
        --base-ckpt Qwen/Qwen3-8B --av-ckpt <sft>/iter_XXXXXXX \
        --parquet <data>/train.parquet --eval-parquet <data>/eval.parquet \
        --save-dir <ckpts>/rl_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from nla.config import load_nla_config
from nla.future_lens.data import (
    encode_prompt, load_docs, load_fl_meta, load_fl_rows, TOPK_COLUMNS, resolve_docs_path,
)
from nla.future_lens.eval import evaluate, stop_ids, write_records
from nla.future_lens.inject import INJECTION_MODES, AffineInjector, prepare_vectors, register_injection
from nla.future_lens.rewards import (
    REWARD_KINDS, exact_match_reward, group_advantages, per_offset_hits, target_logp_sum_reward,
    target_token_logps, truncate_readout,
)
from nla.injection import marker_well_formed
from nla.schema import sidecar_path_for
from nla.train_rl_self_contained import grpo_update_microbatched
from nla.train_sft import _resolve_device_map
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config


# ----------------------------------------------------------------------------
# Rollouts
# ----------------------------------------------------------------------------

@torch.no_grad()
def rollout_batch(actor, tokenizer, jobs: list[dict], *, inject_char, vectors_ref, injection_mode,
                  group_size, max_new_tokens, temperature, device, eos_ids, gen_batch: int = 0):
    """Sample `group_size` readouts for every job in ONE left-padded generate() call
    (or a few, if gen_batch caps the rows per call). Returns a flat list of samples
    {prompt_ids, resp_ids, full_ids, prompt_len, group, job} in group-major order."""
    was_training = actor.training
    actor.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    enc = [encode_prompt(tokenizer, j["prompt"], inject_char) for j in jobs]
    flat = [(gi, enc[gi]) for gi in range(len(jobs)) for _ in range(group_size)]
    samples = [None] * len(flat)   # type: ignore[list-item]
    step = gen_batch if gen_batch > 0 else len(flat)
    for cs in range(0, len(flat), step):
        chunk = flat[cs: cs + step]
        L = max(len(e) for _, e in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, (_, e) in enumerate(chunk):
            ids[r, L - len(e):] = torch.tensor(e, dtype=torch.long)
            attn[r, L - len(e):] = 1
        raw = torch.tensor(np.stack([np.asarray(jobs[gi]["vector"], dtype=np.float32) for gi, _ in chunk]))
        alphas = torch.tensor([float(jobs[gi]["alpha"]) for gi, _ in chunk])
        vectors_ref[0] = prepare_vectors(raw, alphas, injection_mode).to(device)
        try:
            gen = actor.generate(
                input_ids=ids.to(device), attention_mask=attn.to(device),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_p=1.0, top_k=0, repetition_penalty=1.0,
                pad_token_id=pad_id, return_dict_in_generate=True,
            )
        finally:
            vectors_ref[0] = None
        seqs = gen.sequences
        for r, (gi, e) in enumerate(chunk):
            resp = seqs[r, L:].tolist()
            n_real = next((i + 1 for i, t in enumerate(resp) if t in eos_ids), len(resp))
            resp = resp[:n_real]
            samples[cs + r] = {
                "prompt_ids": e, "resp_ids": resp,
                "full_ids": torch.tensor(e + resp, dtype=torch.long),
                "prompt_len": len(e), "group": gi, "job": jobs[gi],
            }
    if was_training:
        actor.train()
    return samples


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--av-ckpt", required=True, help="future-lens SFT LoRA dir (iter_*): policy init + KL reference")
    p.add_argument("--parquet", required=True, help="train.parquet")
    p.add_argument("--sidecar", default=None)
    p.add_argument("--eval-parquet", default=None, help="eval.parquet (doc-disjoint) for the periodic controlled eval")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="none",
                   help="bf16 base by default: the frozen target for target_logp/surprisal must be the model the activations came from.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--injection", choices=INJECTION_MODES, default=None, help="default: read from <av-ckpt>/future_lens.json")
    p.add_argument("--alpha-mult", type=float, default=None, help="default: read from <av-ckpt>/future_lens.json")
    p.add_argument("--label", choices=("text", "greedy"), default=None,
                   help="readout label (see data.py); default: read from <av-ckpt>/future_lens.json")
    p.add_argument("--layers", default=None, help="comma list; default all layers in the parquet")
    p.add_argument("--max-rows", type=int, default=None)
    # --- GRPO ---
    p.add_argument("--reward", choices=REWARD_KINDS, default="exact_match")
    p.add_argument("--length-penalty", type=float, default=0.1, help="subtracted when the readout length != K")
    p.add_argument("--fail-reward", type=float, default=None,
                   help="reward for an unscorable rollout (empty readout). default: -length_penalty (exact_match) / -10 (target_logp)")
    p.add_argument("--target-ctx", type=int, default=1024, help="target_logp: prefix tokens fed to the frozen target")
    p.add_argument("--missing-token-penalty", type=float, default=4.0,
                   help="target_logp: nats charged per readout slot left empty (early EOS)")
    p.add_argument("--target-micro-batch", type=int, default=32, help="target_logp: sequences per frozen-target forward")
    p.add_argument("--num-steps", type=int, default=4000)
    p.add_argument("--batch-prompts", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=0, help="rows per generate() call; 0 = all B*G at once")
    p.add_argument("--adv-norm", choices=["none", "std"], default="none",
                   help="none = Dr. GRPO (group mean only; no length/variance bias). std = classic GRPO.")
    p.add_argument("--seq-agg", choices=["const", "token_mean"], default="const",
                   help="const = Dr. GRPO (sum of token terms / max_new_tokens); token_mean = per-sample mean.")
    p.add_argument("--kl-beta", type=float, default=0.03)
    p.add_argument("--kl-estimator", choices=["k3", "dist"], default="k3")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logp-micro-batch", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    # --- eval / save / resume ---
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-rows", type=int, default=200, help="eval rows PER LAYER")
    p.add_argument("--eval-ks", default=None, help="default: sidecar k_choices")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume-from-lora", default=None)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="rl-future-lens")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="rl")
    p.add_argument("--wandb-tags", default=None)
    p.add_argument("--no-wandb", action="store_true")
    apply_config_defaults(p)
    args = p.parse_args(argv)

    # ---- fail-fast ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(save_dir.glob("iter_*"))
    if existing and args.resume_from_lora is None:
        raise SystemExit(f"[save] {save_dir} already has {len(existing)} iter_* checkpoints "
                         f"(latest {existing[-1].name}); resume with --resume-from-lora or use a fresh --save-dir")
    if args.sidecar is None:
        args.sidecar = args.parquet
    side_src = sidecar_path_for(args.sidecar)
    side_dst = save_dir / "nla_meta.yaml"
    if side_dst.exists():
        prev, cur = yaml.safe_load(side_dst.read_text()), yaml.safe_load(side_src.read_text())
        for k in ("tokens", "extraction", "future_lens"):
            assert prev.get(k) == cur.get(k), f"save-dir sidecar snapshot disagrees with --sidecar on {k!r}"
    else:
        shutil.copy2(side_src, side_dst)
    assert args.temperature == 1.0, "on-policy update scores at T=1; keep --temperature 1.0"
    if args.fail_reward is None:
        args.fail_reward = -args.length_penalty if args.reward == "exact_match" else -10.0

    # injection settings travel with the SFT checkpoint
    fl_json = Path(args.av_ckpt) / "future_lens.json"
    ck = json.loads(fl_json.read_text()) if fl_json.exists() else {}
    if args.injection is None:
        args.injection = ck.get("injection", "replace_embed")
    if args.alpha_mult is None:
        args.alpha_mult = float(ck.get("alpha_mult", 1.0))
    if args.label is None:
        args.label = ck.get("label", "text")
    print(f"[cfg] label={args.label} injection={args.injection} alpha_mult={args.alpha_mult} reward={args.reward} "
          f"adv_norm={args.adv_norm} kl_beta={args.kl_beta}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    on_cuda = device.startswith("cuda")
    dtype = torch.bfloat16 if on_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    fl = load_fl_meta(args.sidecar)
    inj_id, left_id, right_id = (cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                 cfg.injection_right_neighbor_id)

    # ---- actor: base + policy adapter ("default") + frozen SFT copy ("reference") ----
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_use_double_quant=True)
    dmap, max_mem = _resolve_device_map("single", 0, quant_config)
    base = AutoModelForCausalLM.from_pretrained(args.base_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
                                                quantization_config=quant_config, device_map=dmap, max_memory=max_mem)
    if dmap is None:
        base = base.to(device)
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=args.gradient_checkpointing)
    policy_ckpt = args.resume_from_lora or args.av_ckpt
    actor = PeftModel.from_pretrained(base, policy_ckpt, adapter_name="default", is_trainable=True)
    actor.load_adapter(args.av_ckpt, adapter_name="reference")
    actor.set_adapter("default")
    for n_, p_ in actor.named_parameters():
        if ".reference." in n_:
            p_.requires_grad_(False)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        actor.enable_input_require_grads()

    affine = None
    aff_path = Path(policy_ckpt) / "affine.pt"
    if aff_path.exists():
        affine = AffineInjector(cfg.d_model).to(device)
        affine.load_state_dict(torch.load(aff_path, map_location="cpu"))
        print(f"[actor] affine injector loaded from {aff_path}")
    vectors_ref = [None]
    register_injection(actor, args.injection, vectors_ref, inj_id, left_id, right_id, affine)
    eos_ids = stop_ids(tokenizer, actor)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ---- data ----
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    rows = load_fl_rows(args.parquet, n_max=args.max_rows, layers=layers, label=args.label, drop_label_ids=eos_ids)
    for r in rows:
        r["inject_alpha"] = fl.alpha(int(r["activation_layer"]), args.alpha_mult)
    print(f"[data] {len(rows)} train rows, layers={sorted({int(r['activation_layer']) for r in rows})}", flush=True)
    docs = None
    if args.reward == "target_logp":
        dp = resolve_docs_path(args.parquet, fl)
        docs = load_docs(dp) if dp else None
        assert docs is not None, "target_logp reward needs docs.parquet next to the parquet"
    eval_rows = []
    if args.eval_parquet and args.eval_every > 0:
        # seeded random subsample PER LAYER (the parquet is doc-major, so "first N rows"
        # would be a handful of documents, the same ones at every layer)
        all_eval = load_fl_rows(args.eval_parquet, layers=layers, label=args.label, drop_label_ids=eos_ids,
                                columns=["prompt", "activation_vector", "activation_layer", "target_ids",
                                         "target_top5", "k", "doc_idx", "t", "p_top1"]
                                + (TOPK_COLUMNS if (args.label == "greedy" and fl.topk) else []))   # tf_kl in the periodic eval
        by_layer: dict[int, list[dict]] = defaultdict(list)
        for r in all_eval:
            by_layer[int(r["activation_layer"])].append(r)
        _erng = np.random.default_rng(1234 + args.seed)
        for l, lr in sorted(by_layer.items()):
            pick = _erng.choice(len(lr), size=min(args.eval_rows, len(lr)), replace=False)
            eval_rows.extend(lr[i] for i in sorted(pick))
        del all_eval
        print(f"[eval] {len(eval_rows)} eval rows "
              f"({ {l: min(args.eval_rows, len(v)) for l, v in by_layer.items()} }, "
              f"{len({int(r['doc_idx']) for r in eval_rows})} docs)", flush=True)
    eval_ks = [int(x) for x in args.eval_ks.split(",")] if args.eval_ks else fl.k_choices

    # ---- optimizer ----
    try:
        import bitsandbytes as bnb
        adam_cls = bnb.optim.AdamW8bit
    except ImportError:
        adam_cls = torch.optim.AdamW
    trainable = [p_ for p_ in actor.parameters() if p_.requires_grad]
    if affine is not None:
        trainable += list(affine.parameters())
    optim = adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if args.resume_from_lora is not None:
        from nla.utils.resume import find_optim_ckpt, warn_cold_adam
        oc = find_optim_ckpt(args.save_dir, args.resume_from_lora)
        if oc is not None:
            st = torch.load(str(oc), map_location="cpu", weights_only=True)
            _m = re.search(r"iter_(\d+)", str(args.resume_from_lora))
            if _m and int(_m.group(1)) != int(st.get("step", -1)):
                print(f"[resume] WARN: resuming weights from step {int(_m.group(1))} but optim_latest.pt "
                      f"is from step {st.get('step')} — Adam moments will not match these weights", flush=True)
            if args.start_step == 0 and int(st.get("step", 0)) > 0:
                args.start_step = int(st["step"])
            try:
                optim.load_state_dict(st["actor_optim"])
                print(f"[resume] optimizer state restored from {oc} (step {st.get('step')})")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"[resume] WARN optimizer state incompatible ({e}); Adam restarts")
        else:
            warn_cold_adam(args.start_step)
    save_resolved_config(args, args.save_dir)

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group,
                   tags=(args.wandb_tags.split(",") if args.wandb_tags else []) + ["future-lens"],
                   config=vars(args))

    rng = np.random.default_rng(args.seed)
    pending = list(range(len(rows)))
    rng.shuffle(pending)
    cursor = 0
    evals_path = save_dir / "evals.jsonl"

    def run_eval(step):
        recs = evaluate(actor, tokenizer, eval_rows, fl, cfg, injection_mode=args.injection,
                        vectors_ref=vectors_ref, device=device, conditions=["real", "shuffled"],
                        ks=eval_ks, layers=layers, alpha_mult=args.alpha_mult, docs=None,
                        batch_size=args.eval_batch, seed=args.seed, eos_ids=eos_ids, verbose=False,
                        tag={"step": step, "checkpoint": f"rl_step{step}", "reward": args.reward})
        write_records(recs, evals_path)
        log = {}
        # every metric the eval produces: p1 (free-running, PRIMARY), tf_p1 (teacher-forced,
        # confirmatory), tf_kl (reported-only: expected to worsen under a mode-seeking reward),
        # p5, exact, len_ok — per (condition, layer, N), plus real-shuffled gaps and pooled means
        for metric in ("p1", "tf_p1", "tf_kl", "p5", "exact", "len_ok"):
            vals = {(r["condition"], r["layer"], r["N"]): r["value"] for r in recs if r["metric"] == metric}
            if not vals:
                continue
            for (cond, layer, N), v in vals.items():
                log[f"eval/{metric}_{cond}/L{layer}_N{N}"] = v
                if cond == "real" and ("shuffled", layer, N) in vals:
                    log[f"eval/{metric}_gap/L{layer}_N{N}"] = v - vals[("shuffled", layer, N)]
            for N in sorted({k[2] for k in vals}):
                for cond in ("real", "shuffled"):
                    vv = [v for (c, l, n), v in vals.items() if c == cond and n == N]
                    if vv:
                        log[f"eval/{metric}_{cond}/N{N}_mean"] = float(np.mean(vv))
                gg = [v for (c, l, n), v in vals.items() if c == "real" and n == N and ("shuffled", l, n) in vals]
                if gg:
                    log[f"eval/{metric}_gap/N{N}_mean"] = float(np.mean(gg)) - float(np.mean(
                        [vals[("shuffled", l, n)] for (c, l, n) in vals if c == "real" and n == N and ("shuffled", l, n) in vals]))
        # headline = the pre-declared success cell: free-running gap pooled over layers at N=2,3
        hl = [log[k] for k in ("eval/p1_gap/N2_mean", "eval/p1_gap/N3_mean") if k in log]
        if hl:
            log["eval/headline_p1_gap_N2_3"] = float(np.mean(hl))
        hl = [log[k] for k in ("eval/tf_p1_gap/N2_mean", "eval/tf_p1_gap/N3_mean") if k in log]
        if hl:
            log["eval/headline_tf_p1_gap_N2_3"] = float(np.mean(hl))
        actor.train()
        return log

    # ---- training loop ----
    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        if cursor + args.batch_prompts > len(pending):
            rng.shuffle(pending); cursor = 0
        batch = [rows[i] for i in pending[cursor: cursor + args.batch_prompts]]
        cursor += args.batch_prompts
        jobs = [{"prompt": r["prompt"], "k": int(r["k"]), "vector": r["activation_vector"],
                 "alpha": r["inject_alpha"], "row": r} for r in batch]

        samples = rollout_batch(actor, tokenizer, jobs, inject_char=cfg.injection_char, vectors_ref=vectors_ref,
                                injection_mode=args.injection, group_size=args.group_size,
                                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                device=device, eos_ids=eos_ids, gen_batch=args.gen_batch)
        t_gen = time.time() - t0
        n = len(samples)

        # ---- rewards ----
        # the update forward scans prompt+response: a rollout that emits the marker/neighbours
        # must be masked here, not crash there
        marker_ok = [marker_well_formed(s["prompt_ids"] + s["resp_ids"], inj_id, left_id, right_id) for s in samples]
        readouts, viols = [], []
        for s in samples:
            ro, viol = truncate_readout(s["resp_ids"], s["job"]["k"], eos_ids)
            readouts.append(ro); viols.append(viol)
        if args.reward == "exact_match":
            rewards = [exact_match_reward(s["resp_ids"], s["job"]["row"]["target_ids"], s["job"]["k"],
                                          length_penalty=args.length_penalty, eos_ids=eos_ids)[0]
                       for s in samples]
        else:
            prefixes = [docs[int(s["job"]["row"]["doc_idx"])][: int(s["job"]["row"]["t"]) + 1] for s in samples]
            tlp = target_token_logps(actor, prefixes, readouts, device, pad_id=pad_id,
                                     micro_batch=args.target_micro_batch, max_prefix=args.target_ctx)
            rewards = [target_logp_sum_reward(t, s["job"]["k"], missing_token_penalty=args.missing_token_penalty,
                                              length_violation=viol, length_penalty=args.length_penalty)
                       for t, s, viol in zip(tlp, samples, viols)]
        # exact-match statistics are logged for both reward kinds
        em = [exact_match_reward(s["resp_ids"], s["job"]["row"]["target_ids"], s["job"]["k"],
                                 length_penalty=0.0, eos_ids=eos_ids)[0] for s in samples]
        hits_by_off: dict[int, list[int]] = defaultdict(list)
        for s in samples:
            for j, h in enumerate(per_offset_hits(s["resp_ids"], s["job"]["row"]["target_ids"], s["job"]["k"], eos_ids)):
                hits_by_off[j].append(h)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        groups_t = torch.tensor([s["group"] for s in samples], dtype=torch.long, device=device)
        ok_t = torch.tensor(marker_ok, dtype=torch.bool, device=device)
        adv = group_advantages(rewards_t, groups_t, ok_t, n_groups=len(jobs), norm=args.adv_norm)
        group_std = [float(rewards_t[groups_t == g].std()) for g in range(len(jobs))
                     if int((groups_t == g).sum()) > 1]
        n_zero_var = sum(1 for sd in group_std if sd < 1e-6)

        # ---- update (skip samples with zero advantage AND no KL need? no: KL applies to all) ----
        full_ids = [s["full_ids"] for s in samples]
        prompt_lens = [s["prompt_len"] for s in samples]
        raw = torch.tensor(np.stack([np.asarray(s["job"]["vector"], dtype=np.float32) for s in samples]))
        alphas = torch.tensor([float(s["job"]["alpha"]) for s in samples])
        prepared = prepare_vectors(raw, alphas, args.injection)
        acts = [prepared[i] for i in range(n)]
        keep = [i for i in range(n) if marker_ok[i]]
        adv_eff = adv
        if args.seq_agg == "const":
            # grpo_token_loss averages over the sample's tokens; scaling the advantage by
            # n_resp / max_new_tokens turns that into sum(tokens) / constant (Dr. GRPO).
            n_resp_t = torch.tensor([len(s["resp_ids"]) for s in samples], dtype=torch.float32, device=device)
            adv_eff = adv * n_resp_t / float(args.max_new_tokens)
        loss, grad_norm, m = grpo_update_microbatched(
            actor, optim, tokenizer, [full_ids[i] for i in keep], [prompt_lens[i] for i in keep],
            [acts[i] for i in keep], adv_eff[keep], vectors_ref, device,
            micro_batch=args.logp_micro_batch, kl_beta=args.kl_beta, max_grad_norm=args.max_grad_norm,
            kl_estimator=args.kl_estimator, n_total=n,
        )

        # ---- logging ----
        lens = [len(s["resp_ids"]) for s in samples]
        log = {
            "reward/mean": float(np.mean(rewards)), "reward/std": float(np.std(rewards)),
            "reward/group_std_mean": float(np.mean(group_std)) if group_std else 0.0,
            "reward/frac_zero_var_groups": n_zero_var / max(1, len(jobs)),
            "reward/exact_match_mean": float(np.mean(em)),
            "av/kl_to_ref": m.get("kl_mean", 0.0), "av/entropy": m.get("entropy", 0.0),
            "av/grad_norm": grad_norm, "av/loss": loss,
            "av/advantage_abs_mean": float(adv.abs().mean()),
            "av/resp_len": float(np.mean(lens)), "av/len_violation_frac": float(np.mean(viols)),
            "av/marker_bad_count": int(n - len(keep)),
            "rollout/gen_s": t_gen, "wall_s": time.time() - t0,
        }
        for j, hs in sorted(hits_by_off.items()):
            log[f"train/p1_off{j}"] = float(np.mean(hs))
        by_k: dict[int, list[float]] = defaultdict(list)
        for s, r in zip(samples, rewards):
            by_k[s["job"]["k"]].append(r)
        for k, v in by_k.items():
            log[f"reward/k{k}"] = float(np.mean(v))
        # per-layer reward and sampled hit rate at offset 1 (which layers RL is improving)
        by_layer: dict[int, list[float]] = defaultdict(list)
        hit1_by_layer: dict[int, list[int]] = defaultdict(list)
        for s, r in zip(samples, rewards):
            l = int(s["job"]["row"]["activation_layer"])
            by_layer[l].append(r)
            if s["job"]["k"] > 1:
                hit1_by_layer[l].append(per_offset_hits(s["resp_ids"], s["job"]["row"]["target_ids"], s["job"]["k"], eos_ids)[1])
        for l, v in by_layer.items():
            log[f"reward/L{l}"] = float(np.mean(v))
        for l, v in hit1_by_layer.items():
            if v:
                log[f"train/p1_off1_L{l}"] = float(np.mean(v))
        log["train/lr"] = float(optim.param_groups[0]["lr"])
        log["rollout/tokens_per_s"] = float(sum(lens)) / max(t_gen, 1e-6)
        offs = " ".join(f"p1@{j}={log[f'train/p1_off{j}']:.2f}" for j in sorted(hits_by_off))
        print(f"step {step:05d} | r {log['reward/mean']:.3f} (gstd {log['reward/group_std_mean']:.3f}, "
              f"zero-var {n_zero_var}/{len(jobs)}) | em {log['reward/exact_match_mean']:.3f} | "
              f"kl {log['av/kl_to_ref']:.4f} | ent {log['av/entropy']:.2f} | len {log['av/resp_len']:.1f} "
              f"viol {log['av/len_violation_frac']:.0%} | {offs} | gen {t_gen:.1f}s t {log['wall_s']:.1f}s", flush=True)
        if n_zero_var == len(jobs):
            print("[WARN] every group has zero reward variance this step — no policy gradient. "
                  "Consider --reward target_logp or more diverse K.", flush=True)

        if eval_rows and args.eval_every > 0 and (step % args.eval_every == 0 or step + 1 == args.num_steps):
            te = time.time()
            log.update(run_eval(step))
            log["time/eval_s"] = time.time() - te
            gaps = {k: v for k, v in log.items() if k.startswith("eval/p1_gap/")}
            print(f"  [eval@{step}] " + " ".join(f"{k.split('/')[-1]}={v:+.3f}" for k, v in sorted(gaps.items())[:12])
                  + f" ({log['time/eval_s']:.0f}s)", flush=True)

        if not args.no_wandb:
            wandb.log(log, step=step)

        if (step + 1) % args.save_every == 0 or step + 1 == args.num_steps:
            out = save_dir / f"iter_{step + 1:06d}"
            out.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(out))
            if affine is not None:
                torch.save(affine.state_dict(), str(out / "affine.pt"))
            (out / "future_lens.json").write_text(json.dumps({
                "injection": args.injection, "alpha_mult": args.alpha_mult, "affine": affine is not None,
                "label": args.label, "reward": args.reward, "step": step + 1, "av_ckpt": args.av_ckpt,
                "seed": args.seed}, indent=2))
            shutil.copy2(side_dst, out / "nla_meta.yaml")
            tmp, dst = save_dir / "optim_latest.pt.tmp", save_dir / "optim_latest.pt"
            torch.save({"step": step + 1, "actor_optim": optim.state_dict()}, str(tmp))
            os.replace(str(tmp), str(dst))
            print(f"[save] LoRA -> {out} (+ optim_latest)", flush=True)

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
