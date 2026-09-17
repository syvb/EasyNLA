"""Stage 3: GRPO on the frozen-reader reward ("behavioral RL").

Same GRPO skeleton as nla/train_rl_self_contained.py, with the reward swapped:

    reconstruction RL:  reward = -MSE(AR(explanation), activation)
    behavioral RL:      reward =  gain(explanation) = how many nats the frozen
                                  reader saves on the target model's own
                                  continuation when shown the explanation

Both objectives live behind one seam (nla/pred/rewards.py) and run through this
same trainer, so a reconstruction arm and a behavioral arm differ in the reward
and nothing else - not the rollout engine, not the LoRA, not the batch size.
Without that, "behavioral beat reconstruction" could just mean "these two runs
were configured differently".

WHAT IS AND IS NOT TRAINED
    trained : one LoRA on the merged AV (the policy)
    frozen  : the merged AV itself, the reader, the target model
    absent  : the held-out reader. It is never loaded in this process, so it
              cannot influence training or checkpoint selection by accident.

    The KL anchors to the SFT policy. With a fresh LoRA on the merged AV that is
    just the adapter-disabled model; with --init-adapter it is a frozen copy of
    that adapter, because the bare base is then not the warm start.

THE REWARD IS A GAIN, NOT A RAW LOG-PROB. Subtracting the per-position
no-explanation baseline cannot change the GRPO gradient (group-relative
advantages already remove any per-position constant), but it makes the logged
reward mean the same thing as the eval tables - nats per target token - and it
gives failed rollouts a floor that does not drift with how predictable a
particular continuation happens to be.

    python -m nla.pred.train_rl --config configs/pred/rl_behavioral.yaml \
        --positions <positions.parquet> --save-dir <ckpts>/behavioral_rl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from nla.config import load_nla_config
from nla.injection import marker_well_formed
from nla.pred.data import load_positions
from nla.pred.reader import DEFAULT_BUCKETS, HEADLINE_BUCKET, FrozenReader, ReaderTemplates
from nla.pred.rewards import ReaderGainReward, ReconReward
from nla.pred.wandb_util import finish_run, init_run
from nla.schema import extract_explanation
from nla.utils import build_prompt_text, cjk_fraction, register_karvonen_hook
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config


class VecRef(list):
    """vectors_ref plus the injection character, so the rollout helper takes one
    object rather than two arguments that can drift apart."""

    def __init__(self, inject_char):
        super().__init__([None])
        self.inject_char = inject_char


@torch.no_grad()
def rollout_batched(
    model, tokenizer, rows, vectors_ref, *, group_size, max_new_tokens,
    temperature, device, eos_ids, gen_batch,
):
    """Generate `group_size` explanations per row, batching ACROSS prompts.

    The stock single-GPU trainer generates one prompt at a time: B sequential
    generate() calls of ~150 decode steps each, which on this workload IS the
    step. Batching across prompts turns that into ceil(B*G / gen_batch) calls and
    is the difference between a 2-hour pilot and a 2-day one. Correctness rests
    on the injection hook scanning each row for its own marker - the same
    property the held-out eval in the stock trainer already relies on.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    flat = [(i, r) for i, r in enumerate(rows) for _ in range(group_size)]
    out: list[dict] = []
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for c0 in range(0, len(flat), gen_batch):
            chunk = flat[c0 : c0 + gen_batch]
            texts = [build_prompt_text(r["prompt"], vectors_ref.inject_char, tokenizer)
                     for _, r in chunk]
            acts = [torch.as_tensor(r["activation"], dtype=torch.float32) for _, r in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            vectors_ref[0] = torch.stack(acts).to(device).float()
            try:
                gen = model.generate(
                    input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=(temperature > 0),
                    **({"temperature": temperature} if temperature > 0 else {}),
                    # Qwen3's generation_config ships top_p/top_k; leaving them on
                    # would sample from a distribution the on-policy update does
                    # not score under.
                    top_p=1.0, top_k=0, repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            finally:
                vectors_ref[0] = None
            new = gen.sequences[:, enc.input_ids.shape[1] :]
            pid_cache: dict = {}
            for k, (row_i, _row) in enumerate(chunk):
                resp = new[k].tolist()
                n_real = next((j + 1 for j, t in enumerate(resp) if t in eos_ids),
                              len(resp))
                resp = resp[:n_real]
                # Unpadded prompt ids: the update must never train on left padding.
                # The AV template is fixed, so this is one distinct string per batch.
                p_ids = pid_cache.get(texts[k])
                if p_ids is None:
                    p_ids = tokenizer(texts[k], add_special_tokens=False)["input_ids"]
                    pid_cache[texts[k]] = p_ids
                out.append({
                    "row": row_i, "prompt_ids": p_ids, "resp_ids": resp,
                    "text": tokenizer.decode(resp, skip_special_tokens=True),
                    "n_resp": len(resp),
                    "truncated": (len(resp) >= max_new_tokens
                                  and (resp[-1] not in eos_ids if resp else True)),
                })
    finally:
        tokenizer.padding_side = orig_side
    return out


@contextlib.contextmanager
def _reference_policy(model, ref_adapter):
    """The SFT policy, for the KL anchor.

    With a fresh LoRA on the merged AV, the SFT policy IS the adapter-disabled
    model. But when --init-adapter starts the policy from an existing adapter,
    disabling gives the BARE BASE instead, silently moving the KL anchor off the
    warm start - so that case switches to a frozen copy of the init adapter.
    """
    if ref_adapter is None:
        with model.disable_adapter():
            yield
        return
    try:
        model.set_adapter(ref_adapter)
        yield
    finally:
        model.set_adapter("default")


def grpo_update(
    model, optim, tokenizer, samples, advantages, activations, vectors_ref, device,
    *, micro_batch=2, kl_beta=0.01, max_grad_norm=1.0, n_total=None,
    ref_adapter=None,
):
    """Fused micro-batched forward + loss + backward. On-policy GRPO surrogate
    plus a k3 KL toward the SFT policy (see _reference_policy)."""
    optim.zero_grad()
    n = len(samples)
    losses, kls, ents = [], [], []
    advantages = advantages.detach()
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        seqs = [samples[i]["prompt_ids"] + samples[i]["resp_ids"] for i in idxs]
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.eos_token_id
        bs = len(idxs)
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            batch_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[r, : len(s)] = 1
        v_batch = torch.stack([activations[i].to(device).float() for i in idxs])
        # Keep vectors_ref set through backward: under gradient checkpointing the
        # recompute re-fires the injection hook, and clearing early would drop the
        # norm-match Jacobian on exactly the marker pathway.
        vectors_ref[0] = v_batch
        new_logits = model(input_ids=batch_ids, attention_mask=attn).logits
        with torch.no_grad(), _reference_policy(model, ref_adapter):
            ref_logits = model(input_ids=batch_ids, attention_mask=attn).logits
        chunk_losses = []
        for r, i in enumerate(idxs):
            p_len = len(samples[i]["prompt_ids"])
            L = len(seqs[r])
            if L <= p_len:
                continue
            target_ids = batch_ids[r, p_len:L]
            pred_idx = torch.arange(p_len - 1, L - 1, device=device)
            resp_logits = new_logits[r].index_select(0, pred_idx).float()
            lse = torch.logsumexp(resp_logits, dim=-1)
            new_lp = resp_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - lse
            with torch.no_grad():
                p_resp = (resp_logits - lse.unsqueeze(-1)).exp()
                ents.append(float((lse - (p_resp * resp_logits).sum(-1)).mean()))
                del p_resp
            ref_sel = ref_logits[r].index_select(0, pred_idx).float()
            ref_lse = torch.logsumexp(ref_sel, dim=-1)
            ref_lp = (ref_sel.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                      - ref_lse).detach()
            if new_lp.numel() == 0:
                continue
            delta = (ref_lp - new_lp).clamp(max=12.0)
            kl = torch.exp(delta) - delta - 1.0           # k3, spike-clamped
            per_tok = -(advantages[i] * new_lp - kl_beta * kl)
            chunk_losses.append(per_tok.mean())
            kls.append(float(kl.detach().mean()))
        del ref_logits
        if not chunk_losses:
            vectors_ref[0] = None
            del new_logits
            continue
        denom = n_total if n_total is not None else n
        chunk_loss = torch.stack(chunk_losses).sum() / denom
        chunk_loss.backward()
        vectors_ref[0] = None          # only after backward (checkpoint recompute)
        losses.append(chunk_loss.item() * denom / len(chunk_losses))
        del new_logits
    gn = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm)
    gn = float(gn.item() if hasattr(gn, "item") else gn)
    if math.isfinite(gn):
        optim.step()
    else:
        optim.zero_grad(set_to_none=True)
        print(f"[grpo] non-finite grad norm ({gn}) - skipping optimizer step", flush=True)
    return (float(np.mean(losses)) if losses else 0.0, gn,
            {"kl_mean": float(np.mean(kls)) if kls else 0.0,
             "entropy": float(np.mean(ents)) if ents else 0.0})


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p)
    p.add_argument("--positions", required=True)
    p.add_argument("--sidecar", default=None, help="defaults to the positions parquet")
    p.add_argument("--base-ckpt", default="syvb/nanonla-qwen3-8b-L24-av",
                   help="The merged AV. The policy is a LoRA on top, so the "
                        "adapter-disabled model IS the SFT KL reference.")
    p.add_argument("--init-adapter", default=None,
                   help="Optional adapter to initialize the policy LoRA from. The "
                        "KL reference stays the bare base either way.")
    p.add_argument("--init-adapter-subfolder", default=None)
    p.add_argument("--kl-reference-adapter", default=None,
                   help="Adapter the KL anchors to. Default: the bare base, which "
                        "IS the SFT policy when --base-ckpt is the merged AV and "
                        "the policy LoRA is fresh. Defaults to --init-adapter when "
                        "that is given, since the bare base is then NOT the warm "
                        "start. On resume, pass the original init adapter.")
    # --- reward ---
    p.add_argument("--reward", choices=["reader_gain", "recon"], default="reader_gain",
                   help="reader_gain = this experiment. recon = the standard NLA "
                        "objective, for a compute-matched baseline arm.")
    p.add_argument("--reader", default="Qwen/Qwen3-4B-Base",
                   help="TRAINING reader. The held-out reader is never loaded here.")
    p.add_argument("--reader-dtype", default="bfloat16")
    p.add_argument("--reader-batch-rows", type=int, default=48)
    p.add_argument("--reader-batch-tokens", type=int, default=49152)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--fail-gain", type=float, default=1.0,
                   help="reader_gain only: reward for a rollout with no parseable "
                        "explanation or one truncated at the cap, in nats/target "
                        "token BELOW the no-explanation baseline.")
    p.add_argument("--ar-ckpt", default="syvb/nanonla-qwen3-8b-L24-ar",
                   help="recon only: the frozen AR reconstructor.")
    # --- optimization ---
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--batch-prompts", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=192,
                   help="The SFT policy writes a median 119 tokens (p99 149). A cap "
                        "at 128 truncates a fifth of its own explanations, so RL "
                        "would learn brevity before it learned anything else.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kl-beta", type=float, default=0.02,
                   help="KL toward the SFT init. Higher than the reconstruction "
                        "default because this reward can be raised by drifting "
                        "toward reader-pleasing phrasing rather than better content.")
    p.add_argument("--length-penalty", type=float, default=None,
                   help="Hinged penalty per token past --length-threshold, in the "
                        "REWARD's units. Defaults per reward type: the gain reward "
                        "lives on a ~0.01-0.3 scale where the reconstruction "
                        "trainer's 0.01 would dominate the signal.")
    p.add_argument("--length-threshold", type=int, default=0, help="0 => cap - 32.")
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    # --- eval / io ---
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n-positions", type=int, default=200)
    p.add_argument("--eval-temperature", type=float, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume-from-lora", default=None)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rl-rows", type=int, default=0, help="0 = all")
    p.add_argument("--wandb-project", default="pred-nla")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="rl")
    p.add_argument("--wandb-tags", default=None)
    p.add_argument("--no-wandb", action="store_true")
    apply_config_defaults(p)
    return p.parse_args()


def main():
    args = build_args()

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert args.temperature == 1.0, (
        f"--temperature {args.temperature} != 1.0: the on-policy update scores "
        f"sampled tokens under the untempered policy, so any other value biases "
        f"the gradient.")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(save_dir.glob("iter_*"))
    if existing and args.resume_from_lora is None:
        raise SystemExit(
            f"[save] {save_dir} already has {len(existing)} iter_* checkpoints "
            f"(latest {existing[-1].name}) - refusing to overwrite. Resume with "
            f"--resume-from-lora {existing[-1]} or use a fresh --save-dir.")
    if args.length_threshold <= 0:
        args.length_threshold = max(1, args.max_new_tokens - 32)
    sidecar = args.sidecar or args.positions
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(sidecar, tokenizer)
    print(f"[cfg] d_model={cfg.d_model} layer={cfg.extraction_layer_index} "
          f"inj_id={cfg.injection_token_id}", flush=True)
    import shutil

    from nla.schema import sidecar_path_for
    dst = save_dir / "nla_meta.yaml"
    if not dst.exists():
        shutil.copy2(sidecar_path_for(sidecar), dst)

    buckets = DEFAULT_BUCKETS
    branches = tuple(range(args.branches))

    # ---- data ----
    train_rows = load_positions(args.positions, split="rl",
                                limit=args.max_rl_rows or None)
    val_rows = load_positions(args.positions, split="val", limit=args.eval_n_positions)
    assert train_rows, "no rl-split rows in the positions parquet"
    n_br = len(train_rows[0]["cont_text"])
    assert args.branches <= n_br, f"--branches {args.branches} > {n_br} stored"
    print(f"[data] {len(train_rows)} rl positions, {len(val_rows)} val positions, "
          f"{args.branches}/{n_br} branches", flush=True)

    # ---- policy: merged AV + LoRA; disable_adapter() == the SFT policy ----
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
        quantization_config=quant_config)
    if quant_config is None:
        base = base.to(device)
    else:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=args.gradient_checkpointing)
    # A resumed run keeps the KL anchor it started with; --init-adapter implies it.
    if args.kl_reference_adapter is None and args.init_adapter:
        args.kl_reference_adapter = args.init_adapter
        print(f"[policy] KL reference defaults to --init-adapter "
              f"({args.init_adapter}): the bare base is not the warm start here.",
              flush=True)
    init_from = args.resume_from_lora or args.init_adapter
    if init_from:
        kw = ({"subfolder": args.init_adapter_subfolder}
              if (args.init_adapter_subfolder and not args.resume_from_lora) else {})
        model = PeftModel.from_pretrained(base, init_from, adapter_name="default",
                                          is_trainable=True, **kw)
        print(f"[policy] LoRA initialized from {init_from}", flush=True)
    else:
        from nla.utils.arch_adapters import resolve_lora_target_modules
        model = get_peft_model(base, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", use_rslora=args.use_rslora,
            target_modules=resolve_lora_target_modules(base.config)))
        print(f"[policy] fresh LoRA r={args.lora_r} on the merged AV "
              f"(zero-init B, so step 0 IS the SFT policy)", flush=True)
    ref_adapter = None
    if args.kl_reference_adapter:
        kw = ({"subfolder": args.init_adapter_subfolder}
              if args.init_adapter_subfolder else {})
        model.load_adapter(args.kl_reference_adapter, adapter_name="reference", **kw)
        model.set_adapter("default")
        for n_, p_ in model.named_parameters():
            if ".reference." in n_:
                p_.requires_grad_(False)     # the anchor must never train
        ref_adapter = "reference"
        print(f"[policy] KL reference adapter: {args.kl_reference_adapter}", flush=True)
    else:
        print("[policy] KL reference: adapter-disabled base (== the SFT policy)",
              flush=True)
    model.print_trainable_parameters()
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    vectors_ref = VecRef(cfg.injection_char)
    register_karvonen_hook(model, vectors_ref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    eos_ids = {tokenizer.eos_token_id}
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc_eos is not None:
        eos_ids.update(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
    eos_ids.discard(None)

    # ---- reward ----
    if args.reward == "reader_gain":
        reader = FrozenReader.load(
            args.reader, device=device, dtype=args.reader_dtype,
            templates=ReaderTemplates(), max_batch_rows=args.reader_batch_rows,
            max_batch_tokens=args.reader_batch_tokens)
        reward_fn = ReaderGainReward(reader, branches=branches, buckets=buckets,
                                     fail_gain=args.fail_gain)
        headline_key = "gain"
    else:
        from nla.models import NLACriticModel
        from nla.schema import compute_predict_mean_baselines, resolve_target_scale
        mse_scale = resolve_target_scale(cfg.mse_scale, cfg.d_model)
        critic = NLACriticModel.from_pretrained(
            args.ar_ckpt, torch_dtype=dtype).to(device).eval()
        for p_ in critic.parameters():
            p_.requires_grad_(False)
        acts = torch.tensor(
            np.stack([r["activation"] for r in train_rows[: min(4000, len(train_rows))]]),
            dtype=torch.float32)
        _, fve_baseline = compute_predict_mean_baselines(acts, mse_scale)
        del acts
        print(f"[ar] {args.ar_ckpt} FROZEN; predict-the-mean baseline "
              f"{fve_baseline:.4f}", flush=True)
        reward_fn = ReconReward(critic, tokenizer, cfg.critic_prompt_template,
                                mse_scale, device, fve_baseline=fve_baseline)
        headline_key = "reward/mean"
    if args.length_penalty is None:
        args.length_penalty = reward_fn.default_length_penalty
    print(f"[reward] {reward_fn.name} | length penalty {args.length_penalty}/token "
          f"past {args.length_threshold} (cap {args.max_new_tokens})", flush=True)

    # ---- optimizer ----
    try:
        import bitsandbytes as bnb
        adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] bitsandbytes AdamW8bit ({bnb.__version__})", flush=True)
    except ImportError:
        adam_cls = torch.optim.AdamW
        print("[optim] bitsandbytes unavailable - torch AdamW", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if args.resume_from_lora:
        from nla.utils.resume import find_optim_ckpt, warn_cold_adam
        ck = find_optim_ckpt(args.save_dir, args.resume_from_lora)
        if ck is not None:
            st = torch.load(str(ck), map_location="cpu", weights_only=True)
            saved = int(st.get("step", 0))
            if args.start_step == 0 and saved > 0:
                args.start_step = saved
            try:
                optim.load_state_dict(st["actor_optim"])
                print(f"[resume] optimizer restored from {ck} (step {saved})", flush=True)
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"[resume] WARN optimizer incompatible ({e})", flush=True)
        else:
            warn_cold_adam(args.start_step)

    save_resolved_config(args, args.save_dir)
    run = None if args.no_wandb else init_run(
        args, project=args.wandb_project, name=args.wandb_name,
        group=args.wandb_group, job_type=f"rl-{reward_fn.name}",
        extra_config={"reward_name": reward_fn.name,
                      "headline_bucket": list(HEADLINE_BUCKET),
                      "reader_templates": ReaderTemplates().as_dict(),
                      "n_rl_positions": len(train_rows)})

    rng = np.random.default_rng(args.seed)
    pending = list(range(len(train_rows)))
    rng.shuffle(pending)
    cursor = 0
    for _ in range(args.start_step):
        if cursor + args.batch_prompts > len(pending):
            rng.shuffle(pending)
            cursor = 0
        cursor += args.batch_prompts
    eval_table: list[list] = []
    best = {"step": -1, "score": -float("inf")}

    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        if cursor + args.batch_prompts > len(pending):
            rng.shuffle(pending)
            cursor = 0
        rows = [train_rows[i] for i in pending[cursor : cursor + args.batch_prompts]]
        cursor += args.batch_prompts

        model.eval()
        samples = rollout_batched(
            model, tokenizer, rows, vectors_ref, group_size=args.group_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            device=device, eos_ids=eos_ids, gen_batch=args.gen_batch)
        t_gen = time.time()

        expls = [extract_explanation(s["text"]) for s in samples]
        cjk_bad = [cjk_fraction(s["text"]) > 0.05 for s in samples]
        # A silent injection failure is the one bug that looks exactly like "the
        # method does not work", so check the mechanism every step.
        marker_ok = [marker_well_formed(s["prompt_ids"], cfg.injection_token_id,
                                        cfg.injection_left_neighbor_id,
                                        cfg.injection_right_neighbor_id)
                     for s in samples]
        inject_ok = [(not c) and m for c, m in zip(cjk_bad, marker_ok)]
        truncated = [s["truncated"] for s in samples]

        rout = reward_fn.score(rows=rows, samples=samples, explanations=expls,
                               truncated=truncated)
        t_rew = time.time()
        rewards_t = torch.tensor(rout.reward, dtype=torch.float32, device=device)

        shape = {}
        if args.length_penalty > 0:
            n_tok = torch.tensor([s["n_resp"] for s in samples], dtype=torch.float32,
                                 device=device)
            over = (n_tok - float(args.length_threshold)).clamp_min(0.0)
            rewards_t = rewards_t - args.length_penalty * over
            shape["av/len_pen_mean"] = float((args.length_penalty * over).mean())
            shape["av/len_overage_frac"] = float((over > 0).float().mean())

        group = torch.tensor([s["row"] for s in samples], dtype=torch.long, device=device)
        ok_t = torch.tensor(inject_ok, dtype=torch.bool, device=device)
        adv = torch.zeros_like(rewards_t)
        n_degenerate = 0
        for gi in range(len(rows)):
            mask = (group == gi) & ok_t
            if mask.sum() == 0:
                continue
            gr = rewards_t[mask]
            sd = gr.std() if gr.numel() > 1 else torch.tensor(1.0, device=device)
            if float(sd) < 1e-6:
                n_degenerate += 1
            adv[mask] = (gr - gr.mean()) / (sd + 1e-6)

        keep = [i for i, ok in enumerate(inject_ok) if ok]
        if not keep:
            print(f"step {step}: every rollout failed the injection checks - "
                  f"skipping the update", flush=True)
            continue
        model.train()
        acts_all = [torch.as_tensor(rows[s["row"]]["activation"], dtype=torch.float32)
                    for s in samples]
        loss_val, grad_norm, metrics = grpo_update(
            model, optim, tokenizer, [samples[i] for i in keep],
            adv.index_select(0, torch.tensor(keep, device=device)),
            [acts_all[i] for i in keep], vectors_ref, device,
            micro_batch=args.logp_micro_batch, kl_beta=args.kl_beta,
            max_grad_norm=args.max_grad_norm, n_total=len(samples),
            ref_adapter=ref_adapter)
        t_upd = time.time()

        valid_r = rout.reward[rout.valid]
        log = {
            "gain" if reward_fn.name == "reader_gain" else "reward/valid_mean":
                float(np.mean(valid_r)) if valid_r.size else float("nan"),
            "wall_s": time.time() - t0,
            "av/grad_norm": grad_norm,
            "av/kl_to_ref": metrics["kl_mean"],
            "av/entropy": metrics["entropy"],
            "av/advantage_std": float(adv.std()),
            "av/extraction_rate": float(np.mean([e is not None for e in expls])),
            "av/resp_len": float(np.mean([s["n_resp"] for s in samples])),
            "av/frac_cut_off": float(np.mean(truncated)),
            "av/inject_fail_count": int(sum(cjk_bad)),
            "av/marker_bad_count": int(sum(1 for m in marker_ok if not m)),
            "av/inject_masked_count": int(len(samples) - len(keep)),
            "av/failed_reward_count": int((~rout.valid).sum()),
            "av/degenerate_groups": n_degenerate,
            "reward/mean": float(np.mean(rout.reward)),
            "reward/std": float(np.std(rout.reward)),
            "reward/min": float(np.min(rout.reward)),
            "reward/max": float(np.max(rout.reward)),
            "time/gen_s": t_gen - t0, "time/reward_s": t_rew - t_gen,
            "time/update_s": t_upd - t_rew,
            "loss": loss_val,
            **rout.logs, **shape,
        }
        head = log.get(headline_key, float("nan"))
        print(f"step {step:04d} | {headline_key} {head:+.4f} "
              f"| kl {log['av/kl_to_ref']:.4f} | ent {log['av/entropy']:.3f} "
              f"| ext {log['av/extraction_rate']:.0%} | len {log['av/resp_len']:.0f} "
              f"| adv_sd {log['av/advantage_std']:.2f} | t {log['wall_s']:.0f}s "
              f"(gen {log['time/gen_s']:.0f} rew {log['time/reward_s']:.0f} "
              f"upd {log['time/update_s']:.0f})", flush=True)

        # ---- held-out eval, TRAINING READER ONLY ----
        if args.eval_every > 0 and val_rows and step % args.eval_every == 0:
            t_ev = time.time()
            model.eval()
            et = (args.eval_temperature if args.eval_temperature is not None
                  else args.temperature)
            ev = rollout_batched(
                model, tokenizer, val_rows, vectors_ref, group_size=1,
                max_new_tokens=args.max_new_tokens, temperature=et, device=device,
                eos_ids=eos_ids, gen_batch=args.gen_batch)
            e_expl = [extract_explanation(s["text"]) for s in ev]
            e_score = reward_fn.eval_score(val_rows, e_expl,
                                           [s["truncated"] for s in ev])
            fin = e_score[np.isfinite(e_score)]
            mean_s = float(np.mean(fin)) if fin.size else float("nan")
            log["eval/score"] = mean_s
            log["eval/extraction_rate"] = float(np.mean([e is not None for e in e_expl]))
            log["eval/resp_len"] = float(np.mean([s["n_resp"] for s in ev]))
            log["time/eval_s"] = time.time() - t_ev
            if np.isfinite(mean_s) and mean_s > best["score"]:
                best = {"step": step, "score": mean_s}
            print(f"  [eval@{step}] score {mean_s:+.4f} "
                  f"| ext {log['eval/extraction_rate']:.0%} "
                  f"| len {log['eval/resp_len']:.0f} "
                  f"| best {best['score']:+.4f}@{best['step']}", flush=True)
            for k in range(min(3, len(ev))):
                print(f"    [row={val_rows[k]['row_id']} s={e_score[k]:+.3f}] "
                      + (e_expl[k] or "<failed>")[:180].replace("\n", " "), flush=True)
            if run is not None:
                import wandb
                for k in range(min(8, len(ev))):
                    eval_table.append([step, val_rows[k]["row_id"], float(e_score[k]),
                                       ev[k]["n_resp"], (e_expl[k] or "<failed>")[:600]])
                log["eval/samples"] = wandb.Table(
                    columns=["step", "row_id", "score", "n_tokens", "explanation"],
                    data=list(eval_table))

        if run is not None:
            run.log(log, step=step)

        if (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            tmp = save_dir / "optim_latest.pt.tmp"
            torch.save({"step": step + 1, "actor_optim": optim.state_dict()}, str(tmp))
            os.replace(str(tmp), str(save_dir / "optim_latest.pt"))
            (save_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[save] {out_dir}", flush=True)

    (save_dir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"done. best held-out score {best['score']:+.4f} at step {best['step']}",
          flush=True)
    if run is not None:
        run.summary["best/eval_score"] = best["score"]
        run.summary["best/step"] = best["step"]
        finish_run(run)


if __name__ == "__main__":
    main()
