"""Future Lens learned-prompt baseline (Pal et al. 2023) ported to Qwen3.

Their main method: a soft prompt c = [c_1..c_M] (M = 10 learned embeddings) per layer l; the
stored hidden state h_t^l is TRANSPLANTED into the residual stream at block l's output at the
last prompt position (no scaling, no marker token); the prompt is trained with the model
frozen to minimise KL(model after transplant || original model) at N = 1, teacher-forced on
the model's own greedy token (their Eq. 10/11: "optimizing for N=1 works best and
generalizes surprisingly well to other N"). Evaluation: precision@1 / @5 at N = 0..3 given the
model's own previous tokens (teacher-forced), plus surprisal.

Here the KL target is the frozen target's stored top-K distribution at greedy step N
(`greedy_topk_*`, collected with --topk), scored with the same truncated-KL as SFT --distill.
Eval writes the standard JSONL records (group "futurelens") on the SAME seeded position
subsample as nla.future_lens.eval, so the numbers sit next to the SFT/RL decoder's:
    tf_p1 / p5 / tf_kl  teacher-forced at offset N (the paper's convention)
    p1                  free-running: greedy 9-token readout after the transplant, offset N
Conditions: real, shuffled (another position's vector, same layer).

    python -m nla.future_lens.futurelens_prompt --base-ckpt Qwen/Qwen3-8B-Base \
        --train-parquet $D/train.parquet --parquet $D/eval.parquet --layers 8,12,16,20,24,28,32 \
        --n-train 10000 --steps 600 --max-rows 2000 --save-dir $C/futurelens --out $E/futurelens.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nla.future_lens.data import TOPK_COLUMNS, load_fl_meta, load_fl_rows
from nla.future_lens.eval import write_records

INIT_TEXT = "Please, tell me something about the following, continuing the text exactly:"


# ----------------------------------------------------------------------------
# transplant hook: replace the residual stream at block `layer`'s output, position `pos`
# ----------------------------------------------------------------------------

class Transplant:
    def __init__(self, model, layer: int, pos: int):
        self.vec = None            # [B, d] or None
        self.pos = pos
        self.n_fired = 0
        block = model.model.layers[layer]
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.vec is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= self.pos:          # decode steps (S == 1): nothing to do
            return output
        hidden = hidden.clone()
        hidden[:, self.pos, :] = self.vec.to(hidden.dtype)
        self.n_fired += 1
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def remove(self):
        self.handle.remove()


def soft_ce(logits: torch.Tensor, ids: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """Per-row truncated soft CE vs the target's top-K (+ remainder bucket): logits [n,V],
    ids/logp [n,K]. = KL(q||p) + H(q) on the coarsened support (same as train_sft.distill_loss)."""
    lp = F.log_softmax(logits.float(), dim=-1)
    q = logp.float().exp()
    q_rest = (1.0 - q.sum(-1)).clamp(min=0.0)
    lp_k = lp.gather(1, ids)
    lp_rest = torch.logsumexp(lp.scatter(1, ids, float("-inf")), dim=-1)
    return -(q * lp_k).sum(-1) - q_rest * lp_rest


def entropy_q(logp: torch.Tensor) -> torch.Tensor:
    q = logp.float().exp(); q_rest = (1.0 - q.sum(-1)).clamp(min=0.0)
    h = -(q * logp.float()).sum(-1)
    return h - torch.where(q_rest > 0, q_rest * q_rest.clamp(min=1e-12).log(), torch.zeros_like(q_rest))


def build_inputs(model, soft: torch.Tensor, label_ids: torch.Tensor):
    """inputs_embeds = [soft prompt (M)] + [label tokens]; label_ids [B, n] (n may be 0)."""
    B = label_ids.shape[0]
    emb = model.get_input_embeddings()
    parts = [soft.to(emb.weight.dtype).unsqueeze(0).expand(B, -1, -1)]
    if label_ids.shape[1] > 0:
        parts.append(emb(label_ids))
    x = torch.cat(parts, dim=1)
    return x, torch.ones(x.shape[:2], dtype=torch.long, device=x.device)


# ----------------------------------------------------------------------------
# train one layer's prompt
# ----------------------------------------------------------------------------

def train_prompt(model, layer: int, rows: list[dict], *, M: int, steps: int, batch: int, lr: float,
                 offsets: list[int], device, seed: int, init_ids: list[int],
                 max_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    """Train one layer's soft prompt. With `max_seconds`, the step count is set from a short timing
    probe so that arms with different prompt lengths get the SAME wall-clock (compute-matched
    comparison); the cosine schedule then anneals over the remaining steps. Returns (prompt, steps)."""
    torch.manual_seed(seed)
    emb = model.get_input_embeddings()
    with torch.no_grad():
        init = emb.weight[torch.tensor(init_ids[:M], device=emb.weight.device)].float()
        if init.shape[0] < M:
            init = torch.cat([init, emb.weight[torch.randint(0, emb.weight.shape[0], (M - init.shape[0],))].float()])
    soft = torch.nn.Parameter(init.clone().to(device))
    optim = torch.optim.Adam([soft], lr=lr)
    tp = Transplant(model, layer, pos=M - 1)
    rng = np.random.default_rng(seed)
    n_lab = max(offsets)                                   # teacher-forced tokens fed after the prompt
    probe = 30 if max_seconds else 0                       # untimed-schedule warmup used to measure s/step
    sched = None if max_seconds else torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=lr * 0.1)
    t0 = time.time()
    step = 0
    try:
        while step < steps:
            idx = rng.choice(len(rows), size=batch, replace=len(rows) < batch)
            chunk = [rows[i] for i in idx]
            vec = torch.tensor(np.stack([np.asarray(r["activation_vector"], dtype=np.float32) for r in chunk]), device=device)
            lab = torch.tensor(np.stack([np.asarray(r["target_ids"][:n_lab], dtype=np.int64) for r in chunk]), device=device)
            x, attn = build_inputs(model, soft, lab)
            tp.vec = vec
            logits = model(inputs_embeds=x, attention_mask=attn).logits          # [B, M+n_lab, V]
            tp.vec = None
            loss = 0.0
            for N in offsets:                                                     # position M-1+N predicts label N
                ids = torch.tensor(np.stack([np.asarray(r["greedy_topk_ids"][N], dtype=np.int64) for r in chunk]), device=device)
                lpq = torch.tensor(np.stack([np.asarray(r["greedy_topk_logp"][N], dtype=np.float32) for r in chunk]), device=device)
                loss = loss + soft_ce(logits[:, M - 1 + N], ids, lpq).mean()
            loss = loss / len(offsets)
            optim.zero_grad(); loss.backward(); optim.step()
            if sched is not None:
                sched.step()
            step += 1
            if max_seconds and step == probe:              # budget -> step count, then anneal over the rest
                per = (time.time() - t0) / probe
                steps = max(probe + 1, int(max_seconds / per))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps - probe, eta_min=lr * 0.1)
                print(f"[futurelens] L{layer} M={M}: {per*1000:.0f} ms/step -> {steps} steps "
                      f"in {max_seconds/3600:.2f} h", flush=True)
            if step % 500 == 0 or step == steps:
                print(f"[futurelens] L{layer} step {step:5d}/{steps} soft_ce {loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        tp.remove()
    print(f"[futurelens] L{layer} M={M} done: {step} steps in {time.time() - t0:.0f}s", flush=True)
    return soft.detach(), step


# ----------------------------------------------------------------------------
# eval one layer's prompt
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_prompt(model, tokenizer, layer: int, soft: torch.Tensor, rows: list[dict], *, M: int, batch: int,
                nf: int, device, seed: int, conditions=("real", "shuffled"), tag: dict | None = None,
                eos_ids: set[int] | None = None) -> list[dict]:
    tp = Transplant(model, layer, pos=M - 1)
    rng = np.random.default_rng(seed)
    recs = []
    try:
        for cond in conditions:
            vecs = [np.asarray(r["activation_vector"], dtype=np.float32) for r in rows]
            if cond == "shuffled":
                perm = rng.permutation(len(rows)); src = [None] * len(rows)
                for j in range(len(rows)):
                    src[perm[j]] = vecs[perm[(j + 1) % len(rows)]]
                vecs = src
            hits_tf = defaultdict(list); hits5 = defaultdict(list); kls = defaultdict(list); hits_fr = defaultdict(list)
            t0 = time.time()
            for cs in range(0, len(rows), batch):
                chunk = rows[cs: cs + batch]
                vec = torch.tensor(np.stack(vecs[cs: cs + batch]), device=device)
                lab = torch.tensor(np.stack([np.asarray(r["target_ids"][:nf], dtype=np.int64) for r in chunk]), device=device)
                # teacher-forced: one forward with all nf label tokens appended
                x, attn = build_inputs(model, soft, lab)
                tp.vec = vec
                logits = model(inputs_embeds=x, attention_mask=attn).logits.float()
                tp.vec = None
                pred = logits.argmax(-1)
                for i, r in enumerate(chunk):
                    top5 = np.asarray(r["target_top5"]).reshape(-1, 5)
                    for j in range(nf):
                        p = int(pred[i, M - 1 + j]); y = int(lab[i, j])
                        hits_tf[j].append(int(p == y)); hits5[j].append(int(p in set(int(v) for v in top5[j])))
                    if "greedy_topk_ids" in r:
                        ids = torch.as_tensor(np.asarray(r["greedy_topk_ids"][:nf], dtype=np.int64), device=device)
                        lpq = torch.as_tensor(np.asarray(r["greedy_topk_logp"][:nf], dtype=np.float32), device=device)
                        ce = soft_ce(logits[i, M - 1: M - 1 + nf], ids, lpq)
                        kl = (ce - entropy_q(lpq)).cpu().tolist()
                        for j in range(nf):
                            kls[j].append(kl[j])
                # free-running: greedy readout of nf tokens after the transplant
                x0, attn0 = build_inputs(model, soft, lab[:, :0])
                tp.vec = vec
                gen = model.generate(inputs_embeds=x0, attention_mask=attn0, max_new_tokens=nf, min_new_tokens=nf,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                     eos_token_id=None)
                tp.vec = None
                ro = gen[:, -nf:] if gen.shape[1] >= nf else gen   # generate() with inputs_embeds returns new tokens only
                for i in range(len(chunk)):
                    for j in range(nf):
                        hits_fr[j].append(int(int(ro[i, j]) == int(lab[i, j])))
            base = {"layer": layer, "condition": cond, "injection": "transplant", "seed": seed,
                    "checkpoint": f"futurelens_soft{M}", "group": "futurelens", "label": "greedy", **(tag or {})}
            for j in range(nf):
                b = {**base, "N": j, "k": j + 1}
                recs.append({**b, "metric": "tf_p1", "value": float(np.mean(hits_tf[j])), "n": len(hits_tf[j])})
                recs.append({**b, "metric": "p5", "value": float(np.mean(hits5[j])), "n": len(hits5[j])})
                recs.append({**b, "metric": "p1", "value": float(np.mean(hits_fr[j])), "n": len(hits_fr[j])})
                if kls[j]:
                    recs.append({**b, "metric": "tf_kl", "value": float(np.mean(kls[j])), "n": len(kls[j])})
            print(f"[futurelens] L{layer} {cond:9s} n={len(rows)} tf_p1 " + " ".join(f"{np.mean(hits_tf[j]):.3f}" for j in range(4))
                  + " | p1 " + " ".join(f"{np.mean(hits_fr[j]):.3f}" for j in range(4))
                  + (f" | tf_kl {np.mean(kls[1]):.2f}" if kls[1] else "") + f" ({time.time() - t0:.0f}s)", flush=True)
    finally:
        tp.remove()
    return recs


def subsample_like_eval(rows: list[dict], max_rows: int | None, seed: int) -> list[dict]:
    """Identical to nla.future_lens.eval --max-rows: seeded shuffle of the sorted position keys."""
    if not max_rows:
        return rows
    keys = sorted({(int(r["doc_idx"]), int(r["t"])) for r in rows})
    random.Random(seed).shuffle(keys)
    keep = set(keys[:max_rows])
    return [r for r in rows if (int(r["doc_idx"]), int(r["t"])) in keep]


# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--parquet", required=True, help="eval.parquet")
    p.add_argument("--layers", default="8,12,16,20,24,28,32")
    p.add_argument("--prompt-len", type=int, default=10)
    p.add_argument("--n-train", type=int, default=10000, help="positions per layer (paper: 10k)")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="train for this wall-clock instead of --steps (step count from a timing probe), so arms "
                        "with different --prompt-len are compute-matched")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--train-offsets", default="1", help="paper: N=1 only")
    p.add_argument("--max-rows", type=int, default=2000, help="eval positions per layer (same subsample as eval.py)")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--conditions", default="real,shuffled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--group", default="futurelens", help="record group/checkpoint prefix (e.g. futurelens_6k for a longer run)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.base_ckpt, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    fl = load_fl_meta(args.train_parquet)
    assert fl.topk, "needs a split collected with --topk (distillation targets)"
    nf, M = fl.n_future, args.prompt_len
    layers = [int(x) for x in args.layers.split(",")]
    offsets = [int(x) for x in args.train_offsets.split(",")]
    init_ids = tok.encode(INIT_TEXT, add_special_tokens=False)
    cols = ["activation_vector", "activation_layer", "target_ids", "target_top5", "k", "doc_idx", "t"] + TOPK_COLUMNS
    eos = {tok.eos_token_id}
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    all_recs = []
    for layer in layers:
        tr = load_fl_rows(args.train_parquet, layers=[layer], label="greedy", drop_label_ids=eos, columns=cols)
        rng = np.random.default_rng(args.seed)
        if len(tr) > args.n_train:
            tr = [tr[i] for i in sorted(rng.choice(len(tr), size=args.n_train, replace=False))]
        soft, n_steps = train_prompt(model, layer, tr, M=M, steps=args.steps if not args.max_seconds else 10**9,
                                     batch=args.batch, lr=args.lr, offsets=offsets, device=device, seed=args.seed,
                                     init_ids=init_ids, max_seconds=args.max_seconds)
        del tr
        if save_dir:
            torch.save({"soft": soft.cpu(), "layer": layer, "M": M, "steps": n_steps, "init_text": INIT_TEXT},
                       save_dir / f"soft_L{layer}_M{M}.pt")
        ev = subsample_like_eval(load_fl_rows(args.parquet, layers=[layer], label="greedy", drop_label_ids=eos, columns=cols),
                                 args.max_rows, args.seed)
        recs = eval_prompt(model, tok, layer, soft, ev, M=M, batch=args.eval_batch, nf=nf, device=device, seed=args.seed,
                           conditions=args.conditions.split(","), eos_ids=eos,
                           tag={"group": args.group, "checkpoint": f"{args.group}_soft{M}_s{n_steps}"})
        all_recs += recs
        write_records(recs, args.out)
    print(f"[futurelens] wrote {len(all_recs)} records -> {args.out}")


if __name__ == "__main__":
    main()
