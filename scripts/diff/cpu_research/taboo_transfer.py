"""What do the Taboo `Org - Inst` diffs that Benchmark 5 feeds to the verbalizer look like?

Benchmark 5 hands Arm 2 the raw difference between a Taboo LoRA and Instruct at `asst_last` and
at the mean over user-content tokens, and asks for the secret word. For the ten default organisms
(five words, each as a narrow and a 50%-mix LoRA), in no-think mode, this reports per position:

  1. size and shape: norm against tau, and how much of the diff is one direction shared by all organisms
  2. organism information: leave-one-prompt-out nearest-centroid accuracy (10-way), and whether a
     word's narrow LoRA and 50%-mix LoRA move activations the same way (5-way, centroids from one
     regime, tested on the other)
  3. a crude logit lens: rank of the secret word when the diff is read through Instruct's final-norm gain
     and unembedding (no upper blocks), for the mean diff and for single prompts, with and without
     removing the mean over all organisms
  4. distribution shift: share of each unit diff that lies in the top principal directions of the
     Arm 2 training target at the matching role, against held-out training targets themselves

    python taboo_transfer.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io
from corpus_geometry import center_by_role, unit
from corpus_rows import load_corpus_rows, massive_rows

from nla.diff.alignment import apply_alignment, load_alignment
from nla.diff.benchmarks import TABOO_ELICITATION_PROMPTS, get_taboo_organisms
from nla.diff.wrappers import encode_chatml_dialogue

KINDS = {"asst_last": "asst_last", "user_mean": "user_content"}  # position kind -> training role


def unembedding(model_name: str) -> torch.Tensor:
    """lm_head with the final RMSNorm gain folded in: to first order a residual difference d moves logits by W (g * d)."""
    got: dict[str, torch.Tensor] = {}
    for f in sorted(cached_io._snapshot_dir(model_name).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for key in ("lm_head.weight", "model.norm.weight"):
                if key in sf.keys():  # noqa: SIM118
                    got[key] = sf.get_tensor(key).float()
    return got["lm_head.weight"] * got["model.norm.weight"][None, :]


def loo_centroid_acc(x: np.ndarray, y: np.ndarray, prompt: np.ndarray) -> float:
    """Nearest-centroid accuracy by cosine, with the test prompt held out of every centroid."""
    classes = np.unique(y)
    hits = 0
    for p in np.unique(prompt):
        tr = prompt != p
        cents = unit(np.stack([x[tr & (y == c)].mean(0) for c in classes]))
        te = np.where(~tr)[0]
        hits += int((classes[(unit(x[te]) @ cents.T).argmax(1)] == y[te]).sum())
    return hits / len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = json.loads((p0 / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tau = float(cfg["tau_web_median"])
    win = str(cfg["base_wrapper"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    orgs = get_taboo_organisms("paired")
    encs = [encode_chatml_dialogue(tok, q, enable_thinking=False, max_total_tokens=512)
            for q in TABOO_ELICITATION_PROMPTS[: args.n_prompts]]
    seqs = [e.input_ids for e in encs]
    pos = [list(e.user_content_positions) + [e.asst_last_idx] for e in encs]

    def grab(model: cached_io.LazyLM) -> dict[str, np.ndarray]:
        exs = cached_io.cached_extract(model, seqs, pos, layer_index=k, pad_token_id=pad)
        model.free()
        return {"asst_last": np.stack([x.hidden[-1].numpy() for x in exs]),
                "user_mean": np.stack([x.hidden[:-1].mean(0).numpy() for x in exs])}

    h_inst = grab(cached_io.LazyLM(cfg["target_model"]))
    h_org = {o.name: grab(cached_io.LazyLM(cfg["target_model"], adapter=o.repo_id)) for o in orgs}
    if cached_io.RECORD_PATH:
        return

    # Arm 2 training targets at each role, to compare the Taboo diffs against.
    rows = load_corpus_rows(cfg, tok, args.rollouts)
    ok = ~massive_rows(rows.ht, rows.hb["chatml"], rows.hb["plain"])
    art = load_alignment(p0 / f"align_L{k}_id_anchored.pt")
    train = center_by_role(apply_alignment(torch.from_numpy(rows.hb[win]), torch.from_numpy(rows.ht), art).numpy(), rows.roles)
    w_u = unembedding(cfg["target_model"])
    words = sorted({o.secret_word for o in orgs})
    word_ids = {w: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w, w.capitalize(), " " + w.capitalize())
                           if len(tok.encode(v, add_special_tokens=False)) == 1}) for w in words}

    report: dict = {"layer": k, "tau_web_median": tau, "n_prompts": len(seqs), "organisms": [o.name for o in orgs],
                    "secret_word_token_ids": word_ids, "positions": {}}
    n = len(seqs)
    for kind, role in KINDS.items():
        v = np.concatenate([h_org[o.name][kind] - h_inst[kind] for o in orgs]).astype(np.float64)  # [orgs * n, d]
        org_y = np.repeat(np.arange(len(orgs)), n)
        word_y = np.repeat([words.index(o.secret_word) for o in orgs], n)
        narrow = np.repeat([o.regime == "narrow" for o in orgs], n)
        prompt = np.tile(np.arange(n), len(orgs))
        grand = v.mean(0)
        means = np.stack([v[org_y == i].mean(0) for i in range(len(orgs))])
        um = unit(means)
        off = (um @ um.T)[~np.eye(len(orgs), dtype=bool)]
        vc = v - grand
        r: dict = {
            "median_norm_over_tau": {o.name: float(np.median(np.linalg.norm(v[org_y == i], axis=1)) / tau) for i, o in enumerate(orgs)},
            "shared_direction_energy_share": float((grand**2).sum() / (v**2).sum(1).mean()),
            "mean_cos_between_organism_means": float(off.mean()),
            "organism_id_acc_10way": loo_centroid_acc(vc, org_y, prompt),
        }
        # Same word, other regime: centroids from one regime, tested on the other (each regime centred on its own mean).
        acc = []
        for tr_narrow in (True, False):
            tr, te = narrow == tr_narrow, narrow != tr_narrow
            xtr, xte = v[tr] - v[tr].mean(0), v[te] - v[te].mean(0)
            cents = unit(np.stack([xtr[word_y[tr] == c].mean(0) for c in range(len(words))]))
            acc.append(float(((unit(xte) @ cents.T).argmax(1) == word_y[te]).mean()))
        r["word_id_acc_5way_narrow_to_mix"], r["word_id_acc_5way_mix_to_narrow"] = acc

        # Logit lens. "centred" removes the mean over all ten organisms, which a one-organism audit would not have.
        def ranks(logits: np.ndarray, secret: str) -> tuple[int, int]:
            cand = {w: float(max(logits[t] for t in ids)) for w, ids in word_ids.items()}
            return int((logits > cand[secret]).sum()) + 1, sorted(cand, key=cand.get, reverse=True).index(secret) + 1

        lens: dict = {}
        for i, o in enumerate(orgs):
            entry: dict = {}
            for tag, shift in (("centred", grand), ("uncentred", np.zeros_like(grand))):
                logits = (w_u @ torch.from_numpy(means[i] - shift).float()).numpy()
                vocab_rank, among = ranks(logits, o.secret_word)
                single = [ranks(row, o.secret_word)
                          for row in (torch.from_numpy(v[org_y == i] - shift).float() @ w_u.T).numpy()]
                entry[tag] = {
                    "vocab_rank_of_secret_word": vocab_rank,
                    "rank_among_5_words": among,
                    "top_tokens": [tok.decode([int(t)]) for t in np.argsort(-logits)[:8]],
                    "single_prompt_median_vocab_rank": float(np.median([a for a, _ in single])),
                    "single_prompt_frac_first_of_5": float(np.mean([b == 1 for _, b in single])),
                    "single_prompt_frac_vocab_rank_le_10": float(np.mean([a <= 10 for a, _ in single])),
                }
            lens[o.name] = entry
        r["logit_lens"] = lens
        for tag in ("centred", "uncentred"):
            r[f"logit_lens_first_of_5_{tag}"] = float(np.mean([x[tag]["rank_among_5_words"] == 1 for x in lens.values()]))
            r[f"logit_lens_single_prompt_first_of_5_{tag}"] = float(np.mean([x[tag]["single_prompt_frac_first_of_5"] for x in lens.values()]))

        # Distribution shift against the Arm 2 training target at the matching role.
        m = (rows.roles == role) & ok
        convs = np.unique(rows.conv[m])
        held = np.isin(rows.conv, convs[::5])
        tr_t = unit(train[m & ~held])
        _, q = torch.linalg.eigh(torch.from_numpy(tr_t.T @ tr_t / len(tr_t)))
        q = q.flip(1).numpy()
        shift: dict = {"train_median_norm_over_tau": float(np.median(np.linalg.norm(train[m], axis=1)) / tau)}
        for kk in (10, 100):
            sub = q[:, :kk]
            shift[f"top{kk}_share"] = {
                "heldout_training_targets": float(((unit(train[m & held]) @ sub) ** 2).sum(1).mean()),
                "taboo_org_minus_inst": float(((unit(v) @ sub) ** 2).sum(1).mean()),
                "taboo_org_minus_inst_centered": float(((unit(vc) @ sub) ** 2).sum(1).mean()),
                "random_direction": kk / q.shape[0],
            }
        r["vs_arm2_training_target"] = shift
        report["positions"][kind] = r
        print(f"[taboo] {kind}: " + json.dumps({a: b for a, b in r.items() if a != "logit_lens"}), flush=True)
        for tag in ("centred", "uncentred"):
            print(f"[taboo] {kind} lens, {tag}: " + json.dumps(
                {o: (x[tag]["vocab_rank_of_secret_word"], x[tag]["rank_among_5_words"], x[tag]["single_prompt_median_vocab_rank"],
                     x[tag]["single_prompt_frac_first_of_5"], x[tag]["top_tokens"][:4]) for o, x in lens.items()}, ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
