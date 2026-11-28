"""Anatomy of the massive activations at Layer K: which tokens, which coordinates, how much of each objective.

On the web tokens Phase 0 fits its alignment to, for the Target and the Base model:

  1. the norm distribution (how separated the massive tokens are from the rest),
  2. which tokens and sequence positions they are, and whether the two models fire on the same ones,
  3. which residual coordinates carry them, and how constant the vectors are,
  4. how much of the alignment objective (the squared residual h_target - h_base) those rows carry.

On the chat corpus, it scans every position of every conversation for the same thing, to see whether
the rows Phase 2 samples can hit one.

    python massive_anatomy.py --phase0-dir P0 --rollouts P2/rollouts.parquet --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

from nla.diff.benchmarks import load_fineweb_passages
from nla.diff.wrappers import encode_aligned_pair

BINS = (0.0, 0.5, 2.0, 5.0, 20.0, float("inf"))  # norm / median


def load_web_rows(cfg: dict, tok: AutoTokenizer, *, n_train: int = 1600, n_val: int = 400, max_tokens: int = 64) -> dict:
    """The web tokens Phase 0 fits and validates its alignment on (positions > 0), from the activation cache."""
    k = int(cfg["layer_index"])
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    docs = load_fineweb_passages(n_passages=n_train + n_val)
    out: dict = {}
    for split, ds in (("train", docs[:n_train]), ("val", docs[n_train : n_train + n_val])):
        seqs, pos, doc_ids, doc_idx, tok_pos, tok_ids = [], [], [], [], [], []
        for d in ds:
            ids = tok.encode(d["text"], add_special_tokens=False)[:max_tokens]
            if len(ids) < 2:
                continue
            doc_idx.extend([len(seqs)] * (len(ids) - 1))
            seqs.append(ids)
            pos.append(list(range(1, len(ids))))
            doc_ids.extend([d["doc_id"]] * (len(ids) - 1))
            tok_pos.extend(range(1, len(ids)))
            tok_ids.extend(ids[1:])
        h = {}
        for name, key in ((cfg["target_model"], "ht"), (cfg["base_model"], "hb")):
            m = cached_io.LazyLM(name)
            h[key] = torch.cat([e.hidden for e in cached_io.cached_extract(m, seqs, pos, layer_index=k, pad_token_id=pad)])
            m.free()
        out[split] = {"seqs": seqs, "doc_ids": doc_ids, "doc_idx": np.asarray(doc_idx), "pos": np.asarray(tok_pos),
                      "tok": np.asarray(tok_ids), **h}
    return out


def massive_mask(h: torch.Tensor, mult: float = 5.0) -> torch.Tensor:
    n = h.norm(dim=1)
    return n > mult * n.median()


def describe(h: torch.Tensor, m: torch.Tensor) -> dict:
    """Norms and coordinates of the massive rows `m` of `h`."""
    n = h.norm(dim=1)
    med = float(n.median())
    hist = torch.histogram(n / med, bins=torch.tensor(BINS[:-1] + (1e9,))).hist.long().tolist()
    base = {"median_norm": med, "count_by_norm_over_median": {f"{lo:g} to {hi:g}": c for lo, hi, c in zip(BINS[:-1], BINS[1:], hist)},
            "n_massive": int(m.sum()), "largest_norm_over_median": float(n.max() / med)}
    if not m.any():
        return base
    hm = h[m].double()
    energy = (hm**2).sum(0)
    top = torch.argsort(energy, descending=True)[:4]
    u = hm / hm.norm(dim=1, keepdim=True)
    cos = u @ u.T
    off = cos[~torch.eye(len(u), dtype=torch.bool)]
    c0 = int(top[0])
    return {
        **base,
        "massive_norm_min_median_max": [float(n[m].min()), float(n[m].median()), float(n[m].max())],
        "largest_ordinary_norm_over_median": float(n[~m].max() / med),
        "smallest_massive_norm_over_median": float(n[m].min() / med),
        "top_coordinates_share_of_massive_energy": {int(c): float(energy[c] / energy.sum()) for c in top},
        "pairwise_cosine_between_massive_rows_min_mean": [float(off.min()), float(off.mean())] if len(off) else None,
        "top_coordinate_value_on_massive_rows_mean_std": [float(hm[:, c0].mean()), float(hm[:, c0].std()) if len(hm) > 1 else 0.0],
        "top_coordinate_value_on_ordinary_rows_mean_std": [float(h[~m][:, c0].double().mean()), float(h[~m][:, c0].double().std())],
        "share_of_centred_variance_in_massive_rows": float(((hm - h.double().mean(0)) ** 2).sum() / ((h.double() - h.double().mean(0)) ** 2).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-dir", required=True)
    ap.add_argument("--rollouts", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((Path(args.phase0_dir) / "phase0_config.json").read_text())
    k = int(cfg["layer_index"])
    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    web = load_web_rows(cfg, tok)
    report: dict = {"layer": k, "web": {}}

    for split, w in web.items():
        ht, hb = w["ht"], w["hb"]
        mt, mb = massive_mask(ht), massive_mask(hb)
        either = mt | mb
        r = {"n_tokens": len(ht), "target": describe(ht, mt), "base": describe(hb, mb),
             "fires": {"both": int((mt & mb).sum()), "target_only": int((mt & ~mb).sum()), "base_only": int((mb & ~mt).sum())}}
        if not either.any():
            report["web"][split] = r
            print(f"[massive] web {split}: " + json.dumps(r, ensure_ascii=False), flush=True)
            continue
        # Direction shared across models?
        ut, ub = ht[mt].double().mean(0), hb[mb].double().mean(0)
        r["cosine_target_vs_base_mean_massive_vector"] = float(ut @ ub / (ut.norm() * ub.norm()))
        # Which tokens and where.
        idx = torch.where(either)[0].numpy()
        r["tokens"] = Counter(tok.decode([int(t)]) for t in w["tok"][idx]).most_common(12)
        r["position_in_sequence_min_median_max"] = [int(w["pos"][idx].min()), float(np.median(w["pos"][idx])), int(w["pos"][idx].max())]
        r["documents_with_a_massive_token"] = len(np.unique(w["doc_idx"][idx]))
        r["n_documents"] = len(w["seqs"])
        first = 0
        for i in idx:
            seq = w["seqs"][int(w["doc_idx"][i])]
            first += int(seq.index(int(w["tok"][i])) == int(w["pos"][i]))
        r["share_that_are_the_first_occurrence_of_their_token_in_the_document"] = first / len(idx)
        # Same token id at ordinary rows: how often does that token fire at all?
        fire_rate = {}
        for t, _ in Counter(w["tok"][idx].tolist()).most_common(5):
            sel = w["tok"] == t
            fire_rate[tok.decode([int(t)])] = [int(either.numpy()[sel].sum()), int(sel.sum())]
        r["massive_count_over_occurrences_for_top_tokens"] = fire_rate
        # Share of the id-anchored objective (target: h_target - h_base) carried by these rows.
        res = (ht - hb).double()
        res_c = res - res.mean(0)
        r["residual"] = {
            "share_of_centred_squared_residual_in_massive_rows": float((res_c[either] ** 2).sum() / (res_c**2).sum()),
            "median_residual_norm_ordinary": float(res[~either].norm(dim=1).median()),
            "median_residual_norm_both_fire": float(res[mt & mb].norm(dim=1).median()) if (mt & mb).any() else None,
            "median_residual_norm_one_fires": float(res[mt ^ mb].norm(dim=1).median()) if (mt ^ mb).any() else None,
        }
        # Threshold sensitivity of the mask.
        nt, nb = ht.norm(dim=1), hb.norm(dim=1)
        r["n_masked_by_threshold"] = {str(c): int(((nt > c * nt.median()) | (nb > c * nb.median())).sum()) for c in (2, 3, 5, 10, 20, 40)}
        report["web"][split] = r
        print(f"[massive] web {split}: " + json.dumps(r, ensure_ascii=False), flush=True)
    del web

    if args.rollouts:
        tbl = pq.read_table(args.rollouts).to_pylist()
        chat: dict = {}
        plans = []
        for row in tbl:
            kw = {"history": json.loads(row["history_json"]), "user_ids": row.get("prompt_user_ids"),
                  "rollout_ids": row["rollout_token_ids"], "enable_thinking": False, "max_total_tokens": None,
                  "truncate_history": False}
            ie, _ = encode_aligned_pair(tok, row["user_text"], base_wrapper="chatml", **kw)
            _, bp = encode_aligned_pair(tok, row["user_text"], base_wrapper="plain", **kw)
            plans.append((ie, bp))
        for name, model, which in (("target_chatml", cfg["target_model"], 0), ("base_chatml", cfg["base_model"], 0),
                                   ("base_plain", cfg["base_model"], 1)):
            h16 = cached_io._dirs(cached_io.LazyLM(model).tag, k)[0]
            norms, where = [], []
            for p in plans:
                e = p[which]
                h = np.load(h16 / f"{cached_io.seq_key(e.input_ids)}.npy", mmap_mode="r")
                assert h.shape[0] == len(e.input_ids)
                norms.append(np.linalg.norm(np.asarray(h, dtype=np.float32), axis=1))
                sampled = set(e.user_content_positions) | {e.asst_last_idx} | set(e.asst_rollout_positions)
                where.append([(i, e.input_ids[i], i in sampled) for i in range(len(e.input_ids))])
            flat = np.concatenate(norms)
            med = float(np.median(flat))
            hits = [(i, t, s) for n, w in zip(norms, where) for (i, t, s), v in zip(w, n) if v > 5 * med]
            chat[name] = {
                "n_tokens": len(flat), "median_norm": med, "n_massive": len(hits),
                "n_massive_at_position_0": sum(1 for i, _, _ in hits if i == 0),
                "n_massive_at_a_position_phase2_samples": sum(1 for _, _, s in hits if s),
                "massive_after_position_0": Counter(f"pos {i}: {tok.decode([int(t)])!r}" for i, t, _ in hits if i > 0).most_common(8),
                "largest_norm_over_median_after_position_0": float(max(n[1:].max() for n in norms) / med),
            }
            print(f"[massive] chat {name}: " + json.dumps(chat[name], ensure_ascii=False), flush=True)
        report["chat"] = chat

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
