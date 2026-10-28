"""How does the head NLA do on Qwen3-8B's own chats, compared with web text? (chat-viewer data, CPU only)

    python scripts/heads_chat_analysis.py --out <json>

Reads the public HF dataset: chats from l24/chat_viewer/<chat>/ (sites.parquet, writes.npz, attn.npz, token_ids.npy,
mtry.parquet when present) and the 4 web documents from l24/std_rl/doc_viewer/ + l24/mtry/doc_viewer/. Every site is
every token × head. Pooled FVE = 1 − Σerr / Σbase with err = 2(1 − cos) for a unit prediction and the per-site base
recovered from the stored RL (cos, FVE): the head's mean AR-training write on web text, as in the viewer.

  groups        FVE (RL / SFT / matryoshka) by data set and, for chats, by token role (template / user / assistant),
                plus the position-matched comparison: assistant tokens at position ≥ 50 vs web text at position ≥ 50
                (training never sampled positions < 50, where web FVE is inflated by heads parked on the first token)
  baselines     the same FVE against a domain-specific baseline: each head's mean unit write over all tokens of the
                OTHER chats (or other web documents), not renormalized
  sink          cos between each chat's first-token write and the web documents' (per head); secondary sinks: key
                positions other than 0 that get > 0.1 mean attention from queries t ≥ 16
  doubt         share of explanations opening with doubt / hedging words at parked (sink ≥ 0.9) vs unparked (< 0.5) sites
  cheap         per head, the previous token's write and the same-document leave-one-out mean as explanation-free
                guesses (heads where each beats the NLA)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_doc_viewer_html import chat_roles  # noqa: E402

REPO = "syvb/head-nla-qwen3-8b"
WEB = ["01_journalism_ocoee_homicide", "02_cs_programming_languages", "03_medical_depression", "04_law_gainsborough_fraud"]
DOUBT = re.compile(r"doubt|hedg|negat|absen|uncertain|skeptic|specul|conditional", re.I)


def get(f: str) -> str:
    return hf_hub_download(REPO, f, repo_type="dataset")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load(kind: str, name: str, tok) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    base = f"l24/chat_viewer/{name}" if kind == "chat" else f"l24/std_rl/doc_viewer/{name}"
    S = pd.read_parquet(get(f"{base}/sites.parquet"), columns=["pos", "head", "first_w", "v_norm", "rl_cos", "rl_fve", "sft_cos", "rl_text"])
    mt_path = f"{base}/mtry.parquet" if kind == "chat" else f"l24/mtry/doc_viewer/{name}/mtry.parquet"
    try:
        S = S.merge(pd.read_parquet(get(mt_path), columns=["pos", "head", "mtry_cos", "mtry_text"]), on=["pos", "head"], validate="one_to_one")
    except Exception:  # noqa: BLE001 — matryoshka outputs not uploaded yet
        S["mtry_cos"], S["mtry_text"] = np.nan, ""
    ids = np.load(get(f"{base}/token_ids.npy"))
    toks = [tok.decode([int(i)]) for i in ids]
    roles = chat_roles(toks) or ["web"] * len(toks)
    S["role"] = [("template" if roles[p].startswith("~") else roles[p]) for p in S.pos]
    S["tok"] = [toks[p] for p in S.pos]
    S["doc"], S["kind"] = name, kind
    S["base"] = 2 * (1 - S.rl_cos) / np.maximum(1 - S.rl_fve, 1e-6)
    W = unit(np.load(get(f"{base}/writes.npz"))["v"].astype(np.float32)).astype(np.float16)   # [T, H, d], float16 to fit in RAM
    A = np.load(get(f"{base}/attn.npz"))["attn"].astype(np.float32)               # [H, T, T]
    return S.sort_values(["head", "pos"]).reset_index(drop=True), W, A


def fve(S: pd.DataFrame, col: str, base: str = "base") -> float | None:
    s = S[S[col].notna()]
    return None if s.empty else float(1 - (2 * (1 - s[col])).sum() / s[base].sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    chats = sorted({f.split("/")[2] for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("l24/chat_viewer/0") and f.endswith("/DONE")})
    data = {(k, n): load(k, n, tok) for k, names in (("web", WEB), ("chat", chats)) for n in names}
    out = {"chats": chats}

    # per-head mean unit write per document (for the domain baselines) and position-0 writes
    means = {key: W.astype(np.float32).mean(0) for key, (_, W, _) in data.items()}   # [H, d]
    frames = []
    for (kind, name), (S, W, A) in data.items():
        others = [means[k] for k in means if k[0] == kind and k[1] != name]
        m = np.mean(others, axis=0)                                                 # [H, d], not renormalized
        Wf = W.astype(np.float32)
        g = Wf[S.pos.values, S["head"].values]
        S["base_domain"] = ((m[S["head"].values] - g) ** 2).sum(1)
        prev = np.vstack([Wf[:1] * np.nan, Wf[:-1]])                                # previous token's write
        S["prev_cos"] = (prev[S.pos.values, S["head"].values] * g).sum(1)
        n_doc = W.shape[0]
        loo = unit((Wf.sum(0)[S["head"].values] - g) / (n_doc - 1))                 # same-document leave-one-out mean
        S["docmean_cos"] = (loo * g).sum(1)
        frames.append(S)
        del Wf, prev
    D = pd.concat(frames, ignore_index=True)
    has_m = D.loc[D.kind == "chat", "mtry_cos"].notna().any()

    def row(S):
        r = {"n_sites": int(len(S)), "rl": fve(S, "rl_cos"), "sft": fve(S, "sft_cos"), "mtry": fve(S, "mtry_cos"),
             "sink": float(S.first_w.mean()), "parked_share": float((S.first_w >= 0.9).mean())}
        r["rl_vs_domain_mean"] = float(1 - (2 * (1 - S.rl_cos)).sum() / S.base_domain.sum())
        if S.mtry_cos.notna().any():
            r["mtry_vs_domain_mean"] = float(1 - (2 * (1 - S.mtry_cos)).sum() / S.base_domain.sum())
        return r
    W_, C_ = D[D.kind == "web"], D[D.kind == "chat"]
    out["groups"] = {"web: all": row(W_), "web: pos ≥ 50": row(W_[W_.pos >= 50]), "web: pos 1–49": row(W_[(W_.pos >= 1) & (W_.pos < 50)]),
                     "chat: all": row(C_), "chat: template": row(C_[C_.role == "template"]), "chat: user": row(C_[C_.role == "user"]),
                     "chat: assistant": row(C_[C_.role == "assistant"]), "chat: assistant, pos ≥ 50": row(C_[(C_.role == "assistant") & (C_.pos >= 50)])}
    out["per_chat"] = {n: row(C_[C_.doc == n]) for n in chats} | {n: row(W_[W_.doc == n]) for n in WEB}
    out["per_chat_assistant_pos50"] = {n: row(C_[(C_.doc == n) & (C_.role == "assistant") & (C_.pos >= 50)]) for n in chats}

    # sinks
    web0 = np.mean([data[("web", n)][1][0].astype(np.float32) for n in WEB], axis=0)                   # [H, d]
    out["sink"] = {"cos_chat_first_token_write_vs_web": {n: float(np.median((unit(data[("chat", n)][1][0].astype(np.float32)) * unit(web0)).sum(1))) for n in chats},
                   "min_over_heads": {n: float(np.min((unit(data[("chat", n)][1][0].astype(np.float32)) * unit(web0)).sum(1))) for n in chats}}
    sec = []
    for (kind, name), (S, W, A) in data.items():
        T = A.shape[1]
        if T <= 16:
            continue
        mean_in = A[:, 16:, :].mean(1)                                              # [H, T]: mean attention received from t ≥ 16
        toks = S.drop_duplicates("pos").set_index("pos").tok
        for h, k in zip(*np.where(mean_in[:, 1:] > 0.1)):
            sec.append({"kind": kind, "doc": name, "head": int(h), "pos": int(k + 1), "tok": toks.get(int(k + 1), "?"), "mean_attention": float(mean_in[h, k + 1])})
    out["secondary_sinks"] = sec

    # doubt default
    def doubt(S, col):
        t = S[col].fillna("").str[:80]
        return {"parked": float(t[S.first_w >= 0.9].str.contains(DOUBT).mean()), "unparked": float(t[S.first_w < 0.5].str.contains(DOUBT).mean())}
    out["doubt"] = {k: {"rl": doubt(S, "rl_text"), **({"mtry": doubt(S, "mtry_text")} if has_m or k == "web" else {})} for k, S in (("web", W_), ("chat", C_))}

    # explanation-free guesses per head
    cheap = {}
    for kind, S in (("web", W_), ("chat", C_)):
        s = S[S.pos > 0]
        r = {}
        for col in ("prev_cos", "docmean_cos"):
            per = s.groupby("head").apply(lambda x: 1 - (2 * (1 - x[col])).sum() / x.base.sum(), include_groups=False)
            rl = s.groupby("head").apply(lambda x: 1 - (2 * (1 - x.rl_cos)).sum() / x.base.sum(), include_groups=False)
            r[col.replace("_cos", "")] = {"fve": float(1 - (2 * (1 - s[col])).sum() / s.base.sum()), "heads_beating_rl": int((per > rl).sum())}
            if s.mtry_cos.notna().any():
                mt = s.groupby("head").apply(lambda x: 1 - (2 * (1 - x.mtry_cos)).sum() / x.base.sum(), include_groups=False)
                r[col.replace("_cos", "")]["heads_beating_mtry"] = int((per > mt).sum())
        cheap[kind] = r
    out["cheap_baselines"] = cheap

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: out[k] for k in ("groups", "sink", "doubt", "cheap_baselines")}, indent=1, ensure_ascii=False, default=float))
    print("secondary sinks:", len(sec), sec[:12])


if __name__ == "__main__":
    main()
