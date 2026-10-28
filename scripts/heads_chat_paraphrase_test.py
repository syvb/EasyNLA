"""Does the head NLA's reconstruction on chats survive a Haiku 5.5 paraphrase? (the private-code test, on the chat data)

Same question and edits as scripts/heads_private_code_test.py, on the chat viewer's sites: if the AR reads a private
code (exact wording, markup, quotes, odd tokens) rather than the claims, a paraphrase that keeps the claims loses
much of the gain, and a form-only skeleton keeps some. Earlier result on web text (RL pair): the paraphrase kept 97%
of the gain over an empty explanation, the skeleton 2%.

Sites: 1,000 random (head, token) sites of the 4 chats (first token excluded; seeded). Per site, for the RL AV's and
the matryoshka AV's explanation:
  <av>_orig       the explanation as scored in the viewer
  <av>_para       paraphrased by Haiku 5.5 with the earlier test's prompt (same claims, new wording, no markup)
  <av>_skeleton   every content word replaced by "thing"; stopwords, punctuation, markup and layout kept
  empty           no explanation (the AR sees only the head name)
Readers: the RL AR (std iter_000400 critic) and the matryoshka AR (iter_000640 critic), on every variant.
Retention = (variant − empty) / (orig − empty) in mean cos, for each explanation read by its own AR.

    python scripts/heads_chat_paraphrase_test.py prep --chats <dir with the 4 chat folders> [--chats ...] --data chat_paraphrase_data
    python scripts/heads_chat_paraphrase_test.py score --data chat_paraphrase_data --rl-critic <dir> --mtry-critic <dir> --out <dir>
    python scripts/heads_chat_paraphrase_test.py report --data chat_paraphrase_data --results <dir>
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads_private_code_test import LAYER, SCALE, paraphrase_all, skeleton, stopwords  # noqa: E402

AVS = ("rl", "mtry")


def prep(a) -> None:
    rng = random.Random(a.seed)
    sw = stopwords()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    chats = sorted(d for c in a.chats for d in Path(c).iterdir() if (d / "mtry.parquet").exists())
    frames, W = [], {}
    for d in chats:
        s = pd.read_parquet(d / "sites.parquet", columns=["pos", "head", "rl_text", "rl_cos", "rl_fve"])
        s = s.merge(pd.read_parquet(d / "mtry.parquet", columns=["pos", "head", "mtry_text", "mtry_cos"]), on=["pos", "head"], validate="one_to_one")
        frames.append(s.assign(chat=d.name))
        W[d.name] = np.load(d / "writes.npz")["v"]
    allsites = pd.concat(frames, ignore_index=True)
    pool = allsites[allsites.pos > 0].index.tolist()
    sites = allsites.loc[sorted(rng.sample(pool, a.n_sites))].reset_index(drop=True)
    sites.insert(0, "site", range(len(sites)))
    np.save(data / "targets.npy", np.stack([W[c][p, h] for c, p, h in zip(sites.chat, sites.pos, sites["head"])]).astype(np.float16))

    cache = data / "paraphrases.jsonl"
    want = {f"{av}:{r.site}": getattr(r, f"{av}_text") for r in sites.itertuples() for av in AVS}
    have = {json.loads(x)["id"] for x in cache.read_text().splitlines()} if cache.exists() else set()
    todo = {k: v for k, v in want.items() if k not in have}
    if todo:
        key = Path(a.key_file).expanduser().read_text().strip()
        asyncio.run(paraphrase_all(todo, cache, key, a.concurrency))
    para = {json.loads(x)["id"]: json.loads(x)["text"] for x in cache.read_text().splitlines()}
    rows = []
    for r in sites.itertuples():
        rows.append(dict(site=r.site, head=r.head, variant="empty", text=""))
        for av in AVS:
            t = getattr(r, f"{av}_text")
            for k, v in ((f"{av}_orig", t), (f"{av}_para", para.get(f"{av}:{r.site}")), (f"{av}_skeleton", skeleton(t, sw))):
                if v is not None:
                    rows.append(dict(site=r.site, head=r.head, variant=k, text=v))
    V = pd.DataFrame(rows)
    sites.to_parquet(data / "sites.parquet")
    V.to_parquet(data / "variants.parquet")
    print(f"{len(sites)} sites from {sites.chat.nunique()} chats, {len(V)} variant texts → {data}")
    for k in ("rl_orig", "rl_para", "mtry_orig", "mtry_para", "mtry_skeleton"):
        print(f"--- {k}: {V[V.variant == k].text.iloc[5][:260]!r}")


def score(a) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from heads_sft_checks import ar_predict
    from show_nla_generations import load_ar_critic
    from transformers import AutoTokenizer

    from nla.datagen.heads.export_sft import critic_explanation
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites, V = pd.read_parquet(data / "sites.parquet"), pd.read_parquet(data / "variants.parquet")
    tgt = np.load(data / "targets.npy").astype(np.float32)
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    tok = AutoTokenizer.from_pretrained(a.base)

    def run(critic, heads, texts) -> np.ndarray:
        order = np.argsort([len(t) for t in texts], kind="stable")
        ins = [critic_explanation(LAYER, int(heads[i]), texts[i]) for i in order]
        with torch.no_grad():
            p = ar_predict(critic, tok, ins, SCALE, a.device, a.batch_size)
        pred = np.empty_like(p)
        pred[order] = p / np.linalg.norm(p, axis=1, keepdims=True)
        return pred

    for reader, path in (("rl", a.rl_critic), ("mtry", a.mtry_critic)):
        critic = load_ar_critic(path, a.base, a.device).eval()
        V[f"cos_{reader}"] = (run(critic, V["head"].to_numpy(), V.text.tolist()) * tgt[V.site.to_numpy()]).sum(1)
        own = V[V.variant == f"{reader}_orig"].merge(sites[["site", f"{reader}_cos"]], on="site")
        print(f"[{reader}] reproduces the viewer's stored scores for its own text: {own[f'cos_{reader}'].mean():.4f} vs "
              f"{own[f'{reader}_cos'].mean():.4f} (corr {np.corrcoef(own[f'cos_{reader}'], own[f'{reader}_cos'])[0, 1]:.4f})", flush=True)
        del critic
        torch.cuda.empty_cache()
    V.drop(columns="text").to_parquet(out / "scores.parquet")
    (out / "DONE").write_text("ok\n")


def report(a) -> None:
    data, res = Path(a.data), Path(a.results)
    sites = pd.read_parquet(data / "sites.parquet")
    V = pd.read_parquet(res / "scores.parquet").merge(sites[["site", "rl_cos", "rl_fve"]], on="site")
    full = V.groupby("site").variant.nunique()
    V = V[V.site.isin(full[full == full.max()].index)]          # sites where both paraphrases came back
    base = 2 * (1 - V.rl_cos) / (1 - V.rl_fve)                  # per-site FVE denominator, from the viewer
    for r in AVS:
        V[f"fve_{r}"] = 1 - 2 * (1 - V[f"cos_{r}"]) / base
    order = ["rl_orig", "rl_para", "rl_skeleton", "mtry_orig", "mtry_para", "mtry_skeleton", "empty"]
    T = V.groupby("variant")[[f"{m}_{r}" for r in AVS for m in ("cos", "fve")]].mean().reindex(order)
    rep = {"n_sites": int(V.site.nunique()), "table": T.round(4).to_dict("index"), "retention_own_reader": {}}
    for av in AVS:
        lo, hi = T.loc["empty", f"cos_{av}"], T.loc[f"{av}_orig", f"cos_{av}"]
        rep["retention_own_reader"][av] = {k: round(float((T.loc[f"{av}_{k}", f"cos_{av}"] - lo) / (hi - lo)), 3) for k in ("para", "skeleton")}
        P = V[V.variant.isin([f"{av}_orig", f"{av}_para"])].pivot_table(index=["site", "head"], columns="variant", values=f"cos_{av}").reset_index()
        for c in (f"{av}_orig", f"{av}_para"):
            P[c] = P[c] - P.groupby("head")[c].transform("mean")
        rep.setdefault("site_level_corr_orig_vs_para", {})[av] = round(float(np.corrcoef(P[f"{av}_orig"], P[f"{av}_para"])[0, 1]), 3)
    rep["fve_drop_para"] = {av: round(float(T.loc[f"{av}_orig", f"fve_{av}"] - T.loc[f"{av}_para", f"fve_{av}"]), 4) for av in AVS}
    (res / "report.json").write_text(json.dumps(rep, indent=1))
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())
    print(json.dumps({k: v for k, v in rep.items() if k != "table"}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--chats", action="append", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--n-sites", type=int, default=1000)
    q.add_argument("--concurrency", type=int, default=32)
    q.add_argument("--key-file", default="~/.anthropic_key")
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("score")
    q.add_argument("--data", required=True)
    q.add_argument("--rl-critic", required=True)
    q.add_argument("--mtry-critic", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--base", default="Qwen/Qwen3-8B")
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--device", default="cuda:0")
    q = sub.add_parser("report")
    q.add_argument("--data", required=True)
    q.add_argument("--results", required=True)
    a = p.parse_args()
    {"prep": prep, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
