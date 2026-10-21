"""Scan a directory (e.g. a local snapshot of an HF repo) for secrets before publishing it.

    python scripts/hf_secret_scan.py <dir> [--secret-file ~/.hf_token ...] [--runpod-config ~/.runpod/config.toml]

Two checks, over every text file and every string column of every parquet:
- EXACT: the actual secret values (read from the given files; never printed) and 16-character windows of them;
- PATTERNS: HF / Anthropic / RunPod key shapes, private-key blocks, AWS keys, bearer tokens, `key = value`
  secrets, plus things worth a look before going public (40-hex strings, IPv4 addresses, emails, /home paths).
Binary arrays (.npy/.npz/.safetensors/.pt) are skipped: check safetensors headers and pickles separately.
"""

import argparse
import collections
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
import tomllib

PAT = {"hf_token": r"\bhf_[A-Za-z0-9]{30,}", "anthropic": r"sk-ant-[A-Za-z0-9_\-]{20,}", "runpod": r"\brpa_[A-Za-z0-9]{20,}",
       "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "aws": r"\bAKIA[0-9A-Z]{16}\b",
       "bearer": r"Bearer\s+[A-Za-z0-9._\-]{20,}",
       "kv_secret": r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}",
       "hex40": r"(?<![0-9a-fA-F])[0-9a-f]{40}(?![0-9a-fA-F])", "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
       "email": r"[\w.+-]+@[\w-]+\.[\w.]+", "home_path": r"/home/\w+"}
SENSITIVE = {"hf_token", "anthropic", "runpod", "bearer", "kv_secret"}
SKIP = {".npy", ".npz", ".safetensors", ".pt", ".bin"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--secret-file", action="append",
                   default=["~/.hf_token", "~/.anthropic_key", "~/.wandb_key", "~/.runpod/ssh/runpodctl-ssh-key"])
    p.add_argument("--runpod-config", default="~/.runpod/config.toml")
    args = p.parse_args()
    secrets = {}
    for f in args.secret_file:
        f = Path(os.path.expanduser(f))
        if f.exists():
            v = f.read_text().strip()
            if "PRIVATE KEY" in v:     # ssh key: its base64 lines
                for i, line in enumerate(x for x in v.splitlines() if x and not x.startswith("-----")):
                    secrets[f"{f.name}:line{i}"] = line
            else:
                secrets[f.name] = v
    rc = Path(os.path.expanduser(args.runpod_config))
    if rc.exists():
        cfg = {k.lower(): v for k, v in tomllib.loads(rc.read_text()).items()}
        if cfg.get("apikey"):
            secrets["runpod_apikey"] = cfg["apikey"]
    windows = {n: {v[i:i + 16] for i in range(0, max(1, len(v) - 15), 4)} for n, v in secrets.items() if len(v) >= 16}
    hits, examples, exact = collections.defaultdict(collections.Counter), collections.defaultdict(set), []

    def scan(text: str, where: str) -> None:
        for n, v in secrets.items():
            if v in text:
                exact.append((n, where))
        for n, ws in windows.items():
            if any(w in text for w in ws):
                exact.append((n + " (partial)", where))
        for n, pat in PAT.items():
            for m in re.findall(pat, text):
                m = m if isinstance(m, str) else m[0]
                hits[n][where] += 1
                if len(examples[n]) < 12:
                    examples[n].add(m[:12] + "…" if n in SENSITIVE else m[:60])

    root, n_files = Path(args.root), 0
    for f in root.rglob("*"):
        if not f.is_file() or "/.cache/" in str(f) or f.suffix in SKIP:
            continue
        n_files += 1
        rel = str(f.relative_to(root))
        if f.suffix == ".parquet":
            t = pq.read_table(f)
            for col in t.column_names:
                if "string" in str(t.schema.field(col).type):
                    scan("\n".join(json.dumps(v, ensure_ascii=False) for v in t.column(col).to_pylist()), f"{rel}:{col}")
        else:
            scan(f.read_bytes().decode("utf-8", "ignore"), rel)
    print(f"files scanned: {n_files}; secrets checked: {len(secrets)}")
    print("EXACT/PARTIAL SECRET MATCHES:", exact if exact else "none")
    for n in PAT:
        if hits[n]:
            print(f"[{n}] {sum(hits[n].values())} hits in {len(hits[n])} files; e.g. {list(hits[n])[:4]}; examples {sorted(examples[n])[:8]}")


if __name__ == "__main__":
    main()
