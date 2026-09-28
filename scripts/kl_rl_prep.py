"""RL data for the KL-NLA pilot: an RL parquet minus the audit documents.

Drops every row whose doc_id is one of the audit rows' documents (the fvecmp
explanations.json set), so the final checkpoint curves are scored on documents RL
never trained on. Keeps the columns --recon-loss kl needs and copies the sidecar.

    python scripts/kl_rl_prep.py --src rl_full.parquet --explanations explanations.json --out rl_pilot.parquet
"""
import argparse
import json
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--explanations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    held = sorted(set(json.load(open(a.explanations))["doc_id"]))
    t = pq.read_table(a.src)
    for c in ("prompt", "activation_vector", "doc_id", "detokenized_text_truncated", "n_raw_tokens"):
        assert c in t.column_names, f"{a.src} lacks {c}"
    keep = pc.invert(pc.is_in(t["doc_id"], value_set=pa.array(held)))
    out = t.filter(keep)
    pq.write_table(out, a.out, row_group_size=4096)
    shutil.copy2(a.src + ".nla_meta.yaml", a.out + ".nla_meta.yaml")
    print(f"[rl_prep] {t.num_rows} rows -> {out.num_rows} ({t.num_rows - out.num_rows} rows from "
          f"{len(held)} audit docs dropped) -> {a.out}")


if __name__ == "__main__":
    main()
