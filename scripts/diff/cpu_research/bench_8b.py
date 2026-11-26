"""Time the 8B bottom half on CPU and report peak RSS."""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cached_io

name, layer = sys.argv[1], int(sys.argv[2])
t0 = time.time()
part = cached_io.load_partial(name, 0, layer + 1, embed=True, head=False)
print(f"load {time.time() - t0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
g = torch.Generator().manual_seed(0)
for threads in (8, 16):
    torch.set_num_threads(threads)
    for bs, L in ((16, 64), (32, 64), (8, 256)):
        ids = torch.randint(1000, 100000, (bs, L), generator=g)
        mask = torch.ones_like(ids)
        t = time.time()
        with torch.no_grad():
            part.model(input_ids=ids, attention_mask=mask, use_cache=False)
        dt = time.time() - t
        print(f"threads={threads} bs={bs} L={L}: {bs * L / dt:.1f} tok/s ({dt:.1f}s)", flush=True)
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
