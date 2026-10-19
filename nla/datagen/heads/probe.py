"""Cross-fitted description → write probe: the recipe's warm-start filter, and the
yardstick for comparing description sources.

Per head. Targets are write deviations in an orthonormal basis of the head's write
subspace: with W_h = U S Vᵀ, c = S Vᵀ (z − z̄), so ‖c − ĉ‖ is exactly the residual-space
error ‖W_h (z − ẑ)‖. Features are either TF-IDF word uni/bigrams (fit inside each fold)
or a frozen LM's mean-pooled embedding of the description. Ridge with α from leave-one-
out CV; two folds — fit on one half, score the other, swap. FVE is against the held-out
targets' own mean, so an uninformative description scores ≈ 0 (slightly below).

Per-example held-out squared errors come back too: the filter drops the worst.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ALPHAS = np.logspace(-2, 4, 13)


def write_coords(z_dev: torch.Tensor, w_h: torch.Tensor) -> np.ndarray:
    """[N, D] head-output deviations → [N, D] coordinates whose Euclidean geometry is the
    write's (w_h: [d_model, D], this head's o_proj slice)."""
    _, s, vh = torch.linalg.svd(w_h.float(), full_matrices=False)
    return (z_dev.float() @ (s[:, None] * vh).T).numpy()


def _tfidf_model() -> Any:
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"),
                         RidgeCV(alphas=_ALPHAS))


def _dense_model() -> Any:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=_ALPHAS))


def crossfit(features: list[str] | np.ndarray, y: np.ndarray, n_folds: int = 2,
             seed: int = 0) -> tuple[float, np.ndarray]:
    """Held-out FVE and per-example squared error. `features` is a list of texts
    (TF-IDF) or an [N, F] array (dense)."""
    n = len(y)
    err = np.empty(n)
    is_text = isinstance(features, list)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(n)):
        if is_text:
            x_tr = [features[i] for i in train]
            x_te = [features[i] for i in test]
            model: Any = _tfidf_model()
        else:
            x_tr, x_te = features[train], features[test]
            model = _dense_model()
        try:
            model.fit(x_tr, y[train])
            pred = model.predict(x_te)
        except ValueError:      # e.g. empty vocabulary — no usable text at all
            pred = np.broadcast_to(y[train].mean(0), y[test].shape)
        err[test] = ((pred - y[test]) ** 2).sum(-1)
    total = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - err.sum() / max(total, 1e-12)), err


def lm_embedder(model_name: str, device_map: str = "auto", dtype: str = "bfloat16",
                batch_size: int = 32) -> Callable[[list[str]], np.ndarray]:
    """Mean-pooled last-layer hidden states of `model_name` (empty text → zeros)."""
    from transformers import AutoModel

    from nla.datagen._common import load_tokenizer
    tok = load_tokenizer(model_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, device_map=device_map, dtype=getattr(torch, dtype)).eval()
    device = model.get_input_embeddings().weight.device

    @torch.no_grad()
    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
        nonempty = [i for i, t in enumerate(texts) if t.strip()]
        for s in range(0, len(nonempty), batch_size):
            idx = nonempty[s : s + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        return out

    return embed
