"""SAE-style KL objective for the AR: splice the reconstruction back into the
target model and compare next-token distributions.

    KL(p_orig || p_splice)   at the extraction position t

p_orig is the frozen target model's next-token distribution after the source
prefix x_<=t; p_splice is the same forward with the layer-K residual at t
replaced by the AR's prediction. The AR predicts DIRECTION only (MSE is on
vectors normalised to mse_scale), so the prediction is rescaled to the natural
residual's norm before splicing. Only position t is compared: the datasets
keep the prefix up to t (`detokenized_text_truncated`), not the continuation.

Cost: the prefix x_<t runs once per unique prefix, no grad, into a KV cache.
The last token then runs as Q query copies at the SAME position (copy 0
unspliced -> p_orig, copies 1.. spliced), each attending to the cached prefix
and to itself only (4D mask). So one cached prefix serves every candidate
reconstruction (all G rollouts of a prompt) without duplicating the cache, and
gradients flow through the single spliced position only.
"""

import torch
import torch.nn.functional as F

from nla.utils.arch_adapters import resolve_decoder_layers


def tokenize_prefixes(tokenizer, texts, n_raw_tokens, max_len=None):
    """Token ids of each source prefix, or None where it can't be trusted.

    add_special_tokens=True matches stage0's extraction tokenization. The text
    was stored decoded (skip_special_tokens=True), and decode->encode is not
    guaranteed to round-trip, so a row is kept only if re-encoding gives back
    exactly n_raw_tokens tokens (the extraction position is the last one).
    """
    enc = tokenizer(list(texts), add_special_tokens=True)["input_ids"]
    out = []
    for ids, n in zip(enc, n_raw_tokens):
        ok = (n is not None and len(ids) == int(n) and len(ids) >= 2
              and (max_len is None or len(ids) <= max_len))
        out.append(list(ids) if ok else None)
    return out


class SpliceKL:
    """Frozen target model + a splice hook on the output of decoder layer K
    (= the extractor's `layers[K]` hook = HF hidden_states[K+1])."""

    def __init__(self, model, layer_index, micro_batch=8):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.micro_batch = micro_batch
        self._splice = None   # (vectors [c, Q, d], mask [c, Q] bool) while active
        resolve_decoder_layers(model)[layer_index].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self._splice is None:
            return output
        resid, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        vecs, mask = self._splice
        q = vecs.shape[1]
        h = resid[:, -q:]                                     # the query copies
        v = vecs.to(resid.device).float()
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        v = v * h.detach().float().norm(dim=-1, keepdim=True)  # natural norm
        h = torch.where(mask.to(resid.device)[..., None], v.to(resid.dtype), h)
        resid = torch.cat([resid[:, :-q], h], dim=1)
        return resid if rest is None else (resid, *rest)

    def kl(self, prefixes, vectors):
        """KL(p_orig || p_splice) [N] for vector j spliced after prefixes[j].

        prefixes: N token-id lists (full prefix x_<=t; rows sharing a prefix are
        deduplicated). vectors: [N, d]; grad flows into it if it requires grad.
        """
        n = len(prefixes)
        assert vectors.shape[0] == n
        groups = {}
        for j, ids in enumerate(prefixes):
            groups.setdefault(tuple(ids), []).append(j)
        keys = list(groups)
        out = [None] * n
        dev = self.model.get_input_embeddings().weight.device
        for cs in range(0, len(keys), self.micro_batch):
            chunk = keys[cs:cs + self.micro_batch]
            c = len(chunk)
            q = 1 + max(len(groups[k]) for k in chunk)
            T = max(len(k) for k in chunk)
            # Left-pad so every prefix's last token sits at the same index.
            ids = torch.zeros((c, T), dtype=torch.long, device=dev)
            attn = torch.zeros((c, T), dtype=torch.long, device=dev)
            for r, k in enumerate(chunk):
                ids[r, T - len(k):] = torch.tensor(k, dtype=torch.long, device=dev)
                attn[r, T - len(k):] = 1
            pos = (attn.cumsum(-1) - 1).clamp_min(0)
            with torch.no_grad():
                cache = self.model(
                    input_ids=ids[:, :-1], attention_mask=attn[:, :-1],
                    position_ids=pos[:, :-1], use_cache=True,
                ).past_key_values
            dtype = self.model.get_input_embeddings().weight.dtype
            neg = torch.finfo(dtype).min
            keep = torch.cat([attn[:, :-1].bool()[:, None, :].expand(c, q, T - 1),
                              torch.eye(q, dtype=torch.bool, device=dev).expand(c, q, q)], dim=-1)
            mask4d = torch.zeros(keep.shape, dtype=dtype, device=dev).masked_fill(~keep, neg)[:, None]
            vecs = torch.zeros((c, q, vectors.shape[1]), dtype=vectors.dtype, device=vectors.device)
            smask = torch.zeros((c, q), dtype=torch.bool, device=dev)
            slots = []
            for r, k in enumerate(chunk):
                for s, j in enumerate(groups[k], start=1):
                    slots.append((r, s, j))
                    smask[r, s] = True
            if slots:
                rr, ss, jj = (torch.tensor(x, device=vectors.device) for x in zip(*slots))
                vecs = vecs.index_put((rr, ss), vectors[jj])
            self._splice = (vecs, smask)
            try:
                logits = self.model(
                    input_ids=ids[:, -1:].expand(c, q), attention_mask=mask4d,
                    position_ids=pos[:, -1:].expand(c, q), past_key_values=cache,
                    use_cache=True,
                ).logits.float()
            finally:
                self._splice = None
            logp = F.log_softmax(logits, dim=-1)              # [c, q, V]
            lp_orig = logp[:, :1].detach()
            kl = (lp_orig.exp() * (lp_orig - logp)).sum(-1)  # [c, q]; col 0 ~ 0
            for r, s, j in slots:
                out[j] = kl[r, s]
        return torch.stack(out)


def orthogonal_fail_vectors(golds, mean_dir):
    """Per-row vector orthogonal to the gold activation: the dataset mean
    direction with the gold component projected out. The KL of splicing it is
    the KL analogue of the MSE failure floor (-2.0 = an orthogonal prediction).
    """
    g = golds.float() / golds.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    m = mean_dir.float().to(g.device).flatten()
    m = (m / m.norm().clamp_min(1e-12)).expand_as(g)
    u = m - (m * g).sum(-1, keepdim=True) * g
    assert torch.all(u.norm(dim=-1) > 1e-3), "gold activation parallel to the mean direction"
    return u


# Parquet columns --recon-loss kl needs (stage0 provenance; kept by stage3 with
# --keep-debug-metadata, the default).
PREFIX_COLUMNS = ("detokenized_text_truncated", "n_raw_tokens")


def attach_prefix_ids(rows, tokenizer, tag="data"):
    """Set row["prefix_ids"] from the PREFIX_COLUMNS; drop rows whose prefix
    does not re-tokenize to n_raw_tokens (see tokenize_prefixes)."""
    ids = tokenize_prefixes(tokenizer, [r[PREFIX_COLUMNS[0]] for r in rows],
                            [r[PREFIX_COLUMNS[1]] for r in rows])
    kept = []
    for r, i in zip(rows, ids):
        if i is not None:
            r["prefix_ids"] = i
            kept.append(r)
    print(f"[kl] {tag}: {len(kept)}/{len(rows)} rows have a prefix that re-tokenizes "
          f"to n_raw_tokens ({len(rows) - len(kept)} dropped)", flush=True)
    assert kept, f"[kl] {tag}: no usable prefixes"
    return kept


def load_splice_kl(target_ckpt, sidecar_source, layer_index, device, micro_batch=8,
                   dtype=torch.bfloat16):
    """Frozen target model (bf16 by default) wrapped in SpliceKL. target_ckpt=None -> the
    sidecar's extraction.base_model (the model the activations came from)."""
    if target_ckpt is None:
        import yaml
        from nla.schema import sidecar_path_for
        meta = yaml.safe_load(sidecar_path_for(sidecar_source).read_text())
        target_ckpt = (meta.get("extraction") or {}).get("base_model")
        assert target_ckpt, "sidecar has no extraction.base_model; pass --kl-target-ckpt"
    assert layer_index is not None, "sidecar has no extraction.layer_index"
    from transformers import AutoModelForCausalLM
    print(f"[kl] target {target_ckpt}, splice at the output of layer {layer_index}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    return SpliceKL(model, layer_index, micro_batch=micro_batch)
