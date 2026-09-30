"""Gather-then-SDPA attention over the paged cache.

This is the correctness-first path: pages are packed into a contiguous
K/V tensor and handed to PyTorch SDPA. A later kernel can walk the
block table directly; logits must match this reference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import EngineConfig
from .kv_cache import PagedKVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: [B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return (
        x[:, :, None]
        .expand(b, h, n_rep, s, d)
        .reshape(b, h * n_rep, s, d)
    )


def paged_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer: int,
    slot: int,
    *,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Args:
        q: [B, H_q, T_q, D]  T_q=1 for decode, T_q=S for prefill-without-cache
        cache / layer / slot: which pages to read
        causal: only meaningful when T_q > 1 and T_q == seq_len
    Returns:
        attn output [B, H_q, T_q, D]
    """
    if q.dim() != 4:
        raise ValueError(f"q must be [B, H, T, D], got {tuple(q.shape)}")

    k, v = cache.gather(layer, slot)  # [1, H_kv, S, D]
    n_rep = cache.cfg.num_qo_heads // cache.cfg.num_kv_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    if q.shape[0] != 1:
        k = k.expand(q.shape[0], -1, -1, -1)
        v = v.expand(q.shape[0], -1, -1, -1)

    scale = sm_scale if sm_scale is not None else (q.shape[-1] ** -0.5)
    # SDPA uses is_causal only when Q/K lengths match. Decode (T_q=1, S>1)
    # is already "attend to all cached tokens".
    use_causal = bool(causal and q.shape[2] == k.shape[2])
    return F.scaled_dot_product_attention(
        q.to(k.dtype),
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=use_causal,
        scale=scale,
    )


def contiguous_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: EngineConfig,
    *,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference attention against a dense cache. k,v: [B, H_kv, S, D]."""
    n_rep = cfg.num_qo_heads // cfg.num_kv_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    scale = sm_scale if sm_scale is not None else (q.shape[-1] ** -0.5)
    use_causal = bool(causal and q.shape[2] == k.shape[2])
    return F.scaled_dot_product_attention(
        q.to(k.dtype),
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=use_causal,
        scale=scale,
    )
