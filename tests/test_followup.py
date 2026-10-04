"""Tests for follow-up turns. CPU only, no model download.

  * a chunk of new tokens after cached ones must be causal (offset mask)
  * truncate() keeps the kept prefix, frees the pages it should, and in int8
    does not let removed tokens poison the next write

Run:
    python tests/test_followup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.attention import contiguous_attention, paged_attention
from engine.config import EngineConfig
from engine.kv_cache import PagedKVCache


def _cfg(kv_dtype: str) -> EngineConfig:
    return EngineConfig(
        num_layers=2,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        page_size=4,
        max_batch_size=1,
        max_seq_len=32,
        num_pages=16,
        kv_dtype=kv_dtype,
        compute_dtype="float32",
        device="cpu",
    )


def _write(cache, slot, k, v, lo, hi) -> None:
    cache.write_prefill(slot, list(k[:, :, lo:hi]), list(v[:, :, lo:hi]))


def _pages(cache, slot) -> set:
    return {p for p in cache._table_cpu[slot] if p >= 0}


def test_chunk_after_cache_is_causal() -> None:
    """10 tokens cached, then a 6-token follow-up chunk. Row i of the chunk must
    only see keys up to its own position, exactly like a full causal pass."""
    cfg = _cfg("float32")
    cache = PagedKVCache(cfg)
    g = torch.Generator().manual_seed(1)
    shape = (cfg.num_layers, cfg.num_kv_heads, 16, cfg.head_dim)
    k, v = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    q_full = torch.randn(1, cfg.num_qo_heads, 16, cfg.head_dim, generator=g)

    slot = cache.allocate_seq()
    _write(cache, slot, k, v, 0, 10)                       # first turn, committed
    cache.write(0, slot, k[0, :, 10:], v[0, :, 10:])       # follow-up chunk, layer 0

    out = paged_attention(q_full[:, :, 10:], cache, 0, slot, causal=True)
    ref = contiguous_attention(
        q_full, k[0].unsqueeze(0), v[0].unsqueeze(0), cfg, causal=True
    )[:, :, 10:]
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"chunk attention differs from full causal pass: {err:.3g}"
    print(f"  [chunk causal] follow-up chunk matches full causal attention (err {err:.1e})")


def test_truncate(kv_dtype: str) -> None:
    cfg = _cfg(kv_dtype)
    cache = PagedKVCache(cfg)
    tol = 1e-6 if kv_dtype == "float32" else 0.1

    g = torch.Generator().manual_seed(3)
    shape = (cfg.num_layers, cfg.num_kv_heads, 12, cfg.head_dim)
    k, v = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    k[:, :, 6:] *= 100  # tokens 6..11 get huge values on purpose
    v[:, :, 6:] *= 100

    slot = cache.allocate_seq()
    _write(cache, slot, k, v, 0, 12)
    assert len(_pages(cache, slot)) == 3
    free_before = cache.allocator.num_free

    cache.truncate(slot, 6)
    assert int(cache.seq_len[slot]) == 6
    assert len(_pages(cache, slot)) == 2, "page 2 should have been freed"
    assert cache.allocator.num_free == free_before + 1
    assert int(cache.block_table[slot, 2]) == -1

    # Append one fresh token (position 6) into the kept page, over stale slots.
    kn = torch.randn((cfg.num_layers, cfg.num_kv_heads, 1, cfg.head_dim), generator=g)
    vn = torch.randn((cfg.num_layers, cfg.num_kv_heads, 1, cfg.head_dim), generator=g)
    _write(cache, slot, kn, vn, 0, 1)
    assert int(cache.seq_len[slot]) == 7

    for layer in range(cfg.num_layers):
        gk, gv = cache.gather(layer, slot)
        assert gk.shape == (1, cfg.num_kv_heads, 7, cfg.head_dim)
        # the new token must be accurate: stale tokens 6,7 must not inflate the scale
        err_new = max(
            (gk[0, :, 6:7] - kn[layer]).abs().max().item(),
            (gv[0, :, 6:7] - vn[layer]).abs().max().item(),
        )
        assert err_new <= tol, f"{kv_dtype} layer {layer}: new token error {err_new:.4g} > {tol}"
        # tokens 0..3 sit in an untouched page and must be unchanged
        err_old = max(
            (gk[0, :, :4] - k[layer, :, :4]).abs().max().item(),
            (gv[0, :, :4] - v[layer, :, :4]).abs().max().item(),
        )
        assert err_old <= tol, f"{kv_dtype} layer {layer}: kept tokens changed {err_old:.4g}"
        if kv_dtype == "float32":
            # in float32 the kept tokens 4,5 are exact too
            assert (gk[0, :, 4:6] - k[layer, :, 4:6]).abs().max().item() <= tol

    cache.free_seq(slot)
    assert cache.allocator.num_free == cfg.num_pages
    print(f"  [truncate {kv_dtype:7s}] kept prefix intact, page freed, new token accurate")


def test_truncate_edges() -> None:
    cfg = _cfg("float32")
    cache = PagedKVCache(cfg)
    g = torch.Generator().manual_seed(4)
    shape = (cfg.num_layers, cfg.num_kv_heads, 9, cfg.head_dim)
    k, v = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    slot = cache.allocate_seq()
    _write(cache, slot, k, v, 0, 9)

    cache.truncate(slot, 9)  # no-op
    assert int(cache.seq_len[slot]) == 9

    try:
        cache.truncate(slot, 10)
        raise AssertionError("expected ValueError when growing with truncate")
    except ValueError:
        pass

    cache.truncate(slot, 0)
    assert int(cache.seq_len[slot]) == 0
    assert cache.allocator.num_free == cfg.num_pages
    _write(cache, slot, k, v, 0, 5)  # still usable afterwards
    assert int(cache.seq_len[slot]) == 5
    print("  [truncate edges] no-op, invalid length, and truncate to 0 behave")


def main() -> None:
    test_chunk_after_cache_is_causal()
    test_truncate("float32")
    test_truncate("int8")
    test_truncate_edges()
    print("ok")


if __name__ == "__main__":
    main()