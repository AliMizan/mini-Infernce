"""Correctness gate: paged cache must match a contiguous KV tensor.

Run:
    python tests/test_kv_cache.py
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


def _tiny_cfg(kv_dtype: str, device: str) -> EngineConfig:
    return EngineConfig(
        num_layers=2,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=32,
        page_size=16,
        max_batch_size=1,
        max_seq_len=128,
        num_pages=16,
        kv_dtype=kv_dtype if kv_dtype == "int8" else "float32",
        compute_dtype="float32",  # stable compare
        device=device,
    )


def _rand_kv(cfg: EngineConfig, t: int, device: torch.device):
    k = torch.randn(cfg.num_layers, 1, cfg.num_kv_heads, t, cfg.head_dim, device=device)
    v = torch.randn(cfg.num_layers, 1, cfg.num_kv_heads, t, cfg.head_dim, device=device)
    return k, v


def test_prefill_then_decode_matches_contiguous(kv_dtype: str, device: str) -> None:
    cfg = _tiny_cfg(kv_dtype, device)
    torch.manual_seed(0)
    cache = PagedKVCache(cfg)
    slot = cache.allocate_seq()
    dev = cache.device

    prefill_t, decode_t = 40, 7  # 40 crosses two full pages + 8 leftover
    k_pf, v_pf = _rand_kv(cfg, prefill_t, dev)
    cache.write_prefill(slot, list(k_pf[:, 0]), list(v_pf[:, 0]))
    assert int(cache.seq_len[slot]) == prefill_t

    k_dc, v_dc = _rand_kv(cfg, decode_t, dev)
    # write token by token to exercise page-boundary appends
    for t in range(decode_t):
        cache.write_decode(
            slot,
            [k_dc[layer, :, :, t : t + 1] for layer in range(cfg.num_layers)],
            [v_dc[layer, :, :, t : t + 1] for layer in range(cfg.num_layers)],
        )
    total = prefill_t + decode_t
    assert int(cache.seq_len[slot]) == total

    k_ref = torch.cat([k_pf, k_dc], dim=3)  # [L, 1, H, S, D]
    v_ref = torch.cat([v_pf, v_dc], dim=3)

    q = torch.randn(1, cfg.num_qo_heads, 1, cfg.head_dim, device=dev, dtype=torch.float32)

    max_err = 0.0
    for layer in range(cfg.num_layers):
        gk, gv = cache.gather(layer, slot)
        assert gk.shape == (1, cfg.num_kv_heads, total, cfg.head_dim)
        if kv_dtype != "int8":
            torch.testing.assert_close(gk, k_ref[layer], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(gv, v_ref[layer], atol=1e-5, rtol=1e-5)

        out_paged = paged_attention(q, cache, layer, slot, causal=False)
        out_ref = contiguous_attention(q, k_ref[layer], v_ref[layer], cfg, causal=False)
        err = (out_paged.float() - out_ref.float()).abs().max().item()
        max_err = max(max_err, err)
        atol = 5e-2 if kv_dtype == "int8" else 1e-4
        assert err < atol, f"layer {layer} attn err {err} >= {atol}"

    mem = cache.memory_bytes()
    print(
        f"  [{kv_dtype:7s}] prefill={prefill_t} decode={decode_t} "
        f"max_attn_err={max_err:.4e} pages_used="
        f"{mem['pages_total'] - mem['pages_free']}/{mem['pages_total']} "
        f"pool={mem['total'] / 1024:.1f} KiB"
    )
    cache.free_seq(slot)
    assert cache.allocator.num_free == cfg.num_pages


def test_layerwise_write_visible_before_commit(device: str) -> None:
    """Decoder loop: write layer i, attend, then commit once."""
    cfg = _tiny_cfg("float16", device)
    torch.manual_seed(1)
    cache = PagedKVCache(cfg)
    slot = cache.allocate_seq()
    t = 24
    k, v = _rand_kv(cfg, t, cache.device)
    q = torch.randn(1, cfg.num_qo_heads, t, cfg.head_dim, device=cache.device)

    for layer in range(cfg.num_layers):
        cache.write(layer, slot, k[layer], v[layer])
        assert int(cache.seq_len[slot]) == 0
        assert int(cache._pending[slot]) == t
        gk, _ = cache.gather(layer, slot)
        assert gk.shape[-2] == t
        out_p = paged_attention(q, cache, layer, slot, causal=True)
        out_r = contiguous_attention(q, k[layer], v[layer], cfg, causal=True)
        err = (out_p - out_r).abs().max().item()
        assert err < 1e-4, err

    cache.commit(slot)
    assert int(cache.seq_len[slot]) == t
    assert int(cache._pending[slot]) == 0
    print("  [layerwise] write-then-attend before commit matches contiguous")
    cache.free_seq(slot)


def test_page_pool_exhaustion(device: str) -> None:
    cfg = EngineConfig(
        num_layers=1,
        num_qo_heads=4,
        num_kv_heads=1,
        head_dim=8,
        page_size=4,
        max_batch_size=1,
        max_seq_len=32,
        num_pages=2,  # only 8 tokens of capacity
        kv_dtype="float16",
        compute_dtype="float32",
        device=device,
    )
    cache = PagedKVCache(cfg)
    slot = cache.allocate_seq()
    k = torch.randn(1, 1, 9, 8, device=cache.device)
    v = torch.randn(1, 1, 9, 8, device=cache.device)
    try:
        cache.write(0, slot, k, v)
        raise AssertionError("expected pool exhaustion")
    except RuntimeError as e:
        assert "exhausted" in str(e) or "max_seq_len" in str(e)
    print("  [exhaust] raised as expected")


def estimate_qwen_1p5b(device: str) -> None:
    cfg = EngineConfig.qwen25_coder_1p5b_4050(device=device)
    cache = PagedKVCache(cfg)
    mem = cache.memory_bytes()
    print(
        f"  [qwen-1.5b] layers={cfg.num_layers} H_kv={cfg.num_kv_heads} "
        f"page={cfg.page_size} max_seq={cfg.max_seq_len} "
        f"pages={cfg.num_pages} kv={cfg.kv_dtype} "
        f"pool={mem['total'] / 1024**2:.2f} MiB"
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    test_prefill_then_decode_matches_contiguous("float32", device)
    test_prefill_then_decode_matches_contiguous("int8", device)
    test_layerwise_write_visible_before_commit(device)
    test_page_pool_exhaustion(device)
    estimate_qwen_1p5b(device)
    print("ok")


if __name__ == "__main__":
    main()
