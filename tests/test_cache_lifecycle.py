"""Lifecycle tests for PagedKVCache. CPU only, no model download.

Covers what test_kv_cache.py does not:
  * free a sequence, reuse its pages for another, nothing stale leaks through
  * two sequences at once do not disturb each other
  * running out of sequence slots, and reset()
  * int8: stale data left in a reused page must not wreck the new scale

Run:
    python tests/test_cache_lifecycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import EngineConfig
from engine.kv_cache import PagedKVCache


def _cfg(kv_dtype: str, batch: int = 2) -> EngineConfig:
    return EngineConfig(
        num_layers=2,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        page_size=4,
        max_batch_size=batch,
        max_seq_len=32,
        num_pages=16,
        kv_dtype=kv_dtype,
        compute_dtype="float32",
        device="cpu",
    )


def _tol(kv_dtype: str) -> float:
    return 1e-6 if kv_dtype == "float32" else 0.1


def _rand_kv(cfg: EngineConfig, t: int, seed: int, amp: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.num_layers, cfg.num_kv_heads, t, cfg.head_dim)
    return torch.randn(shape, generator=g) * amp, torch.randn(shape, generator=g) * amp


def _write(cache: PagedKVCache, slot: int, k: torch.Tensor, v: torch.Tensor, lo: int, hi: int) -> None:
    """Write tokens [lo, hi) of the reference tensors as one forward step."""
    cache.write_prefill(slot, list(k[:, :, lo:hi]), list(v[:, :, lo:hi]))


def _pages(cache: PagedKVCache, slot: int) -> set:
    return {p for p in cache._table_cpu[slot] if p >= 0}


def _check(cache: PagedKVCache, slot: int, k, v, n: int, tol: float, what: str) -> None:
    cfg = cache.cfg
    assert int(cache.seq_len[slot]) == n, f"{what}: seq_len {int(cache.seq_len[slot])} != {n}"
    for layer in range(cfg.num_layers):
        gk, gv = cache.gather(layer, slot)
        assert gk.shape == (1, cfg.num_kv_heads, n, cfg.head_dim), f"{what}: bad shape {tuple(gk.shape)}"
        err = max(
            (gk[0] - k[layer, :, :n]).abs().max().item(),
            (gv[0] - v[layer, :, :n]).abs().max().item(),
        )
        assert err <= tol, f"{what}: layer {layer} max error {err:.4g} > {tol}"


def test_free_then_reuse(kv_dtype: str) -> None:
    cfg = _cfg(kv_dtype)
    cache = PagedKVCache(cfg)
    tol = _tol(kv_dtype)

    # Sequence A uses deliberately huge values, so leftovers are easy to spot.
    ka, va = _rand_kv(cfg, 12, seed=1, amp=100.0)
    slot = cache.allocate_seq()
    _write(cache, slot, ka, va, 0, 12)
    pages_a = _pages(cache, slot)
    assert len(pages_a) == 3

    cache.free_seq(slot)
    assert cache.allocator.num_free == cfg.num_pages, "pages not returned"
    assert cache.block_table[slot].max().item() == -1, "block_table row not cleared"
    assert int(cache.seq_len[slot]) == 0

    # Sequence B is small and ends mid-page, so stale slots remain in its last page.
    kb, vb = _rand_kv(cfg, 8, seed=2, amp=1.0)
    slot_b = cache.allocate_seq()
    assert slot_b == slot
    _write(cache, slot_b, kb, vb, 0, 6)
    assert _pages(cache, slot_b) <= pages_a, "expected B to reuse A's pages"
    _check(cache, slot_b, kb, vb, 6, tol, "reused pages after prefill")

    # Decode two tokens into slots that used to hold A's data.
    for t in (6, 7):
        _write(cache, slot_b, kb, vb, t, t + 1)
    _check(cache, slot_b, kb, vb, 8, tol, "reused pages after decode")

    cache.free_seq(slot_b)
    assert cache.allocator.num_free == cfg.num_pages
    print(f"  [reuse {kv_dtype:7s}] pages reused, nothing from the old sequence leaks")


def test_two_slots(kv_dtype: str) -> None:
    cfg = _cfg(kv_dtype)
    cache = PagedKVCache(cfg)
    tol = _tol(kv_dtype)

    k0, v0 = _rand_kv(cfg, 12, seed=11)
    k1, v1 = _rand_kv(cfg, 12, seed=12)
    a = cache.allocate_seq()
    b = cache.allocate_seq()
    assert {a, b} == {0, 1}

    _write(cache, a, k0, v0, 0, 7)
    _write(cache, b, k1, v1, 0, 5)
    for t in range(3):  # interleaved decode, one token each
        _write(cache, a, k0, v0, 7 + t, 8 + t)
        _write(cache, b, k1, v1, 5 + t, 6 + t)

    _check(cache, a, k0, v0, 10, tol, "slot A")
    _check(cache, b, k1, v1, 8, tol, "slot B")
    assert _pages(cache, a).isdisjoint(_pages(cache, b)), "slots share a page"

    cache.free_seq(a)
    _check(cache, b, k1, v1, 8, tol, "slot B after A was freed")

    k2, v2 = _rand_kv(cfg, 5, seed=13)
    c = cache.allocate_seq()
    assert c == a
    _write(cache, c, k2, v2, 0, 5)
    _check(cache, c, k2, v2, 5, tol, "new sequence in freed slot")
    _check(cache, b, k1, v1, 8, tol, "slot B after C was written")

    cache.free_seq(b)
    cache.free_seq(c)
    assert cache.allocator.num_free == cfg.num_pages
    print(f"  [2 slots {kv_dtype:7s}] independent, disjoint pages, free/reuse safe")


def test_slot_exhaustion_and_reset() -> None:
    cache = PagedKVCache(_cfg("float32"))
    s0 = cache.allocate_seq()
    s1 = cache.allocate_seq()
    try:
        cache.allocate_seq()
        raise AssertionError("expected 'no free sequence slot'")
    except RuntimeError as e:
        assert "no free sequence slot" in str(e)

    cfg = cache.cfg
    k, v = _rand_kv(cfg, 9, seed=21)
    _write(cache, s0, k, v, 0, 9)
    _write(cache, s1, k, v, 0, 5)
    assert cache.allocator.num_free < cfg.num_pages

    cache.reset()
    assert cache.allocator.num_free == cfg.num_pages
    assert not any(cache._occupied)
    assert cache.allocate_seq() in (0, 1)
    print("  [slots] exhaustion raises, reset() returns every page")


def main() -> None:
    for dt in ("float32", "int8"):
        test_free_then_reuse(dt)
    for dt in ("float32", "int8"):
        test_two_slots(dt)
    test_slot_exhaustion_and_reset()
    print("ok")


if __name__ == "__main__":
    main()