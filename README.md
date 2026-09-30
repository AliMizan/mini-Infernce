# Paged KV cache skeleton

Minimal PyTorch KV cache for a local LLD/SOLID review agent on a 6 GB card
(RTX 4050 class). Batch size 1. No CUDA kernels yet.

## Layout

```
engine/config.py      Qwen2.5-Coder 1.5B / 3B + 4050 defaults
engine/kv_cache.py    BlockAllocator + PagedKVCache (INT8 pages optional)
engine/attention.py   gather pages → SDPA  (reference attention)
tests/test_kv_cache.py
```

## Contract

- Physical pages: `[num_pages, H_kv, page_size, head_dim]`
- `block_table[slot, logical_page] → physical page id | -1`
- Prefill and decode both go through `write()` at the committed cursor
- `gather()` includes the in-flight step (`seq_len + pending`) so a decoder
  layer can attend immediately after it writes
- Call `commit(slot)` once after every layer of a step has written
- `gather(layer, slot)` packs used pages into `[1, H_kv, S, D]`
- INT8 pages use a per-(page, head) absmax scale

## Correctness gate

Paged attention must match a contiguous cache.

```bash
python tests/test_kv_cache.py
```

INT8 is allowed a larger atol (`5e-2`). FP16/FP32 gather is bit-close.

## Drop onto Qwen2.5-Coder-1.5B

```python
from engine import EngineConfig, PagedKVCache, paged_attention

cfg = EngineConfig.qwen25_coder_1p5b_4050(device="cuda")
cache = PagedKVCache(cfg)
slot = cache.allocate_seq()

# inside each decoder layer, after RoPE:
#   q : [1, H_q,  T, D]
#   k, v : [1, H_kv, T, D]
cache.write(layer_idx, slot, k, v)
attn_out = paged_attention(q, cache, layer_idx, slot)

cache.commit(slot)  # once per forward step, after the last layer
```


On a 4050 keep `max_seq_len=4096`, `page_size=16`, `kv_dtype="int8"`.
The 1.5B INT8 page pool is a few hundred MiB — weights, not KV, dominate VRAM.

## Not in this skeleton

- Copy-on-write / beam sharing
- Fused paged-attention kernel (walk block table inside the softmax)
- 4-bit weight packing
- Multi-sequence continuous batching
