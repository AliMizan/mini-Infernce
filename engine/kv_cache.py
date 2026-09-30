"""
Paged KV cache (vLLM-style pages, single-tenant first).

Physical layout per layer
-------------------------
    k_pages, v_pages : [num_pages, num_kv_heads, page_size, head_dim]
    k_scale, v_scale : [num_pages, num_kv_heads, 1, 1]   (INT8 only)

Logical layout per live sequence
--------------------------------
    block_table[b, i] = physical page id for logical page i, or -1
    seq_len[b]        = tokens written so far

INT8 pages store a per-(page, head) absmax scale. That is coarse but
cheap and good enough to prove the 6 GB path. Swap in per-token scales
later without touching the allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .config import EngineConfig


def _torch_device(spec: str) -> torch.device:
    if spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


class BlockAllocator:
    """Free-list of physical page ids. O(1) alloc / free of a single page."""

    def __init__(self, num_pages: int):
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self.num_pages = num_pages
        self._free: List[int] = list(range(num_pages - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"KV page pool exhausted ({self.num_pages} pages). "
                "Lower max_seq_len or raise EngineConfig.num_pages."
            )
        return self._free.pop()

    def alloc_n(self, n: int) -> List[int]:
        return [self.alloc() for _ in range(n)]

    def free(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(f"invalid page id {page_id}")
        self._free.append(page_id)

    def free_many(self, page_ids: Sequence[int]) -> None:
        for pid in page_ids:
            if pid >= 0:
                self.free(int(pid))


@dataclass
class SequenceSlot:
    slot: int
    seq_len: int = 0


class PagedKVCache:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = _torch_device(cfg.device)
        self.compute_dtype = cfg.torch_compute_dtype
        self.quantized = cfg.kv_dtype == "int8"
        self.allocator = BlockAllocator(cfg.num_pages)

        page_shape = (cfg.num_pages, cfg.num_kv_heads, cfg.page_size, cfg.head_dim)
        storage_dtype = (
            torch.int8
            if self.quantized
            else {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[cfg.kv_dtype]
        )

        self.k_pages = [
            torch.zeros(page_shape, dtype=storage_dtype, device=self.device)
            for _ in range(cfg.num_layers)
        ]
        self.v_pages = [
            torch.zeros(page_shape, dtype=storage_dtype, device=self.device)
            for _ in range(cfg.num_layers)
        ]

        if self.quantized:
            scale_shape = (cfg.num_pages, cfg.num_kv_heads, 1, 1)
            self.k_scale = [
                torch.ones(scale_shape, dtype=self.compute_dtype, device=self.device)
                for _ in range(cfg.num_layers)
            ]
            self.v_scale = [
                torch.ones(scale_shape, dtype=self.compute_dtype, device=self.device)
                for _ in range(cfg.num_layers)
            ]
        else:
            self.k_scale = self.v_scale = None

        # block_table[b, logical_page] -> physical page
        self.block_table = torch.full(
            (cfg.max_batch_size, cfg.pages_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # committed tokens from finished forward steps
        self.seq_len = torch.zeros(cfg.max_batch_size, dtype=torch.int32, device=self.device)
        # tokens written in the current step (all layers share this window)
        self._pending = torch.zeros(cfg.max_batch_size, dtype=torch.int32, device=self.device)
        self._occupied = [False] * cfg.max_batch_size

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def allocate_seq(self) -> int:
        for i, used in enumerate(self._occupied):
            if not used:
                self._occupied[i] = True
                self.seq_len[i] = 0
                self._pending[i] = 0
                self.block_table[i].fill_(-1)
                return i
        raise RuntimeError("no free sequence slot")

    def free_seq(self, slot: int) -> None:
        pages = self.block_table[slot].tolist()
        self.allocator.free_many(pages)
        self.block_table[slot].fill_(-1)
        self.seq_len[slot] = 0
        self._pending[slot] = 0
        self._occupied[slot] = False

    def reset(self) -> None:
        for i, used in enumerate(self._occupied):
            if used:
                self.free_seq(i)
        assert self.allocator.num_free == self.cfg.num_pages

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _ensure_page(self, slot: int, logical_page: int) -> int:
        pid = int(self.block_table[slot, logical_page].item())
        if pid >= 0:
            return pid
        pid = self.allocator.alloc()
        self.block_table[slot, logical_page] = pid
        return pid

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [H_kv, T, D]  -> per-head absmax over (T, D) of this write
        # For page-level scales we recompute from the whole page after write.
        x_f = x.to(torch.float32)
        scale = x_f.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.clamp(torch.round(x_f / scale), -127, 127).to(torch.int8)
        return q, scale.to(self.compute_dtype)

    def _write_float_into_page(
        self,
        layer: int,
        page_id: int,
        offset: int,
        length: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """k, v: [H_kv, length, D] compute dtype."""
        if self.quantized:
            # Dequant existing page (if any tokens already there), splice, requant.
            k_page = self.k_pages[layer][page_id].to(self.compute_dtype) * self.k_scale[layer][page_id]
            v_page = self.v_pages[layer][page_id].to(self.compute_dtype) * self.v_scale[layer][page_id]
            k_page[:, offset : offset + length] = k
            v_page[:, offset : offset + length] = v
            k_q, k_s = self._quantize(k_page)
            v_q, v_s = self._quantize(v_page)
            self.k_pages[layer][page_id].copy_(k_q)
            self.v_pages[layer][page_id].copy_(v_q)
            self.k_scale[layer][page_id].copy_(k_s)
            self.v_scale[layer][page_id].copy_(v_s)
        else:
            self.k_pages[layer][page_id][:, offset : offset + length] = k
            self.v_pages[layer][page_id][:, offset : offset + length] = v

    def write(
        self,
        layer: int,
        slot: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append K/V for one sequence at the current cursor.

        k, v shapes accepted:
            [H_kv, T, D]  or  [1, H_kv, T, D]  or  [T, H_kv, D]
        T may be 1 (decode) or many (prefill chunk).
        """
        k, v = self._normalize_kv(k, v)
        t = k.shape[1]
        if t == 0:
            return

        cursor = int(self.seq_len[slot].item())
        if cursor + t > self.cfg.max_seq_len:
            raise RuntimeError(
                f"sequence would exceed max_seq_len={self.cfg.max_seq_len} "
                f"(have {cursor}, writing {t})"
            )

        pending = int(self._pending[slot].item())
        if pending not in (0, t):
            raise RuntimeError(
                f"layer {layer} tried to write {t} tokens but this step "
                f"already opened a window of {pending}"
            )

        written = 0
        while written < t:
            pos = cursor + written
            logical = pos // self.cfg.page_size
            offset = pos % self.cfg.page_size
            take = min(self.cfg.page_size - offset, t - written)
            page_id = self._ensure_page(slot, logical)
            self._write_float_into_page(
                layer,
                page_id,
                offset,
                take,
                k[:, written : written + take],
                v[:, written : written + take],
            )
            written += take

        self._pending[slot] = t

    def commit(self, slot: int) -> None:
        """Call once after every layer of a forward step has written K/V."""
        self.seq_len[slot] = self.seq_len[slot] + self._pending[slot]
        self._pending[slot] = 0

    def write_prefill(
        self,
        slot: int,
        k_layers: Sequence[torch.Tensor],
        v_layers: Sequence[torch.Tensor],
    ) -> None:
        if len(k_layers) != self.cfg.num_layers:
            raise ValueError("expected K/V for every layer")
        for layer in range(self.cfg.num_layers):
            self.write(layer, slot, k_layers[layer], v_layers[layer])
        self.commit(slot)

    def write_decode(
        self,
        slot: int,
        k_layers: Sequence[torch.Tensor],
        v_layers: Sequence[torch.Tensor],
    ) -> None:
        self.write_prefill(slot, k_layers, v_layers)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def gather(self, layer: int, slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Materialize contiguous K, V for attention.

        Returns:
            k, v : [1, H_kv, S, D] in compute dtype
        """
        s = int(self.seq_len[slot].item()) + int(self._pending[slot].item())
        if s == 0:
            empty = torch.empty(
                1,
                self.cfg.num_kv_heads,
                0,
                self.cfg.head_dim,
                dtype=self.compute_dtype,
                device=self.device,
            )
            return empty, empty.clone()

        n_pages = (s + self.cfg.page_size - 1) // self.cfg.page_size
        page_ids = self.block_table[slot, :n_pages].long()

        k = self.k_pages[layer].index_select(0, page_ids)  # [P, H, page, D]
        v = self.v_pages[layer].index_select(0, page_ids)
        if self.quantized:
            ks = self.k_scale[layer].index_select(0, page_ids)
            vs = self.v_scale[layer].index_select(0, page_ids)
            k = k.to(self.compute_dtype) * ks
            v = v.to(self.compute_dtype) * vs
        else:
            k = k.to(self.compute_dtype)
            v = v.to(self.compute_dtype)

        # [P, H, page, D] -> [H, P*page, D] -> trim
        h = self.cfg.num_kv_heads
        k = k.permute(1, 0, 2, 3).contiguous().view(h, n_pages * self.cfg.page_size, self.cfg.head_dim)
        v = v.permute(1, 0, 2, 3).contiguous().view(h, n_pages * self.cfg.page_size, self.cfg.head_dim)
        k = k[:, :s].unsqueeze(0)
        v = v[:, :s].unsqueeze(0)
        return k, v

    def memory_bytes(self) -> dict:
        def _nbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        pages = sum(_nbytes(t) for t in self.k_pages) + sum(_nbytes(t) for t in self.v_pages)
        scales = 0
        if self.quantized:
            scales = sum(_nbytes(t) for t in self.k_scale) + sum(_nbytes(t) for t in self.v_scale)
        tables = (
            _nbytes(self.block_table)
            + _nbytes(self.seq_len)
            + _nbytes(self._pending)
        )
        return {
            "pages": pages,
            "scales": scales,
            "tables": tables,
            "total": pages + scales + tables,
            "pages_free": self.allocator.num_free,
            "pages_total": self.cfg.num_pages,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() == 4:
            if k.shape[0] != 1:
                raise ValueError("only batch=1 writes are supported in this skeleton")
            k, v = k[0], v[0]
        if k.dim() != 3:
            raise ValueError(f"expected 3D or 4D K/V, got {tuple(k.shape)}")
        # Accept [H, T, D] or [T, H, D]
        if k.shape[0] == self.cfg.num_kv_heads and k.shape[-1] == self.cfg.head_dim:
            pass
        elif k.shape[1] == self.cfg.num_kv_heads and k.shape[-1] == self.cfg.head_dim:
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        else:
            raise ValueError(
                f"cannot interpret K shape {tuple(k.shape)} for "
                f"H_kv={self.cfg.num_kv_heads} D={self.cfg.head_dim}"
            )
        return k.to(self.compute_dtype), v.to(self.compute_dtype)
