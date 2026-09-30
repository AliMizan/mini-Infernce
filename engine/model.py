# """Qwen2 / Qwen2.5 decoder that writes K/V into PagedKVCache.

# Uses HuggingFace weights and norms/MLP. Attention is ours.
# """

# from __future__ import annotations

# from typing import Optional, Tuple

# import torch
# import torch.nn as nn

# from .attention import paged_attention
# from .config import EngineConfig
# from .kv_cache import PagedKVCache


# def rotate_half(x: torch.Tensor) -> torch.Tensor:
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     cos: torch.Tensor,
#     sin: torch.Tensor,
#     unsqueeze_dim: int = 1,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Match HuggingFace Qwen2: cos/sin are [B, T, D], unsqueeze onto heads."""
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def _inner(model: nn.Module) -> nn.Module:
#     return model.model if hasattr(model, "model") else model


# def _rope_module(model: nn.Module) -> Optional[nn.Module]:
#     inner = _inner(model)
#     if hasattr(inner, "rotary_emb"):
#         return inner.rotary_emb
#     layer0 = inner.layers[0].self_attn
#     return getattr(layer0, "rotary_emb", None)


# def _position_embeddings(
#     model: nn.Module,
#     hidden: torch.Tensor,
#     position_ids: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     rope = _rope_module(model)
#     if rope is None:
#         raise RuntimeError("could not find rotary_emb on this checkpoint")
#     out = rope(hidden, position_ids)
#     if isinstance(out, tuple) and len(out) == 2:
#         return out
#     raise RuntimeError(f"unexpected rotary_emb output type {type(out)}")


# class PagedQwenEngine:
#     def __init__(
#         self,
#         hf_model: nn.Module,
#         cache: PagedKVCache,
#         slot: int = 0,
#     ):
#         self.model = hf_model
#         self.inner = _inner(hf_model)
#         self.cache = cache
#         self.slot = slot
#         self.cfg = cache.cfg

#     def reset(self) -> None:
#         if self.cache._occupied[self.slot]:
#             self.cache.free_seq(self.slot)
#         self.slot = self.cache.allocate_seq()

#     def _shape_qkv(
#         self,
#         attn: nn.Module,
#         hidden: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         bsz, q_len, _ = hidden.shape
#         head_dim = self.cfg.head_dim
#         q = attn.q_proj(hidden).view(bsz, q_len, self.cfg.num_qo_heads, head_dim).transpose(1, 2)
#         k = attn.k_proj(hidden).view(bsz, q_len, self.cfg.num_kv_heads, head_dim).transpose(1, 2)
#         v = attn.v_proj(hidden).view(bsz, q_len, self.cfg.num_kv_heads, head_dim).transpose(1, 2)
#         return q, k, v

#     def _paged_attn(
#         self,
#         layer_idx: int,
#         attn: nn.Module,
#         hidden: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#     ) -> torch.Tensor:
#         bsz, q_len, _ = hidden.shape
#         q, k, v = self._shape_qkv(attn, hidden)
#         cos, sin = position_embeddings
#         q, k = apply_rotary_pos_emb(q, k, cos, sin)
#         self.cache.write(layer_idx, self.slot, k, v)
#         attn_out = paged_attention(q, self.cache, layer_idx, self.slot, causal=True)
#         attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
#         return attn.o_proj(attn_out)

#     def _decoder_layer(
#         self,
#         layer_idx: int,
#         layer: nn.Module,
#         hidden: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#     ) -> torch.Tensor:
#         residual = hidden
#         hidden = layer.input_layernorm(hidden)
#         hidden = self._paged_attn(layer_idx, layer.self_attn, hidden, position_embeddings)
#         hidden = residual + hidden
#         residual = hidden
#         hidden = layer.post_attention_layernorm(hidden)
#         hidden = layer.mlp(hidden)
#         hidden = residual + hidden
#         return hidden

#     def forward_tokens(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
#         """
#         One prefill or decode step. input_ids: [1, T]
#         Does not commit; caller commits after this returns.
#         Returns logits [1, T, vocab].
#         """
#         hidden = self.inner.embed_tokens(input_ids)
#         pos_emb = _position_embeddings(self.model, hidden, position_ids)
#         for i, layer in enumerate(self.inner.layers):
#             hidden = self._decoder_layer(i, layer, hidden, pos_emb)
#         hidden = self.inner.norm(hidden)
#         if hasattr(self.model, "lm_head"):
#             return self.model.lm_head(hidden)
#         return torch.nn.functional.linear(hidden, self.inner.embed_tokens.weight)

#     @torch.no_grad()
#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         max_new_tokens: int,
#         eos_token_id: Optional[int] = None,
#     ) -> torch.Tensor:
#         if input_ids.dim() != 2 or input_ids.shape[0] != 1:
#             raise ValueError("input_ids must be [1, T]")
#         self.reset()
#         device = input_ids.device
#         prefill_len = input_ids.shape[1]
#         position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
#         logits = self.forward_tokens(input_ids, position_ids)
#         self.cache.commit(self.slot)

#         pieces = [input_ids]
#         for _ in range(max_new_tokens):
#             next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#             pieces.append(next_id)
#             if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
#                 break
#             pos = torch.tensor(
#                 [[int(self.cache.seq_len[self.slot].item())]],
#                 device=device,
#             )
#             logits = self.forward_tokens(next_id, pos)
#             self.cache.commit(self.slot)
#         return torch.cat(pieces, dim=1)

"""Qwen2 / Qwen2.5 decoder that writes K/V into PagedKVCache.

Uses HuggingFace weights and norms/MLP. Attention is ours.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import paged_attention
from .config import EngineConfig
from .kv_cache import PagedKVCache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match HuggingFace Qwen2: cos/sin are [B, T, D], unsqueeze onto heads."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _inner(model: nn.Module) -> nn.Module:
    return model.model if hasattr(model, "model") else model


def _rope_module(model: nn.Module) -> Optional[nn.Module]:
    inner = _inner(model)
    if hasattr(inner, "rotary_emb"):
        return inner.rotary_emb
    layer0 = inner.layers[0].self_attn
    return getattr(layer0, "rotary_emb", None)


def _position_embeddings(
    model: nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rope = _rope_module(model)
    if rope is None:
        raise RuntimeError("could not find rotary_emb on this checkpoint")
    out = rope(hidden, position_ids)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    raise RuntimeError(f"unexpected rotary_emb output type {type(out)}")


class PagedQwenEngine:
    def __init__(
        self,
        hf_model: nn.Module,
        cache: PagedKVCache,
        slot: int = 0,
    ):
        self.model = hf_model
        self.inner = _inner(hf_model)
        self.cache = cache
        self.slot = slot
        self.cfg = cache.cfg

    def reset(self) -> None:
        if self.cache._occupied[self.slot]:
            self.cache.free_seq(self.slot)
        self.slot = self.cache.allocate_seq()

    def _shape_qkv(
        self,
        attn: nn.Module,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden.shape
        head_dim = self.cfg.head_dim
        q = attn.q_proj(hidden).view(bsz, q_len, self.cfg.num_qo_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(bsz, q_len, self.cfg.num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(bsz, q_len, self.cfg.num_kv_heads, head_dim).transpose(1, 2)
        return q, k, v

    def _paged_attn(
        self,
        layer_idx: int,
        attn: nn.Module,
        hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden.shape
        q, k, v = self._shape_qkv(attn, hidden)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        self.cache.write(layer_idx, self.slot, k, v)
        attn_out = paged_attention(q, self.cache, layer_idx, self.slot, causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return attn.o_proj(attn_out)

    def _decoder_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        hidden = self._paged_attn(layer_idx, layer.self_attn, hidden, position_embeddings)
        hidden = residual + hidden
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden
        return hidden

    def forward_tokens(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        last_only: bool = False,
    ) -> torch.Tensor:
        """
        One prefill or decode step. input_ids: [1, T]
        Does not commit; caller commits after this returns.

        last_only=False: logits for every position, [1, T, vocab]
        last_only=True : logits for the final position only, [1, 1, vocab].
                         Use this for generation; it avoids a huge
                         [1, T, vocab] tensor during prefill.
        """
        hidden = self.inner.embed_tokens(input_ids)
        pos_emb = _position_embeddings(self.model, hidden, position_ids)
        for i, layer in enumerate(self.inner.layers):
            hidden = self._decoder_layer(i, layer, hidden, pos_emb)
        hidden = self.inner.norm(hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(hidden)
        return torch.nn.functional.linear(hidden, self.inner.embed_tokens.weight)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must be [1, T]")
        prefill_len = input_ids.shape[1]
        if prefill_len + max_new_tokens > self.cfg.max_seq_len:
            raise ValueError(
                f"prompt ({prefill_len}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        self.reset()
        device = input_ids.device

        position_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
        logits = self.forward_tokens(input_ids, position_ids, last_only=True)
        self.cache.commit(self.slot)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        pieces = [input_ids]
        for step in range(max_new_tokens):
            pieces.append(next_id)
            if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
                break
            if step == max_new_tokens - 1:
                break  # last token picked; no need to run it through the model
            pos = torch.tensor([[self.cache._len_cpu[self.slot]]], device=device)
            logits = self.forward_tokens(next_id, pos, last_only=True)
            self.cache.commit(self.slot)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return torch.cat(pieces, dim=1)