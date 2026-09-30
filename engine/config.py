"""Hardware-aware defaults for a 6 GB RTX 4050-class card."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # Qwen2.5-Coder-1.5B Instruct (override for 3B)
    num_layers: int = 28
    num_qo_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128

    page_size: int = 16
    max_batch_size: int = 1
    max_seq_len: int = 4096

    # Physical page pool. 0 = auto from max_batch * pages_per_seq + slack.
    num_pages: int = 0
    page_slack: int = 8

    compute_dtype: str = "float16"  # float16 | bfloat16 | float32
    kv_dtype: str = "int8"  # float16 | bfloat16 | float32 | int8
    device: str = "cuda"

    def __post_init__(self) -> None:
        pages_per_seq = (self.max_seq_len + self.page_size - 1) // self.page_size
        if self.num_pages <= 0:
            self.num_pages = self.max_batch_size * pages_per_seq + self.page_slack
        if self.num_qo_heads % self.num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads (GQA)")

    @property
    def n_rep(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    @property
    def pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size

    @property
    def torch_compute_dtype(self):
        import torch

        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.compute_dtype]

    @classmethod
    def qwen25_coder_1p5b_4050(cls, device: str = "cuda") -> "EngineConfig":
        return cls(
            num_layers=28,
            num_qo_heads=12,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            max_batch_size=1,
            max_seq_len=4096,
            kv_dtype="int8",
            compute_dtype="float16",
            device=device,
        )

    @classmethod
    def qwen25_coder_3b_4050(cls, device: str = "cuda") -> "EngineConfig":
        # Qwen2.5-3B: 36 layers, 16 Q heads, 2 KV heads, dim 128
        return cls(
            num_layers=36,
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            max_batch_size=1,
            max_seq_len=4096,
            kv_dtype="int8",
            compute_dtype="float16",
            device=device,
        )
