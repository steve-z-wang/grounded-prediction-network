"""Multi-head causal self-attention with RoPE. Internal RMSNorm, no residual."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.torch.modules.rmsnorm import RMSNorm
from src.torch.modules.rope import precompute_rope_cache, apply_rope


class Attention(nn.Module):
    """Llama-style: RMSNorm -> Q/K/V projections -> RoPE -> SDPA -> output proj."""

    def __init__(self, hidden_dim: int, n_heads: int, max_seq_len: int = 8192,
                 rope_base: float = 10000.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim

        self.norm = RMSNorm(hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        H, Dh = self.n_heads, self.head_dim

        x = self.norm(h)
        q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, Dh).transpose(1, 2)

        cos = self.rope_cos[:T].to(dtype=q.dtype)
        sin = self.rope_sin[:T].to(dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, Dh)
        o = o.transpose(1, 2).reshape(B, T, H * Dh)
        return self.W_o(o)
