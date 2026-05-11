"""Rotary Position Embedding (RoPE)."""

import torch


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0,
                          device=None, dtype=torch.float32):
    """Precompute cos/sin caches of shape (max_seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(t, freqs)              # (T, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (T, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, H, T, head_dim).

    cos, sin: (T, head_dim).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)
