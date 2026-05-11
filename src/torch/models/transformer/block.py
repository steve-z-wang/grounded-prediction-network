"""Transformer pre-norm block: attention + FFN."""

import torch
import torch.nn as nn

from src.torch.modules import SwiGLUFFN, Attention


class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = Attention(hidden_dim, n_heads)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(h)
        h = h + self.ffn(h)
        return h
