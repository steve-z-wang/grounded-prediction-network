"""SwiGLU FFN with internal RMSNorm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.torch.modules.rmsnorm import RMSNorm


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: norm(x) -> silu(W_gate(x)) * W_val(x) -> W_down.

    Input and output dimension are the same (d). Hidden dim is d_hidden.
    """

    def __init__(self, d: int, d_hidden: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.w1 = nn.Linear(d, 2 * d_hidden, bias=False)  # gate + val fused
        self.w2 = nn.Linear(d_hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, val = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * val)
