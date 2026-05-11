"""Gated Linear Recurrence (GLR) block.

    h_t = f_t * h_{t-1} + i_t * u_t

where f_t (forget gate), i_t (input gate), u_t (candidate) are all computed
from the input token. Parallel via accelerated-scan.

Internal RMSNorm, no residual (caller adds the residual).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerated_scan.warp import scan

from src.torch.modules.rmsnorm import RMSNorm


class GLR(nn.Module):
    """Gated linear recurrence: h_t = f * h_{t-1} + i * u_t."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        D = hidden_dim
        self.norm = RMSNorm(D)
        self.W_in = nn.Linear(D, D, bias=False)
        self.W_f = nn.Linear(D, D, bias=False)
        self.W_i = nn.Linear(D, D, bias=False)
        self.W_out = nn.Linear(D, D, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D)
        x = self.norm(h)
        u = self.W_in(x)                      # candidate      (B, T, D)
        f = torch.sigmoid(self.W_f(x))        # forget gate    (B, T, D)
        i = torch.sigmoid(self.W_i(x))        # input gate     (B, T, D)

        gated_input = i * u                   # (B, T, D)

        # accelerated_scan expects (B, C, T) layout
        gates = f.transpose(1, 2).contiguous()
        tokens = gated_input.transpose(1, 2).contiguous()

        # state[t] = f[t] * state[t-1] + gated_input[t]
        state = scan(gates, tokens)           # (B, D, T)
        state = state.transpose(1, 2)         # (B, T, D)

        return self.W_out(state)
