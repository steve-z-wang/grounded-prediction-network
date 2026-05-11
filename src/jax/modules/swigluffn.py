"""SwiGLU FFN with internal RMSNorm (Flax Linen)."""
import jax.numpy as jnp
import flax.linen as nn
from src.jax.modules.rmsnorm import RMSNorm


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: norm(x) -> silu(W_gate(x)) * W_val(x) -> W_down.

    Input and output dimension are the same (d). Hidden dim is d_hidden.
    w2 is zero-initialized so FFN starts as identity (zero delta).
    """
    d: int
    d_hidden: int

    def setup(self):
        self.norm = RMSNorm(dim=self.d, name='norm')
        self.w1 = nn.Dense(2 * self.d_hidden, use_bias=False, name='w1')
        self.w2 = nn.Dense(self.d, use_bias=False, name='w2',
                           kernel_init=nn.initializers.zeros)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.norm(x)
        gate_val = self.w1(x)
        gate, val = jnp.split(gate_val, 2, axis=-1)
        return self.w2(nn.silu(gate) * val)
