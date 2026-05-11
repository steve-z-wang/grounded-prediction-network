"""RMS Normalization (Flax Linen)."""
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    """RMS Normalization with learnable scale.

    Requires explicit dim so parameters are created at setup time,
    which is necessary for use inside jax.lax.scan.
    """
    dim: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_f32 = x.astype(jnp.float32)
        norm = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / norm).astype(x.dtype) * self.weight
