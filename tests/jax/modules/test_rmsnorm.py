import jax
import jax.numpy as jnp

from src.jax.modules.rmsnorm import RMSNorm


def test_rmsnorm_shape():
    norm = RMSNorm(dim=16)
    params = norm.init(jax.random.PRNGKey(0), jnp.ones((2, 5, 16)))
    x = jnp.ones((2, 5, 16))
    assert norm.apply(params, x).shape == x.shape


def test_rmsnorm_unit_rms():
    norm = RMSNorm(dim=64)
    params = norm.init(jax.random.PRNGKey(0), jnp.ones((4, 8, 64)))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8, 64)) * 10
    y = norm.apply(params, x)
    rms = jnp.sqrt(jnp.mean(y ** 2, axis=-1))
    assert jnp.allclose(rms, jnp.ones_like(rms), atol=0.05)
