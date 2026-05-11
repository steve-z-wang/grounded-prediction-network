import jax
import jax.numpy as jnp

from src.jax.modules.swigluffn import SwiGLUFFN


def test_swigluffn_shape():
    ffn = SwiGLUFFN(d=32, d_hidden=64)
    params = ffn.init(jax.random.PRNGKey(0), jnp.ones((2, 5, 32)))
    x = jnp.ones((2, 5, 32))
    assert ffn.apply(params, x).shape == x.shape


def test_swigluffn_backward():
    ffn = SwiGLUFFN(d=16, d_hidden=32)
    params = ffn.init(jax.random.PRNGKey(0), jnp.ones((2, 4, 16)))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16))

    def loss_fn(params, x):
        return ffn.apply(params, x).sum()

    grads = jax.grad(loss_fn)(params, x)
    grad_leaves = jax.tree.leaves(grads)
    assert all(not jnp.isnan(g).any() for g in grad_leaves)
