import jax
import jax.numpy as jnp

from src.jax.models.sot import SoT, SoTConfig


def _make(vocab=100, d_state=32, d_emb=16, d_ffn=64):
    cfg = SoTConfig(vocab_size=vocab, d_state=d_state, d_emb=d_emb, d_ffn=d_ffn)
    model = SoT(cfg=cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((2, 4), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    return model, cfg, variables['params']


def test_sot_init_state():
    model, cfg, params = _make()
    s = model.init_state(batch_size=4)
    assert s.shape == (4, cfg.d_state)
    assert jnp.all(s == 0)


def test_sot_step_shapes():
    model, cfg, params = _make()

    def run_step(params, s, tok):
        return model.apply({'params': params}, s, tok, method=model.step)

    s = model.init_state(2)
    tok = jax.random.randint(jax.random.PRNGKey(1), (2,), 0, cfg.vocab_size)
    new_s, predicted = run_step(params, s, tok)
    assert new_s.shape == s.shape
    assert predicted.shape == (2, cfg.d_state)


def test_sot_forward_and_backward():
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)

    def loss_fn(params):
        loss, state, metrics = model.apply({'params': params}, tokens)
        return loss, (state, metrics)

    (loss, (state, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert loss.ndim == 0
    assert state.shape == (2, cfg.d_state)
    assert "pred_ce" in metrics
    # Check that at least some gradients are non-zero
    grad_leaves = jax.tree.leaves(grads)
    has_grad = any(jnp.abs(g).sum() > 0 for g in grad_leaves)
    assert has_grad


def test_sot_state_carries():
    """Passing final state as init_state should not re-init to zero."""
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, cfg.vocab_size)
    _, s1, _ = model.apply({'params': params}, tokens)
    _, s2, _ = model.apply({'params': params}, tokens, init_state=s1)
    assert not jnp.allclose(s1, s2)
