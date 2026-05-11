import jax
import jax.numpy as jnp

from src.jax.models.sot_m import SoTM, SoTMConfig


def _make(vocab=100, d_state=32, d_emb=16, d_ffn=64,
          n_heads=4, d_key=8, d_val=16):
    cfg = SoTMConfig(
        vocab_size=vocab, d_state=d_state, d_emb=d_emb, d_ffn=d_ffn,
        n_heads=n_heads, d_key=d_key, d_val=d_val,
    )
    model = SoTM(cfg=cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((2, 4), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    return model, cfg, variables['params']


def test_sotm_init_state():
    model, cfg, params = _make()
    s, prev_s, M = model.apply({'params': params}, batch_size=4, method=model.init_state)
    assert s.shape == (4, cfg.d_state)
    assert prev_s.shape == (4, cfg.d_state)
    assert M.shape == (4, cfg.n_heads, cfg.d_key, cfg.d_val)
    assert jnp.all(s == 0)
    assert jnp.all(M == 0)


def test_sotm_step_shapes():
    model, cfg, params = _make()
    s, prev_s, M = model.apply({'params': params}, batch_size=2, method=model.init_state)
    tok = jax.random.randint(jax.random.PRNGKey(1), (2,), 0, cfg.vocab_size)

    new_s, new_prev_s, M_new, predicted = model.apply(
        {'params': params}, s, prev_s, M, tok, method=model.step
    )
    assert new_s.shape == (2, cfg.d_state)
    assert new_prev_s.shape == (2, cfg.d_state)
    assert M_new.shape == M.shape
    assert predicted.shape == (2, cfg.d_state)


def test_sotm_forward_and_backward():
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)

    def loss_fn(params):
        loss, state, metrics = model.apply({'params': params}, tokens)
        return loss, (state, metrics)

    (loss, (state, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert loss.ndim == 0
    s, prev_s, M = state
    assert s.shape == (2, cfg.d_state)
    assert M.shape == (2, cfg.n_heads, cfg.d_key, cfg.d_val)
    assert "pred_ce" in metrics
    grad_leaves = jax.tree.leaves(grads)
    has_grad = any(jnp.abs(g).sum() > 0 for g in grad_leaves)
    assert has_grad


def test_sotm_state_carries():
    """Passing final state as init_state should not re-init to zero."""
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, cfg.vocab_size)
    _, state1, _ = model.apply({'params': params}, tokens)
    _, state2, _ = model.apply({'params': params}, tokens, init_state=state1)
    s1, _, _ = state1
    s2, _, _ = state2
    assert not jnp.allclose(s1, s2)


def test_sotm_memory_updates():
    """Memory matrix should change after processing tokens."""
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)
    _, state, _ = model.apply({'params': params}, tokens)
    _, _, M = state
    assert not jnp.all(M == 0)


def test_sotm_memory_gradient():
    """Memory parameters should receive gradient."""
    model, cfg, params = _make()
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)

    def loss_fn(params):
        loss, _, _ = model.apply({'params': params}, tokens)
        return loss

    grads = jax.grad(loss_fn)(params)
    # Check that memory-related params have non-zero gradients
    mem_grads = [v for k, v in grads.items() if 'write_' in k or 'read_' in k]
    assert len(mem_grads) > 0
    has_mem_grad = any(jnp.abs(g).sum() > 0 for g in mem_grads)
    assert has_mem_grad


def test_sotm_checkpoint_gradient():
    """Gradient checkpointing should produce same loss as without."""
    model, cfg, params = _make()
    rng = jax.random.PRNGKey(0)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)

    loss_no_ckpt, _, _ = model.apply({'params': params}, tokens, checkpoint_every=0)
    loss_ckpt, _, _ = model.apply({'params': params}, tokens, checkpoint_every=4)
    assert jnp.allclose(loss_no_ckpt, loss_ckpt, atol=1e-5)
