"""Trainer for SoT+M — JAX.

Full-sequence BPTT via jax.lax.scan with optional gradient checkpointing (remat).
"""

from src.jax.training.trainer import Trainer


class SoTMTrainer(Trainer):

    def make_loss_fn(self, model, cfg):
        """Return a pure function: loss_fn(params, batch, rng) -> (loss, metrics)."""
        checkpoint_every = cfg.checkpoint_every
        bf16 = cfg.bf16

        def loss_fn(params, batch, rng):
            loss, state, metrics = model.apply(
                {'params': params}, batch,
                checkpoint_every=checkpoint_every,
                bf16=bf16,
            )
            return loss, metrics

        return loss_fn
