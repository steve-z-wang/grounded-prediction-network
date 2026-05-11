"""Base trainer with common training logic — JAX/optax."""

import os
import csv
import math
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.jax.training.config import BaseTrainConfig
from src.data.dataset import BaseDataset


def _v(x):
    """Extract scalar from JAX array or return as-is."""
    if isinstance(x, (jnp.ndarray, jax.Array)):
        return float(x)
    return x


def _count_params(params):
    """Count total number of parameters in a pytree."""
    leaves = jax.tree.leaves(params)
    return sum(x.size for x in leaves)


def _make_weight_decay_mask(params):
    """Create mask: True for 2D+ arrays (matrices), False for 1D (biases, norms).
    Also excludes A_log and dt_bias from weight decay."""
    def _mask_fn(path, x):
        name = '/'.join(str(k) for k in path)
        if 'A_log' in name or 'dt_bias' in name:
            return False
        return x.ndim >= 2
    flat = jax.tree_util.tree_map_with_path(_mask_fn, params)
    return flat


class Trainer:
    """Base JAX trainer with optax optimizer, scheduler, checkpointing, logging.

    Subclasses implement make_loss_fn(model, cfg) -> loss_fn(params, batch, rng, state)
    that returns (loss, (new_state, metrics)).
    """

    def __init__(self):
        self.model = None
        self.dataset = None
        self.cfg = None
        self.params = None
        self.opt_state = None
        self.optimizer = None
        self.n_params = 0
        self.start_step = 0
        self.start_data_pos = 0

    # --- Public interface ---

    def setup(self, model, params, dataset: BaseDataset, cfg: BaseTrainConfig) -> None:
        self.model = model
        self.params = params
        self.dataset = dataset
        self.cfg = cfg

        self.n_params = _count_params(params)

        # Build optax optimizer: warmup cosine schedule + grad clip + AdamW
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.total_steps,
            end_value=cfg.learning_rate * cfg.min_lr_ratio,
        )

        self.schedule_fn = schedule_fn
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adamw(
                learning_rate=schedule_fn,
                b1=cfg.adam_beta1,
                b2=cfg.adam_beta2,
                weight_decay=cfg.weight_decay,
                mask=_make_weight_decay_mask(params),
            ),
        )
        self.opt_state = self.optimizer.init(params)

        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    def make_loss_fn(self, model, cfg):
        """Override in subclass. Return a function:
        loss_fn(params, batch, rng) -> (loss, metrics_dict)
        """
        raise NotImplementedError

    def run(self) -> None:
        cfg = self.cfg
        end_step = min(self.start_step + cfg.run_steps, cfg.total_steps)

        self._print_summary()

        loss_fn = self.make_loss_fn(self.model, cfg)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_path = os.path.join(cfg.log_dir, f"{cfg.name}_train_log.csv")
        append_mode = bool(cfg.resume) and os.path.exists(log_path)
        log_file = open(log_path, "a" if append_mode else "w", newline="")
        log_writer = csv.writer(log_file)
        metric_keys = None

        step = self.start_step
        running = {"loss": 0.0, "gn": 0.0}
        pbar = tqdm(total=end_step - self.start_step, desc="Training")
        done = False

        data_pos = self.start_data_pos
        rng = jax.random.PRNGKey(cfg.seed)

        params = self.params
        opt_state = self.opt_state

        # Gradient accumulation state
        accum_count = 0
        accum_grads = None
        accum_loss = 0.0
        accum_metrics = {}

        @jax.jit
        def accum_step(params, batch, rng):
            """Compute loss and grads for one micro-batch."""
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(params, batch, rng)
            return loss, grads, metrics

        @jax.jit
        def apply_grads(params, opt_state, grads):
            """Apply accumulated (averaged) gradients."""
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            return new_params, new_opt_state, grad_norm

        for batch in self.dataset.batches(cfg.batch_size, start=self.start_data_pos):
            rng, step_rng = jax.random.split(rng)
            batch_jnp = jnp.array(batch)

            loss, grads, metrics = accum_step(params, batch_jnp, step_rng)

            if metric_keys is None:
                metric_keys = [k for k in metrics if k != "loss"]
                if not append_mode:
                    log_writer.writerow(["step", "loss", "lr", "grad_norm"] + metric_keys)
                    log_file.flush()

            # Accumulate gradients, loss, and metrics across micro-batches
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = jax.tree.map(lambda a, b: a + b, accum_grads, grads)
            accum_loss += _v(loss)
            for k in metric_keys:
                accum_metrics[k] = accum_metrics.get(k, 0.0) + _v(metrics.get(k, 0.0))

            accum_count += 1
            data_pos += 1

            if accum_count >= cfg.grad_accum:
                # Average gradients across micro-batches, then apply
                avg_g = jax.tree.map(lambda g: g / cfg.grad_accum, accum_grads)
                params, opt_state, grad_norm = apply_grads(params, opt_state, avg_g)

                # Average loss and metrics across micro-batches
                avg_loss = accum_loss / cfg.grad_accum
                avg_metrics = {k: v / cfg.grad_accum for k, v in accum_metrics.items()}

                running["loss"] += avg_loss
                running["gn"] += _v(grad_norm)
                for k in metric_keys:
                    running[k] = running.get(k, 0.0) + avg_metrics[k]
                step += 1
                pbar.update(1)

                accum_count = 0
                accum_grads = None
                accum_loss = 0.0
                accum_metrics = {}

                if step % cfg.log_every == 0:
                    n = cfg.log_every
                    # Get current learning rate from schedule
                    cur_lr = _v(self.schedule_fn(step))
                    pbar.set_postfix(
                        loss=f"{running['loss'] / n:.4f}",
                        gn=f"{running['gn'] / n:.2f}",
                        lr=f"{cur_lr:.2e}",
                    )
                    row = [
                        step,
                        f"{running['loss'] / n:.6f}",
                        f"{cur_lr:.2e}",
                        f"{running['gn'] / n:.4f}",
                    ] + [f"{running.get(k, 0.0) / n:.6f}" for k in metric_keys]
                    log_writer.writerow(row)
                    log_file.flush()
                    running = {"loss": 0.0, "gn": 0.0}

                if step % cfg.save_every == 0:
                    self.params = params
                    self.opt_state = opt_state
                    self._save_checkpoint(step, data_pos)

                if step >= end_step:
                    done = True
                    break
            if done:
                break

        pbar.close()
        log_file.close()
        self.params = params
        self.opt_state = opt_state
        self._save_checkpoint(step, data_pos, final=True)

    # --- Private helpers ---

    def _save_checkpoint(self, step, data_pos, final=False):
        name = f"{self.cfg.name}_final" if final else f"{self.cfg.name}_{step}"
        path = os.path.join(self.cfg.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)
        tmp = os.path.join(path, "checkpoint.pkl.tmp")
        dest = os.path.join(path, "checkpoint.pkl")

        # Serialize params and opt_state as numpy
        params_np = jax.tree.map(lambda x: np.array(x), self.params)
        opt_state_np = jax.tree.map(
            lambda x: np.array(x) if isinstance(x, (jnp.ndarray, jax.Array)) else x,
            self.opt_state,
        )

        with open(tmp, "wb") as f:
            pickle.dump({
                "step": step,
                "data_position": data_pos,
                "params": params_np,
                "opt_state": opt_state_np,
            }, f)
        os.replace(tmp, dest)
        print(f"\nSaved checkpoint to {path}/")

    def _load_checkpoint(self, path):
        with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
            ckpt = pickle.load(f)
        self.params = jax.tree.map(jnp.array, ckpt["params"])
        self.opt_state = jax.tree.map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            ckpt["opt_state"],
        )
        self.start_step = ckpt["step"]
        self.start_data_pos = ckpt["data_position"]
        print(f"Resumed from: {path} (step {self.start_step})")

    def _print_summary(self):
        cfg = self.cfg
        print(f"\n{'=' * 60}")
        print(f"  Run: {cfg.name}")
        print(f"{'=' * 60}")
        devices = jax.devices()
        print(f"\n[Device]\n  JAX devices:          {[str(d) for d in devices]}")
        print(f"\n[Model]\n  Params:               {self.n_params:,}")
        if hasattr(self.model, 'describe'):
            for line in self.model.describe():
                print(f"  {line}")
        print(f"\n[Data]\n  Dataset:              {self.dataset.total_tokens:,} tokens")
        print(f"\n[Training]")
        print(f"  Batch size:           {cfg.batch_size}")
        print(f"  Seq len:              {cfg.seq_len}")
        print(f"  Grad accum:           {cfg.grad_accum}")
        print(f"  Total steps:          {cfg.total_steps:,}")
        print(f"  Run steps:            {cfg.run_steps:,}")
        print(f"  Checkpoint every:     {cfg.checkpoint_every}")
        print(f"  Precision:            {'bf16' if cfg.bf16 else 'fp32'}")
        if self.start_step > 0:
            print(f"  Resuming from step:   {self.start_step:,}")
        print(f"  Learning rate:        {cfg.learning_rate}")
        print(f"\n{'=' * 60}")
