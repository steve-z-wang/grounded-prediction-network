"""Base trainer with common training logic."""

import os
import csv
import math

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.torch.training.config import BaseTrainConfig
from src.data.dataset import BaseDataset


def _v(x):
    return x.item() if isinstance(x, torch.Tensor) else x


class Trainer:
    """Base trainer with optimizer, scheduler, checkpointing, logging.

    Subclasses implement train_step(batch) -> dict with at least 'loss' key
    and must call .backward() themselves (scaling by cfg.grad_accum).
    """

    def __init__(self):
        self.model = None
        self.dataset = None
        self.cfg = None
        self.optimizer = None
        self.scheduler = None
        self.all_params = None
        self.n_params = 0
        self.start_step = 0
        self.start_data_pos = 0

    # --- Public interface ---

    def setup(self, model, dataset: BaseDataset, cfg: BaseTrainConfig) -> None:
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

        decay, no_decay = [], []
        seen = set()
        for param in model.parameters():
            if id(param) in seen or not param.requires_grad:
                continue
            seen.add(id(param))
            if getattr(param, "_no_weight_decay", False) or param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)

        self.all_params = decay + no_decay
        self.n_params = sum(p.numel() for p in self.all_params)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self._lr_lambda(step, cfg, cfg.total_steps),
        )

        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    def train_step(self, batch) -> dict:
        """Override in subclass. Must call .backward() and return dict with 'loss'."""
        raise NotImplementedError

    def run(self) -> None:
        cfg = self.cfg
        end_step = min(self.start_step + cfg.run_steps, cfg.total_steps)

        self._print_summary()

        if cfg.compile:
            self.model = torch.compile(self.model)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.bf16
            else torch.autocast("cuda", enabled=False)
        )

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
        self.model.train()

        data_pos = self.start_data_pos
        self.optimizer.zero_grad()
        accum_count = 0
        accum_loss = 0.0
        accum_metrics = {}

        for batch in self.dataset.batches(cfg.batch_size, start=self.start_data_pos):
            batch = torch.from_numpy(batch).long() if not isinstance(batch, torch.Tensor) else batch
            with autocast_ctx:
                result = self.train_step(batch)

            loss = result["loss"]
            if metric_keys is None:
                metric_keys = [k for k in result if k != "loss"]
                if not append_mode:
                    log_writer.writerow(["step", "loss", "lr", "grad_norm"] + metric_keys)
                    log_file.flush()

            accum_loss += _v(loss)
            for k in metric_keys:
                accum_metrics[k] = accum_metrics.get(k, 0.0) + _v(result.get(k, 0.0))

            accum_count += 1
            data_pos += 1

            if accum_count >= cfg.grad_accum:
                grad_norm = clip_grad_norm_(self.all_params, cfg.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                avg_loss = accum_loss / cfg.grad_accum
                gn = grad_norm.item()

                running["loss"] += avg_loss
                running["gn"] += gn
                for k in metric_keys:
                    running[k] = running.get(k, 0.0) + accum_metrics[k] / cfg.grad_accum
                step += 1
                pbar.update(1)

                accum_count = 0
                accum_loss = 0.0
                accum_metrics = {}

                if step % cfg.log_every == 0:
                    n = cfg.log_every
                    cur_lr = self.scheduler.get_last_lr()[0]
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
                    self._save_checkpoint(step, data_pos)

                if step >= end_step:
                    done = True
                    break
            if done:
                break

        pbar.close()
        log_file.close()
        self._save_checkpoint(step, data_pos, final=True)

    # --- Private helpers ---

    @staticmethod
    def _lr_lambda(step, cfg, total_steps):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = step / max(1, total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    def _save_checkpoint(self, step, data_pos, final=False):
        name = f"{self.cfg.name}_final" if final else f"{self.cfg.name}_{step}"
        path = os.path.join(self.cfg.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)
        tmp = os.path.join(path, "checkpoint.pt.tmp")
        dest = os.path.join(path, "checkpoint.pt")
        torch.save({
            "step": step,
            "data_position": data_pos,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, tmp)
        os.replace(tmp, dest)
        print(f"\nSaved checkpoint to {path}/")

    def _load_checkpoint(self, path):
        ckpt = torch.load(os.path.join(path, "checkpoint.pt"), weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_step = ckpt["step"]
        self.start_data_pos = ckpt["data_position"]
        print(f"Resumed from: {path} (step {self.start_step})")

    def _print_summary(self):
        cfg = self.cfg
        print(f"\n{'=' * 60}")
        print(f"  Run: {cfg.name}")
        print(f"{'=' * 60}")
        if torch.cuda.is_available():
            print(f"\n[Device]\n  CUDA device:          {torch.cuda.get_device_name()}")
        print(f"\n[Model]\n  Params:               {self.n_params:,}")
        # Model-specific description (each model implements describe() -> list[str])
        if hasattr(self.model, "describe"):
            for line in self.model.describe():
                print(f"  {line}")
        print(f"\n[Data]\n  Dataset:              {self.dataset.total_tokens:,} tokens")
        print(f"\n[Training]")
        print(f"  Batch size:           {cfg.batch_size}")
        print(f"  Seq len:              {cfg.seq_len}")
        print(f"  Grad accum:           {cfg.grad_accum}")
        print(f"  Total steps:          {cfg.total_steps:,}")
        print(f"  Run steps:            {cfg.run_steps:,}")
        print(f"  Precision:            {'bf16' if cfg.bf16 else 'fp32'}")
        if self.start_step > 0:
            print(f"  Resuming from step:   {self.start_step:,}")
        print(f"  Learning rate:        {cfg.learning_rate}")
        print(f"\n{'=' * 60}")
