"""Trainer for Gated Delta Net."""

import torch

from src.torch.training.trainer import Trainer


class GDNTrainer(Trainer):

    def train_step(self, batch) -> dict:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        loss, metrics = self.model(batch, ce_chunk_size=self.cfg.ce_chunk_size)
        (loss / self.cfg.grad_accum).backward()
        return {"loss": loss, **metrics}
