"""Transformer++ baseline: standard pre-norm transformer, next-token CE loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.torch.modules import RMSNorm
from src.torch.models.transformer.config import TransformerConfig
from src.torch.models.transformer.block import TransformerBlock


class Transformer(nn.Module):
    """Transformer++ LM."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.n_heads, cfg.ffn_dim)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_dim)

    def forward(self, tokens: torch.LongTensor, ce_chunk_size: int, **kwargs):
        """tokens: (B, T+1). Returns (loss, metrics)."""
        B, T = tokens.shape

        embeds = self.embedding(tokens)
        h = embeds
        for block in self.blocks:
            h = checkpoint(block, h, use_reentrant=False)
        h = self.norm(h)

        n_use = ((T - 1) // ce_chunk_size) * ce_chunk_size
        targets = tokens[:, 1:1 + n_use]

        W = self.embedding.weight
        V = W.shape[0]
        cs = ce_chunk_size
        n_chunks = n_use // cs
        total = torch.tensor(0.0, device=tokens.device)
        for i in range(n_chunks):
            x = h[:, i * cs:(i + 1) * cs]
            t = targets[:, i * cs:(i + 1) * cs]
            logits = checkpoint(lambda x, w: x @ w.T, x, W, use_reentrant=False)
            total = total + F.cross_entropy(
                logits.reshape(-1, V), t.reshape(-1), reduction="none"
            ).sum()
        loss = total / (B * n_chunks * cs)

        metrics = {"ce": loss.detach()}
        return loss, metrics

    def describe(self) -> list[str]:
        c = self.cfg
        return [
            f"Transformer++: {c.n_layers}L, {c.n_heads}h, "
            f"hidden={c.hidden_dim}, ffn={c.ffn_dim}",
        ]
