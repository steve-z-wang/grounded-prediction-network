"""Gated Delta Net baseline — wraps fla's official GatedDeltaNet model.

Uses fla.models.gated_deltanet.GatedDeltaNetForCausalLM internally for the
exact paper config (short conv, L2 norm, gated output, chunkwise kernel).
Our wrapper just adapts it to our training interface (forward returns
(loss, metrics)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel

from src.torch.models.gdn.config import GDNConfig


class GDN(nn.Module):
    """Wraps fla's GatedDeltaNetModel (base, no LM head).

    LM head and CE are computed manually with chunked logits to avoid
    materializing the full (B, T, V) logits tensor at once.
    """

    def __init__(self, cfg: GDNConfig):
        super().__init__()
        self.cfg = cfg

        fla_cfg = GatedDeltaNetConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            expand_v=cfg.expand_v,
            num_hidden_layers=cfg.n_layers,
            intermediate_size=cfg.ffn_dim,
            hidden_ratio=None,
            tie_word_embeddings=False,
        )
        self.model = GatedDeltaNetModel(fla_cfg)
        self.model.gradient_checkpointing_enable()
        # Separate LM head (untied)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.LongTensor, ce_chunk_size: int, **kwargs):
        """tokens: (B, T+1). Returns (loss, metrics)."""
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        out = self.model(inputs)
        h = out.last_hidden_state  # (B, T, D)

        B, T, D = h.shape
        V = self.cfg.vocab_size

        # Chunked CE along T to avoid materializing full (B, T, V) logits
        cs = ce_chunk_size
        assert T % cs == 0, f"T ({T}) must be divisible by ce_chunk_size ({cs})"
        n_chunks = T // cs
        total = torch.tensor(0.0, device=tokens.device)
        for i in range(n_chunks):
            logits_chunk = self.lm_head(h[:, i * cs:(i + 1) * cs])  # (B, cs, V)
            t_chunk = targets[:, i * cs:(i + 1) * cs]
            total = total + F.cross_entropy(
                logits_chunk.reshape(-1, V), t_chunk.reshape(-1), reduction="sum"
            )
        loss = total / (B * T)

        metrics = {"ce": loss.detach()}
        return loss, metrics

    def describe(self) -> list[str]:
        c = self.cfg
        v_dim = int(c.head_dim * c.expand_v)
        return [
            f"GDN (fla): {c.n_layers}L, {c.n_heads}h × {c.head_dim}k/{v_dim}v, "
            f"hidden={c.hidden_dim}, ffn={c.ffn_dim}",
        ]
