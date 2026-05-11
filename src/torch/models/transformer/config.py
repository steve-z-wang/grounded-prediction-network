"""Transformer model config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    ffn_dim: int

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(
            vocab_size=d["vocab_size"],
            hidden_dim=d["hidden_dim"],
            n_layers=d["n_layers"],
            n_heads=d["n_heads"],
            ffn_dim=d["ffn_dim"],
        )
