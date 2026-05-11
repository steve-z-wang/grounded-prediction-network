"""Gated Delta Net model config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    vocab_size: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    ffn_dim: int
    expand_v: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "GDNConfig":
        return cls(
            vocab_size=d["vocab_size"],
            hidden_dim=d["hidden_dim"],
            n_layers=d["n_layers"],
            n_heads=d["n_heads"],
            head_dim=d["head_dim"],
            ffn_dim=d["ffn_dim"],
            expand_v=d.get("expand_v", 2.0),
        )
