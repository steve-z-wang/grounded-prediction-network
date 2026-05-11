"""SoT model config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SoTConfig:
    vocab_size: int
    d_state: int
    d_emb: int
    d_ffn: int
    # If True, adds a learnable (d_emb,) bias to the decode projection
    # before the embedding matmul — explicitly represents the "frequency
    # prior" direction that otherwise emerges as the state's DC attractor.
    use_decoder_bias: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "SoTConfig":
        return cls(
            vocab_size=d["vocab_size"],
            d_state=d["d_state"],
            d_emb=d["d_emb"],
            d_ffn=d["d_ffn"],
            use_decoder_bias=d.get("use_decoder_bias", False),
        )
