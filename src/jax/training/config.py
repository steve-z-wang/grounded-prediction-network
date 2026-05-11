"""Training configs.

BaseTrainConfig: common fields shared across all trainers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseTrainConfig:
    # Identity / paths
    name: str
    data_dir: str
    checkpoint_dir: str
    log_dir: str
    model: object

    # Training
    batch_size: int
    seq_len: int
    grad_accum: int
    total_steps: int
    run_steps: int
    seed: int
    resume: str

    # Optimizer
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    grad_clip_norm: float

    # Logging
    log_every: int
    save_every: int

    # Optimization (doesn't change the math)
    checkpoint_every: int = 0
    bf16: bool = False

    @classmethod
    def _base_fields(cls, config: dict, model_config):
        dirs = config["dirs"]
        t = config["training"]
        o = config["optimizer"]
        l = config["logging"]
        return dict(
            name=config["name"],
            data_dir=dirs["data"],
            checkpoint_dir=dirs["checkpoint"],
            log_dir=dirs["log"],
            model=model_config,
            batch_size=t["batch_size"],
            seq_len=t["seq_len"],
            grad_accum=t["grad_accum"],
            total_steps=t["total_steps"],
            run_steps=t.get("run_steps", t["total_steps"]),
            seed=t["seed"],
            resume=t.get("resume", ""),
            learning_rate=o["learning_rate"],
            adam_beta1=o["adam_beta1"],
            adam_beta2=o["adam_beta2"],
            weight_decay=o["weight_decay"],
            warmup_steps=o["warmup_steps"],
            min_lr_ratio=o["min_lr_ratio"],
            grad_clip_norm=o["grad_clip_norm"],
            log_every=l["log_every"],
            save_every=l["save_every"],
            checkpoint_every=t.get("checkpoint_every", 0),
            bf16=t.get("bf16", False),
        )

    @classmethod
    def from_dict(cls, config: dict, model_config=None) -> "BaseTrainConfig":
        return cls(**cls._base_fields(config, model_config))
