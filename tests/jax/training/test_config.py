from src.jax.training.config import BaseTrainConfig


BASE_CFG = {
    "name": "test",
    "dirs": {"data": "/tmp/data", "checkpoint": "/tmp/ckpt", "log": "/tmp/log"},
    "training": {
        "batch_size": 4,
        "seq_len": 128,
        "grad_accum": 2,
        "ce_chunk_size": 32,
        "total_steps": 1000,
        "seed": 42,
    },
    "optimizer": {
        "learning_rate": 3e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "grad_clip_norm": 1.0,
    },
    "logging": {"log_every": 1, "save_every": 100},
}


def test_base_config_from_dict():
    cfg = BaseTrainConfig.from_dict(BASE_CFG)
    assert cfg.name == "test"
    assert cfg.seq_len == 128
    assert cfg.grad_accum == 2
    assert cfg.learning_rate == 3e-4


def test_base_config_defaults():
    cfg = BaseTrainConfig.from_dict(BASE_CFG)
    assert cfg.resume == ""
    assert cfg.bf16 is False
    assert cfg.run_steps == cfg.total_steps
    assert cfg.checkpoint_every == 0


def test_checkpoint_every():
    d = {**BASE_CFG, "training": {**BASE_CFG["training"], "checkpoint_every": 64}}
    cfg = BaseTrainConfig.from_dict(d)
    assert cfg.checkpoint_every == 64
