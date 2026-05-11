"""Train base SoT — JAX.

Usage:
    PYTHONPATH=. python -m src.models.sot.train --config <config.json>
"""

import json
import argparse

import jax
import jax.numpy as jnp

from src.data import TokenDataset
from src.jax.training import BaseTrainConfig
from src.jax.models.sot.model import SoT
from src.jax.models.sot.config import SoTConfig
from src.jax.models.sot.trainer import SoTTrainer


def main(config: dict) -> None:
    model_config = SoTConfig.from_dict(config["model"])
    cfg = BaseTrainConfig.from_dict(config, model_config)
    dataset = TokenDataset(cfg.data_dir, cfg.seq_len)

    model = SoT(cfg=model_config)
    rng = jax.random.PRNGKey(cfg.seed)
    dummy_tokens = jnp.zeros((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_tokens)
    params = variables['params']

    trainer = SoTTrainer()
    trainer.setup(model, params, dataset, cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)
