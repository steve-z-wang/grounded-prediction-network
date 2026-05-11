"""Train SoT+M — JAX.

Usage:
    PYTHONPATH=. python -m src.models.sot_m.train --config <config.json>
"""

import json
import argparse

import jax
import jax.numpy as jnp

from src.data import TokenDataset
from src.jax.training import BaseTrainConfig
from src.jax.models.sot_m.model import SoTM
from src.jax.models.sot_m.config import SoTMConfig
from src.jax.models.sot_m.trainer import SoTMTrainer


def main(config: dict) -> None:
    model_config = SoTMConfig.from_dict(config["model"])
    cfg = BaseTrainConfig.from_dict(config, model_config)
    dataset = TokenDataset(cfg.data_dir, cfg.seq_len)

    model = SoTM(cfg=model_config)
    rng = jax.random.PRNGKey(cfg.seed)
    dummy_tokens = jnp.zeros((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_tokens)
    params = variables['params']

    trainer = SoTMTrainer()
    trainer.setup(model, params, dataset, cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)
