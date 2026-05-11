"""Train Transformer++.

Usage:
    PYTHONPATH=. python -m src.models.transformer.train --config <config.json>
"""

import json
import argparse

import torch

# cuDNN SDPA has known issues on some H100 setups; use Flash Attention instead
torch.backends.cuda.enable_cudnn_sdp(False)

from src.data import TokenDataset
from src.torch.training import BaseTrainConfig
from src.torch.models.transformer.model import Transformer
from src.torch.models.transformer.config import TransformerConfig
from src.torch.models.transformer.trainer import TransformerTrainer


def main(config: dict) -> None:
    model_config = TransformerConfig.from_dict(config["model"])
    cfg = BaseTrainConfig.from_dict(config, model_config)
    dataset = TokenDataset(cfg.data_dir, cfg.seq_len + 1)
    torch.manual_seed(cfg.seed)

    model = Transformer(model_config).cuda()
    trainer = TransformerTrainer()
    trainer.setup(model, dataset, cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)
