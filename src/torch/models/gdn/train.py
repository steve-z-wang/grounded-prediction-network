"""Train Gated Delta Net.

Usage:
    PYTHONPATH=. python -m src.models.gdn.train --config <config.json>
"""

import json
import argparse

import torch

from src.data import TokenDataset
from src.torch.training import BaseTrainConfig
from src.torch.models.gdn.model import GDN
from src.torch.models.gdn.config import GDNConfig
from src.torch.models.gdn.trainer import GDNTrainer


def main(config: dict) -> None:
    model_config = GDNConfig.from_dict(config["model"])
    cfg = BaseTrainConfig.from_dict(config, model_config)
    dataset = TokenDataset(cfg.data_dir, cfg.seq_len + 1)
    torch.manual_seed(cfg.seed)

    model = GDN(model_config).cuda()
    trainer = GDNTrainer()
    trainer.setup(model, dataset, cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)
