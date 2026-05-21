"""Training script interface for 3D point-cloud iMF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""

    raise NotImplementedError


def resolve_device(name: str) -> torch.device:
    """Resolve `auto`, `cuda`, or `cpu` to a torch device."""

    raise NotImplementedError


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate `base` or `pool_xattn` from config."""

    raise NotImplementedError


def train(cfg: Dict[str, Any]) -> None:
    """Run the basic training loop.

    Expected high-level steps:
        1. Seed.
        2. Build PyG ShapeNet dataloader.
        3. Build model and optimizer.
        4. Sample `(r, t)`, noise `e`, and compute iMF loss.
        5. Log metrics.
        6. Save checkpoints.
        7. Periodically save sample point clouds.

    TODO:
        Fill in only after the model/loss interfaces are settled.
    """

    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    """CLI parser for `--config`."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise NotImplementedError("Training loop skeleton only. Implement `load_config` and `train` first.")
