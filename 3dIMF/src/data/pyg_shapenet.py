from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_transform(num_points: int):


    raise NotImplementedError


def build_shapenet_dataset(
    root: str,
    category: str,
    num_points: int,
    split: str = "train",
):

    raise NotImplementedError


def build_shapenet_loader(
    root: str,
    category: str,
    num_points: int,
    batch_size: int,
    num_workers: int,
    split: str = "train",
    shuffle: Optional[bool] = None,
):

    raise NotImplementedError


def build_loader_from_config(cfg: Dict[str, Any], split: str = "train"):

    raise NotImplementedError


def batch_to_points(batch, num_points: Optional[int] = None) -> torch.Tensor:

    raise NotImplementedError
