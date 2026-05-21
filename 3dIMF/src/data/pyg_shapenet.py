"""Interfaces for PyG ShapeNet loading.

Early experiments use `torch_geometric.datasets.ShapeNet` with
`NormalizeScale` and `FixedPoints(num_points)`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_transform(num_points: int):
    """Build the PyG transform pipeline.

    Args:
        num_points: Number of fixed points per cloud.

    Returns:
        A PyG transform object.

    TODO:
        Compose `NormalizeScale` and `FixedPoints(num_points)`.
    """

    raise NotImplementedError


def build_shapenet_dataset(
    root: str,
    category: str,
    num_points: int,
    split: str = "train",
):
    """Create a PyG ShapeNet dataset for one category.

    Args:
        root: Dataset root directory.
        category: ShapeNet category, e.g. "Chair".
        num_points: Number of points after transform.
        split: PyG split name.

    Returns:
        A PyG ShapeNet dataset.
    """

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
    """Create a dataloader for PyG ShapeNet.

    TODO:
        Keep this thin. It should mostly wire config fields into PyG.
    """

    raise NotImplementedError


def build_loader_from_config(cfg: Dict[str, Any], split: str = "train"):
    """Build a ShapeNet loader from the project config dict."""

    raise NotImplementedError


def batch_to_points(batch, num_points: Optional[int] = None) -> torch.Tensor:
    """Convert a PyG Batch to dense point clouds.

    Args:
        batch: PyG Batch with `pos` shape [B * N, 3] and `batch` indices.
        num_points: Expected N, or None to infer it.

    Returns:
        points: Tensor with shape [B, N, 3].

    TODO:
        Validate that every item has exactly N points before reshaping.
    """

    raise NotImplementedError
