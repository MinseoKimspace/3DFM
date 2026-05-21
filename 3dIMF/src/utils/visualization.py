"""Visualization/export interfaces."""

from __future__ import annotations

from pathlib import Path

import torch


def save_point_cloud_ply(points: torch.Tensor, path: str | Path) -> None:
    """Save one point cloud as `.ply`.

    Args:
        points: Shape [N, 3].
        path: Output path.

    TODO:
        Use Open3D first. Raise a friendly ImportError if Open3D is missing.
    """

    raise NotImplementedError


def save_point_cloud_batch(points: torch.Tensor, out_dir: str | Path, prefix: str) -> None:
    """Save a batch of point clouds.

    Args:
        points: Shape [B, N, 3].
        out_dir: Output directory.
        prefix: File prefix.
    """

    raise NotImplementedError
