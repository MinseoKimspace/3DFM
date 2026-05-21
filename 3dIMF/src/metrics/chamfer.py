"""Chamfer distance metric interface."""

from __future__ import annotations

import torch


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute symmetric Chamfer distance.

    Args:
        x: First point cloud batch, shape [B, N, 3].
        y: Second point cloud batch, shape [B, M, 3].
        squared: Whether to use squared Euclidean distance.
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar tensor or per-sample tensor with shape [B].

    TODO:
        Implement a simple `torch.cdist` version first.
    """

    raise NotImplementedError
