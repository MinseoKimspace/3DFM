"""Placeholders for raw point-cloud transforms."""

from __future__ import annotations

import torch


def ensure_float_points(pos: torch.Tensor) -> torch.Tensor:
    """Validate/convert one point cloud.

    Args:
        pos: Point tensor, expected shape [N, 3].

    Returns:
        Float tensor with shape [N, 3].

    TODO:
        Decide whether this helper should only validate or also normalize.
    """

    raise NotImplementedError
