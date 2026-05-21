"""Path interfaces for unconditional point-cloud iMF.

Raw point clouds use shape [B, N, 3]. Time tensors should be shaped
[B, 1, 1] so they broadcast over points and coordinates.
"""

from __future__ import annotations

from typing import Tuple

import torch


def sample_time_pair(
    batch: int,
    device: torch.device | str,
    dtype: torch.dtype,
    mu: float = -0.4,
    sigma: float = 1.0,
    neq_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(r, t)` time tensors for iMF training.

    Args:
        batch: Batch size B.
        device: Device for returned tensors.
        dtype: Floating dtype for returned tensors.
        mu: Placeholder parameter for future time distribution.
        sigma: Placeholder parameter for future time distribution.
        neq_ratio: Placeholder ratio for non-equal `(r, t)` pairs.

    Returns:
        r: Shape [B, 1, 1].
        t: Shape [B, 1, 1].

    TODO:
        Implement the actual time-pair sampler. Keep `r <= t` if the sampler
        uses the reverse update `x <- x - (t - r) * u`.
    """

    raise NotImplementedError


def linear_path(x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Map data/noise endpoints to an interpolation point.

    Args:
        x: Data point cloud, shape [B, N, 3].
        e: Noise point cloud, shape [B, N, 3].
        t: Time tensor, shape [B, 1, 1].

    Returns:
        z_t: Point cloud on the path, shape [B, N, 3].

    TODO:
        Implement the first baseline path, likely
        `z_t = (1 - t) * x + t * e`.
    """

    raise NotImplementedError
