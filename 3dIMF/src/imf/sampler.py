"""Sampler interfaces for point-cloud iMF models."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


@torch.no_grad()
def sample_mean_velocity(
    model: torch.nn.Module,
    batch_size: int,
    num_points: int,
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    init: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample point clouds with a mean-velocity update.

    Intended update:
        `x <- x - (t - r) * u`

    Args:
        model: Network with `model(z, r, t) -> (u, v)`. Sampling should use
            only the average velocity `u`.
        batch_size: Number of point clouds B.
        num_points: Number of points N.
        steps: Number of reverse-time steps.
        device: Torch device.
        dtype: Floating dtype.
        init: Optional initial noise, shape [B, N, 3].
        return_trajectory: Whether to return intermediate states.

    Returns:
        samples: Shape [B, N, 3].
        trajectory: Optional list of [B, N, 3] tensors.

    TODO:
        Implement the first deterministic sampler once the loss target is
        finalized.
    """

    raise NotImplementedError
