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

    raise NotImplementedError


def linear_path(x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    raise NotImplementedError
