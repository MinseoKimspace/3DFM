from __future__ import annotations

import torch


def sample_time(
    batch: int,
    device: torch.device | str,
    dtype: torch.dtype,
    eps: float = 1e-5,
) -> torch.Tensor:
    # return t: [B, 1, 1]
    return torch.rand(batch, 1, 1, device=device, dtype=dtype) * (1.0 - 2.0*eps) + eps

def linear_path(
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    # x_data:  [B, N, 3]
    # x_noise: [B, N, 3]
    # t:       [B, 1, 1]
    # return:  [B, N, 3]
    return (1-t)*x_noise + t*x_data