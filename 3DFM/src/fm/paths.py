from __future__ import annotations

import torch


def sample_time(
    batch: int,
    device: torch.device | str,
    dtype: torch.dtype,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.rand(batch, 1, 1, device=device, dtype=dtype) * (1.0 - 2.0*eps) + eps # [B, 1, 1]

def linear_path(
    x_data: torch.Tensor, # [B, N, 3]
    x_noise: torch.Tensor, # [B, N, 3]
    t: torch.Tensor, # [B, 1, 1]
) -> torch.Tensor:
    return (1-t)*x_noise + t*x_data # [B, N, 3]