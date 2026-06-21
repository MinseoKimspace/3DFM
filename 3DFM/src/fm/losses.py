from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from fm.paths import linear_path, sample_time

def fm_loss(
    model: torch.nn.Module,
    x_data: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # x_data: [B, N, 3]
    # model(z_t, t) -> velocity: [B, N, 3]
    x_noise = torch.randn_like(x_data)
    t= sample_time(batch=x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
    z_t = linear_path(x_data, x_noise, t)
    target = x_data - x_noise
    pred = model(z_t, t) 
    loss = F.mse_loss(pred, target)

    metrics = {
        "loss": loss.detach(),
        "fm_mse": loss.detach(),
    }

    return loss, metrics
