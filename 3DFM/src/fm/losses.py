from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from fm.paths import linear_path, sample_time

def fm_loss(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    aux_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # x_data: [B, N, 3]
    # model(z_t, t) -> velocity: [B, N, 3]
    x_noise = torch.randn_like(x_data)
    t= sample_time(batch=x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
    z_t = linear_path(x_data, x_noise, t)
    target = x_data - x_noise

    aux_loss = torch.zeros((), device=x_data.device, dtype=x_data.dtype)
    if aux_weight > 0.0:
        if not hasattr(model, "forward_with_aux"):
            raise ValueError("aux_weight > 0 requires model.forward_with_aux.")
        out = model.forward_with_aux(z_t, t)
        pred = out["velocity"]
        aux_pred = out["aux_velocity"]
        aux_loss = F.mse_loss(aux_pred, target)
    else:
        pred = model(z_t, t)

    main_loss = F.mse_loss(pred, target)
    loss = main_loss + aux_weight * aux_loss

    metrics = {
        "loss": loss.detach(),
        "fm_mse": main_loss.detach(),
        "aux_mse": aux_loss.detach(),
    }

    return loss, metrics
