from __future__ import annotations

import torch


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module, # model(x, t) -> velocity: [B, N, 3]
    batch_size: int,
    num_points: int,
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    init: torch.Tensor | None = None,
    model_kwargs: dict | None = None,
) -> torch.Tensor:
    if model_kwargs is None:
        model_kwargs = {}

    if init is None:
        x = torch.randn(batch_size, num_points, 3, device=device, dtype=dtype)
    else:
        x = init.to(device=device, dtype=dtype)
        batch_size = x.shape[0]

    times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    self_cond = None

    for i in range(steps):
        t_now = times[i]
        t_next = times[i + 1]
        dt = t_next - t_now

        t = t_now.expand(batch_size, 1, 1)
        step_kwargs = dict(model_kwargs)
        if getattr(model, "uses_self_condition", False):
            step_kwargs["self_cond"] = self_cond
        v = model(x, t, **step_kwargs)

        if getattr(model, "uses_self_condition", False):
            self_cond = (x + (1.0 - t) * v).detach()
        x = x + dt * v

    return x # [B, N, 3]
