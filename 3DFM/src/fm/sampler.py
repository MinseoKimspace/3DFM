from __future__ import annotations

import torch


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    batch_size: int,
    num_points: int,
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    init: torch.Tensor | None = None,
    model_kwargs: dict | None = None,
) -> torch.Tensor:
    # model(x, t) -> velocity: [B, N, 3]
    # return: [B, N, 3]
    if model_kwargs is None:
        model_kwargs = {}

    if init is None:
        x = torch.randn(batch_size, num_points, 3, device=device, dtype=dtype)
    else:
        x = init.to(device=device, dtype=dtype)
        batch_size = x.shape[0]

    times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

    for i in range(steps):
        t_now = times[i]
        t_next = times[i + 1]
        dt = t_next - t_now

        t = t_now.expand(batch_size, 1, 1)
        v = model(x, t, **model_kwargs)

        x = x + dt * v

    return x 
