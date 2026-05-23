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

    raise NotImplementedError
