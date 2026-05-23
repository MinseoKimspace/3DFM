from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_V(
    model: torch.nn.Module,
    z: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    tangent_z: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> torch.Tensor:

    raise NotImplementedError


def imf_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    e: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    bc_eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    raise NotImplementedError
