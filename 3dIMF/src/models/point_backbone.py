from __future__ import annotations

import torch
from torch import nn


class TimePairEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PointBackbone(nn.Module):

    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
