from __future__ import annotations

import torch
from torch import nn


class TimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1, 1]
        # return: [B, 1, D]
        return self.net(t.reshape(t.shape[0], 1)).unsqueeze(1)


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
        self.point_embed = nn.Linear(3, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # z: [B, N, 3]
        # t: [B, 1, 1]
        # return velocity: [B, N, 3]
        h = self.point_embed(z)
        h = h + self.time_embed(t)
        h = self.blocks(h)
        return self.out(h)
