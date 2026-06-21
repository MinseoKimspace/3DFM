from __future__ import annotations

import torch
from torch import nn
from models.spatial_pma import SpatialPMA
from models.slot_attention import SlotCrossAttentionBlock

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

class SpatialPMABackbone(nn.Module):
    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 4,
        early_layers: int = 2,
        num_heads: int = 4,
        num_slots: int = 16,
        knn_k: int = 32,
        spatial_random_start: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (0 <= early_layers < num_layers):
            raise ValueError("early_layers must be in [0, num_layers).")
        if num_slots > num_points:
            raise ValueError("num_slots must be <= num_points.")
        if knn_k > num_points:
            raise ValueError("knn_k must be <= num_points.")

        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.early_layers = early_layers
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.knn_k = knn_k
        self.dropout = dropout
        self.point_embed = nn.Linear(3, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.spatial_pma = SpatialPMA(
            M=num_slots,
            K=knn_k,
            random_start=spatial_random_start,
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.slot_cross_attn = SlotCrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
            self,
            z: torch.Tensor,
            t: torch.Tensor
        ) -> torch.Tensor:
        
        h = self.point_embed(z)
        h = h + self.time_embed(t)

        for block in self.blocks[:self.early_layers]:
            h = block(h)

        slots = self.spatial_pma(z, h)

        for i, block in enumerate(self.blocks[self.early_layers:]):
            h = block(h)

            if i == 0:
                h = self.slot_cross_attn(h, slots)

        return self.out(h)
