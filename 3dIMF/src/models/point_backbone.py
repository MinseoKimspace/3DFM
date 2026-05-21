"""Baseline point iMF model interfaces."""

from __future__ import annotations

import torch
from torch import nn


class TimePairEmbedding(nn.Module):
    """Embed `(r, t)` scalar pairs.

    Input:
        r: [B, 1, 1]
        t: [B, 1, 1]

    Output:
        emb: [B, D]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PointBackbone(nn.Module):
    """Simple point/set iMF backbone interface.

    Model interface:
        `model(z, r, t) -> (u, v)`

    Shapes:
        z: [B, N, 3]
        r: [B, 1, 1]
        t: [B, 1, 1]
        u: average velocity, shape [B, N, 3]
        v: instantaneous velocity auxiliary head, shape [B, N, 3]

    TODO:
        Add point embedding MLP, time embedding, simple self-attention blocks,
        and two pointwise output heads: `u_head` and `v_head`.
    """

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
