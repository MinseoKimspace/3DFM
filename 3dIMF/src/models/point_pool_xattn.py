"""Pooled cross-attention point model interfaces."""

from __future__ import annotations

import torch
from torch import nn


class LateXAttnBlock(nn.Module):
    """Late point block interface with optional slot cross-attention.

    Shapes:
        points: [B, N, D]
        slots: [B, K, D]
        output: [B, N, D]
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, points: torch.Tensor, slots: torch.Tensor, use_cross: bool) -> torch.Tensor:
        raise NotImplementedError


class PointPoolXAttn(nn.Module):
    """PMA plus late cross-attention model interface.

    Intended pipeline:
        early point blocks -> PMA slots -> late blocks with cross-attention
        -> pointwise `u` and `v` velocity heads.

    Model interface:
        `model(z, r, t) -> (u, v)`

    Shapes:
        z: [B, N, 3]
        r: [B, 1, 1]
        t: [B, 1, 1]
        u: average velocity, shape [B, N, 3]
        v: instantaneous velocity auxiliary head, shape [B, N, 3]

    `code_mode`:
        normal: use own slots.
        shuffle: shuffle slots across batch.
        zero: replace slots with zeros.

    `cross_attn`:
        first: only first late block reads slots.
        all: all late blocks read slots.
        none: no slot read, useful as a control.
    """

    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 128,
        early_layers: int = 2,
        late_layers: int = 2,
        num_slots: int = 8,
        code_mode: str = "normal",
        cross_attn: str = "first",
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.num_slots = num_slots
        self.code_mode = code_mode
        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
