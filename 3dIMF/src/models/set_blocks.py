"""Set Transformer block interfaces."""

from __future__ import annotations

import torch
from torch import nn


class CanonicalMAB(nn.Module):
    """Set Transformer Multihead Attention Block interface.

    Shape:
        X: Query tokens, shape [B, Tq, D].
        Y: Key/value tokens, shape [B, Tk, D].
        output: Updated query tokens, shape [B, Tq, D].

    TODO:
        Implement the canonical MAB structure:
        attention residual, layer norm, feed-forward residual, layer norm.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PMA(nn.Module):
    """Pooling by Multihead Attention interface.

    Shape:
        x: Input tokens, shape [B, T, D].
        output: Slots, shape [B, K, D].

    TODO:
        Add learned seed slots and apply `CanonicalMAB(seed, x)`.
    """

    def __init__(self, dim: int, num_slots: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
