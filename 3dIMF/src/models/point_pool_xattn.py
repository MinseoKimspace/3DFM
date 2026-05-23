"""PMA + cross-attention point-cloud iMF model interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SlotConditioning:

    early_tokens: torch.Tensor
    slots: torch.Tensor
    xattn_slots: torch.Tensor


@dataclass
class PoolXAttnOutput:

    u: torch.Tensor
    v: torch.Tensor
    conditioning: SlotConditioning

class MAB(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_mult: int = 4,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_layer_norm = use_layer_norm

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PMA(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
        ff_mult: int = 4,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_layer_norm = use_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SlotCrossAttentionBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(
        self,
        point_tokens: torch.Tensor,
        slot_tokens: torch.Tensor,
        use_cross_attention: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class PointPoolXAttn(nn.Module):

    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 128,
        early_layers: int = 2,
        late_layers: int = 2,
        num_slots: int = 8,
        slot_mode: str = "normal",
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
        self.slot_mode = slot_mode
        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.dropout = dropout

    def forward_with_aux(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> PoolXAttnOutput:

        raise NotImplementedError

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError
