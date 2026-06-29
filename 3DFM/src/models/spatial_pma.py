# MAB/PMA modules adapted from the Set Transformer architecture and code:
# https://github.com/juho-lee/set_transformer
#
# Reference:
# Lee et al., "Set Transformer: A Framework for Attention-based
# Permutation-Invariant Neural Networks", ICML 2019.
#
# Original implementation licensed under the MIT License.
# This version is modified for this project

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.point_ops import index_points, farthest_point_sample, knn_point

class MAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_o = nn.Linear(dim, dim)
        self.mid = nn.Linear(dim, dim * ff_mult)
        self.end = nn.Linear(dim * ff_mult, dim) 
        self.dropout = nn.Dropout(dropout)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        
        assert self.dim % self.num_heads == 0

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        Q = self.fc_q(q)
        K, V = self.fc_k(x), self.fc_v(x)
        

        dim_split = self.dim // self.num_heads
        
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))


        identity = O
        O = self.dropout(O)
        O = self.ln1(O)
        O = self.mid(O)
        O = F.relu(O)
        O = self.end(O)
        O = O + identity        
        return O


class PMA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.dropout = dropout
        self.seed = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        learnable_s = self.seed.repeat(batch_size, 1, 1)
        return self.mab(learnable_s, x)


class SpatialPMA(nn.Module):
    def __init__(
        self,
        M: int, 
        K: int,
        random_start: bool,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ) -> None:
        super().__init__()
        self.M = M # number of spatial slots
        self.K = K # number of neighbors per anchor
        self.random_start = random_start
        self.pma = PMA(dim, num_heads, dropout=dropout)
        self.local_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(
        self,
        x_t: torch.Tensor, # [B, N, 3]  current point positions
        h: torch.Tensor, # [B, N, D]  intermediate point tokens
        return_anchors: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        anchor_idx = farthest_point_sample(x_t, self.M, self.random_start) # [B, M]

        anchors = index_points(x_t, idx = anchor_idx) # [B, M, 3]

        knn_idx = knn_point(self.K, x_t, anchors) # [B, M, K]

        local_x = index_points(x_t, knn_idx) # [B, M, K, 3]

        local_h = index_points(h, knn_idx) # [B, M, K, D]

        relative_pos = local_x - anchors[:, :, None, :] # [B, M, K, 3]

        rel_emb = self.local_proj(relative_pos) # [B, M, K, D]
        local_token = local_h + rel_emb # [B, M, K, D]
        B, M, K, D = local_token.shape[:]
        local_token = local_token.reshape(B * M, K, D) # [B*M, K, D]
        slots = self.pma(local_token) # [B*M, 1, D]
        slots = slots.squeeze(1)
        slots = slots.reshape(B, M, D)

        if return_anchors:
            return slots, anchors
        return slots

