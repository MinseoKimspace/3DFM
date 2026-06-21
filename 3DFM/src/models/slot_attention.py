from __future__ import annotations
import torch
import torch.nn as nn


class SlotCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        # slots: [B, M, D]
        q = self.norm_q(h)
        kv = self.norm_kv(slots)

        out, _ = self.attn(query = q, key = kv, value = kv, need_weights = False)
        
        return h + self.dropout(out)
