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
        self.collect_stats = False
        self.last_stats: dict[str, float] = {}

    def reset_last_stats(self) -> None:
        self.last_stats = {}

    def _save_stats(self, h: torch.Tensor, delta: torch.Tensor, attn: torch.Tensor) -> None:
        # attn: [B, H, N, M]
        with torch.no_grad():
            probs = attn.detach().clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1) # [B, H, N]
            max_entropy = torch.log(torch.tensor(float(attn.shape[-1]), device=attn.device))
            delta_norm = delta.detach().flatten(1).norm(dim=1)
            h_norm = h.detach().flatten(1).norm(dim=1).clamp_min(1e-8)

            self.last_stats = {
                "attn_entropy": float(entropy.mean().item()),
                "attn_entropy_norm": float((entropy / max_entropy).mean().item()),
                "attn_max": float(attn.detach().amax(dim=-1).mean().item()),
                "delta_h_norm": float(delta_norm.mean().item()),
                "delta_h_rel_norm": float((delta_norm / h_norm).mean().item()),
            }

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        # slots: [B, M, D]
        q = self.norm_q(h)
        kv = self.norm_kv(slots)

        if self.collect_stats:
            out, attn = self.attn(
                query=q,
                key=kv,
                value=kv,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            out, attn = self.attn(query=q, key=kv, value=kv, need_weights=False)

        delta = self.dropout(out)
        if self.collect_stats and attn is not None:
            self._save_stats(h=h, delta=delta, attn=attn)

        return h + delta
