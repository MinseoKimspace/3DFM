from __future__ import annotations
import torch
import torch.nn as nn

from models.point_ops import index_points, knn_point


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


class KNNSlotReadBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        slot_read_k: int = 1,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if slot_read_k < 1:
            raise ValueError("slot_read_k must be >= 1.")

        self.dim = dim
        self.num_heads = num_heads
        self.slot_read_k = slot_read_k
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.assigned_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.rel_proj = nn.Sequential(
            nn.Linear(3, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.assigned_update = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=False),
        )
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.collect_stats = False
        self.last_stats: dict[str, float] = {}

    def reset_last_stats(self) -> None:
        self.last_stats = {}

    def _save_delta_stats(
        self,
        h: torch.Tensor,
        delta: torch.Tensor,
        rel: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            delta_norm = delta.detach().flatten(1).norm(dim=1)
            h_norm = h.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
            self.last_stats = {
                "slot_read_k": float(self.slot_read_k),
                "slot_rel_dist": float(rel.detach().norm(dim=-1).mean().item()),
                "delta_h_norm": float(delta_norm.mean().item()),
                "delta_h_rel_norm": float((delta_norm / h_norm).mean().item()),
            }

    def _save_attention_stats(
        self,
        h: torch.Tensor,
        delta: torch.Tensor,
        attn: torch.Tensor,
        rel: torch.Tensor,
    ) -> None:
        # attn: [B, H, N, R]
        with torch.no_grad():
            probs = attn.detach().clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(attn.shape[-1]), device=attn.device))
            delta_norm = delta.detach().flatten(1).norm(dim=1)
            h_norm = h.detach().flatten(1).norm(dim=1).clamp_min(1e-8)

            self.last_stats = {
                "slot_read_k": float(self.slot_read_k),
                "slot_rel_dist": float(rel.detach().norm(dim=-1).mean().item()),
                "attn_entropy": float(entropy.mean().item()),
                "attn_entropy_norm": float((entropy / max_entropy).mean().item()),
                "attn_max": float(attn.detach().amax(dim=-1).mean().item()),
                "delta_h_norm": float(delta_norm.mean().item()),
                "delta_h_rel_norm": float((delta_norm / h_norm).mean().item()),
            }

    def forward(
        self,
        h: torch.Tensor,
        slots: torch.Tensor,
        point_pos: torch.Tensor,
        anchors: torch.Tensor,
        use_rel_pos: bool = True,
    ) -> torch.Tensor:
        # h: [B, N, D]
        # slots: [B, M, D]
        # point_pos: [B, N, 3]
        # anchors: [B, M, 3]
        slot_idx = knn_point(self.slot_read_k, anchors, point_pos) # [B, N, R]
        local_slots = index_points(slots, slot_idx) # [B, N, R, D]
        local_anchors = index_points(anchors, slot_idx) # [B, N, R, 3]
        rel = point_pos[:, :, None, :] - local_anchors # [B, N, R, 3]

        if use_rel_pos:
            rel_emb = self.rel_proj(rel)
        else:
            rel_emb = torch.zeros_like(local_slots)

        local_tokens = local_slots + rel_emb

        if self.slot_read_k == 1:
            cond = self.assigned_norm(local_tokens.squeeze(2)) # [B, N, D]
            delta = self.dropout(self.assigned_update(cond))
            if self.collect_stats:
                self._save_delta_stats(h=h, delta=delta, rel=rel)
            return h + delta

        batch_size, num_points, read_k, dim = local_tokens.shape
        q = self.norm_q(h).reshape(batch_size * num_points, 1, dim)
        kv = self.norm_kv(local_tokens).reshape(batch_size * num_points, read_k, dim)

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

        delta = self.dropout(out.reshape(batch_size, num_points, dim))
        if self.collect_stats and attn is not None:
            attn = attn.squeeze(2).reshape(
                batch_size,
                num_points,
                self.num_heads,
                read_k,
            )
            attn = attn.permute(0, 2, 1, 3).contiguous()
            self._save_attention_stats(h=h, delta=delta, attn=attn, rel=rel)

        return h + delta
