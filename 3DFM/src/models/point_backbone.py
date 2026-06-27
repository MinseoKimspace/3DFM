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
        return self.net(t.reshape(t.shape[0], 1)).unsqueeze(1) # [B, 1, D]


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
        z: torch.Tensor, # [B, N, 3]
        t: torch.Tensor, # [B, 1, 1]
    ) -> torch.Tensor:

        h = self.point_embed(z)
        h = h + self.time_embed(t)
        h = self.blocks(h)
        return self.out(h) # [B, N, 3]

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
        xattn_every_late_block: bool = False,
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
        self.xattn_every_late_block = xattn_every_late_block
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
            z: torch.Tensor, # [B, N, 3]
            t: torch.Tensor, # [B, 1, 1]
            slot_mode: str = "normal",
        ) -> torch.Tensor:
        
        h = self.point_embed(z) # [B, N, D]
        h = h + self.time_embed(t) # [B, N, D]

        for block in self.blocks[:self.early_layers]:
            h = block(h)

        slots = self.spatial_pma(z, h)
        if slot_mode == "zero": # replace slots by zeros
            slots = torch.zeros_like(slots)
        elif slot_mode == "shuffle": # use slots from another sample in the batch
            batch_size = slots.shape[0]
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=slots.device)
                if torch.equal(perm, torch.arange(batch_size, device=slots.device)):
                    perm = torch.roll(perm, shifts=1)
                slots = slots[perm]
        elif slot_mode == "normal": # use slots from the same sample
            pass
        else:
            raise ValueError(f"Unknown slot_mode: {slot_mode}")

        for i, block in enumerate(self.blocks[self.early_layers:]):
            h = block(h)

            if self.xattn_every_late_block or i == 0:
                h = self.slot_cross_attn(h, slots)

        return self.out(h)

class XHatSelfCondBackbone(nn.Module):
    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 4,
        early_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_xhat_condition: bool = True,
        ) -> None:
        super().__init__()
        if not (0 <= early_layers < num_layers):
            raise ValueError("early_layers must be in [0, num_layers).")

        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.early_layers = early_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_xhat_condition = use_xhat_condition
        self.point_embed = nn.Linear(3, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)

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

        self.aux_head = nn.Linear(hidden_dim, 3)

        self.xhat_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)

    def forward(
            self,
            z: torch.Tensor,
            t: torch.Tensor,
            cond_mode: str = "normal",
            ) -> torch.Tensor:
        return self.forward_with_aux(z, t, cond_mode=cond_mode)["velocity"]
    
    def forward_with_aux(
            self,
            z: torch.Tensor,# [B, N, 3]
            t: torch.Tensor,# [B, 1, 1]
            cond_mode: str = "normal",
            ) -> dict:
        
        h = self.point_embed(z) # [B, N, D]
        h = h + self.time_embed(t) # [B, N, D]
    
        for block in self.blocks[:self.early_layers]:
            h = block(h)
    
        v_aux = self.aux_head(h) # [B, N, 3]
        x_hat1 = z + (1.0 - t) * v_aux # [B, N, 3]

        x_code = None
        if self.use_xhat_condition:
            if cond_mode == "zero": # remove the self-conditioning branch
                cond = torch.zeros_like(h[:, :1, :])
            elif cond_mode in ("normal", "shuffle"):
                x_code = self.xhat_embed(x_hat1) # [B, N, D]
                x_code = x_code.mean(dim=1, keepdim=True) # [B, 1, D]

                if cond_mode == "shuffle": # use x_hat1 code from another sample
                    batch_size = x_code.shape[0]
                    if batch_size > 1:
                        perm = torch.randperm(batch_size, device=x_code.device)
                        if torch.equal(perm, torch.arange(batch_size, device=x_code.device)):
                            perm = torch.roll(perm, shifts=1)
                        x_code = x_code[perm]

                cond = self.global_proj(x_code)
            else:
                raise ValueError(f"Unknown cond_mode: {cond_mode}")

            h = h + cond # broadcast: [B, N, D] + [B, 1, D]
        elif cond_mode != "normal":
            raise ValueError("cond_mode intervention requires use_xhat_condition=True.")
    
        for block in self.blocks[self.early_layers:]:
            h = block(h)
    
        velocity = self.out(h) # [B, N, 3]
        return {
            "velocity": velocity,
            "aux_velocity": v_aux,
            "x_hat1": x_hat1,
            "global_code": x_code,
        }

class XHatSpatialPMABackbone(nn.Module):
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
            xattn_every_late_block: bool = False,
            dropout: float = 0.0,
            ) -> None:
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
        self.xattn_every_late_block = xattn_every_late_block
        self.point_embed = nn.Linear(3, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.aux_head = nn.Linear(hidden_dim, 3)
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

    def forward(self, z, t, slot_mode="normal"):
        return self.forward_with_aux(z, t, slot_mode=slot_mode)["velocity"]
    
    def forward_with_aux(
            self,
            z: torch.Tensor,# [B, N, 3]
            t: torch.Tensor,# [B, 1, 1]
            slot_mode: str = "normal",
            ) -> dict:
        
        h = self.point_embed(z) # [B, N, D]
        h = h + self.time_embed(t) # [B, N, D]
    
        for block in self.blocks[:self.early_layers]:
            h = block(h)
    
        v_aux = self.aux_head(h) # [B, N, 3]
        x_hat1 = z + (1.0 - t) * v_aux # [B, N, 3]
        
        slots = self.spatial_pma(x_hat1, h)
        if slot_mode == "zero": # replace slots by zeros
            slots = torch.zeros_like(slots)
        elif slot_mode == "shuffle": # use slots from another sample in the batch
            batch_size = slots.shape[0]
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=slots.device)
                if torch.equal(perm, torch.arange(batch_size, device=slots.device)):
                    perm = torch.roll(perm, shifts=1)
                slots = slots[perm]
        elif slot_mode == "normal": # use slots from the same sample
            pass
        else:
            raise ValueError(f"Unknown slot_mode: {slot_mode}")
    
        for i, block in enumerate(self.blocks[self.early_layers:]):
            h = block(h)

            if self.xattn_every_late_block or i == 0:
                h = self.slot_cross_attn(h, slots)
    
        velocity = self.out(h) # [B, N, 3]
        return {
            "velocity": velocity,
            "aux_velocity": v_aux,
            "x_hat1": x_hat1,
        }
