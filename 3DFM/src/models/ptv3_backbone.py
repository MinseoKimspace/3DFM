from __future__ import annotations

import torch
from torch import nn

from models.point_backbone import TimeEmbedding

try:
    from third_party.ptv3.model import PointTransformerV3
except ModuleNotFoundError as exc:
    required = ("addict", "spconv", "timm", "torch_scatter")
    missing = exc.name or ""
    if any(missing == name or missing.startswith(f"{name}.") for name in required):
        raise ModuleNotFoundError(
            "PTv3 dependencies are missing. See PTV3_SETUP.md for installation."
        ) from exc
    raise


class PTv3FlowBackbone(nn.Module):
    """PTv3 adapter with the project-wide Flow Matching interface."""

    def __init__(
        self,
        num_points: int = 8192,
        grid_size: float = 0.01,
        time_dim: int = 32,
        patch_size: int = 128,
    ) -> None:
        super().__init__()
        if grid_size <= 0.0:
            raise ValueError("grid_size must be positive.")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        self.num_points = num_points
        self.grid_size = float(grid_size)
        self.time_dim = time_dim
        self.time_embed = TimeEmbedding(time_dim)

        self.backbone = PointTransformerV3(
            in_channels=3 + time_dim,
            enc_patch_size=(patch_size,) * 5,
            dec_patch_size=(patch_size,) * 4,
            shuffle_orders=False,
            enable_flash=False,
            cls_mode=False,
        )
        self.out = nn.Linear(64, 3)

    def forward(
        self,
        z: torch.Tensor,  # [B, N, 3]
        t: torch.Tensor,  # [B, 1, 1]
    ) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != 3:
            raise ValueError(f"Expected z with shape [B, N, 3], got {tuple(z.shape)}.")
        if t.shape[0] != z.shape[0]:
            raise ValueError("z and t must have the same batch size.")

        batch_size, num_points, _ = z.shape
        time_feat = self.time_embed(t).expand(batch_size, num_points, self.time_dim)
        feat = torch.cat((z, time_feat), dim=-1)

        coord = z.reshape(batch_size * num_points, 3).contiguous()
        feat = feat.reshape(batch_size * num_points, -1).contiguous()
        batch = torch.arange(
            batch_size,
            device=z.device,
            dtype=torch.long,
        ).repeat_interleave(num_points)

        point = self.backbone(
            {
                "coord": coord,
                "feat": feat,
                "batch": batch,
                "grid_size": self.grid_size,
            }
        )
        if point.feat.shape[0] != batch_size * num_points:
            raise RuntimeError(
                "PTv3 did not return one feature per input point. "
                "Check grid_size and duplicate voxel coordinates."
            )

        velocity = self.out(point.feat)
        return velocity.reshape(batch_size, num_points, 3)
