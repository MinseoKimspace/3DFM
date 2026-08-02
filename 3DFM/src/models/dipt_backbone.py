from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.slot_attention import SlotCrossAttentionBlock
from models.spatial_pma import SpatialPMA

try:
    from third_party.dipt import DiffusionPointTransformer
    from third_party.dipt.structure import Point
except ModuleNotFoundError as exc:
    required = ("addict", "spconv", "timm", "torch_scatter")
    missing = exc.name or ""
    if any(missing == name or missing.startswith(f"{name}.") for name in required):
        raise ModuleNotFoundError(
            "DiPT dependencies are missing. See DIPT_SETUP.md for installation."
        ) from exc
    raise


def _apply_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "normal":
        return x
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "shuffle":
        if x.shape[0] <= 1:
            return x
        perm = torch.randperm(x.shape[0], device=x.device)
        if torch.equal(perm, torch.arange(x.shape[0], device=x.device)):
            perm = torch.roll(perm, shifts=1)
        return x[perm]
    raise ValueError(f"Unknown intervention mode: {mode}")


class DiPTFlowBackbone(nn.Module):
    """DiPT with the project-wide `[B, N, 3]` Flow Matching interface."""

    def __init__(
        self,
        num_points: int = 2048,
        grid_size: float = 0.02,
        depth: int = 8,
        channels: int = 384,
        num_heads: int = 6,
        patch_size: int | Sequence[int] = (256, 512, 1024, 1024) * 2,
        time_scale: float = 1000.0,
        shuffle_orders: bool = False,
    ) -> None:
        super().__init__()
        if grid_size <= 0.0:
            raise ValueError("grid_size must be positive.")
        if depth <= 0 or channels <= 0 or num_heads <= 0:
            raise ValueError("depth, channels, and num_heads must be positive.")
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive.")

        if isinstance(patch_size, int):
            patch_sizes = (patch_size,) * depth
        else:
            patch_sizes = tuple(int(size) for size in patch_size)
        if len(patch_sizes) != depth:
            raise ValueError("dipt_patch_size must contain one value per layer.")

        self.num_points = num_points
        self.depth = depth
        self.channels = channels
        self.num_heads = num_heads
        self.grid_size = float(grid_size)
        self.time_scale = float(time_scale)
        self.backbone = DiffusionPointTransformer(
            in_channels=3,
            num_classes=1,
            cls_drop=0.0,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            depth=depth,
            channels=channels,
            num_head=num_heads,
            patch_size=patch_sizes,
            frequency_embedding_size=channels,
            attn_drop=0.2,
            proj_drop=0.3,
            drop_path=0.3,
            shuffle_orders=shuffle_orders,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
        )
        self.out_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(channels, 2 * channels),
        )
        self.out_norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(channels, 3)

    def _prepare_point(self, z: torch.Tensor, t: torch.Tensor) -> Point:
        if z.ndim != 3 or z.shape[-1] != 3:
            raise ValueError(f"Expected z with shape [B, N, 3], got {tuple(z.shape)}.")
        if t.shape[0] != z.shape[0]:
            raise ValueError("z and t must have the same batch size.")

        batch_size, num_points, _ = z.shape
        coord = z.reshape(batch_size * num_points, 3).contiguous()
        batch = torch.arange(batch_size, device=z.device).repeat_interleave(num_points)
        point = Point(
            {
                "coord": coord,
                "feat": coord,
                "batch": batch,
                "grid_size": self.grid_size,
                "timesteps": t.reshape(batch_size, -1)[:, 0] * self.time_scale,
                "cls_token": torch.zeros(batch_size, dtype=torch.long, device=z.device),
            }
        )
        point.serialization(
            order=self.backbone.order,
            shuffle_orders=self.backbone.shuffle_orders,
        )
        point.sparsify()
        point.condition = self.backbone.timestep_embedding(point.timesteps)
        point.condition = point.condition + self.backbone.cls_embedding(
            point.cls_token,
            train=self.training,
        )
        return self.backbone.embedding(point)

    def _run_blocks(self, point: Point, start: int, end: int) -> Point:
        blocks = tuple(self.backbone.enc.children())
        for block in blocks[start:end]:
            point = block(point)
        return point

    @staticmethod
    def _as_batch(point: Point, batch_size: int, num_points: int) -> torch.Tensor:
        return point.feat.reshape(batch_size, num_points, -1)

    @staticmethod
    def _replace_feat(point: Point, h: torch.Tensor) -> Point:
        point.feat = h.reshape(-1, h.shape[-1]).contiguous()
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

    def _predict_velocity(self, point: Point, batch_size: int, num_points: int) -> torch.Tensor:
        shift, scale = self.out_modulation(point.condition).chunk(2, dim=-1)
        shift = shift.repeat_interleave(num_points, dim=0)
        scale = scale.repeat_interleave(num_points, dim=0)
        feat = self.out_norm(point.feat) * (1.0 + scale) + shift
        return self.out(feat).reshape(batch_size, num_points, 3)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = z.shape
        point = self._prepare_point(z, t)
        point = self._run_blocks(point, 0, self.depth)
        return self._predict_velocity(point, batch_size, num_points)


class DiPTSpatialPMABackbone(DiPTFlowBackbone):
    def __init__(
        self,
        early_layers: int = 4,
        num_slots: int = 64,
        knn_k: int = 64,
        spatial_random_start: bool = False,
        xattn_every_late_block: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < early_layers < self.depth:
            raise ValueError("early_layers must be in (0, depth).")
        if num_slots > self.num_points or knn_k > self.num_points:
            raise ValueError("num_slots and knn_k must be <= num_points.")

        self.early_layers = early_layers
        self.xattn_every_late_block = xattn_every_late_block
        self.spatial_pma = SpatialPMA(
            M=num_slots,
            K=knn_k,
            random_start=spatial_random_start,
            dim=self.channels,
            num_heads=self.num_heads,
        )
        self.slot_cross_attn = SlotCrossAttentionBlock(self.channels, self.num_heads)

    def _inject_late(self, point: Point, slots: torch.Tensor, batch_size: int, num_points: int) -> Point:
        for index in range(self.early_layers, self.depth):
            point = self._run_blocks(point, index, index + 1)
            if self.xattn_every_late_block or index == self.early_layers:
                h = self._as_batch(point, batch_size, num_points)
                point = self._replace_feat(point, self.slot_cross_attn(h, slots))
        return point

    def forward(self, z: torch.Tensor, t: torch.Tensor, slot_mode: str = "normal") -> torch.Tensor:
        batch_size, num_points, _ = z.shape
        point = self._prepare_point(z, t)
        point = self._run_blocks(point, 0, self.early_layers)
        h = self._as_batch(point, batch_size, num_points)
        slots = _apply_mode(self.spatial_pma(z, h), slot_mode)
        point = self._inject_late(point, slots, batch_size, num_points)
        return self._predict_velocity(point, batch_size, num_points)


class DiPTXHatSpatialPMABackbone(DiPTSpatialPMABackbone):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.aux_head = nn.Linear(self.channels, 3)

    def forward(self, z: torch.Tensor, t: torch.Tensor, slot_mode: str = "normal") -> torch.Tensor:
        return self.forward_with_aux(z, t, slot_mode=slot_mode)["velocity"]

    def forward_with_aux(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        slot_mode: str = "normal",
    ) -> dict[str, torch.Tensor]:
        batch_size, num_points, _ = z.shape
        point = self._prepare_point(z, t)
        point = self._run_blocks(point, 0, self.early_layers)
        h = self._as_batch(point, batch_size, num_points)
        aux_velocity = self.aux_head(h)
        x_hat1 = z + (1.0 - t) * aux_velocity
        slots = _apply_mode(self.spatial_pma(x_hat1, h), slot_mode)
        point = self._inject_late(point, slots, batch_size, num_points)
        velocity = self._predict_velocity(point, batch_size, num_points)
        return {
            "velocity": velocity,
            "aux_velocity": aux_velocity,
            "x_hat1": x_hat1,
        }


class DiPTXHatSelfConditionBackbone(DiPTFlowBackbone):
    uses_self_condition = True

    def __init__(self, early_layers: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 < early_layers < self.depth:
            raise ValueError("early_layers must be in (0, depth).")
        self.early_layers = early_layers
        self.self_cond_embed = nn.Sequential(
            nn.Linear(3, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, self.channels),
        )
        self.self_cond_proj = nn.Linear(self.channels, self.channels)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        self_cond: torch.Tensor | None = None,
        cond_mode: str = "normal",
    ) -> torch.Tensor:
        batch_size, num_points, _ = z.shape
        point = self._prepare_point(z, t)
        point = self._run_blocks(point, 0, self.early_layers)

        if self_cond is not None and cond_mode != "zero":
            if self_cond.shape != z.shape:
                raise ValueError("self_cond must have the same shape as z.")
            conditioned = _apply_mode(self_cond, cond_mode)
            code = self.self_cond_embed(conditioned).mean(dim=1, keepdim=True)
            h = self._as_batch(point, batch_size, num_points)
            point = self._replace_feat(point, h + self.self_cond_proj(code))
        elif cond_mode not in ("normal", "shuffle", "zero"):
            raise ValueError(f"Unknown conditioning mode: {cond_mode}")

        point = self._run_blocks(point, self.early_layers, self.depth)
        return self._predict_velocity(point, batch_size, num_points)
