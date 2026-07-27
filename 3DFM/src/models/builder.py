from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from models.point_backbone import (
    PointBackbone,
    SpatialPMABackbone,
    XHatSelfCondBackbone,
    XHatSpatialPMABackbone,
)


def _get(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_model(config: Any) -> nn.Module:
    arch = _get(config, "arch", "base")
    common = {
        "num_points": _get(config, "num_points", 1024),
        "hidden_dim": _get(config, "hidden_dim", 128),
        "num_layers": _get(config, "num_layers", 4),
        "num_heads": _get(config, "num_heads", 4),
        "dropout": _get(config, "dropout", 0.0),
    }

    if arch == "base":
        return PointBackbone(**common)

    if arch == "ptv3_base":
        from models.ptv3_backbone import PTv3FlowBackbone

        return PTv3FlowBackbone(
            num_points=_get(config, "num_points", 8192),
            grid_size=_get(config, "ptv3_grid_size", 0.01),
            time_dim=_get(config, "ptv3_time_dim", 32),
            patch_size=_get(config, "ptv3_patch_size", 128),
        )

    if arch == "spatial_pma":
        return SpatialPMABackbone(
            **common,
            early_layers=_get(config, "early_layers", 2),
            num_slots=_get(config, "num_slots", 16),
            knn_k=_get(config, "knn_k", 32),
            spatial_random_start=_get(config, "spatial_random_start", False),
            xattn_every_late_block=_get(config, "xattn_every_late_block", False),
        )

    if arch == "xhat_selfcond":
        return XHatSelfCondBackbone(
            **common,
            early_layers=_get(config, "early_layers", 2),
            use_xhat_condition=_get(config, "use_xhat_condition", True),
        )

    if arch == "xhat_spatial_pma":
        return XHatSpatialPMABackbone(
            **common,
            early_layers=_get(config, "early_layers", 2),
            num_slots=_get(config, "num_slots", 16),
            knn_k=_get(config, "knn_k", 32),
            spatial_random_start=_get(config, "spatial_random_start", False),
            xattn_every_late_block=_get(config, "xattn_every_late_block", False),
        )

    raise ValueError(f"Unknown model arch: {arch}")
