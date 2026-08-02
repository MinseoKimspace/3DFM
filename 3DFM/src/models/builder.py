from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


def _get(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_model(config: Any) -> nn.Module:
    from models.dipt_backbone import (
        DiPTFlowBackbone,
        DiPTSpatialPMABackbone,
        DiPTXHatSelfConditionBackbone,
        DiPTXHatSpatialPMABackbone,
    )

    arch = _get(config, "arch", "dipt_base")
    common = {
        "num_points": _get(config, "num_points", 2048),
        "grid_size": _get(config, "dipt_grid_size", 0.02),
        "depth": _get(config, "dipt_depth", 8),
        "channels": _get(config, "dipt_channels", 384),
        "num_heads": _get(config, "dipt_num_heads", 6),
        "patch_size": _get(config, "dipt_patch_size", None)
        or [256, 512, 1024, 1024, 256, 512, 1024, 1024],
        "time_scale": _get(config, "dipt_time_scale", 1000.0),
        "shuffle_orders": _get(config, "dipt_shuffle_orders", False),
    }

    if arch == "dipt_base":
        return DiPTFlowBackbone(**common)

    if arch == "dipt_spatial_pma":
        return DiPTSpatialPMABackbone(
            **common,
            early_layers=_get(config, "early_layers", 4),
            num_slots=_get(config, "num_slots", 64),
            knn_k=_get(config, "knn_k", 64),
            spatial_random_start=_get(config, "spatial_random_start", False),
            xattn_every_late_block=_get(config, "xattn_every_late_block", False),
        )

    if arch == "dipt_xhat_anchor_pma":
        return DiPTXHatSpatialPMABackbone(
            **common,
            early_layers=_get(config, "early_layers", 4),
            num_slots=_get(config, "num_slots", 64),
            knn_k=_get(config, "knn_k", 64),
            spatial_random_start=_get(config, "spatial_random_start", False),
            xattn_every_late_block=_get(config, "xattn_every_late_block", False),
        )

    if arch == "dipt_xhat_selfcond":
        return DiPTXHatSelfConditionBackbone(
            **common,
            early_layers=_get(config, "early_layers", 4),
        )

    raise ValueError(f"Unknown model arch: {arch}")
