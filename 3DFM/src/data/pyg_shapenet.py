from __future__ import annotations

from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet


def load_pyg_shapes(
    root: str | Path,
    category: str,
    num_points: int,
    split: str = "train",
    start_index: int = 0,
    num_shapes: int = 1,
    replace: bool = False,
) -> torch.Tensor:
    transform = T.Compose([
        T.NormalizeScale(),
        T.FixedPoints(num_points, replace=replace)
    ])

    dataset = ShapeNet(
        root=str(root),
        categories=category,
        include_normals=False,
        split=split,
        transform=transform,
    )

    end_index = (
        len(dataset)
        if num_shapes <= 0
        else min(start_index + num_shapes, len(dataset))
    )
    points_list = []

    for i in range(start_index, end_index):
        data = dataset[i]
        points = getattr(data, "pos", None)
        if points is None:
            raise ValueError("Expected PyG ShapeNet data to contain pos.")
        points_list.append(points.float().cpu().contiguous())

    return torch.stack(points_list, dim=0)  # [S, N, 3]
