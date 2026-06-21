from __future__ import annotations

from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

def load_single_pyg_shape(
    root: str | Path,
    category: str,
    num_points: int,
    split: str = "train",
    index: int = 0,
) -> torch.Tensor:
    # shape : [N, 3]
    transform = T.Compose([
        T.NormalizeScale(),
        T.FixedPoints(num_points, replace=False)
    ])

    dataset = ShapeNet(
        root=str(root),
        categories=category,
        include_normals=False,
        split=split,
        transform=transform,
    )

    data = dataset[index]
    points = getattr(data, "pos", None)
    if points is None:
        raise ValueError("Expected PyG ShapeNet data to contain pos.")

    return points.float().cpu().contiguous()


def load_pyg_shapes(
    root: str | Path,
    category: str,
    num_points: int,
    split: str = "train",
    start_index: int = 0,
    num_shapes: int = 1,
) -> torch.Tensor:
    # return points: [S, N, 3]
    transform = T.Compose([
        T.NormalizeScale(),
        T.FixedPoints(num_points, replace=False)
    ])

    dataset = ShapeNet(
        root=str(root),
        categories=category,
        include_normals=False,
        split=split,
        transform=transform,
    )

    end_index = min(start_index + num_shapes, len(dataset))
    points_list = []

    for i in range(start_index, end_index):
        data = dataset[i]
        points = getattr(data, "pos", None)
        if points is None:
            raise ValueError("Expected PyG ShapeNet data to contain pos.")
        points_list.append(points.float().cpu().contiguous())

    return torch.stack(points_list, dim=0)


def build_pyg_point_cache(
    root: str | Path,
    category: str,
    num_points: int,
    split: str,
    out_path: str | Path,
    max_shapes: int | None = None,
) -> None:
    # saved points: [S, N, 3]
    transform = T.Compose([
        T.NormalizeScale(),
        T.FixedPoints(num_points, replace=False)
    ])

    dataset = ShapeNet(
        root=str(root),
        categories=category,
        include_normals=False,
        split=split,
        transform=transform,
    )

    num_shapes = len(dataset)
    if max_shapes is not None:
        num_shapes = min(num_shapes, max_shapes)

    points_list = []

    for i in range(num_shapes):
        data = dataset[i]
        points = getattr(data, "pos", None)
        if points is None:
            raise ValueError("Expected PyG ShapeNet data to contain pos.")

        points_list.append(points.float().cpu().contiguous())

    points_all = torch.stack(points_list, dim=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "points": points_all,
            "category": category,
            "split": split,
            "num_points": num_points,
            "source": "pyg_shapenet"
         }, out_path
    )

def load_point_cache(path: str | Path) -> torch.Tensor:
    # return points: [S, N, 3]
    cache = torch.load(path, map_location="cpu")
    points = cache["points"]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError()
    return points.float().cpu().contiguous()
