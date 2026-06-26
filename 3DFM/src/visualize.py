from __future__ import annotations

import argparse
from pathlib import Path

import torch


def as_single_point_cloud(
        points: torch.Tensor, # [N, 3] or [B, N, 3]
        index: int = 0,
        ) -> torch.Tensor:
    
    if points.ndim == 2:
        return points # N, 3]
    if points.ndim == 3:
        return points[index]
    raise ValueError(f"Expected [N, 3] or [B, N, 3], got shape {tuple(points.shape)}")


def save_point_cloud_ply(points: torch.Tensor, path: str | Path, index: int = 0) -> None:
    # points: [N, 3] or [B, N, 3]
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required to save .ply files. Install it with `pip install open3d`."
        ) from exc

    points = as_single_point_cloud(points.detach().cpu().float(), index=index)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected last dim to be 3, got shape {tuple(points.shape)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.numpy())
    o3d.io.write_point_cloud(str(path), point_cloud)


def load_points(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "points" in obj:
            return obj["points"]
        if "samples" in obj:
            return obj["samples"]
        if "sample" in obj:
            return obj["sample"]
        raise KeyError("Expected dict checkpoint to contain `points`, `samples`, or `sample`.")
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.input)
    save_point_cloud_ply(points, args.output, index=args.index)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
