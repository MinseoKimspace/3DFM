from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.pointflow_pc15k import (
    compute_stats,
    load_pointflow_15k,
    load_pointflow_shapes,
    stats_to_torch,
)
from models.point_ops import farthest_point_sample, index_points


def subsample(points: torch.Tensor, num_points: int, mode: str, seed: int) -> torch.Tensor:
    if num_points <= 0 or num_points == points.shape[1]:
        return points
    if num_points > points.shape[1]:
        raise ValueError(f"requested {num_points}, but pool only has {points.shape[1]}")

    if mode == "first":
        return points[:, :num_points].contiguous()

    if mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        rows = []
        for cloud in points:
            idx = torch.randperm(points.shape[1], generator=generator)[:num_points]
            rows.append(cloud[idx])
        return torch.stack(rows, dim=0).contiguous()

    if mode == "fps":
        idx = farthest_point_sample(points, num_points, random_start=False)
        return index_points(points, idx).contiguous()

    raise ValueError(f"Unknown subsample mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--category", type=str, default="Chair")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--part", choices=["train", "test", "all"], default="test")
    parser.add_argument("--stats-split", type=str, default="train")
    parser.add_argument("--num-shapes", type=int, default=0)
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--subsample", choices=["first", "random", "fps"], default="random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--normalize", choices=["global", "per_shape"], default="global")
    parser.add_argument("--normalize-std-per-axis", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = None
    if args.normalize == "global":
        stats_clouds = load_pointflow_15k(
            root=args.data_root,
            category=args.category,
            split=args.stats_split,
        )
        stats = compute_stats(
            clouds=stats_clouds,
            normalize=args.normalize,
            normalize_std_per_axis=args.normalize_std_per_axis,
        )

    points, used_stats = load_pointflow_shapes(
        root=args.data_root,
        category=args.category,
        split=args.split,
        part=args.part,
        start_index=args.shape_index,
        num_shapes=args.num_shapes,
        normalize=args.normalize,
        normalize_std_per_axis=args.normalize_std_per_axis,
        stats=stats,
    )
    points = subsample(
        points=points,
        num_points=args.num_points,
        mode=args.subsample,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "points": points,
            "source": "pointflow_pc15k",
            "category": args.category,
            "split": args.split,
            "part": args.part,
            "num_points": int(points.shape[1]),
            "subsample": args.subsample,
            "seed": args.seed,
            "stats_split": args.stats_split if args.normalize == "global" else args.split,
            "stats": stats_to_torch(used_stats),
        },
        output,
    )
    print(f"saved {output} {tuple(points.shape)}")


if __name__ == "__main__":
    main()
