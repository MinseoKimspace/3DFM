from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from visualize import as_single_point_cloud, load_points


def collect_input_paths(run_dir: str | None, inputs: list[str]) -> list[Path]:
    paths: list[Path] = []

    if run_dir:
        root = Path(run_dir)
        for name in ("target.pt", "targets.pt"):
            path = root / name
            if path.exists():
                paths.append(path)
        paths.extend(sorted(root.glob("sample_*.pt")))

    for pattern in inputs:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    seen = set()
    unique_paths = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def load_cloud(path: Path, index: int) -> torch.Tensor:
    # return: [N, 3]
    points = load_points(path)
    return as_single_point_cloud(points, index=index).float().cpu()


def orient_cloud(cloud: torch.Tensor, up_axis: str) -> torch.Tensor:
    # Matplotlib uses z as the vertical axis.
    if up_axis == "z":
        return cloud
    if up_axis == "y":
        return torch.stack([cloud[:, 0], cloud[:, 2], cloud[:, 1]], dim=-1)
    raise ValueError(f"Unknown up axis: {up_axis}")


def camera_angles(view: str, elev: float | None, azim: float | None) -> tuple[float, float]:
    presets = {
        "iso": (18.0, -60.0),
        "front": (8.0, -90.0),
        "side": (8.0, 0.0),
        "top": (90.0, -90.0),
    }
    if view not in presets:
        raise ValueError(f"Unknown view: {view}")
    preset_elev, preset_azim = presets[view]
    return (
        preset_elev if elev is None else elev,
        preset_azim if azim is None else azim,
    )


def point_limits(
    clouds: list[torch.Tensor],
    pad: float,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_points = torch.cat(clouds, dim=0)
    if quantile >= 1.0:
        mins = all_points.min(dim=0).values
        maxs = all_points.max(dim=0).values
    else:
        low = (1.0 - quantile) * 0.5
        high = 1.0 - low
        mins = torch.quantile(all_points, low, dim=0)
        maxs = torch.quantile(all_points, high, dim=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()
    radius = radius * pad
    return center - radius, center + radius


def plot_cloud(
    ax,
    cloud: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    title: str,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    xyz = cloud.numpy()
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=point_size,
        c="#2563eb",
        alpha=0.9,
        linewidths=0,
        depthshade=True,
    )
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_facecolor("white")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    ax.set_axis_off()


def render_grid(
    paths: list[Path],
    output: str | Path,
    index: int,
    max_items: int,
    cols: int,
    point_size: float,
    view: str,
    elev: float | None,
    azim: float | None,
    dpi: int,
    pad: float,
    up_axis: str,
    limit_quantile: float,
) -> None:
    paths = paths[:max_items]
    if not paths:
        raise ValueError("No point cloud files were found.")

    clouds = [orient_cloud(load_cloud(path, index=index), up_axis=up_axis) for path in paths]
    lower, upper = point_limits(clouds, pad=pad, quantile=limit_quantile)
    elev, azim = camera_angles(view, elev=elev, azim=azim)

    cols = min(cols, len(paths))
    rows = math.ceil(len(paths) / cols)
    fig = plt.figure(figsize=(3.0 * cols, 3.0 * rows), dpi=dpi)
    fig.patch.set_facecolor("white")

    for i, (path, cloud) in enumerate(zip(paths, clouds), start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        title = path.stem
        if path.name == "targets.pt":
            title = f"target[{index}]"
        plot_cloud(
            ax=ax,
            cloud=cloud,
            lower=lower,
            upper=upper,
            title=title,
            point_size=point_size,
            elev=elev,
            azim=azim,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--view", choices=["iso", "front", "side", "top"], default="iso")
    parser.add_argument("--up-axis", choices=["y", "z"], default="y")
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--azim", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pad", type=float, default=1.08)
    parser.add_argument("--limit-quantile", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_input_paths(args.run_dir or None, args.inputs)
    output = args.output
    if not output:
        output = str(Path(args.run_dir) / "render_grid.png") if args.run_dir else "render_grid.png"

    render_grid(
        paths=paths,
        output=output,
        index=args.index,
        max_items=args.max_items,
        cols=args.cols,
        point_size=args.point_size,
        view=args.view,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        pad=args.pad,
        up_axis=args.up_axis,
        limit_quantile=args.limit_quantile,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
