from __future__ import annotations

import argparse
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from visualize import as_single_point_cloud, load_points


DEFAULT_NFE = [1, 2, 4, 8, 16, 64]


@dataclass
class RenderJob:
    path: Path
    label: str
    model_index: int
    nfe: int | None = None


def parse_nfe(values: list[str]) -> list[int]:
    out = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    if not out:
        raise ValueError("--nfe must contain at least one value.")
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "cloud"


def sample_roots(root: Path, mode: str) -> list[Path]:
    if mode:
        roots = [root / mode, root / "eval" / mode]
    else:
        roots = [root, root / "eval"]
    return list(dict.fromkeys(roots))


def expand_run_dirs(patterns: list[str]) -> list[Path]:
    roots = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            roots.extend(Path(match) for match in matches)
        else:
            roots.append(Path(pattern))
    return roots


def load_cloud(path: Path, index: int, up_axis: str) -> torch.Tensor:
    cloud = as_single_point_cloud(load_points(path), index=index).float().cpu()
    if cloud.ndim != 2 or cloud.shape[-1] != 3:
        raise ValueError(f"Expected [N, 3], got {tuple(cloud.shape)} from {path}")

    # Mitsuba scene below uses Y as up. Swap Y/Z only for Z-up point clouds.
    if up_axis == "z":
        cloud = torch.stack([cloud[:, 0], cloud[:, 2], cloud[:, 1]], dim=-1)
    return cloud


def collect_jobs(args: argparse.Namespace) -> list[RenderJob]:
    jobs = []
    searched = []
    run_dirs = expand_run_dirs(args.run_dirs)

    for model_index, root in enumerate(run_dirs):
        model = args.labels[model_index] if model_index < len(args.labels) else root.name

        for mode in args.modes:
            for mode_root in sample_roots(root, mode):
                for nfe in args.nfe:
                    candidates = [
                        mode_root / f"nfe_{nfe:03d}" / "samples.pt",
                        mode_root / f"nfe_{nfe}" / "samples.pt",
                    ]
                    searched.extend(candidates)
                    path = next((candidate for candidate in candidates if candidate.exists()), None)
                    if path is None:
                        continue

                    suffix = f" / {mode}" if mode else ""
                    jobs.append(
                        RenderJob(
                            path=path,
                            label=f"{model}{suffix} / NFE={nfe}",
                            model_index=model_index,
                            nfe=nfe,
                        )
                    )

    offset = len(run_dirs)
    for i, path_str in enumerate(args.inputs):
        path = Path(path_str)
        label_index = offset + i
        label = args.labels[label_index] if label_index < len(args.labels) else path.stem
        jobs.append(RenderJob(path=path, label=label, model_index=i))

    if not jobs:
        preview = "\n".join(f"  {path}" for path in searched[:24])
        more = "" if len(searched) <= 24 else f"\n  ... and {len(searched) - 24} more"
        raise FileNotFoundError(
            "No matching .pt files found. Expected paths like:\n"
            f"{preview}{more}\n"
            "If you only have root-level sample_*.pt files from train.py, pass them with --inputs."
        )
    return jobs


def bounds(cloud: torch.Tensor, pad: float) -> tuple[torch.Tensor, float]:
    lo = cloud.min(dim=0).values
    hi = cloud.max(dim=0).values
    center = 0.5 * (lo + hi)
    extent = float((hi - lo).max().item()) * pad
    return center, max(extent, 1e-3)


def camera(center: torch.Tensor, extent: float, view: str, distance: float):
    directions = {
        "iso": torch.tensor([1.4, 0.9, 1.8]),
        "front": torch.tensor([0.0, 0.1, 1.0]),
        "side": torch.tensor([1.0, 0.1, 0.0]),
        "top": torch.tensor([0.0, 1.0, 0.001]),
    }
    direction = directions[view].float()
    direction = direction / direction.norm()
    origin = center + direction * extent * distance
    up = [0.0, 0.0, -1.0] if view == "top" else [0.0, 1.0, 0.0]
    return origin.tolist(), center.tolist(), up


def hex_color(value: str) -> list[float]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("--color must be a 6-digit hex color.")
    return [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def color_for_point(point: torch.Tensor, lo_y: float, hi_y: float) -> list[float]:
    t = 0.5 if hi_y <= lo_y else float((point[1].item() - lo_y) / (hi_y - lo_y))
    t = max(0.0, min(1.0, t))
    bottom = torch.tensor([0.06, 0.25, 0.75])
    middle = torch.tensor([0.12, 0.68, 0.55])
    top = torch.tensor([0.95, 0.55, 0.18])
    if t < 0.5:
        rgb = bottom * (1.0 - 2.0 * t) + middle * (2.0 * t)
    else:
        rgb = middle * (2.0 - 2.0 * t) + top * (2.0 * t - 1.0)
    return rgb.tolist()


def make_scene(mi, cloud: torch.Tensor, args: argparse.Namespace):
    center, extent = bounds(cloud, args.pad)
    origin, target, up = camera(center, extent, args.view, args.camera_distance)
    transform = mi.ScalarTransform4f

    scene = {
        "type": "scene",
        "integrator": {"type": args.integrator},
        "sensor": {
            "type": "perspective",
            "to_world": transform.look_at(origin=origin, target=target, up=up),
            "fov": args.fov,
            "sampler": {"type": "independent", "sample_count": args.spp},
            "film": {
                "type": "hdrfilm",
                "width": args.width,
                "height": args.height,
                "rfilter": {"type": "gaussian"},
            },
        },
        "ambient": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [args.ambient] * 3},
        },
        "key_light": {
            "type": "sphere",
            "center": (center + torch.tensor([-0.7, 1.4, 1.1]) * extent).tolist(),
            "radius": extent * 0.12,
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [args.key_light] * 3},
            },
        },
    }

    fixed_color = hex_color(args.color)
    lo_y = float(cloud[:, 1].min().item())
    hi_y = float(cloud[:, 1].max().item())

    for i, point in enumerate(cloud):
        color = color_for_point(point, lo_y, hi_y) if args.height_color else fixed_color
        scene[f"point_{i:04d}"] = {
            "type": "sphere",
            "center": [float(v) for v in point],
            "radius": args.radius,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": color},
            },
        }

    return mi.load_dict(scene)


def render_job(mi, job: RenderJob, args: argparse.Namespace, out_dir: Path) -> Path:
    cloud = load_cloud(job.path, args.index, args.up_axis)
    if args.max_points > 0:
        cloud = cloud[:args.max_points]

    scene = make_scene(mi, cloud, args)
    image = mi.render(scene, spp=args.spp)
    if args.exposure != 1.0:
        image = image * args.exposure

    nfe = f"nfe_{job.nfe:03d}" if job.nfe is not None else "input"
    out_path = out_dir / f"{safe_name(job.label)}_{nfe}_idx{args.index:03d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(out_path), image)
    print(f"saved {out_path}")
    return out_path


def save_grid(image_paths: list[Path], labels: list[str], out_path: Path, cols: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = max(1, cols)
    rows = math.ceil(len(image_paths) / cols)
    fig = plt.figure(figsize=(3.1 * cols, 3.35 * rows), dpi=170)
    fig.patch.set_facecolor("white")

    for i, (path, label) in enumerate(zip(image_paths, labels), start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=8)
        ax.set_axis_off()

    for i in range(len(image_paths) + 1, rows * cols + 1):
        fig.add_subplot(rows, cols, i).set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--nfe", nargs="+", default=[str(v) for v in DEFAULT_NFE])
    parser.add_argument("--modes", nargs="*", default=[""])
    parser.add_argument("--out-dir", default="mitsuba_renders")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0)

    parser.add_argument("--variant", default="scalar_rgb")
    parser.add_argument("--integrator", choices=["path", "direct"], default="path")
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--view", choices=["iso", "front", "side", "top"], default="iso")
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--up-axis", choices=["y", "z"], default="y")
    parser.add_argument("--pad", type=float, default=1.12)

    parser.add_argument("--radius", type=float, default=0.018)
    parser.add_argument("--color", default="#2563eb")
    parser.add_argument("--height-color", action="store_true")
    parser.add_argument("--ambient", type=float, default=0.25)
    parser.add_argument("--key-light", type=float, default=4.0)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--grid-cols", type=int, default=0)

    args = parser.parse_args()
    args.nfe = parse_nfe(args.nfe)
    return args


def main() -> None:
    args = parse_args()
    if not args.run_dirs and not args.inputs:
        raise ValueError("Pass --run-dirs or --inputs.")

    jobs = collect_jobs(args)

    try:
        import mitsuba as mi
    except ImportError as exc:
        raise ImportError("Install Mitsuba first: pip install mitsuba") from exc

    mi.set_variant(args.variant)

    out_dir = Path(args.out_dir)
    image_paths = [render_job(mi, job, args, out_dir) for job in jobs]

    cols = args.grid_cols or len(args.nfe)
    save_grid(
        image_paths=image_paths,
        labels=[job.label for job in jobs],
        out_path=out_dir / "comparison_grid.png",
        cols=cols,
    )


if __name__ == "__main__":
    main()
