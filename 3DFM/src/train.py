from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.pyg_shapenet import load_pyg_shapes
from fm.losses import fm_loss
from fm.sampler import sample_euler
from models.builder import build_model
from visualize import save_point_cloud_ply


def load_config_defaults(path: str) -> dict[str, object]:
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {}
    defaults.update(cfg.get("data", {}))
    defaults.update(cfg.get("model", {}))
    defaults.update(cfg.get("training", {}))
    defaults.update(cfg.get("sampling", {}))
    defaults.update(cfg.get("output", {}))
    return defaults


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--data-mode", choices=["file", "shapenet"], default="shapenet")
    parser.add_argument("--shape-path", type=str, default="")
    parser.add_argument("--data-root", type=str, default="3DFM/src/data/shapenet_pyg")
    parser.add_argument("--category", type=str, default="Chair")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--num-shapes", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--target-order", choices=["fixed", "shuffle"], default="fixed")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--arch",
        choices=["base", "spatial_pma", "xhat_selfcond", "xhat_spatial_pma"],
        default="base",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--early-layers", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--spatial-random-start", action="store_true")
    parser.add_argument("--xattn-every-late-block", action="store_true")
    parser.add_argument("--use-xhat-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/single_shape_fm")

    parser.set_defaults(**load_config_defaults(config_args.config))
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_targets(args: argparse.Namespace) -> torch.Tensor:
    if args.data_mode == "file":
        obj = torch.load(args.shape_path, map_location="cpu")
        points = obj["points"] if isinstance(obj, dict) else obj
        if points.ndim == 2:
            return points.unsqueeze(0).float().contiguous()
        if points.ndim == 3:
            start = args.shape_index
            end = start + args.num_shapes
            return points[start:end].float().contiguous()
        raise ValueError(f"Expected [N, 3] or [S, N, 3], got {tuple(points.shape)}")

    return load_pyg_shapes( # [S, N, 3]
        root=args.data_root,
        category=args.category,
        num_points=args.num_points,
        split=args.split,
        start_index=args.shape_index,
        num_shapes=args.num_shapes,
    )


def make_batch(
        targets: torch.Tensor,# [S, N, 3]
        batch_size: int,
        target_order: str
        ) -> torch.Tensor:
    shape_indices = torch.randint(
        low=0,
        high=targets.shape[0],
        size=(batch_size,),
        device=targets.device,
    )
    batch = targets[shape_indices].clone()
    if target_order == "shuffle":
        batch_count, num_points, coord_dim = batch.shape
        indices = torch.stack(
            [
                torch.randperm(num_points, device=batch.device)
                for _ in range(batch_count)
            ],
            dim=0,
        )
        batch = batch.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(batch_count, num_points, coord_dim),
        )
    return batch # [B, N, 3]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args).to(device)
    torch.save(targets.cpu(), out_dir / "targets.pt")
    try:
        save_point_cloud_ply(targets[0], out_dir / "target_000.ply")
    except ImportError as exc:
        print(exc)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x_data = make_batch(targets, args.batch_size, args.target_order)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = fm_loss(model, x_data, aux_weight=args.aux_weight)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step} loss {float(metrics['loss']):.6f} "
                f"fm {float(metrics['fm_mse']):.6f} "
                f"aux {float(metrics['aux_mse']):.6f}"
            )

        if step % args.sample_every == 0:
            sample = sample_euler(
                model=model,
                batch_size=1,
                num_points=args.num_points,
                steps=args.nfe,
                device=device,
                dtype=targets.dtype,
            )
            sample_path = out_dir / f"sample_{step:06d}.pt"
            ply_path = out_dir / f"sample_{step:06d}.ply"
            torch.save(sample.cpu(), sample_path)
            try:
                save_point_cloud_ply(sample, ply_path)
            except ImportError as exc:
                print(exc)

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
        },
        out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()
