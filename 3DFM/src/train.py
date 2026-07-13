from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml

from data.pointflow_pc15k import load_pointflow_shapes, stats_to_torch
from fm.losses import fm_loss
from fm.sampler import sample_euler
from models.builder import build_model
from models.point_ops import farthest_point_sample, index_points
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
    parser.add_argument(
        "--data-mode",
        choices=["file", "shapenet", "pointflow"],
        default="shapenet",
    )
    parser.add_argument("--shape-path", type=str, default="")
    parser.add_argument("--data-root", type=str, default="3DFM/src/data/shapenet_pyg")
    parser.add_argument("--category", type=str, default="Chair")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--num-shapes", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--fixed-points-replace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-order", choices=["fixed", "shuffle"], default="fixed")
    parser.add_argument("--target-subsample", choices=["none", "random", "fps"], default="none")
    parser.add_argument("--pointflow-part", choices=["train", "test", "all"], default="train")
    parser.add_argument(
        "--pointflow-normalize",
        choices=["global", "per_shape"],
        default="global",
    )
    parser.add_argument("--pointflow-normalize-std-per-axis", action="store_true")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-gamma", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--arch",
        choices=[
            "base",
            "spatial_pma",
            "xhat_selfcond",
            "xhat_spatial_pma",
        ],
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


def load_targets(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, object]]:
    if args.data_mode == "file":
        obj = torch.load(args.shape_path, map_location="cpu")
        points = obj["points"] if isinstance(obj, dict) else obj
        if points.ndim == 2:
            return points.unsqueeze(0).float().contiguous(), {}
        if points.ndim == 3:
            start = args.shape_index
            end = start + args.num_shapes
            return points[start:end].float().contiguous(), {}
        raise ValueError(f"Expected [N, 3] or [S, N, 3], got {tuple(points.shape)}")

    if args.data_mode == "pointflow":
        points, stats = load_pointflow_shapes(
            root=args.data_root,
            category=args.category,
            split=args.split,
            part=args.pointflow_part,
            start_index=args.shape_index,
            num_shapes=args.num_shapes,
            normalize=args.pointflow_normalize,
            normalize_std_per_axis=args.pointflow_normalize_std_per_axis,
        )
        info = {
            "source": "pointflow_pc15k",
            "split": args.split,
            "part": args.pointflow_part,
            "stats": stats_to_torch(stats),
        }
        return points, info

    from data.pyg_shapenet import load_pyg_shapes

    points = load_pyg_shapes( # [S, N, 3]
        root=args.data_root,
        category=args.category,
        num_points=args.num_points,
        split=args.split,
        start_index=args.shape_index,
        num_shapes=args.num_shapes,
        replace=args.fixed_points_replace,
    )
    return points, {}


def make_batch(
        targets: torch.Tensor,# [S, N, 3]
        batch_size: int,
        target_order: str,
        num_points: int,
        target_subsample: str,
        shape_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
    if shape_indices is None:
        shape_indices = torch.randint(
            low=0,
            high=targets.shape[0],
            size=(batch_size,),
            device=targets.device,
        )
    batch = targets[shape_indices].clone()

    pool_points = batch.shape[1]
    if pool_points < num_points:
        raise ValueError(
            f"target pool has {pool_points} points, requested {num_points}."
        )
    if pool_points > num_points:
        batch_count, _, coord_dim = batch.shape
        if target_subsample == "random":
            indices = torch.stack(
                [
                    torch.randperm(pool_points, device=batch.device)[:num_points]
                    for _ in range(batch_count)
                ],
                dim=0,
            )
            batch = batch.gather(
                dim=1,
                index=indices.unsqueeze(-1).expand(batch_count, num_points, coord_dim),
            )
        elif target_subsample == "fps":
            indices = farthest_point_sample(batch, num_points, random_start=True)
            batch = index_points(batch, indices)
        elif target_subsample == "none":
            batch = batch[:, :num_points]
        else:
            raise ValueError(f"Unknown target_subsample: {target_subsample}")

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


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer.detach())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets, data_info = load_targets(args)
    targets = targets.to(device)
    torch.save(targets.cpu(), out_dir / "targets.pt")
    if data_info:
        torch.save(data_info, out_dir / "data_info.pt")
    try:
        save_point_cloud_ply(targets[0, :args.num_points], out_dir / "target_000.ply")
    except ImportError as exc:
        print(exc)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.lr_gamma,
    )

    ema_model = None
    if args.ema_decay > 0.0:
        if not 0.0 < args.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1).")
        ema_model = copy.deepcopy(model).eval()
        ema_model.requires_grad_(False)

    if args.epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if args.epochs == 0 and args.lr_gamma != 1.0:
        raise ValueError("Exponential LR scheduling requires epochs > 0.")

    def train_batch(x_data: torch.Tensor, step: int, epoch: int | None) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = fm_loss(model, x_data, aux_weight=args.aux_weight)
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, model, args.ema_decay)

        if step % args.log_every == 0 or step == 1:
            epoch_text = f"epoch {epoch}/{args.epochs} " if epoch is not None else ""
            print(
                f"{epoch_text}step {step} "
                f"lr {optimizer.param_groups[0]['lr']:.6e} "
                f"loss {float(metrics['loss']):.6f} "
                f"fm {float(metrics['fm_mse']):.6f} "
                f"aux {float(metrics['aux_mse']):.6f}"
            )

        if step % args.sample_every == 0:
            sample = sample_euler(
                model=ema_model if ema_model is not None else model,
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

    step = 0
    completed_epoch = 0
    if args.epochs > 0:
        for epoch in range(1, args.epochs + 1):
            shape_order = torch.randperm(targets.shape[0], device=targets.device)
            for start in range(0, targets.shape[0], args.batch_size):
                step += 1
                shape_indices = shape_order[start:start + args.batch_size]
                x_data = make_batch(
                    targets=targets,
                    batch_size=shape_indices.numel(),
                    target_order=args.target_order,
                    num_points=args.num_points,
                    target_subsample=args.target_subsample,
                    shape_indices=shape_indices,
                )
                train_batch(x_data, step, epoch)
            scheduler.step()
            completed_epoch = epoch
    else:
        for step in range(1, args.steps + 1):
            x_data = make_batch(
                targets=targets,
                batch_size=args.batch_size,
                target_order=args.target_order,
                num_points=args.num_points,
                target_subsample=args.target_subsample,
            )
            train_batch(x_data, step, None)

    torch.save(
        {
            "model": (ema_model if ema_model is not None else model).state_dict(),
            "model_raw": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": completed_epoch,
            "ema_decay": args.ema_decay,
            "args": vars(args),
        },
        out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()
