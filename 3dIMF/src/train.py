from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.pyg_shapenet import load_single_pyg_shape
from fm.losses import fm_loss
from fm.sampler import sample_euler
from models.point_backbone import PointBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mode", choices=["file", "shapenet"], default="shapenet")
    parser.add_argument("--shape-path", type=str, default="")
    parser.add_argument("--data-root", type=str, default="data/shapenet_pyg")
    parser.add_argument("--category", type=str, default="Chair")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--shape-index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--target-order", choices=["fixed", "shuffle"], default="fixed")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/single_shape_fm")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_target(args: argparse.Namespace) -> torch.Tensor:
    # return: [N, 3]
    if args.data_mode == "file":
        obj = torch.load(args.shape_path, map_location="cpu")
        points = obj["points"] if isinstance(obj, dict) else obj
        if points.ndim == 3:
            points = points[args.shape_index]
        return points.float().contiguous()

    return load_single_pyg_shape(
        root=args.data_root,
        category=args.category,
        num_points=args.num_points,
        split=args.split,
        index=args.shape_index,
    )


def make_batch(points: torch.Tensor, batch_size: int, target_order: str) -> torch.Tensor:
    # points: [N, 3]
    # return: [B, N, 3]
    batch = points.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if target_order == "shuffle":
        index = torch.randperm(batch.shape[1], device=batch.device)
        batch = batch[:, index]
    return batch


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(args).to(device)
    model = PointBackbone(
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x_data = make_batch(target, args.batch_size, args.target_order)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = fm_loss(model, x_data)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step} loss {float(metrics['loss']):.6f}")

        if step % args.sample_every == 0:
            sample = sample_euler(
                model=model,
                batch_size=1,
                num_points=args.num_points,
                steps=args.nfe,
                device=device,
                dtype=target.dtype,
            )
            torch.save(sample.cpu(), out_dir / f"sample_{step:06d}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
        },
        out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()
