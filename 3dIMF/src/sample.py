from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fm.sampler import sample_euler
from models.point_backbone import PointBackbone
from visualize import save_point_cloud_ply


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> PointBackbone:
    args = ckpt["args"]
    model = PointBackbone(
        num_points=args["num_points"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        num_heads=args["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def sample_in_batches(
    model: torch.nn.Module,
    noise: torch.Tensor,
    nfe: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    # noise: [S, N, 3]
    samples = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for start_idx in range(0, noise.shape[0], batch_size):
        init = noise[start_idx:start_idx + batch_size].to(device)
        sample = sample_euler(
            model=model,
            batch_size=init.shape[0],
            num_points=init.shape[1],
            steps=nfe,
            device=device,
            dtype=init.dtype,
            init=init,
        )
        samples.append(sample.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return torch.cat(samples, dim=0), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-ply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(ckpt, device=device)

    train_args = ckpt["args"]
    num_points = train_args["num_points"]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    noise = torch.randn(args.num_samples, num_points, 3, generator=generator)
    torch.save(noise, out_dir / "noise.pt")

    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "num_points": num_points,
        "seed": args.seed,
        "nfe": args.nfe,
        "times": {},
    }

    for nfe in args.nfe:
        samples, elapsed = sample_in_batches(
            model=model,
            noise=noise,
            nfe=nfe,
            batch_size=args.batch_size,
            device=device,
        )

        nfe_dir = out_dir / f"nfe_{nfe:03d}"
        nfe_dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples, nfe_dir / "samples.pt")
        torch.save(samples[0:1], nfe_dir / "sample_000000.pt")

        metadata = {
            "nfe": nfe,
            "num_samples": args.num_samples,
            "num_points": num_points,
            "seconds": elapsed,
            "seconds_per_sample": elapsed / args.num_samples,
        }
        with open(nfe_dir / "time.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if args.save_ply:
            try:
                save_point_cloud_ply(samples[0], nfe_dir / "sample_000000.ply")
            except ImportError as exc:
                print(exc)

        summary["times"][str(nfe)] = metadata
        print(
            f"nfe {nfe}: saved {samples.shape[0]} samples "
            f"in {elapsed:.3f}s ({elapsed / args.num_samples:.4f}s/sample)"
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
