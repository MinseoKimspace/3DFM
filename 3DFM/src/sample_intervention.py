from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fm.sampler import sample_euler
from models.builder import build_model
from visualize import save_point_cloud_ply


SLOT_MODES = ("normal", "shuffle", "zero")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    args = ckpt["args"]
    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_or_make_noise(
    path: str,
    num_samples: int,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    # return noise: [S, N, 3]
    if path:
        noise = torch.load(path, map_location="cpu")
        if isinstance(noise, dict):
            noise = noise["noise"]
        return noise.float().cpu()[:num_samples].contiguous()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(num_samples, num_points, 3, generator=generator)


def relative_sample_diff(reference: torch.Tensor, other: torch.Tensor) -> float:
    # reference: [S, N, 3]
    # other:     [S, N, 3]
    ref = reference.flatten(1)
    val = other.flatten(1)
    diff = (ref - val).norm(dim=1) / (ref.norm(dim=1) + 1e-8)
    return float(diff.mean().item())


def sample_mode_in_batches(
    model: torch.nn.Module,
    noise: torch.Tensor,
    nfe: int,
    batch_size: int,
    device: torch.device,
    slot_mode: str,
) -> tuple[torch.Tensor, float]:
    # noise: [S, N, 3]
    samples = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for start in range(0, noise.shape[0], batch_size):
        init = noise[start:start + batch_size].to(device)
        sample = sample_euler(
            model=model,
            batch_size=init.shape[0],
            num_points=init.shape[1],
            steps=nfe,
            device=device,
            dtype=init.dtype,
            init=init,
            model_kwargs={"slot_mode": slot_mode},
        )
        samples.append(sample.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return torch.cat(samples, dim=0), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--noise", type=str, default="")
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(ckpt, device=device)
    if not hasattr(model, "spatial_pma"):
        raise ValueError("Intervention sampling requires a spatial_pma model.")

    train_args = ckpt["args"]
    num_points = train_args["num_points"]
    noise = load_or_make_noise(
        path=args.noise,
        num_samples=args.num_samples,
        num_points=num_points,
        seed=args.seed,
    )
    if noise.ndim != 3 or noise.shape[-1] != 3:
        raise ValueError("noise must have shape [S, N, 3].")
    if noise.shape[1] != num_points:
        raise ValueError(
            f"noise has {noise.shape[1]} points, checkpoint expects {num_points}."
        )

    torch.save(noise, out_dir / "noise.pt")

    summary = {
        "checkpoint": args.checkpoint,
        "arch": train_args.get("arch", "base"),
        "num_samples": int(noise.shape[0]),
        "num_points": num_points,
        "seed": args.seed,
        "noise": args.noise,
        "nfe": args.nfe,
        "modes": SLOT_MODES,
        "results": {},
    }

    for nfe in args.nfe:
        samples_by_mode = {}
        summary["results"][str(nfe)] = {}

        for mode in SLOT_MODES:
            if mode == "shuffle":
                torch.manual_seed(args.seed + nfe)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + nfe)

            samples, elapsed = sample_mode_in_batches(
                model=model,
                noise=noise,
                nfe=nfe,
                batch_size=args.batch_size,
                device=device,
                slot_mode=mode,
            )
            samples_by_mode[mode] = samples

            mode_dir = out_dir / mode / f"nfe_{nfe:03d}"
            mode_dir.mkdir(parents=True, exist_ok=True)
            torch.save(samples, mode_dir / "samples.pt")
            torch.save(samples[0:1], mode_dir / "sample_000000.pt")

            if args.save_ply:
                try:
                    save_point_cloud_ply(samples[0], mode_dir / "sample_000000.ply")
                except ImportError as exc:
                    print(exc)

            summary["results"][str(nfe)][mode] = {
                "seconds": elapsed,
                "seconds_per_sample": elapsed / noise.shape[0],
            }
            print(
                f"nfe {nfe} mode {mode}: saved {samples.shape[0]} samples "
                f"in {elapsed:.3f}s"
            )

        normal = samples_by_mode["normal"]
        for mode in ("shuffle", "zero"):
            summary["results"][str(nfe)][mode]["rel_sample_diff_vs_normal"] = (
                relative_sample_diff(normal, samples_by_mode[mode])
            )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
