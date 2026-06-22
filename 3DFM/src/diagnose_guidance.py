from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.builder import build_model


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


def load_noise(
    path: str,
    num_samples: int,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    # return noise: [S, N, 3]
    if path:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "noise" in obj:
                obj = obj["noise"]
            elif "samples" in obj:
                obj = obj["samples"]
            else:
                raise KeyError("Noise dict must contain 'noise' or 'samples'.")
        noise = obj.float().cpu()
        return noise[:num_samples].contiguous()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(num_samples, num_points, 3, generator=generator)


def relative_velocity_diff(
    reference: torch.Tensor,
    intervention: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # reference:    [B, N, 3]
    # intervention: [B, N, 3]
    # return:       [B]
    ref = reference.flatten(1)
    other = intervention.flatten(1)
    return (ref - other).norm(dim=1) / (ref.norm(dim=1) + eps)


@torch.no_grad()
def diagnose_batch(
    model: torch.nn.Module,
    init: torch.Tensor,
    nfe: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, float]], torch.Tensor]:
    # init: [B, N, 3]
    # return per-step sums and final normal-rollout samples.
    x = init.to(device=device, dtype=dtype)
    batch_size = x.shape[0]
    times = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)
    sums: list[dict[str, float]] = []

    for step in range(nfe):
        t_now = times[step]
        t_next = times[step + 1]
        dt = t_next - t_now
        t = t_now.expand(batch_size, 1, 1)

        v_normal = model(x, t, slot_mode="normal")
        v_zero = model(x, t, slot_mode="zero")
        v_shuffle = model(x, t, slot_mode="shuffle")

        diff_zero = relative_velocity_diff(v_normal, v_zero)
        diff_shuffle = relative_velocity_diff(v_normal, v_shuffle)
        normal_norm = v_normal.flatten(1).norm(dim=1)

        sums.append(
            {
                "count": float(batch_size),
                "diff_zero_sum": float(diff_zero.sum().item()),
                "diff_shuffle_sum": float(diff_shuffle.sum().item()),
                "normal_norm_sum": float(normal_norm.sum().item()),
            }
        )

        # Follow the normal guided trajectory during inference.
        x = x + dt * v_normal

    return sums, x.cpu()


def merge_step_sums(
    total: list[dict[str, float]],
    batch_sums: list[dict[str, float]],
) -> None:
    for step, row in enumerate(batch_sums):
        if step == len(total):
            total.append(
                {
                    "count": 0.0,
                    "diff_zero_sum": 0.0,
                    "diff_shuffle_sum": 0.0,
                    "normal_norm_sum": 0.0,
                }
            )
        for key, value in row.items():
            total[step][key] += value


def finalize_report(
    step_sums: list[dict[str, float]],
    elapsed: float,
    args: argparse.Namespace,
    checkpoint_args: dict,
) -> dict:
    per_step = []
    total_count = 0.0
    total_zero = 0.0
    total_shuffle = 0.0
    total_norm = 0.0

    for step, row in enumerate(step_sums):
        count = row["count"]
        total_count += count
        total_zero += row["diff_zero_sum"]
        total_shuffle += row["diff_shuffle_sum"]
        total_norm += row["normal_norm_sum"]
        per_step.append(
            {
                "step": step,
                "t": step / args.nfe,
                "rel_diff_zero": row["diff_zero_sum"] / count,
                "rel_diff_shuffle": row["diff_shuffle_sum"] / count,
                "normal_velocity_norm": row["normal_norm_sum"] / count,
            }
        )

    return {
        "checkpoint": args.checkpoint,
        "arch": checkpoint_args.get("arch", "base"),
        "nfe": args.nfe,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "noise": args.noise,
        "seed": args.seed,
        "seconds": elapsed,
        "warning": "shuffle is not meaningful when batch_size is 1"
        if args.batch_size == 1
        else "",
        "mean": {
            "rel_diff_zero": total_zero / total_count,
            "rel_diff_shuffle": total_shuffle / total_count,
            "normal_velocity_norm": total_norm / total_count,
        },
        "per_step": per_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--noise", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-final-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nfe < 1:
        raise ValueError("nfe must be >= 1.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = ckpt["args"]
    model = build_model_from_checkpoint(ckpt, device=device)

    if not hasattr(model, "spatial_pma"):
        raise ValueError("Guidance diagnostics require a spatial_pma model.")

    num_points = checkpoint_args["num_points"]
    noise = load_noise(
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

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    step_sums: list[dict[str, float]] = []
    final_samples = []
    for start in range(0, noise.shape[0], args.batch_size):
        batch = noise[start:start + args.batch_size]
        batch_sums, final_batch = diagnose_batch(
            model=model,
            init=batch,
            nfe=args.nfe,
            device=device,
            dtype=batch.dtype,
        )
        merge_step_sums(step_sums, batch_sums)
        if args.save_final_samples:
            final_samples.append(final_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    report = finalize_report(
        step_sums=step_sums,
        elapsed=elapsed,
        args=args,
        checkpoint_args=checkpoint_args,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.save_final_samples:
        torch.save(torch.cat(final_samples, dim=0), out_path.with_suffix(".samples.pt"))

    mean = report["mean"]
    print(
        "saved guidance diagnostic to "
        f"{out_path} | zero={mean['rel_diff_zero']:.6f} "
        f"shuffle={mean['rel_diff_shuffle']:.6f}"
    )


if __name__ == "__main__":
    main()
