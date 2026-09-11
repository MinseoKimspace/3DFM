from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fm.sampler import sample_euler
from models.builder import build_model
from visualize import save_point_cloud_ply


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


class XHatRecorder(torch.nn.Module):
    """Record the same auxiliary prediction used by the normal Anchor forward."""

    def __init__(self, model: torch.nn.Module, nfe: int) -> None:
        super().__init__()
        if not callable(getattr(model, "forward_with_aux", None)):
            raise ValueError("--save-xhat requires an XHatAnchorPMA checkpoint.")
        self.model = model
        self.nfe = nfe
        self.steps = {0, nfe // 4, nfe // 2, 3 * nfe // 4, nfe - 1}
        self.call_index = 0
        self.snapshots = {}

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model.forward_with_aux(x, t, **kwargs)
        step = self.call_index % self.nfe
        if step in self.steps:
            if step not in self.snapshots:
                self.snapshots[step] = {"t": float(t.flatten()[0]), "xt": [], "xhat1": []}
            row = self.snapshots[step]
            row["xt"].append(x.detach().to(device="cpu", copy=True))
            row["xhat1"].append(out["x_hat1"].detach().to(device="cpu", copy=True))
        self.call_index += 1
        return out["velocity"]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for step, row in sorted(self.snapshots.items()):
            record = {"step": step, "t": row["t"]}
            for key in ("xt", "xhat1"):
                name = f"step_{step:04d}_{key}.pt"
                torch.save(torch.cat(row[key], dim=0), directory / name)
                record[key] = name
            records.append(record)
        with open(directory / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"nfe": self.nfe, "snapshots": records}, f, indent=2)
        print(f"saved intermediate predictions: {directory}", flush=True)


def sample_in_batches(
    model: torch.nn.Module,
    noise: torch.Tensor, # [S, N, 3]
    nfe: int,
    batch_size: int,
    device: torch.device,
    xhat_dir: Path | None = None,
) -> tuple[torch.Tensor, float]:
    recorder = XHatRecorder(model, nfe) if xhat_dir is not None else None
    samples = []
    total = noise.shape[0]
    print(
        f"nfe {nfe}: sampling {total} clouds, {noise.shape[1]} points each, "
        f"batch size {batch_size}, device {device}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for start_idx in range(0, noise.shape[0], batch_size):
        init = noise[start_idx:start_idx + batch_size].to(device)
        sample = sample_euler(
            model=recorder if recorder is not None else model,
            batch_size=init.shape[0],
            num_points=init.shape[1],
            steps=nfe,
            device=device,
            dtype=init.dtype,
            init=init,
        )
        samples.append(sample.cpu())
        completed = start_idx + init.shape[0]
        elapsed = time.perf_counter() - start
        remaining = elapsed * (total - completed) / completed
        print(
            f"nfe {nfe}: {completed}/{total} samples ({completed / total:.1%}) | "
            f"elapsed {elapsed:.1f}s | ETA ~{remaining:.1f}s",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if recorder is not None:
        recorder.save(xhat_dir)
    return torch.cat(samples, dim=0), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--save-xhat", action="store_true", help="Save intermediate states and Anchor predictions.")
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or any(nfe < 1 for nfe in args.nfe):
        parser.error("--num-samples, --batch-size and --nfe must be positive.")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {args.checkpoint} (device={device})", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(ckpt, device=device)

    train_args = ckpt["args"]
    if args.save_xhat:
        if train_args.get("arch") != "dipt_xhat_anchor_pma":
            raise ValueError("--save-xhat requires a dipt_xhat_anchor_pma checkpoint.")
        print(f"checkpoint aux_weight: {train_args.get('aux_weight', 'unknown')}", flush=True)
        print("Intermediate capture is enabled; these timings are not sampling benchmarks.", flush=True)
    checkpoint_num_points = train_args["num_points"]
    num_points = args.num_points if args.num_points > 0 else checkpoint_num_points

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    noise = torch.randn(args.num_samples, num_points, 3, generator=generator)
    torch.save(noise, out_dir / "noise.pt")

    summary = {
        "checkpoint": args.checkpoint,
        "arch": train_args.get("arch", "dipt_base"),
        "num_samples": args.num_samples,
        "num_points": num_points,
        "checkpoint_num_points": checkpoint_num_points,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "checkpoint_aux_weight": train_args.get("aux_weight"),
        "save_xhat": args.save_xhat,
        "nfe": args.nfe,
        "times": {},
    }

    for nfe in args.nfe:
        nfe_dir = out_dir / f"nfe_{nfe:03d}"
        samples, elapsed = sample_in_batches(
            model=model,
            noise=noise,
            nfe=nfe,
            batch_size=args.batch_size,
            device=device,
            xhat_dir=nfe_dir / "xhat" if args.save_xhat else None,
        )

        nfe_dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples, nfe_dir / "samples.pt")
        torch.save(samples[0:1], nfe_dir / "sample_000000.pt")

        metadata = {
            "nfe": nfe,
            "num_samples": args.num_samples,
            "num_points": num_points,
            "checkpoint_num_points": checkpoint_num_points,
            "seconds": elapsed,
            "seconds_per_sample": elapsed / args.num_samples,
            "includes_xhat_capture": args.save_xhat,
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
            f"in {elapsed:.3f}s ({elapsed / args.num_samples:.4f}s/sample)",
            flush=True,
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
