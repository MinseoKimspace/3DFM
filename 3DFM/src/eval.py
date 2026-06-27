from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from fm.sampler import sample_euler
from models.builder import build_model


def load_points(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "points" in obj:
            obj = obj["points"]
        elif "samples" in obj:
            obj = obj["samples"]
        elif "sample" in obj:
            obj = obj["sample"]
        else:
            raise KeyError("Expected dict to contain `points`, `samples`, or `sample`.")
    if obj.ndim != 3 or obj.shape[-1] != 3:
        raise ValueError(f"Expected [S, N, 3], got {tuple(obj.shape)}")
    return obj.float().contiguous()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_arg(config: object, name: str, default: object = None) -> object:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args


def load_or_make_noise(
    path: str,
    num_samples: int,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    if path:
        noise = torch.load(path, map_location="cpu")
        if isinstance(noise, dict):
            noise = noise["noise"]
        return noise.float().cpu()[:num_samples].contiguous()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(num_samples, num_points, 3, generator=generator)


@torch.no_grad()
def chamfer_paired(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y).square() # [B, N, N]
    x_to_y = dist.min(dim=2).values.mean(dim=1)
    y_to_x = dist.min(dim=1).values.mean(dim=1)
    return x_to_y + y_to_x # [B]


@torch.no_grad()
def pairwise_cd_matrix(
    samples: torch.Tensor,
    refs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty(samples.shape[0], refs.shape[0])

    for i in range(0, samples.shape[0], batch_size):
        sample_batch = samples[i:i + batch_size].to(device)
        for j in range(0, refs.shape[0], batch_size):
            ref_batch = refs[j:j + batch_size].to(device)
            bs = sample_batch.shape[0]
            br = ref_batch.shape[0]

            sample_pairs = sample_batch[:, None].expand(bs, br, -1, -1)
            ref_pairs = ref_batch[None].expand(bs, br, -1, -1)
            sample_pairs = sample_pairs.reshape(bs * br, sample_batch.shape[1], 3)
            ref_pairs = ref_pairs.reshape(bs * br, ref_batch.shape[1], 3)

            cd = chamfer_paired(sample_pairs, ref_pairs).reshape(bs, br)
            out[i:i + bs, j:j + br] = cd.cpu()

    return out


def compute_local_cd(
    samples: torch.Tensor,
    refs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    cd = pairwise_cd_matrix(samples, refs, batch_size=batch_size, device=device)
    sample_to_ref = cd.min(dim=1).values
    ref_to_sample = cd.min(dim=0).values
    return {
        "CD": float(sample_to_ref.mean().item()),
        "MMD-CD": float(sample_to_ref.mean().item()),
        "Ref-CD": float(ref_to_sample.mean().item()),
        "Pair-CD": float(cd.mean().item()),
    }


def collect_sample_paths(eval_dir: str | Path) -> list[Path]:
    root = Path(eval_dir)
    paths = sorted(root.glob("nfe_*/samples.pt"))
    if not paths:
        raise FileNotFoundError(f"No nfe_*/samples.pt files found in {root}")
    return paths


def metric_name_from_path(path: Path) -> str:
    return path.parent.name.replace("nfe_", "NFE=")


def read_time(path: Path) -> dict[str, float] | None:
    time_path = path.parent / "time.json"
    if not time_path.exists():
        return None
    with open(time_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(row: dict) -> dict[str, float]:
    out = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def supports_xhat_condition(model: torch.nn.Module, train_args: object) -> bool:
    arch = get_arg(train_args, "arch", "base")
    return arch == "xhat_selfcond" and bool(getattr(model, "use_xhat_condition", False))


@torch.no_grad()
def sample_checkpoint(
    model: torch.nn.Module,
    train_args: object,
    noise: torch.Tensor,
    nfe: int,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, float]:
    samples = []
    model_kwargs = {}
    if supports_xhat_condition(model, train_args):
        model_kwargs["cond_mode"] = mode
    elif mode != "normal":
        raise ValueError("Only xhat_selfcond supports normal/zero intervention.")

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
            model_kwargs=model_kwargs,
        )
        samples.append(sample.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return torch.cat(samples, dim=0), elapsed


def eval_checkpoints_cd(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    refs = load_points(args.refs)
    if args.max_refs > 0:
        refs = refs[:args.max_refs]

    labels = args.labels
    if labels and len(labels) != len(args.checkpoints):
        raise ValueError("--labels must have the same length as --checkpoints.")
    if not labels:
        labels = [Path(path).parent.name for path in args.checkpoints]

    first_model, first_train_args = build_model_from_checkpoint(args.checkpoints[0], device)
    num_points = int(get_arg(first_train_args, "num_points"))
    noise = load_or_make_noise(
        path=args.noise,
        num_samples=args.num_samples,
        num_points=num_points,
        seed=args.seed,
    )
    if noise.shape[1] != num_points:
        raise ValueError(f"noise has {noise.shape[1]} points, checkpoint expects {num_points}.")

    results = {
        "refs": args.refs,
        "noise": args.noise,
        "seed": args.seed,
        "num_samples": int(noise.shape[0]),
        "num_points": num_points,
        "nfe": args.nfe,
        "rows": [],
    }

    checkpoint_items = [(args.checkpoints[0], labels[0], first_model, first_train_args)]
    for path, label in zip(args.checkpoints[1:], labels[1:]):
        model, train_args = build_model_from_checkpoint(path, device)
        checkpoint_items.append((path, label, model, train_args))

    for path, label, model, train_args in checkpoint_items:
        arch = str(get_arg(train_args, "arch", "base"))
        modes = args.modes if supports_xhat_condition(model, train_args) else ["normal"]

        for nfe in args.nfe:
            for mode in modes:
                samples, elapsed = sample_checkpoint(
                    model=model,
                    train_args=train_args,
                    noise=noise,
                    nfe=nfe,
                    batch_size=args.batch_size,
                    device=device,
                    mode=mode,
                )
                if args.max_samples > 0:
                    samples = samples[:args.max_samples]

                row = compute_local_cd(
                    samples=samples,
                    refs=refs,
                    batch_size=args.cd_batch_size,
                    device=device,
                )
                row.update(
                    {
                        "label": label,
                        "arch": arch,
                        "checkpoint": str(path),
                        "mode": mode,
                        "NFE": int(nfe),
                        "Time": float(elapsed),
                        "Time/sample": float(elapsed / noise.shape[0]),
                    }
                )
                results["rows"].append(row)
                print(
                    f"{label} mode={mode} NFE={nfe}: "
                    f"CD={row['CD']:.6f} Ref-CD={row['Ref-CD']:.6f}"
                )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--labels", type=str, nargs="*", default=[])
    parser.add_argument("--noise", type=str, default="")
    parser.add_argument("--samples", type=str, nargs="*", default=[])
    parser.add_argument("--eval-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-refs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--modes", type=str, nargs="+", default=["normal", "zero"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cd-only", action="store_true")
    parser.add_argument("--cd-batch-size", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--lion-root", type=str, default="")
    parser.add_argument("--compute-emd", action="store_true")
    parser.add_argument("--accelerated-cd", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoints:
        eval_checkpoints_cd(args)
        return

    device = choose_device(args.device)
    if not args.cd_only:
        from metrics.lion_backend import add_lion_root, compute_lion_metrics
        add_lion_root(args.lion_root)

    refs = load_points(args.refs)
    if args.max_refs > 0:
        refs = refs[:args.max_refs]

    sample_paths = [Path(p) for p in args.samples]
    if args.eval_dir:
        sample_paths.extend(collect_sample_paths(args.eval_dir))
    if not sample_paths:
        raise ValueError("Pass --samples or --eval-dir.")

    results = {}
    for path in sample_paths:
        samples = load_points(path)
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        label = metric_name_from_path(path) if path.name == "samples.pt" else path.stem
        print(f"evaluating {label}: samples={tuple(samples.shape)} refs={tuple(refs.shape)}")

        if args.cd_only:
            row = compute_local_cd(
                samples=samples,
                refs=refs,
                batch_size=args.cd_batch_size,
                device=device,
            )
        else:
            row = compute_lion_metrics(
                samples=samples,
                refs=refs,
                batch_size=args.metric_batch_size,
                compute_emd=args.compute_emd,
                accelerated_cd=args.accelerated_cd,
                verbose=not args.quiet,
            )
            row = to_jsonable(row)

        time_info = read_time(path)
        if time_info is not None:
            row["Time"] = float(time_info["seconds"])
            row["Time/sample"] = float(time_info["seconds_per_sample"])

        results[label] = row
        print(json.dumps(row, indent=2))

    output = args.output
    if not output and args.eval_dir:
        output = str(Path(args.eval_dir) / "metrics.json")
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
