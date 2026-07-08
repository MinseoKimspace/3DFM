from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from fm.sampler import sample_euler
from models.builder import build_model
from models.point_ops import farthest_point_sample, index_points


def load_points_with_meta(path: str | Path) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu")
    meta = obj if isinstance(obj, dict) else {}
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
    return obj.float().contiguous(), meta


def load_points(path: str | Path) -> torch.Tensor:
    points, _ = load_points_with_meta(path)
    return points


def denormalize_pointflow(points: torch.Tensor, meta: dict) -> torch.Tensor:
    stats = meta.get("stats")
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise KeyError("PointFlow denormalization needs `stats.mean` and `stats.std` in refs.")

    mean = stats["mean"].float()
    std = stats["std"].float()
    while mean.ndim < 3:
        mean = mean.unsqueeze(0)
    while std.ndim < 3:
        std = std.unsqueeze(0)

    if mean.shape[0] not in (1, points.shape[0]):
        raise ValueError(
            "Per-shape PointFlow stats do not match points. "
            "Use global stats or export matching refs."
        )
    if std.shape[0] not in (1, points.shape[0]):
        raise ValueError(
            "Per-shape PointFlow stats do not match points. "
            "Use global stats or export matching refs."
        )
    return points * std + mean


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
    include_1nna: bool = False,
) -> dict[str, float]:
    cd = pairwise_cd_matrix(samples, refs, batch_size=batch_size, device=device)
    sample_to_ref = cd.min(dim=1).values
    sample_nn_ref = cd.min(dim=1).indices
    ref_to_sample = cd.min(dim=0).values

    # PointFlow/L-GAN convention:
    # MMD-CD averages over refs; MMD-CD-sample averages over generated samples.
    coverage = sample_nn_ref.unique().numel() / refs.shape[0]
    return {
        "CD": float(sample_to_ref.mean().item()),
        "MMD-CD": float(ref_to_sample.mean().item()),
        "MMD-CD-sample": float(sample_to_ref.mean().item()),
        "COV-CD": float(coverage),
        "Ref-CD": float(ref_to_sample.mean().item()),
        "Pair-CD": float(cd.mean().item()),
        **(
            {"1-NNA-CD": compute_1nna_cd(samples, refs, batch_size, device)}
            if include_1nna
            else {}
        ),
    }


def compute_1nna_cd(
    samples: torch.Tensor,
    refs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    sample_sample = pairwise_cd_matrix(samples, samples, batch_size, device)
    ref_ref = pairwise_cd_matrix(refs, refs, batch_size, device)
    sample_ref = pairwise_cd_matrix(samples, refs, batch_size, device)

    sample_sample.fill_diagonal_(float("inf"))
    ref_ref.fill_diagonal_(float("inf"))

    sample_nn_same = sample_sample.min(dim=1).values
    sample_nn_other = sample_ref.min(dim=1).values
    ref_nn_same = ref_ref.min(dim=1).values
    ref_nn_other = sample_ref.min(dim=0).values

    sample_correct = sample_nn_same < sample_nn_other
    ref_correct = ref_nn_same < ref_nn_other
    acc = torch.cat([sample_correct, ref_correct]).float().mean()
    return float(acc.item())


def parse_cd_points(values: list[str]) -> list[int | str]:
    levels = []
    for value in values:
        if value == "full":
            levels.append("full")
        else:
            levels.append(int(value))
    return levels


@torch.no_grad()
def downsample_fps(
    points: torch.Tensor,
    num_points: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if num_points >= points.shape[1]:
        return points

    out = []
    for start in range(0, points.shape[0], batch_size):
        batch = points[start:start + batch_size].to(device)
        idx = farthest_point_sample(batch, num_points, random_start=False)
        out.append(index_points(batch, idx).cpu())
    return torch.cat(out, dim=0)


def compute_cd_levels(
    samples: torch.Tensor,
    refs: torch.Tensor,
    levels: list[int | str],
    batch_size: int,
    device: torch.device,
    include_1nna: bool = False,
) -> dict[str, float]:
    result = {}

    for level in levels:
        if level == "full":
            level_samples = samples
            level_refs = refs
            suffix = "full"
        else:
            level_samples = downsample_fps(samples, level, batch_size, device)
            level_refs = downsample_fps(refs, level, batch_size, device)
            suffix = str(level)

        row = compute_local_cd(
            samples=level_samples,
            refs=level_refs,
            batch_size=batch_size,
            device=device,
            include_1nna=include_1nna,
        )
        for key, value in row.items():
            result[f"{key}@{suffix}"] = value

        if level == "full":
            result.update(row)

    return result


def permute_points_per_cloud(points: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows = []
    for cloud in points:
        idx = torch.randperm(cloud.shape[0], generator=generator)
        rows.append(cloud[idx])
    return torch.stack(rows, dim=0).contiguous()


def compute_permutation_check(
    samples: torch.Tensor,
    refs: torch.Tensor,
    levels: list[int | str],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    samples_perm = permute_points_per_cloud(samples, seed)
    refs_perm = permute_points_per_cloud(refs, seed + 1)

    orig = compute_cd_levels(samples, refs, levels, batch_size, device)
    perm = compute_cd_levels(samples_perm, refs_perm, levels, batch_size, device)

    out = {}
    for level in levels:
        suffix = "full" if level == "full" else str(level)
        key = f"CD@{suffix}"
        if key in orig and key in perm:
            delta = perm[key] - orig[key]
            out[f"perm_{key}_orig"] = orig[key]
            out[f"perm_{key}_shuffled"] = perm[key]
            out[f"perm_{key}_abs_delta"] = abs(delta)
            out[f"perm_{key}_rel_delta"] = abs(delta) / (abs(orig[key]) + 1e-8)

    for level in levels:
        if level == "full":
            continue
        sample_fps = downsample_fps(samples, level, batch_size, device).to(device)
        sample_perm_fps = downsample_fps(samples_perm, level, batch_size, device).to(device)
        ref_fps = downsample_fps(refs, level, batch_size, device).to(device)
        ref_perm_fps = downsample_fps(refs_perm, level, batch_size, device).to(device)
        out[f"fps_self_CD_samples@{level}"] = float(
            chamfer_paired(sample_fps, sample_perm_fps).mean().item()
        )
        out[f"fps_self_CD_refs@{level}"] = float(
            chamfer_paired(ref_fps, ref_perm_fps).mean().item()
        )

    return out


def density_stats(
    points: torch.Tensor,
    k: int,
    batch_size: int,
    device: torch.device,
    max_values: int,
) -> dict[str, float]:
    values = []
    total_sum = 0.0
    total_count = 0
    near_zero_count = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)

    for start in range(0, points.shape[0], batch_size):
        batch = points[start:start + batch_size].to(device)
        dist = torch.cdist(batch, batch) # [B, N, N]
        knn = dist.topk(k=k + 1, dim=-1, largest=False).values[..., 1:]
        flat = knn.reshape(-1).cpu()

        total_sum += float(flat.sum().item())
        total_count += int(flat.numel())
        near_zero_count += int((flat < 1e-4).sum().item())

        if max_values > 0 and flat.numel() > max_values:
            idx = torch.randperm(flat.numel(), generator=generator)[:max_values]
            flat = flat[idx]
        values.append(flat)

        if max_values > 0:
            current = sum(value.numel() for value in values)
            if current > max_values:
                merged = torch.cat(values, dim=0)
                idx = torch.randperm(merged.numel(), generator=generator)[:max_values]
                values = [merged[idx]]

    d = torch.cat(values, dim=0)
    return {
        "mean": float(total_sum / max(total_count, 1)),
        "p50": float(torch.quantile(d, 0.50).item()),
        "p95": float(torch.quantile(d, 0.95).item()),
        "near_zero_frac": float(near_zero_count / max(total_count, 1)),
    }


def compute_density_comparison(
    samples: torch.Tensor,
    refs: torch.Tensor,
    k: int,
    batch_size: int,
    device: torch.device,
    max_values: int,
) -> dict[str, float]:
    sample = density_stats(
        samples,
        k=k,
        batch_size=batch_size,
        device=device,
        max_values=max_values,
    )
    ref = density_stats(
        refs,
        k=k,
        batch_size=batch_size,
        device=device,
        max_values=max_values,
    )
    out = {}
    for key, value in sample.items():
        out[f"sample_knn{k}_{key}"] = value
    for key, value in ref.items():
        out[f"ref_knn{k}_{key}"] = value
    out[f"density_knn{k}_mean_ratio"] = sample["mean"] / (ref["mean"] + 1e-8)
    out[f"density_knn{k}_p50_ratio"] = sample["p50"] / (ref["p50"] + 1e-8)
    out[f"density_knn{k}_p95_ratio"] = sample["p95"] / (ref["p95"] + 1e-8)
    return out


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


def intervention_arg(model: torch.nn.Module, train_args: object) -> str:
    if hasattr(model, "spatial_pma"):
        return "slot_mode"
    if supports_xhat_condition(model, train_args):
        return "cond_mode"
    return ""


def set_attention_stats(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "collect_stats"):
            module.collect_stats = enabled
        if hasattr(module, "reset_last_stats"):
            module.reset_last_stats()


def collect_attention_stats(model: torch.nn.Module) -> dict[str, float]:
    for module in model.modules():
        stats = getattr(module, "last_stats", {})
        if stats:
            return {f"xattn_{key}": value for key, value in stats.items()}
    return {}


@torch.no_grad()
def sample_checkpoint(
    model: torch.nn.Module,
    train_args: object,
    noise: torch.Tensor,
    nfe: int,
    batch_size: int,
    device: torch.device,
    mode: str,
    attention_stats: bool,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    samples = []
    model_kwargs = {}
    mode_arg = intervention_arg(model, train_args)
    if mode_arg:
        model_kwargs[mode_arg] = mode
    elif mode != "normal":
        raise ValueError("This model does not support intervention modes.")

    set_attention_stats(model, attention_stats)

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
    stats = collect_attention_stats(model) if attention_stats else {}
    set_attention_stats(model, False)

    return torch.cat(samples, dim=0), elapsed, stats


def eval_checkpoints_cd(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    cd_levels = parse_cd_points(args.cd_points)
    refs, refs_meta = load_points_with_meta(args.refs)
    if args.max_refs > 0:
        refs = refs[:args.max_refs]
    if args.pointflow_denorm:
        refs = denormalize_pointflow(refs, refs_meta)

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
        "pointflow_denorm": bool(args.pointflow_denorm),
        "rows": [],
    }

    checkpoint_items = [(args.checkpoints[0], labels[0], first_model, first_train_args)]
    for path, label in zip(args.checkpoints[1:], labels[1:]):
        model, train_args = build_model_from_checkpoint(path, device)
        checkpoint_items.append((path, label, model, train_args))

    for path, label, model, train_args in checkpoint_items:
        arch = str(get_arg(train_args, "arch", "base"))
        modes = args.modes if intervention_arg(model, train_args) else ["normal"]

        for nfe in args.nfe:
            for mode in modes:
                samples, elapsed, attention_stats = sample_checkpoint(
                    model=model,
                    train_args=train_args,
                    noise=noise,
                    nfe=nfe,
                    batch_size=args.batch_size,
                    device=device,
                    mode=mode,
                    attention_stats=args.attention_stats,
                )
                if args.max_samples > 0:
                    samples = samples[:args.max_samples]
                if args.pointflow_denorm:
                    samples = denormalize_pointflow(samples, refs_meta)

                row = compute_cd_levels(
                    samples=samples,
                    refs=refs,
                    levels=cd_levels,
                    batch_size=args.cd_batch_size,
                    device=device,
                    include_1nna=args.one_nna_cd,
                )
                if args.permutation_check:
                    row.update(
                        compute_permutation_check(
                            samples=samples,
                            refs=refs,
                            levels=cd_levels,
                            batch_size=args.cd_batch_size,
                            device=device,
                            seed=args.permutation_seed,
                        )
                    )
                row.update(attention_stats)
                if args.density:
                    row.update(
                        compute_density_comparison(
                            samples=samples,
                            refs=refs,
                            k=args.density_k,
                            batch_size=args.density_batch_size,
                            device=device,
                            max_values=args.density_max_values,
                        )
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
                cd_msg = " ".join(
                    f"CD@{level}={row[f'CD@{level}']:.6f}"
                    for level in args.cd_points
                    if f"CD@{level}" in row
                )
                attn_msg = (
                    f" xattn_delta={row['xattn_delta_h_rel_norm']:.6f}"
                    if "xattn_delta_h_rel_norm" in row
                    else ""
                )
                print(
                    f"{label} mode={mode} NFE={nfe}: "
                    f"{cd_msg}{attn_msg}"
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
    parser.add_argument("--cd-points", type=str, nargs="+", default=["full"])
    parser.add_argument("--pointflow-denorm", action="store_true")
    parser.add_argument("--one-nna-cd", action="store_true")
    parser.add_argument("--permutation-check", action="store_true")
    parser.add_argument("--permutation-seed", type=int, default=12345)
    parser.add_argument("--attention-stats", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--density-k", type=int, default=4)
    parser.add_argument("--density-batch-size", type=int, default=1)
    parser.add_argument("--density-max-values", type=int, default=2_000_000)
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
    cd_levels = parse_cd_points(args.cd_points)
    if not args.cd_only:
        from metrics.lion_backend import add_lion_root, compute_lion_metrics
        add_lion_root(args.lion_root)

    refs, refs_meta = load_points_with_meta(args.refs)
    if args.max_refs > 0:
        refs = refs[:args.max_refs]
    if args.pointflow_denorm:
        refs = denormalize_pointflow(refs, refs_meta)

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
        if args.pointflow_denorm:
            samples = denormalize_pointflow(samples, refs_meta)

        label = metric_name_from_path(path) if path.name == "samples.pt" else path.stem
        print(f"evaluating {label}: samples={tuple(samples.shape)} refs={tuple(refs.shape)}")

        if args.cd_only:
            row = compute_cd_levels(
                samples=samples,
                refs=refs,
                levels=cd_levels,
                batch_size=args.cd_batch_size,
                device=device,
                include_1nna=args.one_nna_cd,
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

        if args.density:
            row.update(
                compute_density_comparison(
                    samples=samples,
                    refs=refs,
                    k=args.density_k,
                    batch_size=args.density_batch_size,
                    device=device,
                    max_values=args.density_max_values,
                )
            )
        if args.permutation_check:
            row.update(
                compute_permutation_check(
                    samples=samples,
                    refs=refs,
                    levels=cd_levels,
                    batch_size=args.cd_batch_size,
                    device=device,
                    seed=args.permutation_seed,
                )
            )

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
