from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from metrics.lion_backend import add_lion_root, compute_lion_metrics


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=str, required=True)
    parser.add_argument("--samples", type=str, nargs="*", default=[])
    parser.add_argument("--eval-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-refs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--lion-root", type=str, default="")
    parser.add_argument("--compute-emd", action="store_true")
    parser.add_argument("--accelerated-cd", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
