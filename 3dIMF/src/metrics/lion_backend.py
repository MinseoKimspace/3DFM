from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def add_lion_root(lion_root: str | None) -> None:
    if not lion_root:
        return
    path = str(Path(lion_root).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_lion_compute_all_metrics():
    # External dependency, not vendored.
    # Source: https://github.com/nv-tlabs/LION
    # Used by OGPP for Shape CD/EMD metrics:
    # https://github.com/swang3081/OGPP/blob/main/BASELINES.md
    from utils.evaluation_metrics_fast import compute_all_metrics

    return compute_all_metrics


def compute_lion_metrics(
    samples: torch.Tensor,
    refs: torch.Tensor,
    batch_size: int,
    compute_emd: bool,
    accelerated_cd: bool,
    verbose: bool,
) -> dict[str, Any]:
    compute_all_metrics = load_lion_compute_all_metrics()
    metric2 = "EMD" if compute_emd else None
    return compute_all_metrics(
        sample_pcs=samples,
        ref_pcs=refs,
        batch_size=batch_size,
        verbose=verbose,
        accelerated_cd=accelerated_cd,
        metric1="CD",
        metric2=metric2,
    )
