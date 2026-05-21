"""Logging helper interfaces."""

from __future__ import annotations

from typing import Dict

import torch


def get_logger(name: str = "3dIMF"):
    """Return a console logger.

    TODO:
        Keep logging lightweight and script-friendly.
    """

    raise NotImplementedError


def format_metrics(metrics: Dict[str, torch.Tensor | float], prefix: str = "") -> str:
    """Format scalar metrics for console logs."""

    raise NotImplementedError
