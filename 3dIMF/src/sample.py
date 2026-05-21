"""Sampling script interface for trained point-cloud iMF checkpoints."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """CLI parser for checkpoint sampling."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Load checkpoint and export generated point clouds.

    TODO:
        Wire checkpoint loading, model construction, sampler, and `.ply`
        export after the core sampler is implemented.
    """

    raise NotImplementedError


if __name__ == "__main__":
    main()
