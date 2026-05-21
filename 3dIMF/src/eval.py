"""Evaluation script interface for point-cloud iMF checkpoints."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """CLI parser for simple Chamfer evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run a minimal Chamfer evaluation.

    TODO:
        Keep this as a sanity check, not a paper-comparison protocol.
    """

    raise NotImplementedError


if __name__ == "__main__":
    main()
