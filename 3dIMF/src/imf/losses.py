"""Loss interfaces for boundary-condition iMF experiments."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_V(
    model: torch.nn.Module,
    z: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    tangent_z: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute the compound iMF velocity `V`.

    Model interface:
        `model(z, r, t) -> (u, v)`, where both tensors have shape [B, N, 3].

    Intended iMF structure:
        `u`: average velocity head.
        `v`: instantaneous velocity auxiliary head.
        `V = u + (t - r) * stopgrad(du_dt)`.

    Args:
        model: iMF network with average and instantaneous velocity heads.
        z: Current point cloud, shape [B, N, 3].
        r: Reference time, shape [B, 1, 1].
        t: Current time, shape [B, 1, 1].
        tangent_z: Optional JVP direction for `z`, shape [B, N, 3]. If omitted,
            the first implementation should use the predicted `v` head.
        create_graph: Whether to retain higher-order gradients.

    Returns:
        V: Compound iMF velocity, shape [B, N, 3].

    TODO:
        Follow the official iMF code path conceptually: compute the JVP of the
        `u` head along the predicted `v` direction, then form `V`. Keep backend
        behavior easy to debug before optimizing memory or speed.
    """

    raise NotImplementedError


def imf_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    e: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    bc_eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Boundary-condition iMF loss entry point.

    Args:
        model: Network with `model(z, r, t) -> (u, v)`.
        x: Data point cloud, shape [B, N, 3].
        e: Noise point cloud, shape [B, N, 3].
        r: Reference time, shape [B, 1, 1].
        t: Current time, shape [B, 1, 1].
        bc_eps: Tolerance for detecting equal-time boundary pairs.

    Returns:
        loss: Scalar tensor.
        metrics: Dict of scalar tensors for logging.

    TODO:
        Build the first readable iMF objective here:
        `v_target = e - x`;
        `loss_u = ||V - stopgrad(v_target)||^2`;
        `loss_v = ||v - stopgrad(v_target)||^2`.
        Derive the flow-matching mask from `abs(t - r) <= bc_eps` unless
        `sample_time_pair` is later extended to return an explicit mask.
        Keep the detach choices explicit and compare against the official
        `Lyy-iiis/imeanflow` implementation while porting from images to points.
    """

    raise NotImplementedError
