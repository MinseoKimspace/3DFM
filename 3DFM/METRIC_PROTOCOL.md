# Metric Protocol

This file records the metric setup used for Rung 0 diagnostics and later
method comparisons. Numeric results go in `RUNG0_RESULTS.md`.

## Data

- Dataset: PointFlow ShapeNetCore.v2 PC15k.
- Initial category: Chair.
- Reference split: test.
- High-resolution evaluation: 8192 points.
- Standard comparison evaluation: 2048 points when EMD is available.
- Normalize/denormalize: compute metrics after applying PointFlow mean/std to
  both generated samples and references.

## Sampling

- Solver: Euler with a uniform time grid.
- NFE definition: number of model evaluations.
- Verdict NFE values: 4, 8, 64.
- Curve NFE values: 1, 2, 4, 8, 16, 64.
- Use the same noise/sample seeds across methods when doing paired diagnostics.

## Metrics

- CD: squared symmetric Chamfer distance, averaged over points.
- MMD-CD: mean nearest generated-sample distance over references.
- MMD-CD-sample: mean nearest reference distance over generated samples.
- COV-CD: fraction of references selected by generated samples as nearest refs.
- 1-NNA-CD: 1-nearest-neighbor accuracy using CD distances.
- Multi-scale CD: CD@64, CD@256, CD@1024, CD@full.
- Density: kNN distance mean, p50, p95, ratios to refs, and near-zero fraction.
- EMD metrics: deferred to 2048-point evaluation when a stable backend is available.

## Subsampling

- Use the same subsampling operator for generated samples and references.
- Use the same seed when random subsampling is used.
- Record the operator used for CD@k or 8192-to-2048 conversion.

## Metric Stages

- Stage 1: CD, 1-NNA-CD, COV-CD, multi-scale CD, and kNN density.
- Stage 2: optional Sinkhorn/OT cross-kernel check.
- Stage 3: exact/standard CUDA EMD for 2048-point paper comparison.

## Rung 0 Diagnostics

- Harness controls: external calibration, null control, jitter control, density
  control, and mode-drop control.
- Existing-model diagnostics: Base, Aux-only, xhat_selfcond, spatial_pma, and
  xhat_anchor_pma.
- xhat_anchor_pma intervention modes: normal, shuffle, zero, and
  oracle_b_gt_anchor.
