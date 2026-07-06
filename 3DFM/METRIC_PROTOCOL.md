# Metric Protocol and Rung 0 Diagnostics

This document freezes the evaluation protocol before adding new coupling or
architecture code. Its job is to make sure later comparisons use the same
metric definitions, data split, normalization, seeds, and reporting tables.

## Scope

Steps covered here:

1. Metric protocol freeze.
2. Harness controls.
3. Corrected diagnostics for existing models.

No new model or coupling implementation is required for these steps. Existing
sampling and evaluation scripts can be used.

## Protocol Freeze

Fill this table before comparing methods.

| Item | Fixed Value | Notes |
| --- | --- | --- |
| Dataset | PointFlow ShapeNetCore.v2 PC15k | Use the same source for all runs. |
| Category | Chair | Extend later only after Chair is stable. |
| Reference split | test | Do not mix train/test refs in the same table. |
| Reference count | TBD | Record exact number. |
| Generated sample count | TBD | Keep fixed across methods. |
| Point count | 8192 | Use 2048 only for EMD-compatible checks. |
| Denormalization | PointFlow mean/std before metrics | Required for samples and refs. |
| CD convention | Squared symmetric Chamfer, mean over points | Do not change without updating this file. |
| MMD-CD | Mean nearest sample distance over refs | PointFlow/L-GAN convention. |
| MMD-CD-sample | Mean nearest ref distance over samples | Useful but not the main MMD number. |
| COV-CD | Unique refs covered by sample nearest-neighbor indices / num refs | Coverage diagnostic. |
| 1-NNA-CD | 1-nearest-neighbor accuracy using CD distances | Same sample/ref count preferred. |
| EMD metrics | 2048-point protocol only | At minimum report 1-NNA-EMD if feasible. |
| NFE values | 4, 8, 64 | Add 1/2/16 only for visual stress tests. |
| Multi-scale CD | 64, 256, 1024, full | Treat as diagnostics, not official PointFlow metrics. |
| Eval resampling | 2-3 sample sets per checkpoint | Report mean and std for COV/1-NNA. |
| Noise seed policy | TBD | Same noise across methods for paired visual/CD diagnostics. |
| Output format | JSON + Markdown table | JSON is source of truth. |

## Harness Controls

Do not interpret method comparisons until these controls pass or failures are
documented.

### External Calibration

Goal: reproduce one public checkpoint/sample metric within a reasonable
tolerance. If exact reproduction is impossible, record the mismatch and why.

| Source | Category | Points | Samples | Reported 1-NNA-CD | Ours 1-NNA-CD | Delta | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TBD | Chair | TBD | TBD | TBD | TBD | TBD | pending |  |

Debug order if calibration fails:

1. Squared vs non-squared CD.
2. Mean vs sum convention.
3. Global vs per-shape denormalization.
4. Reference split and reference count.
5. Generated sample count.
6. EMD implementation details, if using EMD.

### Null Control

Use references as both generated samples and references.

Expected:

- 1-NNA-CD near 50 percent.
- CD/MMD-CD very low.
- COV-CD high.

| Refs | Samples | Points | 1-NNA-CD | MMD-CD | COV-CD | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| TBD | same as refs | 8192 | TBD | TBD | TBD | pending |  |

### Jitter Positive Control

Add Gaussian jitter to a fixed sample set. Metrics should worsen monotonically
or nearly monotonically as sigma increases.

| Sigma | CD | MMD-CD | COV-CD | 1-NNA-CD | kNN Mean Ratio | Status | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.000 | TBD | TBD | TBD | TBD | TBD | pending | baseline |
| 0.001 | TBD | TBD | TBD | TBD | TBD | pending |  |
| 0.005 | TBD | TBD | TBD | TBD | TBD | pending |  |
| 0.010 | TBD | TBD | TBD | TBD | TBD | pending |  |
| 0.020 | TBD | TBD | TBD | TBD | TBD | pending |  |

### Density Control

Duplicate or subsample points to create a known density artifact. CD may miss
part of the artifact; kNN statistics should detect it.

| Perturbation | CD | MMD-CD | kNN Mean Ratio | kNN P50 Ratio | Near-Zero NN Fraction | Status | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| none | TBD | TBD | TBD | TBD | TBD | pending | baseline |
| duplicate points | TBD | TBD | TBD | TBD | TBD | pending | expect near-zero increase |
| subsample/repeat | TBD | TBD | TBD | TBD | TBD | pending | expect density shift |

### Mode-Drop Control

Reduce generated diversity intentionally. Coverage and 1-NNA should detect the
drop even if CD changes mildly.

| Perturbation | MMD-CD | COV-CD | 1-NNA-CD | Status | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| none | TBD | TBD | TBD | pending | baseline |
| repeat subset | TBD | TBD | TBD | pending | expect COV drop |
| keep half modes | TBD | TBD | TBD | pending | expect COV drop |

## Existing-Model Corrected Diagnostics

Run these after the harness controls are acceptable. Use the frozen protocol
above and PointFlow denormalization.

### Diagnostic 1: xhat_anchor_pma Intervention

Purpose: test whether xhat-anchor spatial PMA is used structurally.

| Run | Checkpoint | Mode | NFE | CD@64 | CD@256 | CD@1024 | CD@full | MMD-CD | COV-CD | 1-NNA-CD | Time/Sample | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| xhat_anchor_pma | TBD | normal | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | shuffle | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | zero | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | normal | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | shuffle | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | zero | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | normal | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | shuffle | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | zero | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |

Interpretation:

- normal ~= shuffle ~= zero supports the same-pass redundancy diagnosis.
- normal clearly better than zero means the slot path is load-bearing.
- shuffle worse than zero may indicate harmful or unstable slot routing.

### Diagnostic 2: Attention Mass and Delta H

Purpose: separate "slots are not read" from "slots are read but weak".

| Run | Mode | NFE | Attention Entropy | Attention Entropy Norm | Attention Max | Delta H Norm | Delta H Rel Norm | CD@full | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| xhat_anchor_pma | normal | 4 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | shuffle | 4 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | zero | 4 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | normal | 8 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | shuffle | 8 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | zero | 8 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | normal | 64 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | shuffle | 64 | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | zero | 64 | TBD | TBD | TBD | TBD | TBD | TBD |  |

Interpretation:

- Low Delta H relative norm means weak injection.
- High attention entropy with low Delta H means diffuse and weak readout.
- Strong Delta H but little metric effect suggests redundant information.

### Diagnostic 3: Existing Model Multi-Scale CD

Purpose: re-evaluate existing 8192 models under the corrected protocol.

| Model | Checkpoint | NFE | CD@64 | CD@256 | CD@1024 | CD@full | MMD-CD | COV-CD | 1-NNA-CD | Time/Sample | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base | TBD | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Aux-only | TBD | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | TBD | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| spatial_pma | TBD | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Base | TBD | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Aux-only | TBD | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | TBD | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| spatial_pma | TBD | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Base | TBD | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Aux-only | TBD | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | TBD | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| spatial_pma | TBD | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | TBD | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |

Interpretation:

- selfcond > aux-only > base suggests self-conditioning is load-bearing.
- aux-only ~= selfcond > base suggests deep supervision/capacity effect.
- PMA ~= base supports the current weak-guidance diagnosis.
- CD@64 poor but CD@full good suggests coarse/fine mismatch.

### Diagnostic 4: kNN Density Distribution

Purpose: detect density collapse, clumping, or oversparse regions that CD can
hide.

| Model | Mode | NFE | k | Sample Mean | Ref Mean | Mean Ratio | Sample P50 | Ref P50 | P50 Ratio | Sample P95 | Ref P95 | P95 Ratio | Near-Zero Fraction | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base | normal | 4 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | normal | 4 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | normal | 4 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Base | normal | 8 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | normal | 8 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | normal | 8 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| Base | normal | 64 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond | normal | 64 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma | normal | 64 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |  |

Interpretation:

- Very low near-neighbor distances indicate clumping/duplicates.
- Large mean or P95 ratio indicates oversparse regions or outliers.
- Similar CD with worse density means the model matches surface location but
  not point distribution.

## Decision Gates

| Gate | Pass Condition | If Failed |
| --- | --- | --- |
| Harness calibration | External/null/positive controls behave as expected | Fix eval before method comparisons. |
| Existing diagnostics | Corrected metrics confirm PMA weak use and/or 8192 issue | If not confirmed, revise motivation before coupling. |
| Density diagnostics | kNN stats are interpretable and stable | Add stronger density controls before Phase A. |
| Rung 0.5 readiness | Coupling variance can be measured without training | Implement coupling-only ladder next. |

## Notes

- Multi-scale CD is a diagnostic, not an official PointFlow metric.
- EMD should be reported at 2048 if feasible to avoid CD-only conclusions.
- Do not add new knobs unless a decision gate explicitly triggers them.
- Keep JSON files as source of truth; Markdown tables are summaries.
