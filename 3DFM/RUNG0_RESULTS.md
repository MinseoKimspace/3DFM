# Rung 0 Results

Use this file as the short result board. Keep detailed definitions in
`METRIC_PROTOCOL.md`.

## Status

| Item | Status | Result | Notes |
| --- | --- | --- | --- |
| External calibration | pending | TBD | 1-NNA-CD within 1-2 pp |
| Null control | pending | TBD | 1-NNA-CD in [45, 55]% |
| Jitter control | pending | TBD | metrics worsen with sigma |
| Density control | pending | TBD | kNN detects duplicate/subsample |
| Mode-drop control | pending | TBD | COV-CD drops |

## Existing Models

Verdict NFE: 4, 8, 64.

| Model | NFE | CD@64 | CD@256 | CD@1024 | CD@full | MMD-CD | COV-CD | 1-NNA-CD | Density | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Base | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Base | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Base | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Aux-only | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Aux-only | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Aux-only | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_selfcond | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_selfcond | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_selfcond | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| spatial_pma | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| spatial_pma | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| spatial_pma | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_anchor_pma | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_anchor_pma | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xhat_anchor_pma | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## xhat_anchor_pma Intervention

| Mode | NFE | CD@64 | CD@256 | CD@full | MMD-CD | COV-CD | 1-NNA-CD | Delta H Rel | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| normal | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| shuffle | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| zero | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| oracle_b_gt_anchor | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| normal | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| shuffle | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| zero | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| oracle_b_gt_anchor | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| normal | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| shuffle | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| zero | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| oracle_b_gt_anchor | 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## NFE Curve

Use for figures only. Verdict tables use NFE 4, 8, 64.

| Model | NFE=1 | NFE=2 | NFE=4 | NFE=8 | NFE=16 | NFE=64 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base CD@full | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_selfcond CD@full | TBD | TBD | TBD | TBD | TBD | TBD |  |
| xhat_anchor_pma CD@full | TBD | TBD | TBD | TBD | TBD | TBD |  |

## Short Conclusions

| Question | Answer | Evidence |
| --- | --- | --- |
| Is the metric harness trusted? | TBD | controls |
| Does PMA affect output? | TBD | normal/shuffle/zero |
| Does Oracle B help? | TBD | GT anchor swap |
| Does selfcond still win? | TBD | existing model table |
| Is there density collapse? | TBD | kNN stats |
| Can we move to Rung 0.5? | TBD | all above |
