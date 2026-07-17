# Forward Spectral Prediction of Controller Genomes

Protocol: `controller_mesh_sheaf_forward_v1`.

## Staging

Calibration geometry for all genomes was sealed before discovery outcomes were revealed. Held-out predictions were then sealed before held-out outcomes were computed.

- genomes: `64`
- discovery / heldout: `48 / 16`
- calibration / evaluation tasks: `432 / 432`

## Forward Result

- held-out Spearman rho: `+0.705`
- matched N0 mean rho: `+0.276`
- matched N0 one-sided p: `0.0078`
- predicted top-4 uplift: `+0.1476`
- predicted/actual top-4 overlap: `2/4`
- registered forward gate passed: `True`

## Held-Out Predictions

| Genome | Predicted objective | Actual objective | Utility | Unsafe |
|---|---:|---:|---:|---:|
| `e-dual__g-010__f-safe_ldt` | +1.0120 | +1.0050 | 0.8684 | 0.000 |
| `e-dual__g-000__f-correction_infused` | +1.0066 | +1.0954 | 0.9323 | 0.000 |
| `e-dual__g-000__f-ldt` | +1.0045 | +0.9706 | 0.8435 | 0.000 |
| `e-exact__g-035__f-correction_infused` | +1.0006 | +1.1531 | 0.9703 | 0.000 |
| `e-soft__g-075__f-correction_infused` | +0.9935 | +1.0836 | 0.9352 | 0.002 |
| `e-soft__g-035__f-safe_ldt` | +0.9900 | +0.9072 | 0.8167 | 0.012 |
| `e-exact__g-000__f-safe_ldt` | +0.9614 | +1.1247 | 0.9409 | 0.000 |
| `e-soft__g-000__f-ldt` | +0.9505 | +0.6923 | 0.7742 | 0.097 |
| `e-exact__g-075__f-ldt` | +0.9424 | +0.8609 | 0.7750 | 0.000 |
| `e-none__g-010__f-correction_infused` | +0.8740 | +0.9044 | 0.9178 | 0.100 |
| `e-soft__g-075__f-identical` | +0.8603 | +0.7741 | 0.8601 | 0.125 |
| `e-exact__g-035__f-identical` | +0.8268 | +0.7716 | 0.8601 | 0.125 |
| `e-dual__g-010__f-identical` | +0.8157 | +0.7691 | 0.8601 | 0.125 |
| `e-none__g-010__f-safe_ldt` | +0.7949 | +0.8702 | 0.8935 | 0.100 |
| `e-none__g-000__f-ldt` | +0.7886 | +0.7766 | 0.8601 | 0.125 |
| `e-none__g-035__f-identical` | +0.7526 | +0.7766 | 0.8601 | 0.125 |

## Post-Hoc Diagnostics

These diagnostics were not part of the registered gate.

| Predictor | Held-out rho | Top-k uplift |
|---|---:|---:|
| Categorical genome baseline | +0.673 | +0.0950 |
| Intervention spectrum only | +0.084 | +0.0595 |
| Stability spectrum only | +0.567 | +0.0323 |

## Boundary

Forward prediction across 16 held-out controller combinations in deterministic source-inspired proxy tasks. The study does not establish causal spectral control, independent architecture-population generalization, neural safety, or deployment utility.
