# Checkpoint: LSPG Learning-rate Diagnostic Grid v1

## Grid

Three five-step S0 diagnostics held clip norm at 100 and varied learning rate:

| Learning rate | Finite updates | First non-finite step |
| ---: | ---: | ---: |
| `3e-4` | 1 | 1 |
| `3e-5` | 2 | 2 |
| `3e-6` | 2 | 2 |
| `1e-12` | 2 | 2 |

The near-zero update still fails on the same step as `3e-5` and `3e-6`.
Parameter movement is therefore not sufficient to explain the failure. All
diagnostics are resource-only and have distinct cell hashes.

## Interpretation

The deterministic step-2 failure under a near-zero update points to an FP16
forward/backward overflow associated with that batch, not optimizer divergence.
The next discriminating control is identical FP32 execution. Precision is added
as a diagnostic cell-hash field; the campaign remains AMP FP16 until the control
is measured and any amendment is frozen.
