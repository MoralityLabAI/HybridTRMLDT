# Checkpoint: LSPG Gradient Clip Grid v1

## Grid

Two five-step, resource-only S0 diagnostics varied only global gradient clip:

| Clip norm | Finite updates | First non-finite step | Step-0 raw norm |
| ---: | ---: | ---: | ---: |
| 10 | 1 | 1 | 323.9803 |
| 1 | 1 | 1 | 323.9803 |

Both runs used the registered S0 learning rate `3e-4`, produced distinct cell
hashes, emitted no task metrics, and exited cleanly through the wrapper.

## Interpretation

The identical post-update failure is not sensitive to a 100x clip range. For
AdamW, uniform gradient scaling is largely canceled by moment normalization on
the first update. The evidence therefore points to update magnitude, controlled
by learning rate, rather than the clip threshold itself.

## Next Step

Hold clip norm at the v1.1 value 100 and run a five-step S0 diagnostic at
`3e-5`. If finite, extend that setting to the full 50-step calibration before
amending the registered scale-wise rates.
