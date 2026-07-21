# Kappa Data-Order Intervention v0.2.1 Recovery Registration

## Scope

This is a no-training, post-completion recovery from the malformed baseline exposure-set assertion. It was frozen after all four cells and their individual outcomes existed, but before the recovery code computed an aggregate branch.

## Admitted Evidence

- source records SHA-256: `84419b04bbb59ceba0a87f91b67f8e143bcf3f2378b71ce9f8a47b86cb3518b3`
- records: exactly `20`, all unique
- order seeds: exactly `{103,211,223,227}`
- exposures per order: exactly `{0,512,1024,2048,4096}`
- model, task, measurement, and probe seeds: exactly `103`
- model regime: tied `R=64` only
- new training cells: `0`

## Correction

The new grid must be `{0,512,1024,2048,4096}` and the sealed parent kappa grid must be `{0,1024,2048,4096}`. Replay compares all four parent kappa checkpoints at the unchanged `1e-12` tolerance. The new `E=512` kappa is retained but excluded from replay because the parent `E=512` row is gradient-only.

All scientific rules remain unchanged:

- per-order trough: `log(K2048/K1024)<=-0.1` and `log(K4096/K2048)>=0.1`
- persistence: at least `2/3` fresh orders pass
- data-order sensitivity: at most `1/3` fresh orders pass
- replay failure: `instrument_drift`

## Identity

- recovery config: `configs/loop_schedule_kappa_data_order_v0_2_1_recovery1.json`
- recovery config SHA-256: `21717c2f67e32f35424c60e53c7bd86c7afadc23c018a01a27ecdc49b9b38de4`
- source commit: `7ef610ac68f34674893e549d590def8c99b89c01`
- registration status: `frozen_after_cell_completion_before_aggregate_classification`
- full repository verification: `324 passed in 37.19s`

The original post-outcome seed selection and claim boundary remain in force.
