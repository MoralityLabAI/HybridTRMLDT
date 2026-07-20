# Checkpoint: LSPG Non-finite Receipt Failure v1

## Result

S0 calibration attempt 3 entered the training loop under execution amendment
v1.1 and reached a non-finite raw gradient. The stop rule fired, but result
sealing then rejected the IEEE `inf` diagnostic because canonical JSON forbids
non-finite numbers. The wrapper retained `process_exit_1`, 1,066.383 MB peak
RAM, 0 MB reported process VRAM under WDDM telemetry, 7.586 MB/s peak I/O, and
clean process exit.

## Correction

The result schema now preserves the largest finite raw norm and records a
boolean non-finite flag plus the exact failing step. The events ledger receives
an explicit stop row. No non-finite float is serialized, and the v1.1 behavior
still stops before any update associated with a non-finite gradient.

## Next Step

Run a separate resource-only S0 diagnostic with the corrected receipt path.
Determine whether the event is initial, post-update, or AMP-scale related before
changing any registered training behavior.
