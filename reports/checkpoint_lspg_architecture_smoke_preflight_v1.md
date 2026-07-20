# Checkpoint: LSPG Architecture Smoke Preflight v1

## Attempt

- Cell: `LSAD-C-K2L8-calibration-S0-s991`.
- Intended exposure: one optimizer step, resource-only.
- Wrapper: registered RAM, CPU, I/O, VRAM, and wall-time caps.

## Result

The wrapper returned `construction_failure` before creating a child process.
GPU utilization was 40%, above the registered 5% launch threshold, and an
unrelated `TheyAreBillions.exe` compute process was present. The wrapper did not
start training and did not terminate or modify the foreign process.

The resource receipt records `owned_pid: null`, `cleanup_passed: true`, and
`abort_reason: external_gpu_contention`. No task outcome was observed.

## Next Step

Retain this receipt as a preflight negative control. Retry the same smoke cell
only after the device satisfies all three idle samples; do not relax the frozen
contention policy.
