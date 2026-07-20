# Checkpoint: LSPG Clip-100 Diagnostic v1

## Result

An isolated resource-only S0 diagnostic reproduced the v1.1 trajectory with a
canonical result receipt:

| Measurement | Value |
| --- | ---: |
| finite optimizer steps | 1 |
| step-0 raw gradient norm | 323.9803 |
| applied clipped norm | 100.0 |
| first non-finite gradient step | 1 |
| token-visit exposures | 16,384 |
| peak allocated CUDA memory | 129,403,904 bytes |

The failure is therefore post-update rather than an initialization-only AMP
overflow. The wrapper recorded `process_exit_3`, 1,056.711 MB peak RAM, 7.146
MB/s peak I/O, and clean cleanup. No task outcome was emitted.

## Diagnostic Decision

Add a diagnostic-only clip override to the hard-cap wrapper and bind the clip
value into the cell hash. Run five-step resource-only probes at clip norms 10
and 1. The registered campaign remains at 100 until those diagnostics support a
minimal stable correction and a new amendment is frozen.
