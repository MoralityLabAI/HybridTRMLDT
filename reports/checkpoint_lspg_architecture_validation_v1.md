# Checkpoint: LSPG Architecture Validation v1

## Completed

- Sealed the proposal corpus and execution addendum before task outcomes.
- Added resource-only calibration, stage manifests, bounded resumptions, campaign caps, per-example prediction receipts, and paired final inference.
- Parsed both PowerShell launchers and compiled all Python modules.

## Tests

The complete repository suite returned:

```text
244 passed, 1 failed in 81.81 seconds
```

The only failure is the pre-existing
`test_frozen_attestation_registration_and_local_sources` mismatch. The frozen
attestation registration expects v1 results hash
`92894ee80266434e12e629fface8abc16971637ed58d9b1eeca7dce6d7eb05cf`, while
this isolated branch contains
`428c0f4692e8226b34890d29a28ba46b59f61a9b2206aaa743de75badecfe019`.
That mismatch existed at branch baseline and belongs to the separate attested
provenance workstream. No LSPG file caused it, and this branch does not modify,
stage, or reconcile those frozen artifacts.

## Execution State

The one-step GPU smoke remains unexecuted because the registered preflight sees
an unrelated `TheyAreBillions.exe` process and GPU utilization above 5%. The
blocked preflight receipt is committed. No task outcome has been observed.

## Next Step

When the GPU is idle, rerun the one-step resource-only smoke, then execute S0,
S1, and S2 resource calibration. Commit the calibration receipts and selected
budget profile before sealing the A1 stage manifest.
