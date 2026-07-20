# Loop Schedule Kappa Surface v1: Attempt-1 Orchestration Failure

## Status

Attempt 1 is sealed as `external_orchestration_failure`, not as a scientific result. The command host terminated its PowerShell invocation after approximately 20 minutes, before the registered wrapper's 1800-second phase timeout and before the wrapper's `finally` block could write its sampled resource receipt.

The child Python process remained alive as PID `25180`. It was the exact PID launched by the wrapper and was terminated explicitly; the subsequent process and GPU-registry audit found no lingering owned process or GPU entry. No broad process-name cleanup was used.

## Retained Evidence

The append-only event ledger is preserved byte-for-byte:

- path: `experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl`
- SHA-256: `ad4c4b7a40945b7086994705f21abdfb7598074c96d7b64c79fd9e6124b74c09`
- events: 56
- measurements: 31
- fully completed cells: 6
- checkpoints: 12

The six complete cells are all tied `R=16` and tied `R=32` seeds. The final event is the initialization measurement for tied `R=64`, seed 101; that cell lacks a completion event and is inadmissible as a completed cell. No canonical `surface_records.jsonl`, `surface_result.json`, or final benchmark receipt was emitted.

## Interpretation

No model comparison, curvature onset, gradient-coupling result, or scientific classification is licensed from attempt 1. The completed cells may be reused only by a separately registered recovery protocol that:

1. binds to the exact partial-ledger hash;
2. admits only cells with a finite `cell_complete` event and the full registered measurement set;
3. excludes the incomplete `R=64` cell entirely;
4. states that the recovery was designed after the tied `R=16/R=32` trajectories were observed; and
5. retains the original hard caps and aggregate one-GPU-hour accounting.

The failure also reveals a cost-model error: the five-iteration kappa estimator scales approximately quadratically with visit count, so a 90-measurement monolithic phase was not feasible inside the outer command host's 20-minute window.
