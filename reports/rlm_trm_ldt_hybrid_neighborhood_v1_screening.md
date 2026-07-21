# RLM x TRM/LDT Hybrid Neighborhood v1 Screening Result

The registered campaign stopped at screening. All eight calibration tasks and eleven arms completed under the resource wrapper, producing 88 records. The typed membrane condition passed with zero unsafe executions, but the full promotion gate failed: five API cells ended in registered token/error-limit failures and 15 of 16 conductor cells did not complete the required propose, verify, and typed-commit sequence.

## What Passed

- Provider capability transport passed before the shard.
- Typed architectures executed zero unsafe actions.
- Resource cleanup passed with 4.168 MB peak owned-process RAM, zero measured VRAM, and no I/O violation.
- Every record, event log, trajectory, capability receipt, and resource receipt is hash-bound.
- No held-out evaluation task was observed.

## What Failed

- Conductor manipulation failures: 15/16 (93.75%).
- API cell errors: 5/48 (10.42%).
- Error classes: three `ErrorThresholdExceededError` and two `TokenLimitExceededError`.
- The common conductor failure was incomplete control flow: no accepted `hybrid_commit`, often after no tool call or only a partial propose/verify sequence.

The host-side LDT fallback did its safety job, but that is not evidence that the conductor architecture worked. The screen therefore distinguishes safety containment from useful orchestration: the former passed and the latter did not.

## Calibration Signals

Screen utility is descriptive calibration evidence only. The fixed trained-ControlTRM-to-LDT arm had the highest screen accuracy (0.875), while the RLM membrane remained safe and reached 0.500 accuracy. The untyped RLM-only, proxy-TRM-only, and trained-ControlTRM-only controls executed five unsafe actions in aggregate. These eight calibration tasks are too few for architecture ranking and were not part of the registered held-out analysis.

## Decision

The v1 held-out stages will not run because `promotion_passed` is false. A successor may use this screen to set its evaluation policy, but must be labeled calibration-informed, retain the untouched 24-task evaluation split, and freeze its rule before opening any evaluation cell.

Claim boundary: generated long-context gym tasks, official RLM control flow, gym proxy/ControlTRM proposal sources, and exact typed LDT fallback only. This is not a TinyRecursiveModels, Conductor-HRM, VPD, general-alignment, or model-superiority result.
