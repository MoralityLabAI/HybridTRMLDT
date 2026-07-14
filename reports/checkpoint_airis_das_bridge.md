# Checkpoint: AIRIS/DAS Hybrid Sequencer Bridge

Date: `2026-07-13`

## Completed

- Exported 28 calibration-derived context plans as `airis_storyworld_rule_v1` rules.
- Added embedded and HTTP AIRIS/DAS forecast adapters.
- Added a fail-closed topology membrane for exact protocol/context/route/orientation authorization.
- Sealed AIRIS forecasts and decisions into all 667 Verifiers v1 replay tasks.
- Exercised the actual metta-storyworld DAS-shaped service without changing its dirty worktree.

## Quantitative Result

Forecast coverage, topology acceptance, and `control_math` parity are all `1.000` on 667 valid evaluation
receipts. AIRIS/DAS macro utility is `0.900085`, exactly matching `control_math`; this is a compatibility result,
not an improvement claim. The local service materializes 28 rules into 392 facts and agrees with the embedded
best-rule forecast.

The stale-protocol negative control is rejected as `partial_airis_match` and `protocol_not_bound`, then falls
back to the original `control_math` sequence.

## Boundary

The exercised service reports `in_memory_das_shaped`; native DAS/Hyperon backends are not installed. AIRIS/DAS
retrieves proposals but never receives execution authority. No neural model is trained or modified in this step.
