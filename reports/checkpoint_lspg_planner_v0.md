# LSPG-v0 Planner Checkpoint

## Completed

- Added exact, left-, right-, and interval-censored observations.
- Ingested both P2 cells as `left_censored, upper=0.15`.
- Added separate theory, empirical, and acquisition records.
- Added six-rung materialization and local-resource filtering.
- Added deterministic proposal, model, and run receipts.
- Added sealed promotion and stopping policy schemas.
- Added proposal and receipt-ingestion command-line tools.

## Tests

The planner-focused suite passes nine tests. It verifies censor round-tripping,
CDF likelihoods, censor-aware promotion blocking, deterministic proposal
hashes, twelve diverse S0 proposals, all-rung costing, resource elimination,
and trainability-only claim scope.

## Decisions

- The current empirical channel is a transparent additive approximation over
  sparse receipts, not a validated scaling surrogate.
- Tied and untied gamma fits calibrate separate prior means; P1's tied miss
  widens uncertainty rather than changing the algebraic formula.
- Two edge-censored P2 cells make lower-p grid extension the leading decision.
- Width solving uses the candidate's physical tie-class count so every variant
  stays within 3% of its unique-parameter target.
- Larger rungs are materialized and costed but explicitly excluded from Stage
  A execution.

## Open Work

Implement the real LoopedDecoderLM schedule executor and capped S0 trainer.
No proposal batch has been generated or sealed yet.
