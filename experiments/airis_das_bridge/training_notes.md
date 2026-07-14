# AIRIS/DAS Bridge Experiment Notes

## Design

- One AIRIS-compatible rule is exported per calibration-derived context plan.
- Evaluation outcomes are not used to construct rules, confidence, support, or routing.
- DAS performs rule persistence and forecast retrieval; it does not receive execution authority.
- The topology membrane accepts only exact, protocol-bound, context-bound, supported, route-consistent,
  orientation-safe proposals. Every failed check falls back to `control_math`.
- Verifiers v1 replays sealed forecasts and decisions, so benchmark evaluation has no mutable service
  dependency. The actual local HTTP service is exercised by a separate smoke test.

## Quantitative Replay

- episodes: `667`
- rules: `28`
- forecast coverage: `1.000000`
- topology acceptance: `1.000000`
- fail-closed fallback: `0.000000`
- control-math parity: `1.000000`
- macro utility delta: `+0.000000`

## Service Contract

- expected service schema: `airis_das_service_v1`
- expected forecast schema: `airis_das_forecast_v1`
- backend exercised separately: `in_memory_das_shaped`

## Boundary

Conformance and replay benchmark for calibration-derived AIRIS rules in a DAS-shaped index. It does not measure independent AIRIS causal learning or native distributed DAS performance.
