# AIRIS/DAS Forecast Integrity Experiment Notes

## Design

- Paired deterministic fault injection: every evaluation episode receives the same ten conditions.
- Clean retrieval is the positive control; nine protocol, context, receipt, and outage conditions are
  negative controls.
- `topology_only` preserves the prior protocol/context/route/orientation membrane without a rule digest.
- `integrity_sealed` additionally binds rule ID, confidence, support, counterexample count, preconditions,
  and predicted outcome to the frozen calibration-derived rule registry.
- Candidate outcomes are held fixed. No model, solver, or task policy is retrained.

## Results

- episodes: `667`
- total paired trials: `6670`
- attack trials: `6003`
- topology-only clean acceptance: `1.000000`
- topology-only attack acceptance: `0.444444`
- integrity-sealed clean acceptance: `1.000000`
- integrity-sealed attack acceptance: `0.000000`
- integrity-sealed attack fallback: `1.000000`
- attack acceptance delta: `-0.444444`

## Interpretation

Topology checks reject semantic route, authority, protocol, context, and orientation conflicts, but they
cannot identify a syntactically valid forecast whose rule material changed in transit. The digest seal
closes that transport-integrity gap while preserving clean acceptance.

## Boundary

Deterministic paired fault-injection over sealed AIRIS/DAS replay receipts. It measures authorization resilience, not adversarial robustness of a learned AIRIS model or native distributed DAS.
