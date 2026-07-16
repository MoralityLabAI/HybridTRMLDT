# Checkpoint: AIRIS/DAS Forecast Integrity

Date: `2026-07-16`

## Completed

- Canonically hashed rule ID, confidence, support, counterexample count, preconditions, and predicted outcome.
- Bound accepted forecasts to the frozen calibration-derived rule registry.
- Made `predicts.outcome` authoritative and redundant `outcome_fields` consistency-checked.
- Added paired clean, distribution-shift, receipt-tamper, and outage trials over every evaluation episode.
- Extended the live HTTP smoke with altered-rule rejection.
- Anchored the live ruleset to the independently saved bridge-result SHA-256 before registry construction.

## Commands Run

```powershell
python -m research_gym.scripts.bench_airis_das_bridge
python -m research_gym.scripts.bench_airis_das_resilience
python scripts/smoke_airis_das_bridge.py
python -m pytest -q
```

## Results

The benchmark contains 6,670 paired trials: 667 clean controls and 6,003 negative controls. Topology-only
arbitration accepts 44.4% of altered receipts and changes the selected sequence in 11.1%; maximum observed regret
relative to `control_math` is 1.0037. Integrity sealing retains 100% clean acceptance while rejecting 100% of
negative controls, always falling back to `control_math`.

## Decisions Made

Rule integrity is a separate typed gate, not another confidence term. High confidence or support cannot purchase
authorization after the frozen rule material changes. The registry is derived only from calibration-exported
rules and does not use evaluation outcomes.

## Boundary

This validates deterministic transport and index integrity under specified faults. It does not establish learned
AIRIS robustness, native distributed DAS behavior, or the behavioral quality of an intact rule.
The ruleset hash or equivalent signature must remain outside the mutable rule/index store.

## Recommended Next Step

Train or induce AIRIS rules on a calibration split rather than exporting the topology plan directly, then compare
accepted proposal utility, fallback coverage, and intact-but-wrong rule detection on held-out contexts.
