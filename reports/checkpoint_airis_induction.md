# Checkpoint: AIRIS Calibration-Outcome Induction

Date: `2026-07-16`

## Completed

- Froze the induction target, tie order, support threshold, confidence sweep, and endpoint.
- Induced 28 context rules from 455 calibration episode utility winners.
- Preserved calibration disagreements as concrete rule counterexamples.
- Kept topology features in the authorization membrane rather than using topology sequence labels for learning.
- Evaluated intact rules over all 667 held-out sequencer episodes.
- Replayed an induced rule through the actual metta-storyworld DAS-shaped HTTP service.

## Commands Run

```powershell
python -m research_gym.scripts.bench_airis_induction
python scripts/smoke_airis_das_bridge.py --rules data/airis_das/induced_rules.json --episodes data/benchmarks/sequencer_control_episodes.jsonl --bridge-results data/benchmarks/airis_induction_results.json --out data/airis_das/induced_live_service_smoke.json
python -m pytest -q
```

## Results

Raw held-out utility-winner precision is `0.940030`; the 40 errors comprise four routing, 22 secret-ending, and
14 moral-optimization episodes. Sudoku, ARC-1, and ARC-2 context rules have perfect winner precision. At the
primary threshold `0.5`, acceptance is `1.000`, accepted-rule precision is `0.940030`, and all intact wrong rules
remain accepted.

Every induced proposal equals the frozen `control_math` context choice. Raw and guarded macro utility is
`0.900085`, proposal/control parity is `1.000`, and macro utility delta is `+0.000000`. Raising confidence to
`1.0` rejects all wrong rules but still executes the same fallback, so utility, accuracy, and cost are unchanged.
The live in-memory DAS-shaped service loads 28 induced rules as 420 facts and agrees with the embedded forecast.
The full regression suite passes: `117 passed in 15.24s`.

## Decision

Record this as a negative-result baseline. Confidence is useful for identifying uncertain contexts, but a gate
cannot protect behavior when its proposal and fallback are identical. The next architecture must condition on
episode state and provide a typed fallback capable of selecting a different sequence.

## Boundary

This is deterministic context-majority induction from calibration utilities. It is not causal identification,
neural AIRIS training, native distributed DAS performance, or evidence that confidence alone improves control.
