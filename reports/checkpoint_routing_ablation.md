# Checkpoint Routing Ablation

Date: 2026-06-19

## Work Completed

- Added a hard-filter routing ablation, `hybrid_hard_filter`.
- Added a confidence-arbitration routing ablation, `hybrid_confidence_arbitration`.
- Kept the ablation outside the main `ldt`, `trm`, `hybrid` result set so aggregate benchmark comparison remains stable.
- Fixed routing candidate refinement determinism by sorting token and candidate iteration.
- Regenerated routing outputs, experiment notes, aggregate comparison report, and paper text.

## Deterministic Routing Result

| Router | Accuracy | Correct | Abstained | Avg Candidates |
|---|---:|---:|---:|---:|
| `ldt` | 0.474 | 82/173 | 15 | 0.00 |
| `trm` | 0.815 | 141/173 | 0 | 0.00 |
| `hybrid` | 0.815 | 141/173 | 0 | 2.09 |
| `hybrid_hard_filter` | 0.584 | 101/173 | 0 | 2.09 |
| `hybrid_confidence_arbitration` | 0.815 | 141/173 | 0 | 2.09 |

## Architecture Variants

| Architecture | Accuracy | Correct | Control Policy |
|---|---:|---:|---|
| `typed_membrane` | 0.815 | 141/173 | TRM proposes; LDT evidence stays soft unless sound |
| `hard_gate` | 0.584 | 101/173 | LDT candidates hard-filter TRM scoring |
| `confidence_arbitration` | 0.815 | 141/173 | TRM acts above train-selected margin; saved gamma `0.00` |

## Interpretation

The ablation strengthens the paper's membrane claim. Token-derived LDT candidate sets are useful as soft telemetry, but hard-eliminating TRM routes from this evidence is unsafe. Soft hybrid preserves TRM's `0.815` accuracy; hard filtering drops to `0.584`. Confidence arbitration is implemented as a second alternative, but the train-selected gamma is `0.00`, so it degenerates to the TRM boundary on this routing slice.

## Files Updated

- `research_gym/benchmarks/routing_bench.py`
- `research_gym/scripts/bench_routing.py`
- `research_gym/scripts/compare_benchmarks.py`
- `tests/test_routing_bench.py`
- `tests/test_compare_benchmarks.py`
- `papers/overleaf/figures/architecture_comparison.tex`
- `papers/overleaf/figures/control_dynamics.tex`
- `reports/routing_bench.md`
- `reports/benchmark_comparison.md`
- `experiments/routing_env_pointer/results.json`
- `experiments/routing_env_pointer/training_notes.md`
- `papers/hybrid_ldt_trm_draft.md`
- `papers/overleaf/main.tex`

## Next Research Step

Train or tune ARC-2 proposal ordering. The current static hybrid ordering still has per-task proposal-count regressions even though aggregate ARC-2 score and proposal count are favorable.
