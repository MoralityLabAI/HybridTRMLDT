# LDT/TRM/Hybrid Benchmark Comparison

This report compares the saved benchmark outputs. VPD is not part of this comparison.

## Score Summary

| Benchmark | LDT | TRM | Hybrid | Best |
|---|---:|---:|---:|---|
| `sudoku` | 0.333 | 1.000 | 1.000 | hybrid, trm |
| `arc1` | 1.000 | 1.000 | 1.000 | hybrid, ldt, trm |
| `arc2` | 1.000 | 1.000 | 1.000 | hybrid, ldt, trm |
| `routing` | 0.474 | 0.815 | 0.815 | hybrid, trm |
| `storyworld` | 1.000 | 0.172 | 1.000 | hybrid, ldt |

## Where TRM Is Effective

- `sudoku`: TRM solves search-heavy puzzles that LDT propagation cannot solve, but uses many more guesses than hybrid.
- `routing`: TRM is the strongest hard router. It reaches `0.815` accuracy while deterministic LDT reaches `0.474` because token-lattice evidence is not sound enough for hard elimination.
- `arc1` and `arc2`: TRM solves all tasks, but it is less efficient than hybrid on aggregate because it searches a wider proposal space.
- `storyworld`: TRM is weak as a standalone policy because greedy local deficits lose modeled reachability under the rival policy.

## Where LDT Is Effective

- `arc1`, `arc2`, and `storyworld`: LDT reaches perfect task success because the transition/rule mechanics are explicit and checkable.
- `sudoku`: LDT is excellent when naked-single propagation is sufficient, but it abstains/fails on puzzles requiring search.
- `routing`: LDT is inferior as a hard router. Its token-derived candidate sets are model/experience evidence, not environment-sound deductions.

## Hybrid Behavior

- `sudoku`: hybrid matches TRM's solve rate and cuts guesses from `31` to `6` by using LDT propagation after proposals.
- `arc1`: hybrid matches the best score and uses fewer steps than LDT and fewer proposals than TRM.
- `arc2`: hybrid matches the best score and reduces aggregate proposals versus TRM (`32` vs `38`), but is inferior to TRM on two individual task proposal counts because the current proposal ordering is heuristic, not learned.
- `routing`: hybrid matches TRM accuracy only after treating LDT candidate sets as soft guidance. The hard-filter ablation drops to `0.584` accuracy.
- `storyworld`: hybrid matches LDT success and slightly reduces average steps, using `124` overrides to repair unsafe TRM proposals.

## Inferior Hybrid Cases

- No aggregate benchmark has hybrid below the best score.
- ARC-2 has individual efficiency regressions: hybrid uses more proposals than TRM on `arc2_flip_then_color` and `arc2_color_then_flip` due to non-learned pair ordering.
- Routing has a design caveat: hybrid is not better than TRM on accuracy yet; LDT is useful only as soft candidate telemetry unless calibrated.

## ARC-2 Efficiency Regressions

| Task | TRM Proposals | Hybrid Proposals | Delta | Interpretation |
|---|---:|---:|---:|---|
| `arc2_color_then_flip` | 12 | 14 | +2 | hybrid worse |
| `arc2_fill_then_rotate` | 21 | 5 | -16 | hybrid better |
| `arc2_flip_then_color` | 5 | 13 | +8 | hybrid worse |

## Routing Ablations

| Router | Accuracy | Correct | Abstained | Avg Candidates |
|---|---:|---:|---:|---:|
| `hybrid_hard_filter` | 0.584 | 101/173 | 0 | 2.09 |
| `hybrid_confidence_arbitration` | 0.815 | 141/173 | 0 | 2.09 |

## Hybrid Architecture Variants

Confidence arbitration gamma: `0.00`

| Architecture | Accuracy | Correct | Avg Candidates | Control Policy |
|---|---:|---:|---:|---|
| `typed_membrane` | 0.815 | 141/173 | 2.09 | TRM proposes; LDT evidence stays soft unless sound. |
| `hard_gate` | 0.584 | 101/173 | 2.09 | LDT candidates hard-filter TRM scoring. |
| `confidence_arbitration` | 0.815 | 141/173 | 2.09 | TRM acts above margin; LDT constrains low-margin cases. |

## Practical Map

| Regime | Best Current Tool | Reason |
|---|---|---|
| Fully checkable local mechanics | LDT or hybrid | Hard certification is reliable. |
| Search required beyond local propagation | TRM or hybrid | Latent/proposal search supplies candidates. |
| Search plus checkable constraints | Hybrid | TRM proposes; LDT prunes/certifies. |
| Noisy lexical environment routing | TRM, hybrid equal | LDT token evidence is not hard-sound. |
| Adversarial/dynamic storyworld policy | LDT or hybrid | Reachability checks prevent greedy traps. |

## Next Fixes

- Train ARC-2 hybrid proposal ordering rather than using the current static order.
- Replace the routing confidence grid with a richer calibration signal; the current trained threshold degenerates to TRM on this slice.
- Add per-instance comparison tables for hybrid regressions, especially ARC-2 proposal counts and routing confusions.
