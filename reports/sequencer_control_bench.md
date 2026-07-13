# Control Math on Infused Hybrid Skill Sequencers

Evaluation episodes: `667`
Protocol SHA-256: `a80d6c729c0005d26f0eccac24e27f564c21cfad4b97417c2bcbc66f6f1007b8`
Context section rate: `1.000`
Orientation reversal rate: `0.143`

| Sequencer | Macro utility | Macro accuracy | Macro cost | Constraint violations |
|---|---:|---:|---:|---:|
| `global_signed` | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| `lineage_only` | 0.8998 | 0.9667 | 6.453 | 0.0000 |
| `fixed_typed` | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| `control_math` | 0.9001 | 0.9667 | 6.436 | 0.0000 |
| `local_calibrated` | 0.9001 | 0.9667 | 6.436 | 0.0000 |

## Paired comparisons

| Control | Utility delta | 95% bootstrap CI | Sign-flip p | Holm p | Accuracy delta | Cost delta |
|---|---:|---:|---:|---:|---:|---:|
| `global_signed` | +0.0069 | [+0.0049, +0.0092] | 0.0003 | 0.0005999 | +0.0000 | -1.238 |
| `lineage_only` | +0.0003 | [+0.0000, +0.0009] | 0.5023 | 0.5023 | +0.0000 | -0.017 |
| `fixed_typed` | +0.0069 | [+0.0049, +0.0092] | 0.0002 | 0.0005999 | +0.0000 | -1.238 |

## Post-registration error-budget sensitivity

| Error budget | Section rate | Macro utility | Macro accuracy | Macro cost |
|---:|---:|---:|---:|---:|
| 0.25 | 1.000 | 0.9001 | 0.9667 | 6.436 |
| 0.50 | 1.000 | 0.9001 | 0.9667 | 6.436 |
| 0.75 | 0.821 | 0.9001 | 0.9667 | 6.436 |
| 1.00 | 0.714 | 0.8967 | 0.9667 | 6.737 |
| 1.50 | 0.214 | 0.8949 | 0.9667 | 6.885 |
| 2.00 | 0.143 | 0.8943 | 0.9667 | 7.223 |

## Family breakdown

| Family | Episodes | Utility | Accuracy | Cost | Violations |
|---|---:|---:|---:|---:|---:|
| `arc1` | 168 | 0.9722 | 1.0000 | 2.429 | 0.0000 |
| `arc2` | 160 | 0.9851 | 1.0000 | 9.125 | 0.0000 |
| `routing` | 115 | 0.7812 | 0.8000 | 1.000 | 0.0000 |
| `story_moral` | 64 | 0.7235 | 1.0000 | 5.766 | 0.0000 |
| `story_secret` | 64 | 0.9869 | 1.0000 | 5.766 | 0.0000 |
| `sudoku` | 96 | 0.9516 | 1.0000 | 14.531 | 0.0000 |

Claim boundary: Matched deterministic solver and policy replay over procedural Sudoku/ARC-style tasks, local Tesseract routing trajectories, and the finite-state storyworld. It measures sequencer control, not neural weight infusion or official ARC/INTELLECT-3 leaderboard performance.
