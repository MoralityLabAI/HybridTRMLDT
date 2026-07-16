# Independently Induced AIRIS Rules

Calibration-derived rules: `28`
Held-out episodes: `667`
Primary confidence threshold: `0.50`

| Controller | Macro utility | Macro accuracy | Macro cost |
|---|---:|---:|---:|
| `raw_airis` | 0.9001 | 0.9667 | 6.436 |
| `guarded_airis` | 0.9001 | 0.9667 | 6.436 |
| `control_math` | 0.9001 | 0.9667 | 6.436 |
| `episode_oracle` | 0.9065 | 0.9725 | 6.324 |

Raw held-out oracle-label accuracy: `0.940`
Guarded acceptance rate: `1.000`
Accepted-rule oracle-label accuracy: `0.940`
Intact-but-wrong held-out rules: `40`
Guarded harmful changes: `0`
Raw proposals below control: `0`
Fallback save rate: `0.000`
Macro utility delta vs control: `+0.000000`

## Held-Out Rule Precision

| Family | Episodes | Rule precision | Wrong rules |
|---|---:|---:|---:|
| `arc1` | 168 | 1.000 | 0 |
| `arc2` | 160 | 1.000 | 0 |
| `routing` | 115 | 0.965 | 4 |
| `story_moral` | 64 | 0.781 | 14 |
| `story_secret` | 64 | 0.656 | 22 |
| `sudoku` | 96 | 1.000 | 0 |

## Confidence Sensitivity

| Threshold | Acceptance | Macro utility | Harmful changes | Wrong-rule acceptance |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.9001 | 0 | 1.000 |
| 0.50 | 1.000 | 0.9001 | 0 | 1.000 |
| 0.60 | 1.000 | 0.9001 | 0 | 1.000 |
| 0.70 | 1.000 | 0.9001 | 0 | 1.000 |
| 0.80 | 0.904 | 0.9001 | 0 | 0.450 |
| 0.90 | 0.808 | 0.9001 | 0 | 0.100 |
| 1.00 | 0.760 | 0.9001 | 0 | 0.000 |

Claim boundary: AIRIS-style empirical context-rule induction from deterministic calibration outcomes; not causal identification, neural AIRIS training, or native distributed DAS performance.
