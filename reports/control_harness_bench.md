# Cross-Project Hybrid Control Harness Benchmark

These are deterministic source-inspired proxy tasks, not direct performance claims about the neighboring repositories.
They preserve the control distinction found in those repos: exact mechanics/provenance versus model- or replay-derived guidance.

Train cases per application: `48`
Held-out cases per application: `48`
Calibrated confidence margin: `0.10`
Calibrated beam width: `2`

## Held-out aggregate

| Architecture | Accuracy | Utility | Unsafe Rate | Regret | Cost | LDT Consult |
|---|---:|---:|---:|---:|---:|---:|
| `trm` | 0.725 | 0.833 | 0.157 | 0.366 | 1.00 | 0.000 |
| `ldt` | 0.528 | 0.756 | 0.000 | 0.244 | 2.00 | 1.000 |
| `hard_gate` | 0.465 | 0.729 | 0.146 | 0.463 | 2.00 | 1.000 |
| `confidence_arbitration` | 0.808 | 0.887 | 0.111 | 0.260 | 1.22 | 0.225 |
| `typed_membrane` | 0.845 | 0.932 | 0.000 | 0.068 | 1.36 | 0.157 |
| `typed_confidence` | 0.928 | 0.967 | 0.000 | 0.033 | 1.57 | 0.169 |
| `counterfactual_beam` | 0.907 | 0.965 | 0.000 | 0.035 | 3.50 | 1.000 |
| `skill_router` | 0.935 | 0.972 | 0.000 | 0.028 | 1.55 | 0.245 |

## Learned skill routes

| Skill | Selected architecture |
|---|---|
| `coalition_planning` | `typed_membrane` |
| `control_profile_selection` | `confidence_arbitration` |
| `moral_optimization` | `trm` |
| `opponent_modeling` | `trm` |
| `provenance_control` | `typed_confidence` |
| `reachability` | `typed_confidence` |
| `skill_orchestration` | `ldt` |
| `strategic_tick_control` | `typed_confidence` |
| `treaty_channel_control` | `typed_confidence` |

## Held-out skill behavior

| Skill | Route | Accuracy | Utility | Unsafe Rate | Cost |
|---|---|---:|---:|---:|---:|
| `coalition_planning` | `typed_membrane` | 0.812 | 0.919 | 0.000 | 1.40 |
| `control_profile_selection` | `confidence_arbitration` | 0.875 | 0.933 | 0.000 | 1.23 |
| `moral_optimization` | `trm` | 1.000 | 1.000 | 0.000 | 1.10 |
| `opponent_modeling` | `trm` | 0.979 | 0.992 | 0.000 | 1.10 |
| `provenance_control` | `typed_confidence` | 1.000 | 1.000 | 0.000 | 1.83 |
| `reachability` | `typed_confidence` | 0.979 | 0.992 | 0.000 | 1.90 |
| `skill_orchestration` | `ldt` | 1.000 | 1.000 | 0.000 | 2.10 |
| `strategic_tick_control` | `typed_confidence` | 0.833 | 0.937 | 0.000 | 1.71 |
| `treaty_channel_control` | `typed_confidence` | 0.938 | 0.976 | 0.000 | 1.54 |

## Interpretation

- Hard gating tests the failure mode where conditional evidence is promoted into elimination authority.
- Typed confidence composes provenance checks with calibration rather than choosing one arbitration mechanism globally.
- Counterfactual beam tests whether a small proposal budget can recover utility after certification.
- Skill routing tests whether architecture choice itself is an optimizable game-playing skill.
