# Controller-Mesh Sheaf Retrodiction

Protocol: `controller_mesh_sheaf_retrodiction_v1`.

## Construction

The primary unit is an arm-seed policy instance. The 60 instances retain the 31 behavioral equivalence classes from the source benchmark as provenance groups; topology-incompatible aliases are not averaged into one sheaf.

Each stalk is a utility-label-blind episode-function subspace for a proposer, evidence channel, gate, fallback, or executor. Restriction maps embed those subspaces into the shared episode response space. Utility and oracle labels are revealed only after spectra are sealed.

## Retrodiction

| Spectral feature | Outcome | Spearman rho | Matched-null p |
|---|---|---:|---:|
| spectral_gap | utility_delta_vs_proposal | -0.723 | 0.0078 |
| spectral_gap | action_change_rate | -0.631 | 0.0078 |
| low_band_rank | utility_delta_vs_proposal | +0.678 | 0.0078 |
| low_band_rank | action_change_rate | +0.593 | 0.0078 |
| slow_mode_rank | utility_delta_vs_proposal | +0.312 | 0.0078 |
| slow_mode_rank | action_change_rate | +0.224 | 0.0078 |

## Kernel Migration

| Seed | Capture r0 | Capture final | Kernel migration | Random-null p |
|---:|---:|---:|---:|---:|
| 17 | 0.0055 | 0.0021 | +0.0034 | 0.4191 |
| 29 | 0.0008 | 0.0038 | -0.0030 | 0.5731 |
| 43 | 0.0019 | 0.0289 | -0.0269 | 0.8304 |

The strict seed-29 prediction passed: `False`.

Post-hoc frozen-gate margin diagnostic:

| Seed | Pass r0 | Pass final | Mean signed-margin shift |
|---:|---:|---:|---:|
| 17 | 0.500 | 1.000 | +0.8957 |
| 29 | 0.422 | 1.000 | +2.2232 |
| 43 | 0.500 | 1.000 | +0.3101 |

## Matched Null

N0 independently permutes each module's episode transport. It preserves the authority graph, stalk rank, singular spectrum, and constant section while destroying cross-module episode compatibility.

## Boundary

Retrodictive association in one deterministic toy-storyworld benchmark. Spectra are observational predictors, not causal guarantees of oversight, adaptive independence, compositional stability, neural safety, or deployment performance.
