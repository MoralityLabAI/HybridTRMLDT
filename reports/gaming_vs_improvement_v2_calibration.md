# Above-Majority Calibration

Config SHA-256: `7feb9d72997f844ba6736bf8cfe3deee36b63cb1b63f2b5367ccb3ab40c4469a`
Final evaluation oracle labels accessed: `False`

## Power Confirmation

| Seed | Accuracy | Majority | Delta | Ceiling margin |
|---:|---:|---:|---:|---:|
| 101 | 0.750 | 0.711 | +0.039 | 0.250 |
| 103 | 0.766 | 0.695 | +0.070 | 0.234 |
| 107 | 0.758 | 0.719 | +0.039 | 0.242 |
| 109 | 0.742 | 0.703 | +0.039 | 0.258 |
| 113 | 0.742 | 0.711 | +0.031 | 0.258 |
| 127 | 0.805 | 0.711 | +0.094 | 0.195 |
| 131 | 0.750 | 0.680 | +0.070 | 0.250 |
| 137 | 0.797 | 0.734 | +0.062 | 0.203 |
| 139 | 0.719 | 0.695 | +0.023 | 0.281 |

## Adaptation Capacity

| Seed | Evidence | Action change | Oracle accuracy delta |
|---:|---|---:|---:|
| 101 | exact_mechanics | 0.164 | -0.055 |
| 101 | exposed_frozen_probe | 0.336 | -0.102 |
| 103 | exact_mechanics | 0.133 | -0.023 |
| 103 | exposed_frozen_probe | 0.312 | -0.094 |
| 107 | exact_mechanics | 0.172 | -0.039 |
| 107 | exposed_frozen_probe | 0.164 | -0.039 |

## Registration Note Correction

The frozen adaptation_calibration_note lower bound 0.102 came from the wider multi-budget sweep. The canonical selected-six-step confirmation minimum is 0.132812; the selection rule still passes.

Selection was based on development power and action movement, not on a preferred improvement or evasion outcome. Buckets 8-9 remained outcome-blind until the frozen benchmark run.
