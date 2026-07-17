# Measured-Stalk Controller Bridge

Protocol: `controller_mesh_measured_stalk_bridge_v1`.

## Source

- model / site: `Qwen/Qwen3.5-0.8B-Base` / `model.layers.23`
- measured rank: `1`
- restriction edges: `10`
- source receipt: `61bf8342937b6e83d221aad86874af5f40028aac5ed7925ab1885e3047cb18d0`

## Result

- combined / controller / categorical rho: `+0.7523 / +0.5688 / +0.6402`
- combined minus categorical rho: `+0.1121` with 95% interval `[-0.0855, +0.3121]`
- combined minus controller rho: `+0.1834` with 95% interval `[+0.0075, +0.3755]`
- matched N0 p: `0.9380`
- top-k uplift delta versus best baseline: `-0.0112`
- registered bridge gate passed: `False`

## Gate Checks

| Check | Passed |
|---|---:|
| `combined_rho` | `True` |
| `rho_delta_vs_categorical` | `True` |
| `rho_delta_vs_controller` | `True` |
| `bootstrap_lower_vs_categorical` | `False` |
| `bootstrap_lower_vs_controller` | `True` |
| `matched_null` | `False` |
| `top_k_uplift_delta` | `False` |

## Boundary

Fresh-seed proxy-task test of whether one fixed, target-blind measured Qwen restriction graph improves controller ranking beyond controller-only spectra and categorical labels. A pass would support this external-product feature construction only; failure must be retained without tuning and cannot be repaired by changing source rank, site, edge weights, or heat times.
