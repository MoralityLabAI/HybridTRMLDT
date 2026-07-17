# Independent Whole-Family Spectral Replication

Protocol: `controller_mesh_sheaf_forward_replication_v2`.

## Design

- genomes: `128`
- discovery / heldout: `64 / 64`
- discovery fallback families: `['identical', 'ldt']`
- heldout fallback families: `['safe_ldt', 'correction_infused']`
- spectral and categorical predictions were sealed together before heldout outcomes

## Co-Primary Result

- spectral rho: `+0.568`
- categorical rho: `+0.598`
- paired rho delta: `-0.030`
- paired bootstrap 95% interval: `[-0.317, +0.246]`
- matched N0 p: `0.0078`
- spectral top-16 uplift: `+0.0928`
- categorical top-16 uplift: `+0.1162`
- top-k uplift delta: `-0.0234`
- registered incremental gate passed: `False`

## Gate Checks

| Check | Passed |
|---|---:|
| `spectral_rho` | `True` |
| `rho_delta` | `False` |
| `bootstrap_lower` | `False` |
| `matched_null` | `True` |
| `top_k_uplift_delta` | `False` |

## Held-Out Families

| Fallback family | Genomes | Objective | Utility | Unsafe |
|---|---:|---:|---:|---:|
| `safe_ldt` | 32 | 0.9011 | 0.8319 | 0.0265 |
| `correction_infused` | 32 | 1.0440 | 0.9325 | 0.0265 |

## Boundary

Independent-seed whole-fallback-family replication in deterministic proxy tasks. Passing would support incremental spectral prediction within this generator; failing any gate is a publishable negative and must not be tuned away.
