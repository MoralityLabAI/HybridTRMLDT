# RSITopology-Aware HRM Training Review

| Scenario | Mechanism | Request | Identity | Route | Bound |
|---|---|---|---|---|---:|
| clean_signed | signed_global_control | model_promotion | holonomy_clean | authorize | 0.095 |
| high_holonomy | signed_global_control | signed_control | lineage_certified | section | 0.757 |
| orientation_reversal | signed_global_control | signed_control | lineage_certified | section | 2.000 |
| unmeasured_loop | signed_global_control | signed_control | lineage_certified | audit | 2.000 |
| long_high_retention_path | signed_global_control | signed_control | lineage_certified | section | 0.530 |
| utility_failure | signed_global_control | model_promotion | holonomy_clean | reject | 0.095 |
| ordinary_high_holonomy | ordinary_optimizer | model_promotion | lineage_certified | authorize | 0.757 |
| bundle_high_holonomy | bundle_allocation | model_promotion | lineage_certified | authorize | 0.757 |
| resource_incomplete | signed_global_control | model_promotion | holonomy_clean | audit | 0.095 |
| aborted_run | signed_global_control | model_promotion | holonomy_clean | reject | 0.095 |

The clean signed candidate is authorized. High measured holonomy and orientation reversal are routed
to sectioning rather than global signed control. An unmeasured loop routes to audit. Strong local edge
retention does not rescue a long path whose conservative bound exceeds the error budget.

Ordinary optimizer promotion remains a behavioral decision even when signed internal coordinates are not
globally identifiable. Bundle allocation needs lineage but not globally flat signed coordinates. Geometry
alone never authorizes promotion: held-out utility, matched controls, damage, provenance, and resource
receipts remain separate gates.

Claim boundary: Deterministic contract exercise over synthetic sealed receipts; no neural model was trained, promoted, or edited by this benchmark.
