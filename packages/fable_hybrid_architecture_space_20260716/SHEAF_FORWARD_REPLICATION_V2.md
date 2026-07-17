# Controller-Mesh Sheaf Forward Replication v2

This sidecar records the preregistered independent replication of the v1 forward spectral result. It does not
modify the frozen Fable ZIP.

The study uses a new task seed and 128 controller genomes. Complete fallback families are held out: 64
`identical`/`ldt` genomes fit the models, while 64 `safe_ldt`/`correction_infused` genomes are evaluated. Spectral
and categorical predictions were sealed together before held-out outcomes.

The registered incremental gate fails. Spectral prediction is nonrandom (`rho=+0.568`, matched N0 `p=0.0078`),
but the preregistered categorical baseline is stronger (`rho=+0.598`). Spectral-minus-categorical rho is `-0.030`
with paired 95% interval `[-0.317, +0.246]`. Spectral top-16 uplift is `+0.0928`, below the categorical baseline's
`+0.1162`.

This is evidence for descriptive spectral signal, not unique information or a validated architecture-acquisition
function. The failed result is retained without feature tuning. Correction infusion transfers to the new seed,
but it remains an oracle-backed synthetic controller arm.

Full construction, math, boundaries, and receipts are in
`docs/controller_mesh_sheaf_forward_replication.md` and
`reports/checkpoint_controller_mesh_sheaf_forward_replication_v2.md`. Independent external audit is pending.
