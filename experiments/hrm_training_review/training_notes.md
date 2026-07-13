# HRM Training Review Experiment Notes

## Scope

This is a deterministic review-contract experiment over synthetic sealed receipts. It ports scalar
lineage and holonomy authorization math from the local RSITopology program into Conductor-HRM flow.
It does not launch training, inspect live weights, or authorize model promotion outside the saved cases.

## Frozen policy

- signed operator-error budget: `0.5`
- simultaneous false-authorization delta: `0.05`
- minimum conservative edge retention: `0.8`
- maximum realized KL for bundle allocation: `0.05`
- minimum grouped replicates: `3`
- hard RAM cap: `2048 MB`
- hard CPU cap: `50.0%`
- hard I/O cap: `50.0 MB/s`

## Review order

1. Seal target-blind spectral geometry and bind model, dataset, and operator hashes.
2. Audit checkpoint/context lineage, loops, orientation, uncertainty, and matched noise nulls.
3. Reveal grouped held-out utility and matched-rank Haar controls only after the geometry seal.
4. Join held-out damage and capped-run resource receipts.
5. Route to authorize, local sectioning, additional audit, or rejection.

## Interpretation

- `holonomy_clean` is required for global signed coordinate control.
- `lineage_certified` is sufficient for invariant bundle-energy allocation under its KL budget.
- Ordinary model promotion is not blocked solely by non-identifiable signed internal coordinates.
- Positive geometry cannot compensate for failed utility, damage, or resource gates.
- The information coefficient is descriptive and is not used as the primary gate.
- Aborts are retained as structured evidence but cannot be promoted as completed training runs.

## Training status

No model training or weight mutation was run in this experiment.
