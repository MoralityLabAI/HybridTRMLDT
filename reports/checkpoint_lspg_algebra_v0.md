# LSPG-v0 Algebra Checkpoint

## Completed

- Added split parameter-visit and retained-state-edge gradient policy.
- Preserved v0 masks through an explicit compatibility mapping.
- Added normalization topology and residual double-normalization validation.
- Added context-aware decoder cost accounting.
- Added the six-rung 5M-400M iso-shape ladder and width solver.
- Added canonical algebra, model, and run hashes.
- Added bounded mutation records with rankability checks.

## Tests

`20 passed` across the new algebra, scale, mutation, identity, and legacy
word-count tests.

## Decisions

- Edge indices identify transitions from visit `i` to visit `i+1`.
- A legacy last-k visit mask retains only graph edges internal to the selected
  tail. New experiments should specify both masks explicitly.
- Residual scale is excluded from directional kappa and applied once in the
  stability functional. A double application is representable only as a
  `known_invalid` control and cannot be ranked.
- The initial scale family fixes two physical blocks, vocabulary size 2,048,
  head dimension 32, and solves only width. Every rung is within the declared
  3% unique-parameter tolerance.

## Open Work

Implement censor-native receipt ingestion and the three-channel proposal
planner before adding the neural executor.
