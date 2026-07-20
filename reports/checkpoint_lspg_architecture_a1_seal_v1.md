# Checkpoint: LSPG Architecture A1 Seal v1

## Manifest

- Stage: `A1`.
- Status: `sealed_before_stage_outcomes`.
- Manifest hash: `612c750c8302af876c26f1baff0e1902799157a3a0c45388ddc54bd49c4790aa`.
- Construction commit: `e8caa044a3eacffbc05f6a8657d8935e0ae2e149`.
- Selected profile: `minimum`.
- Profile selection hash: `28324027699340a077508471b813f074e8798f844b944360b4221b10276bb4a4`.
- Proposal manifest SHA-256: `8cdc824a9a39ac7bf6800fd5bd4b84bfdb4cc78843e28d0602c79f757ea660c8`.

## Cells

The manifest contains 18 cells: 12 first-batch candidates and six deduplicated
periodic controls. Every cell runs at S0 with seed 401 and 524,288 token-visit
exposures. L=6 cells require 43 optimizer steps; L=8 cells require 32. Locked
evaluation is disabled for every cell.

## Selection Boundary

A1 is a single-seed integrity and ranking screen. It may retain at most six
candidates. Positive delta is not required at this stage; three-seed positive
gating begins at A2. No A1 task outcome existed when this manifest was sealed.

## Next Step

Commit this manifest, then execute it sequentially through the hard-cap stage
runner. Seal all 18 result, prediction, resource, and stage receipts before
computing the A1 transition.
