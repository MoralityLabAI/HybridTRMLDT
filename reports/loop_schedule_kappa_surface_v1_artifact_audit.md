# Loop Schedule Kappa Surface v1: Artifact Audit

## Audit Result

Pass for the labeled post-partial Recovery 2 artifact chain.

- Combined records: 81 records, 81 unique record IDs.
- Combined records SHA-256 rehash: pass (`69c1b1ded04426159d0da666ca34a2eaa22ff53ea3a13b0843775ec512dc9409`).
- Recovery result SHA-256 rehash: pass (`cf6835e56d38a8255c0c66e650e6871e63feb49e95988c604c8e4ec49e714597`).
- Final receipt SHA-256: `294d06ede48cb7a2b6ef48bcd6db75afd3028305edbe20a5cb9503d7fde76cd8`.
- Figure SHA-256: `c41153825310a9a5633b6d4413520fce62ae07c68bb17a8c9630720a116c4cd8`.
- Finalizer resource status: completed.
- Finalizer cleanup: passed; no lingering owned process or GPU app.
- Outcome-producing aggregate execution: 1964.109 seconds under the 3600-second cap.

## Regression Gates

The focused recovery suite passes 44 of 44 tests. It locks down:

- the post-partial config and source hashes;
- exclusion of the incomplete original `R=64` cell;
- zero admission from the failed untied I/O-abort attempt;
- checkpoint-before-pacing ordering;
- the paced peak-I/O result (`42.331 < 50 MB/s`);
- exact terminal replication and the untied terminal equivalence control;
- the registered conflict: separable global winner, exposure-2048 local onset, gradient co-localization, and final `form_unresolved` classification; and
- byte-deterministic SVG rendering.

The full repository suite reports 295 passed and 8 failed. The eight failures are inherited Windows line-ending rehash mismatches in older frozen attestation, LSA v0, architecture-discovery, and LSPG artifacts. None is in the new kappa-surface code or artifacts, and no older sealed file was rewritten during this release.

## Claim Boundary

This audit establishes artifact integrity and reproducibility of the registered decision logic. It does not upgrade the post-partial recovery to an as-preregistered completion of the original protocol, restore the missing untied-kappa trajectory control, or establish causality between gradient stress and the transient alignment trough.
