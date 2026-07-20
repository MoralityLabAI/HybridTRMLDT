# Receipt Hash Portability Audit

## Problem

Eight inherited tests failed on Windows because Git checkout translation changed LF text files to CRLF while receipt verification hashed raw working-tree bytes. The failures affected attestation source verification, legacy LSA and LSPG receipt replay, and architecture-discovery proposal replay. They did not indicate changed JSON content or failed scientific controls.

The sealed corpus also contains one legacy exception: `gaming_vs_improvement_results.json` was sealed with its raw CRLF digest, while most text receipts record LF bytes. Rewriting that file or its receipt would break provenance.

## Repair

`research_gym.integrity` now defines two explicit rules:

1. New textual artifact hashes canonicalize CRLF and lone CR to LF before SHA-256.
2. Replay accepts either the canonical digest or an already-sealed raw digest for recognized text artifacts. Binary artifacts remain raw-only.

The attestation source verifier and architecture-discovery planner use this shared implementation. Direct receipt tests use the same verifier rather than duplicating raw `read_bytes()` hashes.

## Regression Coverage

The new tests establish that:

- LF and CRLF forms of the same text have the same canonical digest.
- a pre-existing raw CRLF digest remains admissible for a sealed text receipt.
- binary payloads do not receive newline normalization.

No sealed config, record, result, receipt, package, or recorded digest was modified.

## Verification

- affected non-attestation suites: `17 passed`
- attestation and RSITopology conformance suite: `5 passed`
- full repository suite: `306 passed in 152.88s`
