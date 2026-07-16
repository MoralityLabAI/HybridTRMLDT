# Hybrid Architecture Space Transcript Package

This package contains three unmodified readable Silico transcripts and an independent analysis note prepared
for Fable review.

Read in this order:

1. `ANALYSIS.md`
2. `chats/01_hybrid_design_space_thread.md`
3. `chats/02_membrane_internalization_experiment.md`
4. `chats/03_membrane_internalization_review.md`
5. `source_manifest.json`
6. `REVIEW_RESPONSE.md` (post-package sidecar)
7. `BADCF7B_ARTIFACT_AUDIT.md` (post-commit benchmark audit sidecar)

The transcripts are readable exports containing visible user and assistant messages only. Hidden reasoning,
tool calls, tool outputs, and authentication material are excluded by the recovery process. The source manifest
records both readable-export and raw-session hashes; the ZIP contains only readable exports.

`REVIEW_RESPONSE.md` (this directory, not in the ZIP) records the post-package review chain: Fable's
synthesis, the external GPT Pro audit, and the reconciled position, including the corrected
false-rejection arithmetic and the four-condition decomposition. The ZIP is intentionally left
byte-identical to preserve its recorded SHA-256
(`c4c220fea7cf800a1813696b36078a8755233d80642d4e12568b10a74cf96ef1`).

`BADCF7B_ARTIFACT_AUDIT.md` independently reconciles the committed gaming benchmark and records the sharper
oversight-leverage result plus its majority-baseline power caveat. It also remains outside the frozen ZIP.
