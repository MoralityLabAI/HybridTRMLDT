# MCP-05: ECAN-inspired attention ledger for MCP-04 working sets

**Status:** experimental CPU-only policy. No Bonsai 27B, Inky Mini, native Goose, or SWE-bench model gains are claimed. This is a child of the MCP-04 typed-working-set branch (PR #18). Raw and historical goblin arms remain untouched.

## Evidence custody and memory bands

- **Pinned state (VLTI analogue):** existing user constraints, chronological edit calls and anchors; now all historical observed-empty read atoms from completed, non-error tool returns. Each pinned atom carries its original read-result handle, tool-argument digest, supplied upstream snapshot if any, and local observed edit epoch. A no-results output is a historical tool observation, never proof of subtree irrelevance. Overflow of pinned state causes WorkingSetError.
- **Off-prompt memory (LTI analogue):** unchanged content-addressed full-result and edit-argument handles in the in-process per-tag store. Repeated identical reads share an observation lineage; different queries, result bodies, and edit epochs remain distinct.
- **Prompt-resident reads (STI analogue):** optional read handles rank by task-term overlap, age, saturated reuse and edited-path co-use. They are packed within an estimated envelope. The upstream applied-template plus tokenizer measurement remains the final admission decision. This heuristic is neither native Hyperon PathMap nor a trained ECAN.

The existing verifier membrane holds sole authority for eligibility, pruning, edits and task completion. This compactor does not call certify_and_apply_verified. Tool results are untrusted.

## Tests

CPU regression file: MCP/test_mcp05_attention.py. It covers same-path distinct queries, repeat deduplication, changed-result lineages, supplied snapshots, observed local edit epochs, negative pinning, error-vs-empty separation, and deterministic ranking. The parent test_mcp04.py covers call/result closure, anchors, exact token preflight and provider parity.

    python -m pytest -q MCP/test_mcp05_attention.py MCP/test_mcp04.py MCP/test_hubrun.py

Isolated unit tests passed locally before pushing integration. Full integrated suite, live server parity and native-model replay still need verification.

## Paired evaluation

Replay the same tasks against raw, MCP-04 and MCP-05 with a fixed model checkpoint, prompt window, server template, tool schema, sampling configuration, split and seeds. Retain case-level receipts with total failures.

Measure solved cases, edit-site correctness, finalization, exact upstream prompt tokens, provider usage parity, repeated equivalent reads, distinct queries, result-handle recoveries, stale-edit spans, token overflow and fail-closed rate. Verify that historical empty observations are not reused as pruning certificates. Compare against the held-out MCP-03 repositories without incorporating test data into the policy.

**Acceptance:** match MCP-04 solved tasks and finalization, improve repeated reads and stale edit-site outcomes, maintain zero unauthorized pruning, and preserve exact upstream parity. Read-count reduction alone is insufficient.

## Follow-ons

After held-out evaluation, compare a native Hyperon frame producer against research_gym/adapters/metta_adapter.py and its FakeMettaAdapter. Train AIRIS-style causal suggestions and read/expand utility heads on eligible training traces only; route any proposed exclusions through ResearchMembrane and the existing LDT checker. This branch does not implement those follow-ons.
