# MCP-03-RU01: First CPU platformer retrieval pilot

**Status: saturated development benchmark, no demonstrated retrieval lift.**

- Run: https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/44917038640
- Receipt archive: workflow artifact `mcp03-readaware-cpu-receipts` (ID `14712480131`)
- 12 train / 4 dev / 8 generated test roots, seed `117`, 24 calls, 1024 packet-token *UTF-8-estimate* budget.
- Pinned torch CPU 2.5.1; read-aware probe: **617,877,504 bytes (589.3 MiB peak RSS)**, **0.1643 s/step**, 320,578 total parameters (57,922 non-embedding).
- Training: binary 332 decision states, 42 updates; read-aware 160 decision states, 20 updates. These are **different label selection distributions**.
- Replay: 16 checked episodes, 0 packet mismatches.
- Independent generated root comparison (mean file / symbol recall):

| Controller | File recall | Symbol recall | Symbol reads | Mean packet tokens |
| --- | ---: | ---: | ---: | ---: |
| Identifier rules | 1.000 | 1.000 | 16.50 | 968.75 |
| BM25 lexical | 1.000 | 1.000 | 16.50 | 964.50 |
| Looped p4 d64 binary | 1.000 | 1.000 | 16.125 | 966.875 |
| Looped p4 d64 read-aware | 1.000 | 1.000 | 17.50 | 966.50 |

Paired symbol recall delta = **0.000**, seed-412 bootstrap 95% interval **[0, 0]**. Promote gate **FAIL**. All controllers read the target symbol in all eight roots. The repository generator supplied a directly named symbol in each one-target issue and small easily searchable codebases, so the task could not discriminate the trained policies.

Next: retain all receipts; train on harder root-disjoint cases containing multi-file related defects and distractor symbols. Keep 24 MCP calls and the 1024-token packet budget fixed, and report oracle reachability plus action-type ablations. Train on the new roots; reserve a separately generated seed for confirmation. Later independent native SWE grading remains outstanding.
