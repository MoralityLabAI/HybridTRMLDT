# Snapshot-First RLM Mesh v0.1 Artifact Audit

## Verdict

The sealed artifacts reproduce. The run supports a containment result and a routing null: the opaque snapshot
did not leak registered action tokens, no typed unsafe action executed, and the forced fallback matched the fixed
mesh in all eight tasks. No completed RLM wrapper cell changed the fixed-mesh action, despite a posthoc policy
ceiling of `+0.0850` equal-family macro utility on the sealed panel.

It does not support a claim that snapshot-first atomic routing is more reliable than task-first atomic routing.
The registered contract rates were 0/8 and 1/8, respectively. Snapshot-text improved interface completion but
not action routing: 4/8 responses parsed, three selected `trained_first`, and all three resolved to the consensus
action.

## Recomputed Integrity

- Config SHA-256: `918f44e80a94f28576fd4d0dc2f2b0720c72541b27914062a2bd6a436249def4`.
- Result SHA-256: `7582e91957bb6bc0ee6a8052d35a6bad0c52bd9e53669549466617ec1aff206e`.
- Records SHA-256: `1f9af5ea4b571bd721f8188234c69ab47f50fadd50d59cb92633e5245f1b6a6c`.
- Trajectory SHA-256: `b993348e1cfb0001bc5664ef490ddc0f5c25280e5edf78eee34cb7b7ee0347ba`.
- Snapshot contract SHA-256: `8a275c38498bee4cae385d2d8f1f4271310e76543093c97ec4fdb41ece1aa75c`.
- Record count: 56, comprising eight tasks by seven architectures.
- Registration status: `registered_before_provider_outcomes`; the fresh 152-task corpus and proposal table were
  sealed separately in commit `786f337`.

## Failure Decomposition

| Wrapper | Completed | Contract | Cell errors | Non-consensus hints | Action changes | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Task-first atomic | 6/8 | 1/8 | 2 token-limit | 0/8 | 0/8 | 40,468 |
| Snapshot-first atomic | 6/8 | 0/8 | 2 error-threshold | 0/8 | 0/8 | 59,717 |
| Snapshot-first text | 8/8 | 4/8 | 0 | 3/8 | 0/8 | 69,539 |

Snapshot-first atomic produced five completed responses without a tool call and one completed response with an
invalid call. The invalid call echoed the correct snapshot reference but supplied a structured policy object
where the tool required one literal policy hint. This is a tool-contract failure before certificate selection,
not a certificate or typed-executor failure.

The task-first and snapshot-first atomic macro-utility scores (`0.4482` and `0.4458`) include their registered
zero-utility error cells. On every commonly completed cell, both executed the fixed consensus action. The
snapshot-text score (`0.5871`) equals fixed consensus exactly because it completed all cells and changed no
action.

## Routing Headroom

The three deterministic mesh policies were action-divergent on six of eight selected tasks. Their equal-family
macro utilities were:

- `consensus`: `0.587089`.
- `trained_first`: `0.672089`.
- `ldt_conservative`: `0.491097`.
- Per-task policy oracle: `0.672089`.

The entire `+0.0850` oracle gain came from the two provenance tasks. Snapshot-text failed its output contract on
both, while its three `trained_first` selections occurred in latest-rule and storyworld cells where the selected
action aliased consensus. The result is therefore a selective-interface failure on a panel with measurable
routing value, not a no-op benchmark.

## Scope And Successor

This is an eight-task synthetic diagnostic using the gym ControlTRM, not official TinyRecursiveModels. It does
not establish general safety or model superiority. A successor should preserve this run unchanged and alter two
interface details only: place pure snapshot JSON in the official RLM context with all instructions in the
prologue, and bind the immutable snapshot inside a single-argument `mesh_replay(policy_hint)` tool rather than
requiring the model to echo the snapshot hash. Those changes target the observed JSON-extraction and structured-
argument failures; they require a new registration and fresh panel.
