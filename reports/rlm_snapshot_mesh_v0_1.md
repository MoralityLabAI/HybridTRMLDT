# Snapshot-First RLM-Wrapped Controller Mesh v0.1

## Construction

The snapshot-first RLM receives only opaque candidate references, module confidence and provenance, exact
admissibility, disagreement, and deterministic policy previews. It receives no task text, transcript, action
token, utility, or optimal action. The RLM selects a policy hint; the host verifies the immutable snapshot hash,
replays the complete typed mesh, and commits only a certificate-bound exact-admissible action.

The matched task-first atomic arm retains the v0 raw-task interface. The fresh panel contains
8 tasks selected before provider outcomes by policy-action divergence and a fixed hash salt.

## Results

| Architecture | Utility | Accuracy | Unsafe | Contract | Fallback | Non-consensus | Action change | Errors | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trained_trm_ldt_fixed` | 0.6721 | 0.8750 | 0 | - | 0.1250 | 0.0000 | 0.2500 | 0 | 0 |
| `mesh_fixed_consensus` | 0.5871 | 0.6250 | 0 | - | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `mesh_forced_no_tool_fallback` | 0.5871 | 0.6250 | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `rlm_mesh_snapshot_text` | 0.5871 | 0.6250 | 0 | 0.5000 | 0.5000 | 0.3750 | 0.0000 | 0 | 69,539 |
| `ldt_only` | 0.4911 | 0.1250 | 0 | - | 0.0000 | 0.0000 | 0.5000 | 0 | 0 |
| `rlm_mesh_task_first_atomic` | 0.4482 | 0.5000 | 0 | 0.1250 | 0.8750 | 0.0000 | 0.0000 | 2 | 40,468 |
| `rlm_mesh_snapshot_atomic` | 0.4458 | 0.5000 | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 2 | 59,717 |

Typed unsafe executions: `0`. Cell errors remain zero-utility outcomes:
`4`. The forced no-tool control reproduced fixed-mesh actions and utility:
`true`.

## Interpretation

The snapshot-first construction preserved containment but did not produce useful routing. Among completed API
cells, every executed action matched `mesh_fixed_consensus`. The snapshot-text arm completed 8/8 cells and
returned a parseable policy in 4/8; three parseable choices were `trained_first`, but all three occurred where
that policy resolved to the same action as consensus. The arm therefore reproduced fixed-mesh utility exactly.

The two atomic arms are availability failures rather than evidence of worse selected actions. Task-first atomic
completed 6/8 cells, passed its tool contract once, and had two `TokenLimitExceededError` outcomes. Snapshot-first
atomic completed 6/8 cells, passed 0/8 contracts, had two `ErrorThresholdExceededError` outcomes, made no valid
tool call, and made one malformed call whose policy argument was a structured object rather than a registered
hint. Every completed cell in both arms fell back to or selected the fixed consensus action. Their negative
utility deltas arise solely because the registered rule scores errors as zero utility.

The selected panel nevertheless had routing headroom. In a posthoc policy-oracle calculation, `trained_first`
was non-inferior to consensus on all eight tasks and improved both provenance tasks, raising equal-family macro
utility from `0.5871` to `0.6721` (`+0.0850`). This is not a registered efficacy endpoint; it demonstrates that
the routing null was not forced by an action-identical panel. The wrapper spent 169,724 provider tokens without
realizing that available contrast.

## Matched Contrasts

- `rlm_mesh_snapshot_atomic` vs `rlm_mesh_task_first_atomic`: -0.0024; latest_rule_action=+0.3455, multi_hop_reachability=-0.3550, provenance_gate=+0.0000, storyworld_control=+0.0000.
- `rlm_mesh_snapshot_atomic` vs `mesh_fixed_consensus`: -0.1412; latest_rule_action=+0.0000, multi_hop_reachability=-0.3550, provenance_gate=-0.2100, storyworld_control=+0.0000.
- `rlm_mesh_snapshot_text` vs `mesh_fixed_consensus`: +0.0000; latest_rule_action=+0.0000, multi_hop_reachability=+0.0000, provenance_gate=+0.0000, storyworld_control=+0.0000.
- `rlm_mesh_snapshot_atomic` vs `rlm_mesh_snapshot_text`: -0.1412; latest_rule_action=+0.0000, multi_hop_reachability=-0.3550, provenance_gate=-0.2100, storyworld_control=+0.0000.

## Boundary

This fresh eight-task synthetic pilot tests whether an official RLM can review an opaque typed-mesh policy snapshot more reliably than the prior raw-task wrapper while preserving certificate-bound containment. It uses the gym ControlTRM rather than official TinyRecursiveModels and does not establish general orchestrator safety, neural alignment, task-level model superiority, or a sheaf-spectral claim.

## Integrity

- Result SHA-256: `7582e91957bb6bc0ee6a8052d35a6bad0c52bd9e53669549466617ec1aff206e`
- Records SHA-256: `1f9af5ea4b571bd721f8188234c69ab47f50fadd50d59cb92633e5245f1b6a6c`
- Trajectory manifest SHA-256: `b993348e1cfb0001bc5664ef490ddc0f5c25280e5edf78eee34cb7b7ee0347ba`
- Config SHA-256: `918f44e80a94f28576fd4d0dc2f2b0720c72541b27914062a2bd6a436249def4`
- Snapshot contract SHA-256: `8a275c38498bee4cae385d2d8f1f4271310e76543093c97ec4fdb41ece1aa75c`
