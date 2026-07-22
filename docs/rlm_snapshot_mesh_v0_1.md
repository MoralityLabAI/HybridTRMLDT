# Snapshot-First RLM-Wrapped Controller Mesh v0.1 Protocol

## Question

Can an official RLM review a fixed typed-mesh snapshot and select a useful routing policy without receiving the
raw task or acquiring action authority?

## Architecture

```text
task -> fixed mesh snapshot -> RLM policy review -> atomic mesh replay -> typed executor
                |                         |                    |
                | opaque candidate refs   | policy hint only   | certificate-bound action
                + confidence/provenance/admissibility/disagreement
```

The snapshot-first RLM sees opaque candidate references, module confidence and provenance, exact
admissibility, disagreement, and the candidate reference each registered policy would select. It does not see
the task text, transcript, action tokens, utilities, or optimal action. The host accepts one of `consensus`,
`trained_first`, or `ldt_conservative`, reconstructs and verifies the snapshot, replays the complete mesh, and
commits only a certificate-bound exact-admissible action.

## Matched Arms

- `rlm_mesh_task_first_atomic` preserves the v0 raw-task atomic wrapper on the fresh panel.
- `rlm_mesh_snapshot_atomic` requires one `mesh_replay(snapshot_ref, policy_hint)` call.
- `rlm_mesh_snapshot_text` tests whether custom-tool compliance, rather than policy review, limits the wrapper.
- `mesh_fixed_consensus` is the no-RLM policy baseline.
- `mesh_forced_no_tool_fallback` must remain action- and utility-identical to fixed consensus.
- `ldt_only` and `trained_trm_ldt_fixed` retain local reference points.

## Fresh Panel

The task generator seed, task identities, ControlTRM proposal table, and materialization receipt were sealed in
commit `786f337` before provider outcomes. Two tasks per family are selected from the fresh evaluation split.
Selection prioritizes tasks where the three registered mesh policies produce more than one action and then uses
a fixed hash salt. Selection never reads task utility.

## Interpretation

The co-primary endpoints are contract pass rate relative to task-first atomic, realized action change relative
to fixed consensus, and equal-family macro utility delta relative to fixed consensus. API errors remain
zero-utility cells. A policy hint that differs from consensus but resolves to the same action is not counted as
realized routing. Typed unsafe execution is a hard stop.

This is a small synthetic architecture diagnostic. It cannot establish general safety, alignment, model
superiority, or a sheaf-spectral result.

## Run

The config and generated registration must be committed before any provider outcome. Validation and execution
use the official RLM virtual environment and the bounded resource wrapper.
