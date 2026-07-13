# Checkpoint: Conductor-HRM Architecture

## Completed

- Invented Conductor-HRM, formally described as Hierarchical Typed Module Flow.
- Defined slow and fast HRM managerial states over a graph of typed TRM/LDT modules.
- Defined module receipts, control tokens, branch/join behavior, scoped certificates, and flow budgets.
- Preserved the existing invariant that only module membranes can mutate module-owned public lattice state.
- Added training losses, failure modes, a TheySing worked flow, and a staged benchmark plan.

## Commands run

```powershell
git diff --check
```

This turn is architecture formalization only. No benchmark was claimed or fabricated.

## Files changed

- `docs/hrm_conductor_architecture.md`
- `docs/hybrid_architecture.md`
- `reports/checkpoint_hrm_conductor.md`

## Tests

Documentation-only change. Existing code was not modified.

## Decisions made

- HRM management is flow authority, not state-mutation authority.
- Module-private recurrent states are not shared directly with the manager or other modules.
- Cross-module transfers cannot strengthen provenance.
- Destination modules must revalidate receipts before issuing environment-sound certificates.
- Speculative branches use isolated lattice views and commit only through typed joins.
- Structural safety constraints remain outside the learned HRM objective.
- VPD remains outside this architecture exercise.

## Open questions

- Should the first learned conductor use the published HRM architecture directly or a dependency-free recurrent
  analogue to validate the flow contract first?
- Which receipt summary is sufficient for scheduling without leaking too much module-private state?
- How should delayed game reward be divided between flow decisions and worker-module proposals?

## Recommended next step

Implement the dependency-free `ModuleCard`, `ModuleReceipt`, `ControlToken`, `FlowLedger`, and deterministic
conductor state machine, then benchmark oracle flow recovery before training an HRM scheduler.
