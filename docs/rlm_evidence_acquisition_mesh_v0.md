# RLM Adaptive Evidence-Acquisition Mesh v0 Protocol

## Question

Can an official RLM improve a typed control mesh by choosing which evidence to acquire, without receiving task
text or action authority?

## Control Flow

```text
opaque snapshot -> RLM query -> typed receipt -> optional RLM query -> deterministic arbiter -> exact executor
```

The RLM receives pure canonical JSON containing opaque candidate references, module confidence and provenance,
the universal query catalog, and any receipt from the first query. It may return one registered query ID or
`STOP`. It cannot return an action, candidate, threshold, schedule, or executable plan.

The four queries are an independent soft probe (`0.01`), identity-bound proposal attestation (`0.02`), exact
mechanics (`0.04`), and calibrated counterfactual value bands (`0.04`). Net utility subtracts every executed
query cost. Duplicate or malformed queries are not executed or charged.

## Failure Semantics

Any provider error, timeout, or malformed response ends acquisition and executes fixed consensus while charging
only queries already completed. An explicit `STOP` is a valid decision and executes using receipts already
acquired. With no exact-mechanics receipt, acquired evidence cannot override fixed consensus. The exact executor
verifies every changed action and falls back to the fixed mesh on any inconsistency. API availability therefore
cannot erase baseline raw task utility.

## Calibration And Registration

Seed `294117` was opened during construction and is explicitly retired. Seed `394117` is the fresh corpus.
Only its calibration split may be used for interface or construction gates before registration. Evaluation
contains six tasks per family and is run only after the calibration receipt and registration are committed.

Calibration must show at least `+0.03` mean sequence-oracle net gain, ten positive tasks, three distinct
two-query oracle sequences with no sequence above 70% of two-query choices, and 75% RLM interface compliance.

## Arms

- Fixed no-query mesh.
- Always-exact one-query mesh.
- Fixed exact-then-rollout mesh.
- Seeded random query-count-matched two-query mesh; actual query costs remain explicit.
- Deterministic snapshot-only value-of-information heuristic.
- Official-RLM adaptive two-query acquisition.
- Forced-RLM-failure no-op control.

## Scope

This is a synthetic mixed-control experiment using the existing gym ControlTRM checkpoint and exact host
executor. It does not train a model, use VPD, test official TinyRecursiveModels, or establish general safety.
