# RLM Uncertainty Pre-Gate v0.1 Protocol

## Question

Can a deterministic ambiguity gate preserve the useful decisions of an RLM evidence controller while reducing
provider calls enough to improve all-in utility on a fresh panel?

## Frozen Gate

The rule was selected from the sealed v0 development receipts: invoke the RLM only when the proxy TRM, trained
ControlTRM, and exact LDT name three distinct opaque top-candidate references. No confidence threshold was fitted.
The gate sees no task text, action token, utility, truth, or acquired evidence. Its evaluation invocation task IDs
are frozen in registration before provider outcomes.

## Utility

For cost regime `c`,

```text
U_all_in(c) = U_raw - C_evidence
              - lambda_token(c) * total_tokens / 1000
              - lambda_latency(c) * provider_wall_seconds
```

The primary coefficients are `0.001` utility per 1,000 provider tokens and `0.0005` utility per provider wall
second. Low and high regimes are registered sensitivity analyses. These coefficients are benchmark-local exchange
rates, not dollar prices.

## Arms

- Fixed no-query mesh.
- Gated always-exact local acquisition.
- Gated deterministic-VOI local acquisition.
- Ungated official RLM on every task.
- Gated official RLM, with no provider call when the gate is closed.
- Gated forced-failure identity control.

Where both API arms run, their execution order rotates by task index. Provider or parser failure executes fixed
consensus and charges completed evidence and measured provider resources. Every task is atomically checkpointed.

## Scope

This is a fresh synthetic panel using an existing ControlTRM checkpoint and the official RLM runtime. It tests the
economics of a controller role, not model training, general alignment, or a claim about actual API prices.
