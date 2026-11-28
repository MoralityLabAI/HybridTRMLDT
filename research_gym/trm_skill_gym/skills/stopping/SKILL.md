# Skill: Stopping TRM: decide whether more reasoning, retrieval or verification is worth its cost

## Trigger

An agent holds an answer with a confidence, a remaining step budget and known action costs/gains.

## Objective

Choose the first action of the value-maximising policy over the remaining budget (stop pays confidence; reasoning gains decay; retrieval helps once; verification ends the episode on success and resets confidence on failure).

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: is_stop
- col 1: is_reason
- col 2: is_retrieve
- col 3: is_verify
- col 4: cost
- col 5: gain
- col 6: confidence
- col 7: budget_over_6
- col 8: reason_decay
- col 9: confidence_after_rejection
- col 10: already_retrieved
- Relation: unused
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [one action index: 0 stop, 1 reason, 2 retrieve, 3 verify]}

## Procedure

1. For each action estimate the value of taking it once and then stopping.
2. Choose the action with the highest one-step value.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `stopping`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.
