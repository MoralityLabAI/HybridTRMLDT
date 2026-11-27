# Skill: Deduction TRM: guide candidate-lattice reduction

## Trigger

A candidate lattice must be reduced: a candidate is eliminated iff some surviving candidate dominates it.

## Objective

Return exactly the surviving candidates (the kernel of the dominance relation).

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: prior_score
- col 1: n_dominators_over_3
- col 2: n_dominated_over_3
- col 3: undominated
- Relation: rel[i][j]=1: candidate j dominates candidate i
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [surviving candidate indices]}

## Procedure

1. Keep every undominated candidate; eliminate the rest.
2. After verification, drop survivors that have a surviving dominator and restore eliminated candidates that have no surviving dominator.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `deduction`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.
