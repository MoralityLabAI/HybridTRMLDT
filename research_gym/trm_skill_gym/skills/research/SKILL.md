# Skill: Research TRM: rank proposed experiments by expected information gain

## Trigger

Several experiments are proposed over a finite hypothesis set with a prior; each has a binary outcome likelihood per hypothesis and a cost.

## Objective

Choose the experiment maximising expected information gain (bits) minus cost_weight * cost.

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: p_outcome1_given_h0
- col 1: p_outcome1_given_h1
- col 2: p_outcome1_given_h2
- col 3: p_outcome1_given_h3
- col 4: p_outcome1_given_h4
- col 5: p_outcome1_given_h5
- col 6: cost
- col 7: prior_h0
- col 8: prior_h1
- col 9: prior_h2
- col 10: prior_h3
- col 11: prior_h4
- col 12: prior_h5
- col 13: cost_weight
- Relation: unused
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [one experiment index]}

## Procedure

1. For each experiment compute the prior-weighted variance of its outcome likelihoods.
2. Score = 8 * variance - cost_weight * cost; choose the maximum.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `research`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.
