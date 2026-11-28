# Skill: Repair TRM: repair a rejected candidate using verifier diagnostics

## Trigger

A candidate solution (a 2-colouring of a constraint graph) was rejected by the verifier, which reports per-node conflict counts.

## Objective

Return the minimum set of nodes whose colour must flip so that every edge joins different colours.

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: candidate_colour
- col 1: conflict_fraction
- col 2: degree_over_6
- col 3: has_conflict
- Relation: rel[i][j]=1: nodes i and j are adjacent and must receive different colours
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [node indices to flip]}

## Procedure

1. Flip every node where more than half of its incident edges conflict.
2. Re-verify; flip any node the verifier still reports with a majority of conflicting edges.
3. Prefer the smaller of a flip set and its complement.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `repair`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.

## Trained module

- Implementation: `recurrent_proxy_gru` (12119 parameters), config `37af7332d15f` in
  `module_config.json`, weights in `weights.pt`.
- Protocol: 3 visits, 1 in-loop verifier call(s) plus the final receipt; halting at
  confidence >= 0.85 or predicted value of another recurrence <= 0.
- Acceptance basis: `secondary_rule_beats_existing_skill_not_preregistered_promotion`.
- Frozen evaluation (fresh roots, mean over seeds [0, 1]): dev exact 0.941, withheld-family exact
  0.733; existing skill A: dev 0.268, withheld
  0.264; strongest of A/B/C on dev 0.941.
