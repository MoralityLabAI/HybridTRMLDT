# Skill: Curation TRM: identify inconsistent or low-value training examples

## Trigger

A small training set may contain label noise and duplicates.

## Objective

Flag every example whose label disagrees with its cluster's majority label, and every exact duplicate of an earlier example.

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: x0
- col 1: x1
- col 2: x2
- col 3: x3
- col 4: x4
- col 5: x5
- col 6: label
- col 7: position
- col 8: identical_earlier_twin
- Relation: rel[i][j]=similarity weight between examples i and j (1.0 = identical)
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [flagged example indices]}

## Procedure

1. Flag an example when its label disagrees with the similarity-weighted vote of its neighbours.
2. Flag an example that has an identical earlier twin.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `curation`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.

## Trained module

- Implementation: `native_trm_update` (11683 parameters), config `8e7d449b5e14` in
  `module_config.json`, weights in `weights.pt`.
- Protocol: 3 visits, 1 in-loop verifier call(s) plus the final receipt; halting at
  confidence >= 0.85 or predicted value of another recurrence <= 0.
- Acceptance basis: `secondary_rule_beats_existing_skill_not_preregistered_promotion`.
- Frozen evaluation (fresh roots, mean over seeds [0, 1]): dev exact 0.747, withheld-family exact
  0.644; existing skill A: dev 0.406, withheld
  0.283; strongest of A/B/C on dev 0.759.
