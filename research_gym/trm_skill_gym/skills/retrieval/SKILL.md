# Skill: Retrieval TRM: identify relevant documents, symbols and proof dependencies

## Trigger

A query item needs every document/symbol/lemma it transitively depends on.

## Objective

Return exactly the transitive dependency set of the query (not the query itself).

## Required inputs

- Candidate table: up to 16 candidates, each with typed features:
- col 0: is_query
- col 1: lexical_overlap_with_query
- col 10: out_degree_over_4
- col 11: in_degree_over_4
- Relation: rel[i][j]=1: item i directly depends on item j
- Global observation vector and the verifier feedback channel (verified, ok, diag0, diag1).

## Action schema

Return one JSON object: {"selected": [item indices]}

## Procedure

1. Select the query's direct dependencies.
2. Add items with high lexical overlap with the query.
3. After verification, add items the verifier marks as missing frontier dependencies and drop items it marks as orphans.

## Verifier contract

The host verifier (`research_gym.trm_skill_gym.tasks.verify`, skill `retrieval`) returns a receipt verdict and two
structured diagnostics per candidate. The skill never produces its own receipt.

## Termination

Stop after the visit budget, or earlier when the module's confidence reaches the halting threshold or its
predicted value of another recurrence is not positive.
