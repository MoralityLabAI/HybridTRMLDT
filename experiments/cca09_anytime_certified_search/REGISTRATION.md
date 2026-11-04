# CCA-09: certified anytime TRM search

Status: exploratory successor to CCA-05. The CCA-05 checkpoint tapes and outcomes are already published; this is a retrospective method comparison rather than a newly sealed held-out result.

## Protocol frozen for CCA-09

- Domain: CCA-05's 1,750 nonterminal D_bench decisions, both scenarios, horizon 6.
- Source: all six existing CCA-05 checkpoint output tapes, pass-four full-mask logits. Record SHA-256 of each tape at runtime.
- Comparator: fixed symbolic priority order (M.MORAL_PRIORITY). Both arms use identical verified action evaluators and proof checkers, with candidate order as the only difference.
- Budgets: 1,2,3,4,5 distinct action evaluations. An exhaustive fallback completes absent or duplicated proposer lists.
- Global per-action upper bound: U<=1, from the environment's [0,1] terminal values and max aggregation; each bound is bound to the current context and checked by environment-owned code.
- Every negative verdict is independently rechecked by M.admissible. Every positive verdict carries a CCA-05 ActionCertificate checked by execute_certified. Action utility is checked against M.utility.
- Metrics: optimality certificates by budget, mean achieved utility with abstentions scored zero on viable starts, abstentions, regret against hindsight oracle, distinct evaluator checks, certified suboptimality gaps, scenario/checkpoint groups.
- Negative controls: false/stale upper bounds, false negative labels, wrong action binding, utility exceeding an issued bound, duplicates and unsupported action nominations.
- Inference status: retrospective descriptive. The modeled rival remains a conditional assumption, the TRM logits are pre-recorded, and the number of evaluator calls excludes model-forward time. Do not use this experiment as an independent confirmatory test.

## Predictions declared before running new CCA-09 replay

H1: checked incumbent utility is nondecreasing with evaluation budget, and the certified upper envelope is nonincreasing.
H2: no epsilon_optimal claim has actual regret above epsilon.
H3: the learned ranking may improve finite-budget achieved certified utility against fixed priority. Empirical result may fail or be null and should be reported.
H4: whenever an evaluated feasible action reaches the global upper bound 1, exact optimality is certified without inspecting all actions.
