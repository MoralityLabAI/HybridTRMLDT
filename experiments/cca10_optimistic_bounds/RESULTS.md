# CCA-10 Results: sound bound, null search improvement

**Verdict: null exploratory result.** The derived action-count upper bound passed the full finite-domain soundness test, yet never certified optimality earlier than the CCA-09 global bound.

[Successful CI run](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41814668216). The test checks the analytical envelope against the exact CCA-05 dynamic-programming value on all 1,050 states and horizons 0 through 5 (6,300 comparisons). The full retrospective replay spans all 1,750 D_bench inputs and all six original X/R checkpoint tapes.

Artifact: GitHub Actions `cca10-full` ID `13313997090`. The SHA-256 of `metrics.json` is `8979f7f734b40f26bf5e8cf65628e4991666dc8b7b75a96a639a8df9fb5176a3`.

## Matched outcome

In moral optimization, all 585 viable states per checkpoint require five action evaluations for **exact** epsilon-zero certification under both upper-bound methods. At K=1,2,3,4, there are zero exact certificates for both the TRM-ranked and fixed-priority strategies. K=5 proves all 585. The bound leaves every executed action, utility, abstention rate and evaluator call count identical to CCA-09. Secret ending remains unchanged because its optimistic bound is the global value 1.

Across all 120 aggregate cells (six checkpoints, two scenarios, two proposal orders, five evaluation budgets), the metrics for action utility, abstention, exact certificates and mean action evaluations match the original CCA-09 replay.

## Interpretation and next constraint

The relaxation preserves individual upper bounds for trust, evidence and scene progression, and a lower bound for heat, indexed by the counts of self-actions. These extrema may come from incompatible chronological trajectories. The bound is sound but too loose to prove that the high-utility TRM incumbent is optimal earlier, and its additional arithmetic brings no observed reduction in evaluator calls. Runtime overhead was not measured.

The next candidate should be a **relational** reachability abstraction that retains the tradeoff between evidence collection, heat removal, trust and the scene clock. Prove that its reachable-state overapproximation contains every actual successor before measuring utility-gap tightness. Evaluate generalization on new generated environments rather than tuning against this same inspected benchmark.

The result remains retrospective and assumes the CCA-05 environment and modeled rival. No claim about adversarial natural-language semantics follows.
