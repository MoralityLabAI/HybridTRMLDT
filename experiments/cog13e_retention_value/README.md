# COG-13E — marginal value of retained proof states

Stacked on [COG-13D #38](https://github.com/MoralityLabAI/HybridTRMLDT/pull/38). Native COG-13C/13D receipts were not available when this experiment was designed. This is an executed **synthetic prefix-trace development pilot** for the prospective LDT retention head, with arbitrary generated work units and declared payload bytes, not measured Lean CPU time or memory. It does not use actual theorem proofs.

## Value target

For cache C and request q, let S_C(q) be the cumulative work of the longest matching cached prefix (zero for the root). A candidate state s with payload b_s has a static-set horizon-12 marginal value

V_C(s,t) = sum_{j=t+1}^{t+12} [S_C(q_j)-S_{C\{s}}(q_j)] / b_s.

The target is log(1+100 V). It accounts for redundant ancestors: if a retained descendant covers every matching request, the ancestor adds no marginal saved work for those requests. The label assumes the current candidate set remains fixed over the horizon. It is not the optimal dynamic cache value, a causal counterfactual of actual future eviction trajectories, or an additive value when multiple states are removed together. The future trace is available only to training label construction and the explicitly labeled oracle12 diagnostic.

## Protocol

144 distinct generated roots: 64 train, 16 calibration, 32 ordinary, 32 deeper, each with 60 requests. Four public trace regimes, six branches sharing a prefix tree, heterogeneity of per-command cost and declared snapshot bytes. Deeper branches are longer. Different seeds and exact generated fingerprints enforce distinct roots within the registry, not semantic independence across all possible environments. Seed 2026112607; no held-out fitting or calibration tuning. The experiment generator is public, so none of these are sealed tests.

Persistent cache budgets: 128, 256, 512 declared bytes; the working candidate may temporarily exist outside that persistent limit. Only successful open-state requests can publish snapshots; closed and rejected requests cannot. The failed final command is charged. All policies receive the same trace. Restore the deepest retained exact prefix, execute the suffix, then admit the new state when it fits individually; evict until the persistent budget holds. Every score sees the same candidate set for a particular eviction step.

Train a 13-coordinate ridge scorer on 51,984 candidate-state labels from 384 training rollouts (LRU and work-density at each capacity), root-balanced loss, regularization 0.15. Input: depth, declared bytes, observed cumulative and marginal work, recency, hit counts, observed prefix-request counts, recent reuse, coverage by descendants and lifetime. Root identity, split, regime and future trace are excluded. The model learns a log density; its score is clamped at zero. Within-root shuffled labels are a training control, not a different evaluation policy. Training collection charges 23,040 synthetic requests and 715,001 modeled work units; label oracle computation and regression cost are offline and not measured CPU costs.

Controls: root replay, LRU, LFU, depth times hits, marginal work per byte, marginal work times (1 + recent prefix matches) per byte, trained retention head, shuffled-label head, and oracle12. The oracle uses future requests and current-set marginal labels; it is a diagnostic, not a globally optimal bound.

## Executed results

2,000 evaluation rollouts, 120,000 request receipts audited. The structural checker independently reconstructs prefix paths, execution costs, admission, cache contents, byte limits and terminal counters. It checks that the recorded scores select the stated eviction; it does not independently reconstruct the score calculation or every feature operation. Four negative mutations rejected (work, byte count, admission, missing request). Exact overlap marginal invariant checked. Two full executions reproduce the same frozen summary.

Lower work is better. Ordinary (32 roots):

| Persistent modeled bytes | LRU | Recent density | Learned | Shuffled | Oracle12 |
|---:|---:|---:|---:|---:|---:|
| 128 | 75,625 | 63,431 | 63,546 | 70,443 | 56,222 |
| 256 | 61,597 | 44,884 | 45,608 | 49,717 | 39,349 |
| 512 | 44,281 | 28,673 | 28,653 | 32,022 | 27,020 |

Root replay: 109,568 work units.

Deeper (32 roots):

| Persistent modeled bytes | LRU | Recent density | Learned | Shuffled | Oracle12 |
|---:|---:|---:|---:|---:|---:|
| 128 | 128,325 | 114,940 | 119,452 | 116,562 | 101,557 |
| 256 | 109,594 | 87,390 | 93,754 | 91,528 | 72,682 |
| 512 | 84,184 | 60,927 | 66,428 | 66,431 | 47,799 |

Root replay: 163,926 work units. Learned saves 6.91%, 14.45%, 21.09% versus LRU but costs 3.93%, 7.28%, 9.03% more than recent density. At 256 bytes, learned saves 4,196 command executions vs recent density's 4,073, yet costs more work; its 1,737 cache hits vs 1,540 do not imply better retention. Learned is worse than the shuffled-label head at 128/256 and essentially tied at 512 deeper bytes. The ordinary set supports a narrow gain versus shuffled labels, not a useful superiority over the simple recent heuristic. This is a negative transfer result.

Post-readout descriptive analysis: 4,000 root-stratified bootstrap resamples per contrast, fixed deterministic seeds. At deeper 256 bytes the learned-vs-recent saved-work contrast is -7.28% (95% descriptive percentile interval [-11.04,-3.55] percentage points); 14 wins and 18 losses. No confirmatory claim or multiplicity-adjusted gate. The learned-vs-shuffle interval crosses zero at every deeper capacity. See paired_results.json for every contrast and hypothetical overhead sensitivity. Charging one work unit per ranked candidate plus two more per learned feature evaluation further favors the recent heuristic at all deeper capacities. These coefficients are illustrative, not hardware timings. Real scoring costs are not measured.

## Reproduce and next gate

Node 22, standard library only: `node experiments/cog13e_retention_value/run.mjs /tmp/cog13e-new`. The runner checks both published numerical receipts, generates full root registry and compressed trace/candidate-label exports, verifies train-only custody, and hashes all artifacts. No DataGoblin repository mutation or neural LDT integration yet.

COG-13F should use measured per-tactic incremental CPU cost and native retained-state memory accounting from a passed COG-13D run, then compare the recent-density analytic prior against a residual learner trained directly on eviction ranking or realized delayed regret. Freeze the heuristic and model before new native-root testing. Marginal labels must condition on competing cached descendants and retention horizon. Preserve the current negative result; do not retune on this deeper set. The next gate is a measured improvement in certified theorems per CPU second at matched physical memory limits, including feature and model overhead. The present work provides an executable label definition and accounting simulator only.
