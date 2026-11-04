# Dual-LDT-06 RESULTS: trajectory-valued Jinn, strong symbolic comparator

Status: completed registered synthetic experiment. Six NEW value students were trained using the frozen recipe. The previous paper's persona heads are not these models. The report consists of 64 new configurations from the same finite world family, with four authored preference vectors per configuration. It supplies no independent validation of human values, natural-language action interpretation, or real shutdown authority.

## Run and scope

Source commit: 3c1734574af530ba85eedddf728e0500b598ab04. The preexisting implementation was 49f45624b6099164a3bdc5ed9e611751a7e75c93. The launch commit changed CI execution and receipts only; it did not change the registered data generator, optimizer, budgets, weights, model architecture or endpoints.

[Completed CPU training, development, report, tests and Lean](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41868351947). Nine Python contract tests passed. The Lean kernel checked the conditional uniform-error regret inequality and finite-path recovery invariant. Six trained weights were frozen before the report run.

Training: 128 worlds, 56,068 feasible world/persona/plan examples; seeds 17,23,41; 800 Adam updates per model. Each model has 8,865 parameters. The one-pass and four-pass students share architecture size and input information; computation differs with recurrence depth. Development uses 32 worlds. Report uses 64 worlds, four persona vectors, 343 three-step plans and eight search methods, yielding 256 world-persona cases and 10,240 recorded method/budget checkpoints. Candidate budgets are 1,4,16,64,343 and include the initially checked idle fallback and every infeasible proposal.

## Registered primary comparison

Mean oracle regret on the report set, in expected trajectory-reward units; smaller is better. Ranges span the three seeds, without treating correlated plans or personas as independent samples.

| Plan evaluations | Fixed enumeration | Symbolic upper-bound order | One-pass value student | Four-pass value student |
|---|---:|---:|---:|---:|
| 4 | 46.8211 | 31.4983 | 35.6913-38.7409 | 37.5296-39.2833 |
| 16 | 38.4910 | 13.4150 | 24.3860-26.9784 | 24.7324-27.8999 |
| 64 | 18.6072 | 3.2976 | 7.2991-9.5754 | 7.1045-9.1919 |
| 343 | 0 | 0 | 0 | 0 |

The learned ordering improves upon fixed lexicographic enumeration. The sound symbolic count relaxation has lower mean regret than every learned model at each nontrivial registered budget. At full enumeration all methods attain the same finite-grammar oracle, whose mean return is 46.82110595703125.

At budget 16, symbolic ordering attains the actual optimum in 168/256 cases; one-pass students attain it in 58-85/256 and four-pass students in 56-75/256. At budget 64 the counts are 229/256, 145-171/256 and 147-164/256 respectively. Actual optimum attainment uses the hindsight oracle for scoring. Mathematical exact-optimality certificates at budget 64 occur in 144/256 symbolic cases, 48-67/256 one-pass cases and 44-60/256 four-pass cases. Attainment and certification are distinct endpoints.

No consistent four-pass advantage was found. At budget 64 its range overlaps the one-pass range, with seed-dependent differences; at budget 16 each matched four-pass seed has greater mean regret than its one-pass counterpart. This is evidence about the frozen small-model recipe and finite environment, not a universal limitation on recurrence.

## Runtime and extrapolation

One measured hosted CPU session, Python 3.12.16, PyTorch 2.10.0+cpu, two Torch threads, AMD EPYC 9V45 host. At budget 16 mean end-to-end search times are 1.9758 ms for symbolic ordering, 5.3565-5.3947 ms for one-pass students and 5.5625-5.6090 ms for four-pass students. At budget 64 these are 4.1665 ms, 7.5209-7.5706 ms and 7.7598-7.8295 ms. These measurements include input construction, learned scoring, sorting, candidate evaluation, the rational checker, and bound computation; they exclude training and model loading. They are one-run observations, not hardware-general performance guarantees.

The registered secondary audit runs each frozen four-pass model at eight passes. Its feasible-plan mean absolute error worsens in every seed: 7.7154 -> 8.7987; 8.4827 -> 12.1049; 7.4069 -> 11.4733. Four-pass primary MAE ranges 7.4069-8.4827, compared with one-pass MAE 7.1746-8.5366. Additional untrained iterations do not yield a monotonic value improvement in this evaluation.

## Safety, provenance and independent replay

The host checks every positive-probability execution branch and every current authority update independently of the neural ranking. No upper-bound or acceptance assertion failed in the completed run. The integer-mass evaluator and recursive Fraction checker agree under the same synthetic transition semantics.

A separate post-run standard-library Fraction implementation replayed all 21,952 report world/plan combinations from the exported manifest. It confirmed 7,875 feasible world/plans, all 31,500 feasible world/persona/plan upper-bound inequalities, and all 10,240 recorded selected-plan utilities and regret inequalities. It also checked 2,048 anytime sequences for monotone incumbents and shrinking upper bounds, summary reconciliation, split disjointness, six frozen model file hashes and raw report hashes. Violations: zero. This is an independent calculation of the specified toy model; it cannot validate the model against external reality.

The recorded neural scores never authorize an action. Infeasible plans still consume query budget; at budget 16, learned models average 12.01-12.49 infeasible proposals and the symbolic order 11.18. Improving a learned value approximation alone does not establish efficient discovery of feasible programs.

## Post-hoc exact outcome compression

After the registered results were known, the independent audit formed each certified plan's vector of expected completion features and resource cost. Equal vectors were deduplicated and dominated vectors were removed. Across report worlds the mean sizes were 123.046875 certified plans, 12.46875 unique outcome/cost vectors and 4.96875 nondominated vectors (range 2-13).

For a fixed certified domain, if one plan has every expected feature at least as high and cost no higher than another, it weakly improves utility for every nonnegative linear persona vector. Every discarded record has a retained dominating witness. The audit tested exact optimum preservation on 81 nonnegative weight vectors per world; the dominance argument applies to the whole nonnegative cone. This compression is post hoc, retains no claim of minimal convex coverage, and first requires evaluation of the input plans. Rebuild the cover after a policy/state change removes a retained witness.

This suggests a precise next learning target: predict the shared outcome vector and prioritize missing useful certified programs, with persona weights applied externally. Such a design can reuse a physical prediction across many preferences. It is related to successor-feature and multi-objective planning methods; novelty for those foundations is not claimed. No training or report rerun was performed to tune this next idea.

## Receipts

Full GitHub artifact: `dual-ldt-06-reproduction`, ID `13343636356`, run `41868351947`.

- ZIP SHA-256: `7e07701d8f7541ae5ec6d9162bd87ba7a92f52b044d3aba3be31dade2e62b077`
- Report raw SHA-256: `943a2618257bba58299070d992aced747365892aba7418d7f2a48e787ea9a5b9`
- Report summary SHA-256: `67aed0c8c3876d44af57b4e0816d66d2d621249e3e1d58aba9f118be50fd1f2f`
- Weight-freeze SHA-256: `7b2058775d42522129c56642deee03f6d5d7ee8f3318facc9362bb4e2d6ade9b`

The artifact contains all six saved models, the weight freeze, train/dev/report manifests, full search records, evaluation summaries, exact source commit, package versions and host details. The study is confined to the registered open-loop horizon-three grammar; adaptive policies, stronger symbolic baselines and different learning recipes remain unevaluated. The Jinn or Beast submission manuscript was not edited.
