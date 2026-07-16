# AIRIS Calibration-Outcome Induction

## Objective

Learn AIRIS-style sequence rules from frozen calibration outcomes, then test intact rules on the existing 667
held-out sequencer episodes. This separates behavioral rule quality from the AIRIS/DAS transport-integrity
benchmark. The learner is empirical and context-level; it is not a causal AIRIS learner or a neural model.

## Frozen Protocol

The protocol in `configs/airis_induction_v1.json` fixes the calibration receipt, utility-winner target, tie order,
minimum support, primary confidence threshold, sensitivity thresholds, and endpoint before held-out evaluation.
The induction function accepts calibration episodes only and rejects a calibration hash mismatch.

For context `c`, calibration episode `e`, sequence set `Q`, and measured utility `u_e(q)`, define

```math
y_e=\underset{q\in Q}{\arg\max}\;u_e(q),
\qquad
q_A(c)=\underset{q\in Q}{\arg\max}
\sum_{e\in C_c}\mathbf 1[y_e=q].
```

Ties use the frozen order `typed_propose_certify`, `deduction_only`, `proposal_only`. The induced rule stores

```math
s_c=\sum_{e\in C_c}\mathbf 1[y_e=q_A(c)],
\qquad
\gamma_c=s_c/|C_c|,
\qquad
K_c=\{e\in C_c:y_e\ne q_A(c)\}.
```

`s_c` is support, `gamma_c` is confidence, and `K_c` is the retained set of concrete calibration
counterexamples. Topology plans contribute protocol, route, orientation, and risk features for authorization;
they do not supply the induced sequence or its confidence.

## Guarded Execution

For held-out context `c`, confidence threshold `theta`, integrity predicate `I`, and topology authorization gate
`G`, execution is

```math
q(c)=
\begin{cases}
q_A(c), & \gamma_c\ge\theta\land s_c\ge 2\land I(c)\land G(c),\\
q_T(c), & \text{otherwise},
\end{cases}
```

where `q_T` is the frozen `control_math` fallback. This study exposes an important degenerate case: all 28
induced context-majority proposals equal `q_T`. Confidence can demote uncertain rules, but fallback does not
change the action.

## Held-Out Result

The learner uses 455 calibration episodes and emits 28 rules. On 667 held-out episodes, raw rule precision
against the episode utility winner is `0.940030`.

| Family | Episodes | Rule precision | Wrong rules |
|---|---:|---:|---:|
| Sudoku | 96 | 1.000000 | 0 |
| ARC-1 | 168 | 1.000000 | 0 |
| ARC-2 | 160 | 1.000000 | 0 |
| Routing | 115 | 0.965217 | 4 |
| Secret ending | 64 | 0.656250 | 22 |
| Moral optimization | 64 | 0.781250 | 14 |

At the preregistered threshold `0.5`, every rule is accepted and all 40 intact-but-wrong episode decisions pass
the confidence gate. At threshold `1.0`, acceptance falls to `0.760` and wrong-rule acceptance falls to zero.
Macro utility remains `0.900085` at every threshold because raw proposals and fallback are identical. There are
no proposal changes, harmful changes, or fallback saves relative to `control_math`.

The result is a negative control result, not a robustness gain. Context confidence identifies storyworld
heterogeneity but does not locate which state needs another skill. A useful next comparison must add
state-conditioned story reachability or routing features and a behaviorally distinct typed fallback.

## Reproduction

```powershell
python -m research_gym.scripts.bench_airis_induction
python scripts/smoke_airis_das_bridge.py --rules data/airis_das/induced_rules.json --episodes data/benchmarks/sequencer_control_episodes.jsonl --bridge-results data/benchmarks/airis_induction_results.json --out data/airis_das/induced_live_service_smoke.json
python -m pytest tests/test_airis_induction_bench.py -q
```

## Artifacts

- `data/airis_das/induced_rules.json`: induced rules, support, confidence, and calibration counterexamples.
- `data/airis_das/induction_calibration_episodes.jsonl`: frozen learner input.
- `data/airis_das/induced_live_service_smoke.json`: actual HTTP service parity and fail-closed receipt.
- `data/benchmarks/airis_induction_results.json`: aggregate metrics and artifact hashes.
- `data/benchmarks/airis_induction_records.jsonl`: per-episode proposals and threshold decisions.
- `experiments/airis_induction/`: mirrored rules, input, output, and training notes.
