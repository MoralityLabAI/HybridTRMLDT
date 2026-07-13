# Trade Offs between TRM/LDT Hybrids

## Abstract

We study a small hybrid reasoning architecture that combines a Tiny Recursive Model style proposal mechanism with an explicit Lattice Deduction Transformer style state. The central design is a typed membrane: the latent side proposes refinements, while the explicit lattice side certifies, rejects, or soft-stores them according to provenance. The goal is not to claim that a lattice guarantees correct abstention in all settings. The narrower claim is that an explicit lattice makes abstention, bottom, monotonicity, and provenance inspectable in domains where the relevant mechanics are checkable.

We evaluate this design on five lightweight benchmarks: Sudoku, ARC-1, ARC-2, environment-pointer routing, and a coupled storyworld playing task. Across the saved runs, hybrid matches the best aggregate score in every task family. It improves proposal efficiency in Sudoku and aggregate ARC-2, repairs unsafe TRM actions in storyworld play, and avoids hard-eliminating routes when LDT evidence is only lexical and therefore not environment-sound. The main negative result is that hybrid is not automatically superior: in routing it only ties TRM when LDT candidate sets are treated as soft evidence, and in two ARC-2 instances its heuristic proposal order uses more proposals than raw TRM.

We also specify a Conductor-HRM layer for review of newly trained checkpoints. It combines target-blind spectral bundles, lineage, holonomy, and a conservative signed-control bound with independent held-out utility, damage, provenance, and resource gates. The saved receipt-contract matrix tests review flow only; it is not neural training performance.

We then apply the same control mathematics directly to fixed hybrid skill sequencers in a registered 667-episode replay. Relative to global signed sequencing, topology-controlled localization improves family-macro utility by `0.0069` (95% context-clustered bootstrap interval `[0.0049,0.0092]`, Holm-adjusted `p=0.0006`) with unchanged accuracy and lower cost. Its `0.0003` advantage over lineage-only control is not significant (`p=0.5023`). This measures sequencer control over deterministic skills, not neural weight infusion or official leaderboard performance.

## 1. Motivation

The motivating question is practical:

```text
Can persistent latent recurrence improve explicit lattice deduction
without destroying the interpretability and soundness benefits
of an explicit abstract state?
```

We separate two roles:

- `TRM`: private recurrent or heuristic workspace that proposes actions, refinements, routes, or candidate rules.
- `LDT`: explicit public lattice state that supports monotone refinement, bottom detection, serialization, and typed provenance.

The hybrid architecture is useful only if these roles remain separated. If latent proposals can directly mutate durable state, the system loses the main benefit of explicit deduction. If the lattice is the only mechanism, the system struggles when local propagation is insufficient and search or heuristic proposal is needed.

## 2. Architecture

### 2.1 Two States

At step `t`, the hybrid keeps:

```text
a_t: explicit abstract state
h_t: private latent or heuristic state
```

`a_t` is public and checkable. In this implementation it is often a candidate-set lattice, but the same interface also covers intervals, symbolic constraints, reachable regions, possible rules, and possible environment IDs.

`h_t` is private workspace. It can remember correlations, infer regimes, rank proposals, and track search context. It does not directly mutate `a_t`.

### 2.2 Hybrid Loop

```mermaid
flowchart LR
  X[Task state / prompt] --> A[Explicit state a_t]
  X --> H[TRM latent state h_t]
  H --> P[Proposal p_t]
  A --> M[Typed membrane]
  P --> M
  M -->|environment-sound + monotone| A2[Hard update a_{t+1}]
  M -->|model/experience-sound| S[Soft store / telemetry]
  M -->|unknown/non-monotone/bottom| R[Reject / abstain]
  A2 --> Y[Action / prediction / next state]
  S --> Y
  R --> Y
```

The membrane is the core object. It enforces the distinction between proposal and deduction.

## 3. Lattice Formulation

For the candidate-state implementation, let slots be indexed by `i in I`. Each slot has a finite universe `U_i`. A lattice state is:

```math
a = \{ C_i \subseteq U_i \}_{i \in I}
```

The top state is:

```math
\top = \{ U_i \}_{i \in I}
```

The bottom condition is:

```math
\bot(a) \iff \exists i \in I: C_i = \emptyset
```

Meet is pointwise intersection:

```math
(a \wedge b)_i = C_i^a \cap C_i^b
```

A proposal `p_t` contains a proposed state `\hat{a}_{t+1}`. It is monotone only if:

```math
\forall i \in I,\quad \hat{C}_{i,t+1} \subseteq C_{i,t}
```

The implementation rejects non-monotone proposals before applying meet:

```math
\operatorname{reject}(p_t) \quad \text{if} \quad \exists i: \hat{C}_{i,t+1} \nsubseteq C_{i,t}
```

This matters because otherwise a widening proposal could be hidden by a subsequent meet and appear harmless.

## 4. Typed Soundness

Each proposal carries provenance:

```text
environment-sound: derived from trusted environment mechanics
model-sound: derived from a model of another agent or learned predictor
experience-sound: derived from replay or discovered successes
unknown: no usable provenance
```

The default membrane policy is:

```math
\operatorname{hard}(p) =
\begin{cases}
1 & \text{if provenance}(p)=\text{environment-sound}\\
0 & \text{otherwise}
\end{cases}
```

The durable transition is:

```math
a_{t+1} =
\begin{cases}
a_t \wedge \hat{a}_{t+1} & \text{if hard}(p_t) \land \hat{a}_{t+1}\sqsubseteq a_t \land \neg\bot(a_t \wedge \hat{a}_{t+1})\\
a_t & \text{otherwise}
\end{cases}
```

Bottom is allowed only in explicit abstain/conflict mode:

```math
\bot(a_t \wedge \hat{a}_{t+1}) \Rightarrow \text{accept only if mode}=\text{ABSTAIN}
```

Soft proposals are not discarded. Model-sound and experience-sound proposals can be soft-stored as telemetry:

```math
s_{t+1} = s_t \cup \{p_t\}
```

but this does not change the durable lattice state.

## 5. Control Dynamics and Alternative Architectures

The typed membrane is one point in a larger design space. We compare it with two implementable alternatives to separate the value of hybridization from the risks introduced by the control policy.

Let `q_theta` denote the TRM proposal or scoring mechanism. In the typed membrane:

```math
p_t = q_\theta(h_t,a_t,x_t), \qquad p_t=(\hat a_{t+1},\tau_t,m_t)
```

Durable state changes only through the membrane:

```math
a_{t+1} =
\begin{cases}
a_t \wedge \hat a_{t+1} & \tau_t=\text{environment-sound},\ \hat a_{t+1}\sqsubseteq a_t,\ \neg\bot(a_t\wedge\hat a_{t+1})\\
a_t & \text{otherwise}
\end{cases}
```

Alternative 1 is a hard-gated cascade:

```math
C_t = \operatorname{LDT}(x_t), \qquad
y_t = \arg\max_{y\in C_t} q_\theta(y\mid x_t)
```

This is attractive when `C_t` is environment-sound, but unsafe when `C_t` is lexical or learned.

Alternative 2 is confidence arbitration. Let `s_1(x_t)` and `s_2(x_t)` be the top two TRM scores:

```math
\Delta_t=s_1(x_t)-s_2(x_t)
```

Then:

```math
y_t =
\begin{cases}
\arg\max_y q_\theta(y\mid x_t) & \Delta_t\ge\gamma\\
\arg\max_{y\in C_t} q_\theta(y\mid x_t) & \Delta_t<\gamma
\end{cases}
```

The threshold `gamma` is selected on the training split from `[0.0, 0.5, 1.0, 2.0, 4.0]`. On the saved routing run, `gamma=0.0`, meaning any positive use of noisy LDT routing candidates reduces training accuracy; confidence arbitration degenerates to TRM on this slice.

## 6. Practical Hybrid Dynamics

### 6.1 Sudoku

LDT performs naked-single propagation. TRM performs heuristic MRV search. Hybrid lets TRM propose a branch and then applies LDT propagation after the proposal.

```mermaid
flowchart TD
  G[Partial Sudoku grid] --> C[Candidate lattice]
  C -->|singletons| L[LDT propagation]
  C -->|no singleton| T[TRM MRV proposal]
  T --> H[Place candidate]
  H --> L
  L -->|conflict| B[Backtrack/reject]
  L -->|solved| S[Solved grid]
```

Observed result:

```text
LDT:    1/3 solved
TRM:    3/3 solved, 31 guesses
Hybrid: 3/3 solved, 6 guesses
```

Interpretation: TRM is useful for escaping propagation plateaus. LDT is useful immediately after proposals because it collapses implied consequences and reduces further guessing.

### 6.2 ARC-1 and ARC-2

ARC-1 is a single-rule grid transformation benchmark. ARC-2 uses ordered two-rule compositions.

LDT derives broad rule-family constraints from input/output invariants. TRM searches primitive rules or ordered pairs. Hybrid restricts proposals by LDT families and certifies them against training examples.

```math
\mathcal{R}: \text{primitive rule set}
```

For ARC-1:

```math
\hat{r} = \operatorname{TRM}(x), \quad
\operatorname{accept}(\hat{r}) \iff \forall (x_j,y_j)\in D_\text{train}: \hat{r}(x_j)=y_j
```

For ARC-2:

```math
\hat{r} = r_b \circ r_a
```

and the same train-pair certification applies.

Observed aggregate results:

```text
ARC-1: LDT 1.000, TRM 1.000, Hybrid 1.000
ARC-2: LDT 1.000, TRM 1.000, Hybrid 1.000
```

Efficiency:

```text
ARC-1: hybrid 5 proposals, TRM 9 proposals
ARC-2: hybrid 32 proposals, TRM 38 proposals
```

Negative subcase: on two ARC-2 tasks, hybrid uses more proposals than TRM because the current pair ordering is heuristic. This is not a soundness failure. It is a proposal-ranking failure.

### 6.3 Environment-Pointer Routing

Routing maps a prompt to an environment ID. The Tesseract TRM reference trains a TF-IDF plus MLP router. The dependency-light benchmark here uses a lexical TRM analogue.

LDT builds token-derived candidate sets:

```math
C_t = \bigcap_{w \in \operatorname{tokens}(x)} E_w
```

where `E_w` is the set of environments strongly associated with token `w`.

This evidence is not environment-sound. It is model/experience-derived. Therefore the final hybrid design uses it as soft guidance only:

```math
\operatorname{route}_\text{hybrid}(x)=
\begin{cases}
\arg\max_{e \in C_t} \operatorname{TRM}(e \mid x) & \text{if TRM top route is in } C_t\\
\arg\max_{e} \operatorname{TRM}(e \mid x) & \text{otherwise}
\end{cases}
```

Observed result:

```text
LDT:    0.474
TRM:    0.815
Hybrid: 0.815
Hybrid hard filter: 0.584
```

Design lesson: when LDT evidence is lexical and learned from data, hard elimination is unsafe. The hybrid must distinguish soft route telemetry from environment-sound pruning. In this deterministic run, forcing token-lattice candidates as hard filters reduces accuracy from `0.815` to `0.584`.

### 6.4 Storyworld Playing

The storyworld has exact transitions and a modeled rival policy. LDT uses finite-horizon reachability. TRM uses a greedy local-deficit heuristic. Hybrid lets TRM propose an action, then checks whether the action preserves modeled reachability.

```math
\operatorname{safe}(s,a,H) =
\operatorname{Reachable}_{\text{model}}(T(s,a,\pi_\text{rival}(s)), H-1)
```

Hybrid action selection:

```math
a_t =
\begin{cases}
a^\text{TRM}_t & \text{if safe}(s_t,a^\text{TRM}_t,H-t)\\
a^\text{LDT}_t & \text{if } a^\text{LDT}_t \text{ exists}\\
a^\text{TRM}_t & \text{otherwise}
\end{cases}
```

This is also where the richer storyworld intuition enters the paper. Secret endings and morality optimization are not the same control problem. A secret ending is a typed gate: the system either preserves reachability to a specific latent ending predicate or it does not. A morality objective is a soft preference surface: it can reward trust, evidence, low heat, and scene progress without being an environment-sound proof obligation. This mirrors the difference between worlds such as medical triage or bioethics councils, where some constraints are hard safety gates while others are soft value tradeoffs.

Observed result:

```text
LDT:    64/64 solved
TRM:    11/64 solved
Hybrid: 64/64 solved, 124 overrides
```

Interpretation: TRM local heuristics often rush into terminal states that fail the secret-ending predicate. LDT reachability prevents those traps. Hybrid preserves the proposal interface while repairing unsafe choices.

## 7. Benchmark Summary

| Benchmark | LDT | TRM | Hybrid | Best |
|---|---:|---:|---:|---|
| Sudoku | 0.333 | 1.000 | 1.000 | TRM, Hybrid |
| ARC-1 | 1.000 | 1.000 | 1.000 | all |
| ARC-2 | 1.000 | 1.000 | 1.000 | all |
| Routing | 0.474 | 0.815 | 0.815 | TRM, Hybrid |
| Storyworld | 1.000 | 0.172 | 1.000 | LDT, Hybrid |

The aggregate pattern is:

- TRM helps when proposal/search/routing is needed.
- LDT helps when mechanics are checkable.
- Hybrid helps when proposal can be followed by checkable pruning or safety checks.
- Hybrid does not automatically improve learned routing if the LDT side has only soft lexical evidence.

Routing architecture variants:

| Architecture | Accuracy | Correct | Control Policy |
|---|---:|---:|---|
| Typed membrane | 0.815 | 141/173 | soft telemetry unless sound |
| Hard gate | 0.584 | 101/173 | LDT candidates hard-filter TRM |
| Confidence arbitration | 0.815 | 141/173 | train-selected `gamma=0.0` |

Policy-selection rule of thumb:

| Regime | Preferred Policy | Reason |
|---|---|---|
| Environment-sound gate | Typed membrane or hard gate | Proof-like state should dominate confidence |
| Soft preference objective | Confidence arbitration | Local score can optimize value tradeoffs |
| Noisy lexical evidence | Typed membrane | Keep learned candidates soft |
| Checkable search space | Typed membrane | Propose with TRM, certify with LDT |

Storyworld confidence/type split:

| Scenario | Policy | Success Rate | Avg Score |
|---|---|---:|---:|
| Secret ending | Typed membrane | 1.000 | 1.00 |
| Secret ending | Confidence arbitration | 0.938 | 0.94 |
| Moral optimization | Typed membrane | 1.000 | 9.95 |
| Moral optimization | Confidence arbitration | 1.000 | 11.06 |

Transcript slice:

One secret-ending run starts at `(trust=3, evidence=1, heat=4, scene=1)`. Confidence arbitration sees high local margins and repeatedly selects `defuse`: `defuse, defuse, defuse, defuse`. It ends at `(trust=3, evidence=1, heat=0, scene=5)` and fails because evidence never reaches the secret gate. The typed membrane instead accepts the need to preserve reachability and follows `defuse, wait, wait, investigate, defuse, investigate`, ending at `(trust=1, evidence=3, heat=2, scene=5)` and satisfying the secret-ending predicate.

### 7.1 Topology-Controlled Skill Sequencing

This experiment isolates the control policy while holding the skill implementations fixed. Every context exposes
the same proposal-only TRM, deduction-only LDT, and typed propose/certify hybrid sequences. Sequence utilities are
fitted on calibration data only. Two calibration folds produce transport estimates `T_c^(1)` and `T_c^(2)` over
the preference vector `(u_typed - u_TRM, u_typed - u_LDT)`. Lineage contraction `lambda_c`, loop disagreement
`h_c`, and orientation reversal `o_c` define

```math
\widehat R_c = \min\{2, \lambda_c + h_c\},
\qquad
\operatorname{global}(c) \iff o_c=0 \land \widehat R_c \leq \epsilon,
```

after simultaneous-coverage and matched-noise checks. An orientation reversal forces `R_hat_c=2`. Unauthorized
contexts invoke the best calibration-fitted local section.

The frozen evaluation contains 667 paired episodes: 96 Sudoku, 168 procedural ARC-1, 160 procedural ARC-2, 115
held-out local Tesseract routing trajectories, and 128 storyworld starts split evenly between secret and moral
objectives. Families receive equal macro weight. Inference uses 5,000 paired family-stratified hierarchical
context-cluster bootstrap samples, 10,000 context-cluster sign flips, Holm correction over three registered
comparisons, family-balanced cluster Cohen's `d_z`, and exact McNemar accuracy tests. ARC instances are retained
only when all candidate sequences solve them, isolating sequencer efficiency rather than solver coverage.

| Sequencer | Macro Utility | Accuracy | Macro Cost | Violations |
|---|---:|---:|---:|---:|
| Global signed | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| Lineage only | 0.8998 | 0.9667 | 6.453 | 0.0000 |
| Fixed typed | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| Control math | **0.9001** | **0.9667** | **6.436** | **0.0000** |

Against global signed control, paired macro utility improves by `+0.006943` (clustered 95% CI
`[+0.004909,+0.009232]`, Holm-adjusted `p=0.000600`, family-balanced cluster `d_z=0.876`). Accuracy is unchanged
(exact McNemar `p=1`), while macro cost falls by `1.238`. Against lineage-only control, utility improves by only
`+0.000310` (95% CI `[0,+0.000864]`, clustered `p=0.502350`, `d_z=0.209`) and macro cost falls by `0.017`. This
slice does not establish incremental utility from holonomy/orientation checks beyond lineage localization.

The registered controller invokes a local section for 100% of contexts and matches the local-calibrated
reference. The positive result is therefore conservative localization, not broad evidence that global signed
transport is safe. Orientation reversal is detected in 14.3% of contexts.

Post-registration sensitivity:

| Error Budget | Local-Section Rate | Macro Utility | Macro Cost |
|---:|---:|---:|---:|
| 0.25 | 1.000 | 0.9001 | 6.436 |
| 0.50 | 1.000 | 0.9001 | 6.436 |
| 0.75 | 0.821 | 0.9001 | 6.436 |
| 1.00 | 0.714 | 0.8967 | 6.737 |
| 1.50 | 0.214 | 0.8949 | 6.885 |
| 2.00 | 0.143 | 0.8943 | 7.223 |

Relaxing authorization reduces local sectioning but erodes utility and raises cost. The 667 immutable replay
tasks are also exported as a Verifiers `0.1.14` v1 Taskset/Harness package. Taskset owns tasks and scoring;
Harness owns sequencer rollout. This is an LLM-free portability artifact, not an additional model result.

## 8. HRM Review of New Model Training

Conductor-HRM reviews training as a graph of typed modules. The slow HRM selects an ordinary, invariant-bundle, global-signed, or sectioned-signed review regime. The fast HRM schedules spectral geometry, checkpoint lineage, loop holonomy, grouped utility, damage, and resource audits. A typed join routes the candidate to `authorize`, `section`, `audit`, or `reject`. The manager never writes weights or upgrades a source receipt.

For context stratum $p$, define the normalized reliability-weighted Laplacian

$$
\widetilde L_p = \frac{\delta_p^\top W_p\delta_p}{\lambda_{\max}(\delta_p^\top W_p\delta_p)}.
$$

For frozen band $I_b$, let $P_b(p)=\mathbf 1_{I_b}(\widetilde L_p)$ and infer a consensus bundle from occupancy:

$$
\overline P_b=\frac{1}{|\mathcal P|}\sum_pP_b(p),
\qquad
U_b=\operatorname{im}\mathbf 1_{[\rho,1]}(\overline P_b).
$$

Jacobians must register sites into a common edit space before cross-site projectors are compared. The geometry, rank, band, thresholds, model hash, dataset hash, and operator hash are sealed before outcome reveal.

Adjacent orthonormal frames satisfy

$$
U_b^\top U_a=Q_{b\leftarrow a}S_{ba}.
$$

$S_{ba}$ measures local retention and $Q_{b\leftarrow a}$ transports signed coordinates. Around a context-by-checkpoint loop,

$$
H_{\square}=Q_{00\leftarrow01}Q_{01\leftarrow11}Q_{11\leftarrow10}Q_{10\leftarrow00},
\qquad
\kappa_{\square}=1-\frac{\operatorname{tr}(H_{\square})}{r}.
$$

For edge worst-direction squared retention $W_e$ and net polar transport $H$, the signed-coordinate error obeys

$$
\|M-I\|_2\leq\sum_e(1-\sqrt{W_e})+\|H-I\|_2.
$$

If orientation is preserved, $\|H-I\|_2=2\sin(\alpha_{\max}/2)$. Orientation reversal or an unmeasured loop forces the conservative value 2. With simultaneous one-sided coverage at least $1-\delta$, signed control is authorized only if the uncertainty-adjusted bound $\widehat R\leq\epsilon$. This limits false authorization for registered signed coordinates; it does not certify behavioral safety.

Invariant bundle energy

$$
e_b(x)=\frac{\|U_b^\top x\|_2^2}{\|x\|_2^2}
$$

can guide allocation without choosing a signed direction. Its exponential tilt remains inside a frozen KL budget:

$$
\pi_\eta(i\mid s)\propto\pi_0(i\mid s)\exp(\eta\widehat\Delta_{u,i}),
\qquad
D_{\mathrm{KL}}(\pi_\eta\|\pi_0)\leq K_{\max}.
$$

Certificate requirements are mechanism-specific:

| Mechanism | Required identity | High-holonomy response |
|---|---|---|
| Ordinary optimizer | engineering evidence | behavioral review can proceed; signed authority is withheld |
| Bundle allocation | lineage certified | proceed only under utility and KL gates |
| Global signed control | holonomy clean | section or reject global signed use |
| Sectioned signed control | lineage plus clean patches | audit or rebuild incomplete patches |

Model promotion always separately requires grouped held-out gain, matched-rank Haar controls, held-out damage, independent replicates, OS-enforced RAM/CPU/I/O caps, chunk and checkpoint policy, structured resource logs, abort status, process-owned cleanup, and hash-bound provenance.

Representative frozen-policy contract outcomes under $\epsilon=0.5$ and $\delta=0.05$:

| Receipt case | Request | Route | Bound |
|---|---|---|---:|
| Clean signed candidate | promotion | authorize | 0.095 |
| High measured holonomy | signed control | section | 0.757 |
| Orientation reversal | signed control | section | 2.000 |
| Unmeasured loop | signed control | audit | 2.000 |
| Failed grouped utility | promotion | reject | 0.095 |

These are deterministic synthetic receipt cases. No neural model was trained, promoted, or edited.

## 9. Design Decisions and Alternatives

### 9.1 Hard vs Soft Application

Decision: hard-apply only environment-sound refinements by default.

Alternative: allow model-sound or experience-sound hard updates. This would make routing and replay-derived pruning more aggressive, but it risks false eliminations. The routing benchmark demonstrates the risk: hard LDT filtering from token evidence underperformed TRM, with the hard-filter hybrid ablation at `0.584` versus `0.815` for TRM and soft hybrid.

### 9.2 Reject Non-Monotone Proposals Before Meet

Decision: reject proposals that widen candidate sets before applying meet.

Alternative: apply meet regardless and accept the result if the final state does not widen. This hides proposal defects. Rejecting early gives a cleaner training signal: the proposal itself violated the membrane contract.

### 9.3 Bottom Requires Explicit Abstain

Decision: bottom-producing proposals are rejected unless the proposal is in explicit abstain mode.

Alternative: treat bottom as an ordinary conflict result. We avoid this because bottom is semantically important. It should become a visible conflict/abstention frame, not an accidental deduction.

### 9.4 LDT as Certifier vs LDT as Planner

Decision: use LDT differently by domain.

- Sudoku: propagator after proposals.
- ARC: rule certifier over examples.
- Routing: soft candidate telemetry.
- Storyworld: reachability planner and safety checker.

Alternative: force a single LDT role across all benchmarks. This would be cleaner architecturally but less honest: the meaning of checkability differs across domains.

### 9.5 Hybrid Override Policy

Decision: in storyworld play, override TRM when its proposed action loses modeled reachability and a certified alternative exists.

Alternative: always route through LDT first. That collapses the hybrid into LDT and removes the proposal role. Another alternative is to let TRM override LDT for speed, but then the system loses the safety benefit on dynamic tasks.

### 9.6 Proposal Ordering

Decision: current ARC proposal ordering is simple and hand-coded.

Alternative: train the proposal order. The ARC-2 regressions are evidence that this matters. A learned TRM proposal ranker should reduce individual proposal-count failures while retaining LDT certification.

### 9.7 Topology as Typed Authority

Decision: keep identity geometry outside the scalar HRM objective. Low holonomy cannot substitute for held-out model quality, and high utility cannot purchase an unauthorized signed operation.

Alternative: reject every high-holonomy candidate. We instead withhold global signed authority and permit local sectioning, with each patch subject to its own identity, utility, damage, and trust-radius receipts.

## 10. Limitations

The benchmarks are small and synthetic. They test contracts, not scale.

The TRM implementations in this repo are dependency-light analogues, not full recurrent neural models. The Tesseract router reference uses a TF-IDF plus MLP architecture, and the local routing benchmark mirrors the classification objective without importing its full PyTorch/sklearn stack.

The LDT side is symbolic and domain-specific. A learned LDT head is future work.

The hybrid has no aggregate score regression in the saved benchmarks, but it does have efficiency regressions in ARC-2 subcases and no routing accuracy gain over TRM yet.

The sequencer benchmark replays deterministic known-task skills. Its significant gain over global control is
mainly lower cost and better preference selection, not higher accuracy, and it shows no significant advantage
over lineage-only control. ARC tasks are filtered for common solver success. It is not evidence of neural weight
infusion, official ARC performance, or official INTELLECT-3 performance.

The HRM training-review matrix uses synthetic sealed receipts and supplies no evidence of neural training gains. The signed-control theorem applies only to registered coordinate error under its simultaneous-coverage assumptions. It does not establish general behavioral safety or self-improvement.

## 11. Next Experiments

1. Train ARC-2 proposal ordering and compare proposal counts against the static hybrid.
2. Replace the routing confidence grid with a richer calibration signal; the current trained threshold degenerates to TRM on this slice.
3. Convert membrane decisions into SFT records and train a small learned membrane policy.
4. Run the INTELLECT-3 `logic-env` under WSL/Linux, since the Windows smoke command currently fails before environment loading due to a Unix-only `fcntl` import in `prime_tunnel`.
5. Scale storyworld play to larger GPTStoryworld/SweepWeave tasks after this finite-state harness is stable.
6. Replace deterministic sequencer candidates with frozen learned TRM/LDT checkpoints while retaining the registered partitions and paired control-policy ablations.
7. Wrap a capped HRM/TRM checkpoint run as a preregistered receipt producer, then evaluate context-by-checkpoint lineage, matched loop nulls, grouped held-out utility, and local section persistence.

## 12. Conclusion

The core result is not that hybrid reasoning is always better. The result is more specific: a typed membrane lets latent proposal and explicit deduction cooperate without conflating their evidence types. TRM-style proposal is effective for search and routing. LDT-style explicit state is effective when mechanics are checkable. Hybrid works best when proposals can be followed by monotone refinement, certification, or reachability checks. When the LDT side has only learned or lexical evidence, it should remain soft. Applied to fixed sequencers, topology-controlled localization outperforms global signed transfer, but this slice does not distinguish full holonomy/orientation control from lineage-only localization. At the multi-module level, Conductor-HRM applies the same discipline to training review: topology bounds internal-coordinate authority, while held-out utility, damage, provenance, and resources independently govern model promotion.
