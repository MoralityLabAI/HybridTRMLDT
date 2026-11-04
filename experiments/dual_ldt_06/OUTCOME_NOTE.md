# Outcome-valued Jinn: exact compression on a certified feasible set

Status: post-run mathematical application and finite representation audit. The registered Dual-LDT-06 learning/search comparison is unchanged. This note makes no novelty claim for multi-objective dominance, successor features or coverage sets.

## Outcome representation

For a plan P, let psi(P) be its expected sum of four completion-feature rewards under the fixed transition probability model. Let c(P) be its resource cost. The authored persona objective is

    U_theta(P) = theta dot psi(P) - c(P), theta >= 0.

The Beast supplies the feasible domain S. The Jinn's weights theta never change membership in S. In this experiment all quantities share denominator 64, so dominance checks operate on exact integers.

If psi(Q) is coordinatewise at least psi(P) and c(Q) <= c(P), then

    U_theta(Q) >= U_theta(P) for every nonnegative theta.

Proof: every weighted feature difference is nonnegative; the cost difference has the same favorable sign. A nonempty finite set has a maximal dominating element above each of its elements. Remove duplicates and retain a representative of every nondominated outcome/cost vector. Each removed plan then has a retained dominating witness, and

    max_{P in S} U_theta(P) = max_{P in C(S)} U_theta(P)

for every nonnegative linear persona preference. This set is sufficient and need not be the smallest convex coverage set. The Python verifier checks actual retained IDs and exact outcome values, together with complete witness coverage. The Lean file proves nonnegative-integer weighted dominance and the general covering-maximum implication; the Python extraction algorithm is checked by tests rather than a Lean implementation-refinement proof.

## Report-set audit

After the registered results were known, an independent Fraction-based replay enumerated all 343 plans in every report world. Mean per-world sizes:

| Representation | Mean count |
|---|---:|
| Original candidate programs | 343 |
| Beast-feasible programs | 123.046875 |
| Distinct exact outcome/cost vectors | 12.46875 |
| Nondominated representatives | 4.96875 |

The final count ranges from 2 to 13. A grid of 81 nonnegative persona weight vectors per world preserves the exact optimum under compression, and every discarded record has an exact dominating witness. This is a retrospective representation-size result. Building the cover requires the input evaluations; the experiment establishes no faster discovery procedure or neural advantage from compression.

## Corrigibility and scope

The cover is bound to a fixed state, environment model and feasible set. A later human policy can remove a retained dominator while leaving a previously discarded plan admissible. The cover must then be rebuilt from current certified plans, or updated with a complete witness dependency system. The supplied checker rejects a stale cover against the new feasible records.

Negative preference weights can reverse the dominance relation. Nonlinear fairness, risk sensitivity and worst-case objectives require their own order and sufficient statistics. This implementation permits nonnegative rational weights and exact integer outcome numerators. The host evaluator remains the evidence source; a learned outcome vector alone supplies no safety permission.

## What a learned component would need to establish

A shared vector predictor could expose a reusable physical/outcome estimate to many external preference weights. If its feature error is uniformly bounded by epsilon in infinity norm on the current feasible domain and costs are exact, then

    |Uhat_theta(P)-U_theta(P)| <= ||theta||_1 epsilon.

Selecting an argmax of Uhat on that domain consequently has regret at most

    2 ||theta||_1 epsilon.

This follows by inserting the estimate at the true and predicted maxima and applying the two error bounds. Empirical MAE is not this uniform guarantee. Likewise this inequality alone does not supply a fixed-query search guarantee when the learned choice has not been independently certified.

A prospective follow-up can train vector predictions and compare them with the strong count-relaxation order, retaining the exact Beast checker. Useful endpoints would include discovery cost for nondominated certified vectors and transfer to newly specified preference weights. The result must include compilation, inference and verification costs.

## Precedents and reproducibility

Barreto et al., Successor Features for Transfer in Reinforcement Learning, NeurIPS 2017: https://papers.nips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html

Roijers, Efficient Methods for Multi-Objective Decision-Theoretic Planning, IJCAI 2015: https://ijcai.org/proceedings/2015/642

Files: outcome_frontier.py, audit_artifact.py, tests/test_dual_ldt_outcome.py, formal/CCA/DualLDTOutcome.lean.

To audit the retained full result, unpack artifact 13343636356 from run 41868351947 and run:

    python experiments/dual_ldt_06/audit_artifact.py /path/to/unpacked/artifact

The audit reads the saved frozen weights, manifests and report records, performs its own Fraction replay, and writes INDEPENDENT_AUDIT.json. It does not retrain the models or change the original report.
