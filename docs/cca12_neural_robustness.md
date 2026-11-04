# CCA-12: neural-output and hidden-state control margins

Status: exact margin theorem and Python regressions; CCA-12A registered
retrospective six-tape audit in experiments/cca12_neural_robustness.
CCA-12B runs on original local checkpoints; .pt files are external to Git.

## CCA-T: exact L-infinity action-logit robustness

Given an environment-owned certified nonempty set C subset A, fixed state,
and unmasked logits z indexed by A, define

    Delta = max_{a in C} z_a - max_{u in A\C} z_u.

For C=A, every output is safe, hence radius=infinity. For C empty, no
action can execute; shield abstains. Otherwise, every adversarial additive
per-logit perturbation with norm_infinity <= epsilon has certified winner
provided Delta > 2epsilon. Proof: the maximum certified logit decreases
by at most epsilon and the maximum uncertified logit increases by at most
epsilon. At epsilon > Delta/2, the perturbation setting every safe logit
to z-epsilon and every unsafe logit to z+epsilon forces an unsafe winner.
Thus the positive radius is max(0,Delta/2), with strict inequality to
rule out a tie favoring an unsafe action. At epsilon=Delta/2 ties depend
on vocabulary priority. A negative Delta means baseline already selects
an unsafe action. This is exact within the specified independent-logit
adversary class; it is NOT an assertion about the GRU hidden-state metric.

The independent execution shield certifies its chosen action after
perturbation. It remains safe even if the neural model's original
unshielded action changes, conditional on the existing CCA-05 verifier.

## CCA-U: sufficient 48D final-hidden robustness

For last-layer action head z=Wh+b and additive final-hidden noise d,
let a be the baseline argmax of the UNMASKED action scores and suppose
a is certified. For each uncertified u the perturbed gap is

    (z_a-z_u) + (W_a-W_u) . d.

By Cauchy-Schwarz it remains strictly positive whenever

    ||d||_2 < min_{u not in C} (z_a-z_u)/||W_a-W_u||_2.

Ignore competitors with identical row weights and a strictly positive
initial gap; their gaps cannot change. If all unsafe competitor rows
have zero norm difference the radius is infinity. The minimum is a
conservative anchored sufficient radius: another certified action may
remain ahead when the anchor is displaced. For a baseline unsafe winner
the certified radius is zero. The bound applies to noise injected AFTER
the final GRU update. For noise injected earlier, one additionally needs
a sound bound for the GRU's propagation to the final hidden state.

## CCA-V: disconnected geometry and the need for a bridge

CCA-11's Manhattan state viability radius is a property of the finite
environment state and the trusted verifier. CCA-T and CCA-U are
properties of the neural action-selection mechanism. Their connection
is mediated by the verified action certificate and actual transition.

To claim a joint perturbation budget for a recurrent controller,
we must identify a trusted mapping from perturbed neural behavior to
environment action/transition outcomes and establish a state-dependent
error-envelope bound. Merely correlating hidden-state distance with
environment Manhattan distance supplies no certificate.

A useful next experiment modifies intermediate GRU hidden states under
declared input and weight norms, measures local Jacobian gains, and
compares a certified analytic global Lipschitz upper bound with actual
worst-case/seeded perturbation outcomes. The outer CCA-05 checker remains
unchanged.

## Reproduction

    python -m pytest -q tests/test_cca_neural_margins.py
    python experiments/cca12_neural_robustness/run_tapes.py --out /tmp/cca12_real_tapes.json

With the original registered private checkpoints available on Windows:

    python experiments/cca12_neural_robustness/latent_probe.py \
      --checkpoint-dir "C:\Users\patri\Documents\Codex\jinn-vpd\part_b_real" \
      --out cca12_latent_private.json

The latent probe reports only aggregates and expected hashes, keeping
checkpoints and raw activations outside the public repository.
