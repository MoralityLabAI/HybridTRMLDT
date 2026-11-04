# CCA-11: Recoverability thresholds for certified recurrent control

Status (2026-11-04): theoretical envelope, exact-rational CPU tests, Lean
discrete-integer invariant pending CI. No neural checkpoint evaluation.
Code: research_gym/core/cca_recoverability_threshold.py
Tests: tests/test_cca_recoverability_threshold.py
Lean: formal/CCA/RecoveryThreshold.lean
Registration: experiments/cca11_recoverability_threshold/REGISTRATION.md

## Motivation

Aaronson's account of quantum fault tolerance motivates an AI control
question: how much recurrent disturbance can a certified controller
absorb before its useful continuations disappear?

CCA-05 already provides a conditional *outer action-safety* guarantee:
when the host checker, binding, environment model, and execution membrane
are sound, only certified actions are executed, regardless of the
upstream neural proposal. That guarantee has no intrinsic threshold in
neural perturbation amplitude; the membrane can abstain. CCA-11
studies the distinct property of *continued viability without abstention*.

Robust-invariant sets and runtime shields are established subjects in
control theory. We do not present this recurrence induction as a new
general mathematical theorem. The proposed research contribution is
connecting trusted viability certificates to empirical recurrent-model
behavior, quantifying utility retention, and finding counterexamples.

## CCA-O: sharp uniform recoverability certificate

Suppose an independent proof or verifier establishes a nonnegative
deviation e_t, nonnegative perturbation magnitude eta_t, and a
stepwise envelope

    e_(t+1) <= rho * e_t + c * eta_t + sigma,

with 0 <= rho < 1, c > 0, sigma >= 0. Assume a viability margin m > 0
such that every state with e_t <= m has at least one soundly certified
action whose continuation is permitted. This viability implication is
an extra environment-owned assumption, not a consequence of the formula.

For all eta_t <= eta and eta_crit >= 0, define

    eta_crit = ((1-rho)*m - sigma)/c.

If e_0 <= m and eta <= eta_crit, induction yields e_t <= m at
every step, since rho*m+c*eta_crit+sigma = m.

The constant-noise envelope obeys

    e_t <= rho^t*e_0 + [(c*eta+sigma)/(1-rho)]*(1-rho^t).

For arbitrary disturbances, the upper envelope is

    e_t <= rho^t*e_0 +
           sum_{j=0}^{t-1} rho^(t-1-j)*(c*eta_j+sigma).

SHARPNESS WITHIN THE WORST-CASE ENVELOPE CLASS:
the equality recurrence exits immediately from e_0=m when
eta > eta_crit. From e_0<=m, sustained above-threshold disturbance
eventually exits since its fixed point exceeds m. This does not prove
a given neural controller must fail above that value. A larger isolated
burst can remain viable when the state begins with enough slack.

If sigma > (1-rho)*m, the uniform argument cannot certify even eta=0.
Rho>=1 is outside the certificate's contracting-model assumptions.

The Lean file proves induction for a discrete scaled-integer
recurrence and exact boundary inequality. It does not yet formalize
rational arithmetic or the independent neural/semantic hypotheses.

## CCA-P: conditional utility regret

Let s0 and s be nominal and perturbed states with d(s,s0)<=e.
Assume the same nonempty certified action set A exists at both states.
For each a in A, U(s,a) is L-Lipschitz in s. Suppose the nominal
action a0 maximizes U(s0,a). For an optimum a* at s,

    U(s,a*)-U(s,a0)
       <= U(s0,a*)+L*e - (U(s0,a0)-L*e)
       <= 2*L*e.

The 2*L*e regret bound is tight for two actions with utility functions
-L*x and +L*x at a nominal tie x=0. It depends on the stable
certified action set and the *nominal oracle* premise. For actual
checkpoint proposals, replace the oracle premise using CCA-09's
independently checked utility bound. Action-vocabulary discontinuities
can defeat the Lipschitz assumption.

## CCA-Q: stochastic viability and monitor risk

Expected contraction gives weaker guarantees than pathwise contraction.
For a nonnegative adapted barrier B_t and a first-hitting time tau for
B_t>=m, assume until tau,

    E[B_(t+1) | F_t] <= B_t+delta,       B_0 <= b0.

Stopping at tau and applying Markov gives

    P(tau <= T) <= min(1, (b0+T*delta)/m).

A zero-drift nonnegative supermartingale permits a horizon-independent
b0/m bound. Persistent positive drift yields a bound that grows with T.
This is a bound on reaching the barrier, and thus bounds unsafe visits
when unsafe states imply B>=m. A one-step rare, large spike falsifies
any attempt to infer pathwise safety from expectation alone.

Let G_t denote correctness of environment-owned state coverage,
certificate verification, the modeled transition, and safe fallback at
execution t. If P(not G_t | history)<=q_t uniformly over adaptive
histories, the outer membrane's conditional safety gives

    P(contract violation before T) <= sum_{t<T} q_t.

No independence assumption is needed. Calibration-set SAE recall does
not itself certify a uniform conditional q_t under adaptive attack.
Adding the barrier-exit risk produces a bound on the composite event
of external violation OR loss of continued viability. Keep these
events separate in evaluation.

## Falsification and checkpoint protocol

CCA-11A is a CPU reference with exact rational arithmetic. At rho=1/2,
c=1, sigma=1/10, m=1, eta_crit=2/5. The 2/5 boundary is viable;
41/100 exits the envelope at the next step from e0=m. An isolated
3/4 burst in a different zero-residual envelope is absorbed from e0=0.

CCA-11C (future checkpoint study) should pin the six CCA-05 TRMs,
states and trusted environment digests. Perturb latent states, memory
and action proposals under a predeclared disturbance norm. Fit a
candidate (rho,c,sigma,m) on calibration states only, then seek an
independently verified uniform upper bound on held-out states and
attack families. Report every falsifying trajectory as a receipt.
Empirical quantiles alone cannot establish a worst-case certificate.

CCA-11D should compare raw TRM, structural shield, CCA-05 recurrent
repair, and certified oracle on identical initial conditions. At each
disturbance level, report external violations, abstention, progress to
goal, actual utility, CCA-09 regret certificates, semantic monitor
coverage, and verifier/repair cost. Host-owned tool permissions and
safe abstention are part of the threat model.

Key ways the connection can fail: no valid distance between neural
states and viability; noncontractive repair; missing concrete state
from Gamma(observation); changing certified action support; adaptive
unbounded disturbances; or an unsafe abstention fallback.

## Prior art

- Alshiekh et al., Safe Reinforcement Learning via Shielding, AAAI
  2018: https://doi.org/10.1609/aaai.v32i1.11797
- Jansen et al., Safe Reinforcement Learning via Probabilistic
  Shields, 2018: https://arxiv.org/abs/1807.06096
- Hamel-De le Court et al., Probabilistic Shielding for Safe
  Reinforcement Learning, AAAI 2025:
  https://doi.org/10.1609/aaai.v39i15.33767
- Goodall & Belardinelli, Recovery-based Shielding with Gaussian
  Process Dynamics Models, AAMAS 2026:
  https://doi.org/10.65109/EAHU8566


## CCA-R: finite-state viability-depth selection (CCA-11B)

Pinned source: research_gym/envs/coupled_storyworld.py, blob
545b2a1c283c1da8ac6c9fdb375a1bafd695e9b2.
Code: experiments/cca11_recoverability_threshold/finite_viability.py
Tests: tests/test_cca_finite_viability.py.
This is a retrospective structural computation using the already published
CCA-05 state graph, not new prospective checkpoint data.

For the finite state space X (1,050 states), define the horizon-k winning
set K_k from the exact CCA-05 secret-ending verifier. On integer state
coordinates, the local Manhattan viability radius is

    m_k(s) = min_{y in X \ K_k} ||s-y||_1 - 1,   for s in K_k.

This is the largest integer radius whose full coordinate ball stays in
the winning set, including the state-space boundary. As a result,
all coordinate perturbations with ||Delta||_1 <= m_k(s)
preserve the existence of a certified continuation at that horizon.

At horizon six, the modeled rival has 226 viable states and 618 certified
state-action pairs, exactly matching CCA-05's reported action count.
Radius histogram: m=0:134, m=1:69, m=2:23. For a nondeterministic
adversarial rival there are 40 viable states and 44 certified
state-action pairs, with m=0:34, m=1:6. These counts are for all states
including goal stops; actionable nonterminal states number 208 and 22.

The global minimum is zero in both settings. Consequently a positive
*uniform* state-perturbation margin cannot be inferred on the entire
winning set. CCA-O becomes useful through local margins or by
restricting the operational domain to an interior subset.

**CCA-R (max-min successor depth):** among actions certified by the
host verifier at (s,k), select an action maximizing

    min_{r in allowed_rivals} m_{k-1}(T(s,a,r)),

with goal stops treated as completed and safe from subsequent control
decisions. Every choice remains certified because its admissible set
is unchanged. By finite maximization its minimum successor radius is
at least that of any fixed certified-action ordering. This establishes
a one-step robustness ordering in the declared perturbation metric;
it does not guarantee future optimality or handle an altered rival model.

In the exact six-step model, the first-certified-action baseline produces
a post-action state resilient to any one-coordinate perturbation at 85
of 208 decision states. The depth-maximizing choice does so at 99 states,
with 20 strict depth improvements overall. Against the nondeterministic
rival the counts rise from 13 to 15 of 22, with two strict improvements.
Every selected action still meets the original secret-ending verifier,
whose binary utility is identical for all certified actions.

These results are arithmetic on a symbolic environment. Their metric
has no validated correspondence to SAE activations, recurrent hidden
states, human values, or open-world semantic trust.


## CCA-S: action-local admissible disturbance budget

Let s be a trusted nominal state with a certified action a and possible
nominal successors y=T(s,a,r). Let e denote current bounded deviation and
suppose repair dynamics satisfy the CCA-O inequality for all specified
rival/environment branches. Define

    m_a = min_r m_(k-1)(T(s,a,r)),

omitting successful goal stops that require no subsequent decision. If all
branches terminate successfully, next-step recoverability is vacuous.

If m_a >= rho*e + sigma, every disturbance satisfying

    0 <= eta <= (m_a - rho*e - sigma)/c

preserves viability of all nonterminal successors. If the numerator is
negative, even zero disturbance is uncertified. The reference
next_step_disturbance_budget returns None in this case rather than reporting
zero as an admissible certificate.

CCA-S follows by substitution into the one-step envelope. For
action-independent rho, c, sigma, maximizing m_a maximizes the
admissible disturbance budget. With action-dependent dynamics, optimize
the full local budget. An unverified state/error metric or an unmodeled
rival response invalidates this guarantee.
