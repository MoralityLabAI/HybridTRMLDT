# Event-clock credit and success objectives

## 1. Separate the decision clock from serialization

Let j index retained actor positions. The trusted host supplies a boundary indicator b_j. Use gamma_j = gamma_event^b_j and lambda_j = lambda_event^b_j, with true terminal d_j. For no boundary, both factors are one. Observations condition the critic but do not become extra recursion nodes.

    delta_j = r_j + gamma_j (1-d_j) v_(j+1) - v_j
    A_j = delta_j + gamma_j lambda_j (1-d_j) A_(j+1).

This specializes the paper's TB-GAE to a host-owned clock. Every attempted tool decision counts, including reads, rejected requests and explicit no-ops. A model cannot create or suppress a boundary by printing a tag. Recurrent internal computation remains separately metered. Learned halting choices are policy decisions and must remain in the trajectory law.

Fixed-input stutter lemma: replace one step by any finite sequence whose intermediate rewards are zero and whose intermediate discount/trace factors are one. Preserve the old value at the first anchor and put the old reward/boundary on the final substep. The original anchor advantage is unchanged because intermediate values telescope. The reference checks 200 randomized refinements, including arbitrary intermediate values.

This identity concerns a fixed numerical trajectory. Adding meaningful observations can change the critic and policy. Actor-generated filler can also alter loss weighting, likelihoods or entropy. Those changes require separate behavioral experiments. Treating all latent updates as semantically inert would violate the premise.

At an existing boundary with lambda < 1, inserting a different bootstrap value before the old successor can alter A by gamma(1-lambda)(v_inserted-v_successor). A negative-control test checks this exact failure.

## 2. Chunk continuation

For a cut before position b, compute the suffix first. The prefix requires both v_b and A_b. If the value is retained and the trace is reset, the missing contribution at j is

    (product over i=j,...,b-1 of gamma_i lambda_i (1-d_i)) A_b.

A memory partition is not task termination. True terminals erase both continuation inputs. The test suite checks 200 randomized cut locations.

## 3. Count computation explicitly

For a bounded episode, optimize strict success subject to an external compute budget. A Lagrangian alternative is J = E[S - eta C], where C is measured work. This alternative changes the objective and must be named in every receipt. Neither event-clock invariance nor a unit discount makes extra computation free. A cost-bearing stutter test decreases value by its actual cost.

Use gamma=1 in the first finite-horizon learning experiment to hold the success objective fixed while varying trace decay. Later gamma<1 experiments optimize discounted success and require separate interpretation.

## 4. A concrete SPR counterexample

Use the same family with D=100 and 100 checks. All successful outcomes pass 100 checks and receive 1.5. Policy A succeeds with probability 0.80; its failures pass zero checks. Policy B succeeds with probability 0.75; its failures pass 99 checks. Every successful outcome outranks every failed outcome.

    E[R_A] = .80 * 1.5 = 1.2
    E[R_B] = .75 * 1.5 + .25 * 99/199 = 1.2493718593.

Expected SPR therefore prefers B despite its lower strict success probability. This is a constructed two-action stochastic MDP, not an empirical claim about Terminal-RSI. The paper already cautions that outcome ordering does not establish success-probability optimality.

## 5. Preserve the objective with a potential

Let the base reward be protected terminal success. For bounded state potential Phi and the same transition discount, add

    F(s,a,s') = gamma(s,a,s') Phi(s') - Phi(s).

For a complete episode with zero potential at every true terminal, the discounted shaping sum is -Phi(s_0). A fixed initial-state distribution gives every policy the same additive expectation. At gamma=1 this preserves expected success ordering. At gamma<1 it preserves the discounted objective. Truncation bootstraps require their potential term; setting a nonterminal potential to zero would break the identity.

The CPU suite checks the telescoping equality on 200 random complete episodes. A learned potential must be frozen within the comparison; moving the reward or leaking protected verifier labels invalidates attribution. Authorization remains enforced outside the optimizer.

## 6. Compact recurrent stopping

For latent state z_l, define the action-value of one further internal update as its expected downstream value gain minus its measured cost. A controller should continue only when a calibrated lower confidence bound on that gain exceeds the cost, under the fixed remaining budget. This is a proposed metareasoning experiment. A small Bellman residual alone does not reveal the value of additional thought, and extra loops can alter the information state.
