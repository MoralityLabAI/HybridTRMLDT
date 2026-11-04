# Dual-LDT-04: Certified action synthesis and bounded Jinn preference regret

## Fixed model and state semantics

64 new SHA-derived synthetic worlds (32 development, 32 report) with four hazardous root actions, two maintenance-inducing actions, and two actions preserving active state. The fixed four Jinn heads are the actual paper checkpoints, applied to symbolic route tags and optional host-owned guard tags. No language model generates actions; synthesis enumerates a finite typed grammar of 25 candidates. The true transition model, guard capabilities, resource costs and authorization evidence are supplied by the synthetic host.

A plan consists of a root action and at most one guard primitive. Interlock suppresses an isolated successor only when the host world says a functioning mechanism is installed and there are enough resources. Witness requires available supervision and the active state, moves to maintenance where RESTORE then HALT is possible. Both modify audited tags by adding exactly one declared guard tag; they inherit every original tag, so a constitution tightening cannot be bypassed by merely renaming a plan.

## Beast correction and proof-carrying execution

At each decision (world, state, time, remaining credits), the host constructs a BeastSnapshot across all 25 plans. Immediate I permits only original roots meeting current hard tags. Temporal T further requires all modeled successors to lie in W1. Synthesis S permits host-certified guarded plans as well and preserves every plain action T permits.

The host issues PlanProof for a selected action only when its current snapshot authorizes it. The checker independently recomputes the current world digest, policy epoch, budget, root and guard tags, guard cost, installed capability, all successor states and allowed set from the host's original source configuration. It rejects stale/forged/missing proofs and unknown plans. The certificate is a checkable witness rather than a cryptographic signature; host-owned execution remains a trust assumption. Both issuer and checker share transition semantics, so an error in that common implementation is outside the demonstrated guarantee.

T and S are safe under the game's transition assumptions because every executed action's possible successors lie in W1. The host can restore from maintenance and HALT within two host actions. Lean formalizes an abstract authorization/subset theorem; exact full-game obligations are checked in Python.

## Jinn optimization and quantitative regret

Let u_theta(a) be log P(train|tags(a)) of the real fixed three-lane head. Charge resource cost c(a) credits, rho=0.10, and define v_theta(a)=u_theta(a)-rho*c(a). The Jinn chooses the exact argmax of v_theta among host-certified actions through conditional exponential-family sorting, with a deterministic action ordering on ties.

Because S contains T's plain certified actions in the same state and with the same remaining budget,

  max_{a in S} v_theta(a) >= max_{a in T} v_theta(a).

In particular, any S action chosen by a maximizer has raw Jinn score at least the T argmax's raw score (T plain actions cost 0; all guard costs are nonnegative). This is a state-local statement, not a guarantee that the two trajectories reach identical states.

Let a_star be the immediate Beast's preferred original root at the same state. For any certified candidate w with raw preference shortfall delta=max(0,u(a_star)-u(w)) and cost c(w), the S argmax y has shortfall

  max(0,u(a_star)-u(y)) <= delta+rho*c(w).

Proof: u(y)-rho*c(y)>=u(w)-rho*c(w), so u(y)>=u(w)-rho*c(w) and truncation of the regret at zero gives the bound. Enumerating the entire certified finite plan set identifies the minimum such upper bound and emits an exact finite-grammar witness. The source head's logits are interpreted as a preference score; this is not a calibrated measure of moral value.

Both T and S use the same 25-action reference distribution with q(a)=exp(-rho*c(a)), so S's expanded certified support weakly decreases the conditional KL preference tax -log p(S). The script also tracks the fraction of S decisions achieving a Jinn regret at most epsilon=0.25, fixed in registration.

## Tradeoffs and measurement

Every selected guard spends its host-assigned 1-3 credits. The net task reward is the root's independent task reward minus 0.10 per guard credit. We report task outcome separately from persona preference and intervention safety, by persona, split, disturbance and policy. Every report world is fresh under the frozen SHA namespace; the tag bundles remain in distribution for the original trained heads.

The experiment records the number of proof checks attempted, accepted guarded plans, actual guarded executions, preference regret, upper-bound witness, lost/retained conditional KL mass, resource budget, action trajectories and host HALT outcome. A null preference-recovery result remains valid. No adaptive semantic monitor is assessed, and future supervisory action costs are fixed by this toy model.

Code: research_gym/core/dual_ldt_synthesis.py and experiments/dual_ldt_04/src/{world,host,episode,run}.py.
Formal: formal/CCA/DualLDTActionSynthesis.lean.
Tests: tests/test_dual_ldt_04.py.
