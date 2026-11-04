# Dual-LDT-02: Preserve the host's capacity to intervene

The original Beast filter constrains actions at the current decision. This extension studies a different corrigibility property: **an apparently authorized action may make future authorized intervention impossible.** The new finite game filters such actions whenever its host-owned transition model proves they can lead outside the region in which the host can regain direct HALT control.

## Finite robust intervention game

Let X be finite states and H subset of X the states where an authenticated host HALT command is available immediately. Let A_host(x) be possible recovery actions under the host's control and T(x,a) be a nonempty set of possible successors over permitted environmental disturbances.

Build the winning regions:

\[
W_0=H,\qquad
W_{k+1}=W_k\cup\{x:\exists a\in A_{\mathrm{host}}(x),
\varnothing\ne T(x,a)\subseteq W_k\}.
\]

The existential quantifier is host choice; universal inclusion demands recovery for every represented disturbance. Under a sound complete transition description, membership x in W_k means the host has an adaptive policy reaching direct-HALT availability within at most k actions. The host must subsequently exercise that control when requested.

With immediate Beast-certified actions S(x), preserve the option to intervene after ordinary action a by allowing only

\[
S^{\mathrm{int}}_k(x)=
\{a\in S(x)\cap A_{\mathrm{agent}}(x):
T(x,a)\ne\varnothing,\ T(x,a)\subseteq W_k\}.
\]

The reference implementation conservatively also requires x in W_k, so it cannot grant a new permission from an already uncorrectable state. Missing/empty transitions fail closed.

**Dual-C1 (current safety):** S_int is a subset of the current Beast set S.
**Dual-C2 (future corrigibility):** for every allowed a and represented successor y, y belongs to W_k.
**Dual-C3 (deadline relaxation):** W_k subset W_(k+1), and the intervention shield permits weakly more actions as the recovery deadline expands, given fixed Beast evidence.

The three interface lemmas are kernel-checked in formal/CCA/DualLDTIntervention.lean. Finite fixed-point construction and state-horizon monotonicity are executable and exhaustively checked over small games; an independent machine-checked implementation refinement proof is not yet supplied.

## Concrete test

A host can issue HALT directly from active or halted states, and can recover from a maintenance state by taking restore. From active, the actor has four options: continue -> active, degrade -> maintenance, disable -> isolated, risky -> {active, isolated}.

Every option is initially permitted by the immediate Beast tag gate. At one-step recovery horizon, the temporal shield permits continue and degrade. It removes disable and risky, since at least one possible successor is isolated and has no host recovery action. Under an immediate-only recovery deadline, even degrade is excluded.

A trusted HALT command still has priority over all actor proposals. The protocol relies on independently verified environment transitions and on the executor honoring the host-control channel. A model-authored successor set cannot establish its own soundness.

This is a necessary interface property for future corrigibility, not a complete solution to deceptive behavior, oversight manipulation, or practical shutdown.

Implementation: research_gym/core/dual_ldt_intervention.py.
Tests: tests/test_dual_ldt_intervention.py.
Lean: formal/CCA/DualLDTIntervention.lean.
