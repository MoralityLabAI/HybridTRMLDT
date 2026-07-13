# Hybrid LDT/TRM Architecture

## Claim

The immediate research object is a hybrid LDT/TRM, not a VPD edit loop.

The architecture carries two states:

```text
a_t: explicit lattice or abstract deduction state
h_t: recurrent latent state
```

The lattice state is public, typed, and checkable. The latent state is private recurrent workspace for correlation memory, heuristic proposal, and skill-regime inference.

## Loop

```text
h_{t+1} = TRM_theta(problem, a_t, h_t)
q_t     = proposal_head(h_{t+1}, a_t)
c_t     = conflict_head(h_{t+1}, a_t)
m_t     = mode_head(h_{t+1}, a_t)
r_t     = project_to_lattice(q_t, a_t, mechanics)
a_{t+1} = meet(a_t, r_t)
```

The latent may propose. Durable state change passes through a projection or certification layer.

## Modes

- Deduce: lattice mechanics support monotone refinement.
- Branch: lattice stalls but search is available.
- Expand: current abstraction is too weak.
- Explore: latent or model-based proposal is useful but not hard-sound.
- Abstain: conflict head fires or no safe move exists.

## Soundness typing

- Environment-sound: derived from exact mechanics.
- Model-sound: derived from exact mechanics plus a model of other agents or latent dynamics.
- Experience-sound: derived from replay or discovered successes.

Only environment-sound judgments should become hard eliminations by default.

## VPD position

VPD is an instrumentation layer until proven otherwise. It may label weights, adapters, or components by skill/deduction behavior. The first bridge to training should be data routing, adapter specialization, loss weighting, or regularization. Direct weight-edit hill climbing is not assumed.

## Multi-module extension

`docs/hrm_conductor_architecture.md` defines Conductor-HRM, a hierarchical typed module-flow architecture in
which an HRM schedules and joins multiple hybrid modules. The HRM controls execution flow but cannot bypass a
module membrane, mutate module-owned lattice state, or upgrade receipt provenance.

`docs/rsi_hrm_training_review.md` applies RSITopology bundle, lineage, holonomy, and signed-control mathematics
to Conductor-HRM review of newly trained checkpoints. Geometry controls which internal-coordinate operations are
identifiable; grouped held-out utility, damage, and capped-run receipts separately control model promotion.
