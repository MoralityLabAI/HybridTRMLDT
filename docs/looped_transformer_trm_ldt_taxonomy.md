# Looped Transformers, TRMs, HRMs, and LDTs

## Working taxonomy

Treat a looped transformer as the substrate: the same learned transition operator is revisited across recurrent depth,

[
h_{t+1}=F_	heta(h_t,x).
]

A conventional transformer instead assigns distinct parameters to successive depth steps,

[
h_{ell+1}=F_{	heta_ell}(h_ell).
]

This makes recurrence depth and state structure separate design axes.

### Architecture family

- **Plain looped transformer:** repeated application of a shared transformer block or stack. Fixed loop schedules mainly buy parameter reuse and extra effective depth.
- **TRM:** looped computation plus structured latent state and explicit refinement. A useful abstraction is (h=(y,z)), with repeated (z)-updates followed by (y)-updates.
- **HRM:** recurrent computation with multiple clocks or latent levels. Fast and slow state updates make the recurrence schedule itself part of the architecture.
- **LDT:** recurrent transformer dynamics plus a structured state space and an inter-pass projection or constraint operator. The loop is intended to implement successive deduction/refinement rather than generic repeated depth.
- **PTRM-style variants:** recurrent refinement plus stochasticity or trajectory branching, allowing multiple paths through the same shared dynamics.

## Fixed looping versus scalable recurrence

A fixed schedule such as two traversals of the same block stack can be unrolled into a fixed-depth feed-forward network. Its main advantages are parameter sharing, inference-time weight reuse, and an inductive bias toward repeated computation.

The stronger regime begins when recurrent depth grows with the task or input, or when an external or learned controller can allocate additional iterations. Then the model is better viewed as an iterative solver whose state transition is reused for an open-ended number of computational steps.

Adaptive halting is an efficiency/control mechanism over recurrence. It is not required for scalable recurrence itself: an external scheduler can allocate (T(n)) recurrent steps without a learned halt head.

## Two useful axes

### 1. Recurrence depth

[
1 ightarrow 2 ightarrow 16 ightarrow 1024 ightarrow 	ext{adaptive/scalable}
]

### 2. State structure

[
h ightarrow (y,z) ightarrow (z_H,z_L) ightarrow mathcal{L} ightarrow 	ext{richer world/model state}
]

A model can move far along either axis independently. Fixed Astra-style layer reuse, if that is what the architecture turns out to use, would sit near the low-structure end. TRM, HRM, and LDT move further because repeated visits have different computational roles.

## Why weight tying can become algorithmic

Weight tying alone does not make recurrence interesting. Training can turn repeated visits into stages of a learned algorithm.

With a shared transition,

[
S_{t+1}=F_	heta(S_t,X),
]

visit (t=17) can represent the seventeenth step of a solver rather than a second copy of "layer 17." The architecture becomes more expressive when state structure and training pressure make successive visits perform operations such as constraint propagation, belief revision, latent search, or controller-driven refinement.

A general hybrid form is

[
S_{t+1}=Pi_{m_t}!left[F_{	heta,m_t}(S_t,X)ight],
]

where (m_t) selects a computational mode and (Pi_{m_t}) optionally projects the state into a structured domain.

Under this view:

- plain looping uses a nearly constant (m_t) and identity projection;
- TRM alternates refinement roles for latent and answer state;
- HRM assigns updates to fast and slow recurrent levels;
- LDT gives the inter-pass projection semantic content;
- stochastic recursive variants perturb or branch the state trajectory.

## Implication for HybridTRMLDT

The repo should treat looped transformers as the base recurrent-depth family rather than a competing architecture. The research question is which state factorization, projection, controller schedule, and supervision make repeated shared-weight transitions behave like useful iterative computation.

VPD remains downstream instrumentation in this picture. It can characterize components, heads, adapters, or recurrent stages after a model exists, then test whether those labels support routing, specialization, regularization, or intervention.