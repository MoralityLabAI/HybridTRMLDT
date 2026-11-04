# FWC-01: fixed-weight conditioning, minimum state and typed supervisory control

**Research note** (2026-11-04). Source inspiration: Michael Levin's discussions of associative conditioning in small molecular networks and transferable organization, and a 2026 video on metacognition / selective learning (transcripts supplied in the research conversation). These statements below are **new finite computational examples and elementary deductions**, not reproductions of Levin's gene-regulatory-network experiments. No model training, Qwen measurement, or human-memory analogy is asserted.

## 1. A fixed recurrent circuit with experience-dependent behavior

Let x_t∈{0,1} denote a conditioned cue, u_t∈{0,1} an unconditioned stimulus, and h_t∈{0,1} persistent state. Consider the **fixed** transition / readout:

    y_t = u_t OR (x_t AND h_t)
    h_{t+1} = h_t OR (x_t AND u_t).                                (1)

There are no learned weights, no parameter updates and no persistent replay buffer. Starting h_0=0, a cue-only input (x=1,u=0) has y=0. After one paired input (1,1), state becomes 1. The same cue-only input thereafter has y=1. Resetting h to 0 erases the association. This is a *minimal synthetic analogue* of associative conditioning. It is not evidence of a biological placebo effect, nor of a mind.

**Theorem FWC-01/T1 (minimum information state).** Any deterministic transducer required to output y=0 on cue-only input following an unpaired history, and y=1 on that identical cue-only input following a paired history, needs at least two distinguishable internal states, hence at least one bit of mutable state if encoded by fixed-length bits.

*Proof.* A deterministic readout depending only on present input and a single identical state gives the same output in both cases. The required outputs differ, so the two histories must map to distinct internal states. One bit suffices by construction (1). QED.

This is a distinguishable-histories lower bound (Myhill–Nerode style) for the *specified behavioral contract*. General notions of biological learning and neural capacity are outside its scope.

## 2. Why this matters for hybrid TRM/LDT models

The repo's hybrid contract has an explicit typed lattice state a_t and a recurrent latent state h_t. Conditioning can live entirely in h_t, while a_t enforces what the system is authorized to do. Let

    proposal_t = R(h_t, observation_t)
    action_t = Gate(a_t, proposal_t),                              (2)

where Gate either returns an action in Allowed(a_t) or abstains. Assume Allowed(a_t) is supplied by a trusted typed-state mechanism, independently of the latent proposal.

**Theorem FWC-01/T2 (typed gate invariant).** If for every a,p,

    Gate(a,p) != abstain  =>  Gate(a,p) ∈ Allowed(a),               (3)

then any finite sequence of decisions made by (2) satisfies the per-step typed authorization condition regardless of how the recurrent state evolves.

*Proof.* Apply assumption (3) separately at each step. QED.

This is a boundary-enforcement theorem. It **does not** establish the fidelity of the trusted type system, environmental safety after a permitted action, robustness to compromised gate code, or goal alignment of h_t.

If the latent state is reinitialized between calls, a training effect in (1) vanishes. If recurrent hidden state is carried forward, "no weight update" should never be reported as "no learning"; history has changed the system's state.

## 3. Dynamics and transfer

Let S be the full hybrid machine state (including a,h), Z a proposed abstract task representation, and phi:S→Z. The companion [IDQ-01 note](https://github.com/MoralityLabAI/Compactification-Theory/tree/research/ingression-quotients-20261104/research/ingression_dynamics_v1) specifies controlled semiconjugacy

    phi(F_action(s)) = G_action(phi(s))

and a uniform finite-horizon stochastic approximation bound. In this repo the question is whether *one and the same* phi survives:
- held-out tasks and altered input encodings;
- resets of h versus carrying h across episodes;
- weight-frozen inference versus adapter updates;
- typed-permission updates; and
- damage or intervention in one recurrent submodule.

Projection learned on the evaluation outcomes is circular; fit phi and its hyperparameters on construction data only, then estimate commutation error on a separately generated intervention panel.

Existing controller-mesh experiments provide an important **negative control**: [the measured stalk bridge](controller_mesh_measured_stalk_bridge.md) did not show source-specific Qwen geometry benefit after typed weight-shuffle controls. Any new geometry-based controller should beat a label-only or source-scrambled null at matched compute.

## 4. Preregistered minimal trial

**Stage 0 (CPU, runnable here):** verify the circuit, one-bit lower bound witness, reset-induced loss, and typed gate invariant with dependency-free unit tests.

**Stage 1 (small recurrent network):** replace Boolean h with a learned dynamical h and freeze its weights. Train stimulus sequences, then test cue-only readout on held-out sequence variants, comparing shuffled history and hidden-state-reset controls. Record hidden-state bytes, all weight changes (expected zero), performance and calibration.

**Stage 2 (cross-substrate):** test a frozen abstract readout or quotient phi on a new digital state machine. Report micro intervention commutation error and target recovery relative to a surface-matched random recurrent baseline.

**Decision:** progress to a large-model SAE/VPD protocol only if Stage 1 shows retained history dependence under a reproducible state transfer and Stage 2 shows improvement under blind remapping. No current empirical benefit is claimed.

Run Stage 0: \`python -m unittest discover -s tests -p 'test_fixed_weight_conditioning.py' -v\`.
