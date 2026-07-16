# Hybrid LDT/TRM Contract

The repo treats the hybrid as a two-state reasoner, not as a VPD edit loop.

- `a_t`: explicit lattice or candidate state. It is public, serializable, and checkable.
- `h_t`: TRM-style latent recurrent state. It is private workspace for proposal, memory, correlation tracking, and skill-regime inference.

The contract is:

```text
TRM proposes a typed lattice refinement.
The lattice membrane accepts only certified hard refinements by default.
Soft provenance can be logged, branched on, or used for training, but not silently converted into hard deduction.
```

## Core interface

The membrane is represented by:

- `HybridProposal` / `LatticeProposal`: proposed lattice refinement plus typed provenance.
- `MembranePolicy`: explicit flags for whether model-sound or experience-sound proposals may hard-apply.
- `HybridDecision` / `HybridStepResult`: accepted/rejected decision, before/after states, rejection reason, and optional soft-store payload.

These records serialize through `to_jsonable()` and `from_jsonable()` so membrane traces can become training or audit frames.

`MembranePolicy.provenance_verifier` is an optional dependency-free seam. A verifier receives the proposal and
an environment-owned context. If it returns a `SoundnessType`, that verified type governs gating and the result
records claimed provenance, verified provenance, and disagreement. If it returns `None`, the historical claim
path is used. With no verifier configured, decision behavior and serialized trace keys are unchanged.

## Claimed provenance is an attack surface

Provenance is evidence only when its source has authority for the property. A proposer-authored `env_sound_dead`
label is a claim, not an environment proof. The recovered expert-iteration experiment found that strict gating
increased false environment-sound claims while ground-truth soundness stayed flat. The gaming-versus-improvement
benchmark reproduces the direction in all three seeds.

Reference verifier roles are intentionally separated:

- `exact_mechanics_verifier` adapts an environment-owned mechanics callback and has authority only for the
  property that callback computes.
- `probe_verifier` adapts a latent linear probe. It is level-3 evidence: it can gate through the explicit seam,
  create disagreement, or trigger fallback, but it never silently overrides exact mechanics.
- dual-channel policies require claim/checker agreement and route disagreement to rejection or abstention.

The earlier `allow_model_sound` permissive arm was behaviorally identical to no membrane on its task. A nominal
policy name is not evidence of a new controller; accepted and executed action sets must differ before cross-arm
statistics are interpreted.

## Provenance rules

- `env_sound_dead`: can be a hard elimination when derived from engine/interpreter mechanics.
- `model_sound_dead`: rejected by default as a durable update and soft-stored by default; hard-applies only when `MembranePolicy.allow_model_sound` is enabled.
- `experience_sound_dead`: rejected by default as a durable update and soft-stored by default; hard-applies only when `MembranePolicy.allow_experience_sound` is enabled.
- `unknown`: rejected and not soft-stored by default because it has no usable provenance.

## Monotonicity

Accepted proposals must be monotone refinements of the current public lattice state.

For `CandidateState`, every proposed candidate set must be a subset of the corresponding current candidate set. A proposal that introduces a new candidate widens the state and is rejected before any meet is applied.

Bottom-producing proposals are rejected unless the proposal explicitly uses `HybridMode.ABSTAIN`. This makes conflict/bottom detection visible instead of silently converting it into normal deduction.

## Why this matters

This keeps the hybrid honest. A latent recurrent system may discover useful proposals, but durable state updates pass through a typed interface. That preserves the LDT benefit while testing whether TRM-style cross-step latent state improves lossy or partial abstractions.
