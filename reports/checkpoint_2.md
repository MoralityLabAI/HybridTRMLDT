# Checkpoint 2

## Completed

- Added explicit `MembranePolicy` for hard-apply permissions.
- Added `HybridProposal` and `HybridDecision` aliases over the existing proposal/result records for the strengthened interface.
- Added optional soft-store payloads for rejected model-sound and experience-sound proposals.
- Added monotonicity checks so proposals that widen candidate sets are rejected.
- Added round-trip serialization for proposals, decisions, and policies.
- Updated the hybrid contract documentation.
- Added focused membrane tests.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
```

## Files changed

```text
research_gym/core/hybrid.py
research_gym/core/__init__.py
tests/test_hybrid_membrane.py
docs/hybrid_contract.md
reports/status.md
reports/checkpoint_2.md
data/frames.jsonl
data/metta_frames.jsonl
data/symbolic_report.md
tasks/generated/00_agent_protocol.md
tasks/generated/01_lit_clearance.md
tasks/generated/02_formalization.md
tasks/generated/03_experiment_specs.md
tasks/generated/04_implementation_next.md
```

## Tests

```text
17 passed in 6.19s
```

Direct `make all` and bare `pytest -q` remain unavailable in this shell because `make` and `pytest` are not on PATH. The Python module equivalents pass.

## Decisions made

- `unknown` proposals are rejected and not soft-stored by default.
- Model-sound and experience-sound proposals are rejected as durable deductions by default but soft-stored when `MembranePolicy.store_soft_proposals` and proposal-level `soft_store` are enabled.
- Explicit policy flags are separate for model-derived and experience-derived proposals.
- Non-monotone proposals are rejected before `meet` to avoid hiding widening proposals as no-ops.

## Open questions

- Whether future soft stores should persist to a separate trace file or remain in-memory decision payloads until the frame schema work lands.

## Recommended next step

Proceed to Task 3: generalize frame schemas under a common JSONL-compatible `Frame` shape.
