# Checkpoint 3

## Completed

- Added common `Frame` schema with JSONL read/write helpers.
- Added `to_frame()` exporters for reachability, execution, deduction, repair, and routing frames.
- Kept legacy specialized JSONL formats unchanged.
- Updated generators to emit common-schema sidecars under `data/generated/`.
- Added frame schema documentation.
- Added tests for common-schema export and round-trip behavior.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
```

## Files changed

```text
research_gym/core/frames.py
research_gym/core/metta_frames.py
research_gym/core/__init__.py
research_gym/scripts/generate_frames.py
research_gym/scripts/generate_metta_frames.py
tests/test_frame_schema.py
docs/frame_schema.md
reports/checkpoint_3.md
data/frames.jsonl
data/metta_frames.jsonl
data/generated/reachability_frames.jsonl
data/generated/metta_frames.jsonl
data/symbolic_report.md
tasks/generated/00_agent_protocol.md
tasks/generated/01_lit_clearance.md
tasks/generated/02_formalization.md
tasks/generated/03_experiment_specs.md
tasks/generated/04_implementation_next.md
```

## Tests

```text
19 passed in 6.14s
```

Symbolic frame evaluation after regeneration:

```text
Total frames: 128
env_sound_dead: 58
live: 28
model_sound_dead: 42
```

## Decisions made

- The common frame schema is an adapter layer, not a replacement for specialized dataclasses.
- Legacy `data/frames.jsonl` and `data/metta_frames.jsonl` remain stable for existing scripts.
- `unknown` is used for execution, repair, and routing frames unless a specialized generator has typed provenance.
- `data/generated/reachability_frames.jsonl` and `data/generated/metta_frames.jsonl` are the first common-schema outputs.

## Open questions

- Whether all future scripts should consume only common frames or continue supporting specialized frame files in parallel.

## Recommended next step

Proceed to Task 4: add symbolic baseline evaluators for exact-match, conflict precision/recall, membrane decision accuracy, and typed attribution accuracy.
