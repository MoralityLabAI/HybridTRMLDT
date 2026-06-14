# Checkpoint 5

## Completed

- Expanded the synthetic MeTTa directive corpus with mechanically distinct cases:
  - arithmetic
  - comparison
  - type checking
  - precondition failure
  - postcondition update
  - routing among named skills
  - repair of malformed rule
  - deduction closure
  - conflict and bottom detection
- Updated `examples/metta_rules/toy_skills.metta`.
- Updated MeTTa frame documentation.
- Added dedicated synthetic generator tests.
- Regenerated legacy and common-schema MeTTa frame artifacts.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.eval_frames --source data/generated --report reports/baseline_eval.md
```

## Files changed

```text
research_gym/envs/metta_synthetic.py
examples/metta_rules/toy_skills.metta
tests/test_metta_synthetic.py
docs/metta_frames.md
reports/checkpoint_5.md
data/metta_frames.jsonl
data/generated/metta_frames.jsonl
data/frames.jsonl
data/generated/reachability_frames.jsonl
data/symbolic_report.md
reports/baseline_eval.md
tasks/generated/00_agent_protocol.md
tasks/generated/01_lit_clearance.md
tasks/generated/02_formalization.md
tasks/generated/03_experiment_specs.md
tasks/generated/04_implementation_next.md
```

## Tests

```text
30 passed in 6.41s
```

Generated MeTTa frame counts:

```text
deduction: 5
execution: 6
repair: 2
routing: 3
total: 16
```

Baseline report:

```text
deduction_frame_exact_match: 1.000 (5/5)
transition_frame_exact_match: 1.000 (134/134)
conflict_precision: 1.000 (2/2)
conflict_recall: 1.000 (2/2)
typed_attribution_accuracy: 1.000 (144/144)
Frames evaluated: 144
```

## Decisions made

- Kept the synthetic generator interpreter-free.
- Represented precondition failure as an execution frame with `passed=false` and unchanged state.
- Represented type checking, deduction closure, and bottom detection as deduction frames.
- Used empty candidate lists to represent bottom in the directive lattice syntax.

## Open questions

- Whether Task 6 SFT export should consume only common-schema files or also preserve specialized legacy frame fields in metadata.

## Recommended next step

Proceed to Task 6: add model-ready SFT dataset export from generated common frames.
