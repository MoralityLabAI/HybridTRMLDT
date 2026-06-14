# Checkpoint 4

## Completed

- Added dependency-free symbolic baseline metrics.
- Added common-frame evaluation CLI.
- Added generated baseline report.
- Added tests for exact match, conflict precision/recall, membrane decision accuracy, typed attribution accuracy, and common-frame summaries.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.eval_frames --source data/generated --report reports/baseline_eval.md
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
```

## Files changed

```text
research_gym/eval/__init__.py
research_gym/eval/baselines.py
research_gym/scripts/eval_frames.py
tests/test_eval_baselines.py
reports/baseline_eval.md
reports/checkpoint_4.md
data/frames.jsonl
data/metta_frames.jsonl
data/generated/reachability_frames.jsonl
data/generated/metta_frames.jsonl
data/symbolic_report.md
```

## Tests

```text
24 passed in 6.26s
```

Baseline report:

```text
deduction_frame_exact_match: 1.000 (2/2)
transition_frame_exact_match: 1.000 (130/130)
conflict_precision: 1.000 (1/1)
conflict_recall: 1.000 (1/1)
typed_attribution_accuracy: 1.000 (134/134)
Frames evaluated: 134
```

## Decisions made

- Baselines are deterministic oracle checks over generated labels for now.
- The evaluator consumes common-schema frames, not legacy specialized files.
- Membrane decision accuracy is exposed over `HybridStepResult` sequences; common-frame membrane data can plug in later when Task 6 or SFT export produces those records.

## Open questions

- Whether Task 5 should emit explicit membrane-decision common frames so `membrane_decision_accuracy` can be reported from `data/generated` without constructing `HybridStepResult` objects in tests.

## Recommended next step

Proceed to Task 5: expand the synthetic MeTTa frame generator with more mechanically distinct cases.
