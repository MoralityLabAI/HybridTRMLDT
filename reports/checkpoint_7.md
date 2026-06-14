# Checkpoint 7

## Completed

- Added `MettaAdapterProtocol`.
- Added `ParsedMettaProgram` placeholder.
- Added directive-backed `FakeMettaAdapter`.
- Added adapter package exports.
- Added tests for protocol shape, typecheck, execute, and frame extraction.
- Added real MeTTa integration documentation.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.export_sft_dataset --source data/generated --out data/sft/hybrid_frames.jsonl
python -m research_gym.scripts.eval_frames --source data/generated --report reports/baseline_eval.md
```

## Files changed

```text
research_gym/adapters/__init__.py
research_gym/adapters/metta_adapter.py
tests/test_metta_adapter.py
docs/real_metta_integration.md
reports/checkpoint_7.md
data/sft/hybrid_frames.jsonl
reports/baseline_eval.md
```

## Tests

```text
38 passed in 6.85s
```

SFT export:

```text
wrote 144 SFT records to data/sft/hybrid_frames.jsonl
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

- The fake adapter validates directives by attempting frame extraction.
- Fake execution returns a state wrapper and does not claim real MeTTa semantics.
- The protocol preserves an opaque parsed-program boundary so a real runtime can be added later without rewriting downstream frame pipelines.

## Open questions

- Which concrete MeTTa runtime API should back the first real adapter implementation.
- Whether real adapter typecheck diagnostics should emit repair frames directly.

## Recommended next step

The AGENTS task queue through Task 7 is complete. Next work should follow the evaluation ladder: E0 through E4 before any VPD decomposition work.
