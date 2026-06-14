# Checkpoint 6

## Completed

- Added model-ready SFT dataset export script.
- Added SFT export documentation.
- Added tests for decision targets, record structure, source loading, and JSONL writing.
- Exported `data/sft/hybrid_frames.jsonl` from common-schema generated frames.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.export_sft_dataset --source data/generated --out data/sft/hybrid_frames.jsonl
```

## Files changed

```text
research_gym/scripts/export_sft_dataset.py
docs/sft_export.md
tests/test_sft_export.py
reports/checkpoint_6.md
data/sft/hybrid_frames.jsonl
```

## Tests

```text
34 passed in 7.01s
```

SFT export:

```text
wrote 144 SFT records to data/sft/hybrid_frames.jsonl
```

## Decisions made

- SFT input is the full common-frame JSON encoded as a string.
- SFT output is a compact typed decision target containing family, soundness type, label, and output state.
- Metadata preserves id, family, source, and soundness type for filtering.

## Open questions

- Whether later exports should add prompt variants per frame family.
- Whether membrane-specific decision records should be generated explicitly before tiny-model training.

## Recommended next step

Proceed to Task 7: add a real MeTTa adapter protocol and fake adapter tests without requiring a real MeTTa runtime.
