# GPT-5.4-mini Operator Card

You are operating a starter research gym. Prefer small, reversible edits.

## First commands

```bash
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 64 --horizon 6 --seed 1
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
```

## Rules

- Do not change more than three files in one pass.
- Do not add heavy dependencies without writing why.
- Do not claim direct VPD weight editing exists.
- If a novelty claim lacks a citation, write `CITE_NEEDED`.
- Stop at checkpoint workorders.

## Handoff template

```text
Files changed:
Commands run:
Results:
Problems:
Next recommended step:
```
