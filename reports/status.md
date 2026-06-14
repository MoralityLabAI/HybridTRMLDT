# Status

## Repo Tree

```text
.
├── configs/
├── data/
│   ├── frames.jsonl
│   ├── metta_frames.jsonl
│   └── symbolic_report.md
├── docs/
├── examples/
│   └── metta_rules/
├── research_gym/
│   ├── core/
│   ├── envs/
│   └── scripts/
├── tasks/
│   ├── agent_cards/
│   ├── generated/
│   └── workorders/
├── tests/
├── Makefile
├── README.md
└── pyproject.toml
```

## Commands Run

```text
make all
pytest -q
python --version
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
python -m pytest tests -q
```

## Test Results

```text
python -m pytest tests -q
8 passed in 6.02s
```

The Makefile target bodies also passed when run directly through Python.

## Known Broken Pieces

```text
make all
```

failed because `make` is not installed or not on PATH in the current Windows shell.

```text
pytest -q
```

failed because `pytest` is not available as a direct shell executable. The module form works:

```text
python -m pytest tests -q
```

This appears to be an environment/tooling gap, not a project test failure.

## Next Recommended Patch

Proceed to Task 2 from the repo instructions:

```text
Strengthen the hybrid membrane in research_gym/core/hybrid.py,
research_gym/core/typed_soundness.py, and research_gym/core/lattice.py.
```

Required deliverables:

```text
tests/test_hybrid_membrane.py
docs/hybrid_contract.md update
reports/checkpoint_2.md
```
