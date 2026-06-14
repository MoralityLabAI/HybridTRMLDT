# Checkpoint Routing Bench

## Completed

- Located the existing Tesseract TRM router reference:
  - `C:/projects/Tesseract/Tesseract/scripts/train_router_trm.py`
- Confirmed Tesseract's TRM router is an MLP over TF-IDF prompt features trained to predict environment IDs.
- Added dependency-light env-pointer routing benchmark in this repo.
- Added three trainable routing models:
  - `ldt`: token lattice router that refines candidate environment IDs.
  - `trm`: lexical TRM analogue over bag-of-token evidence.
  - `hybrid`: LDT candidates as soft guidance plus TRM scoring.
- Saved experiment outputs and notes under `experiments/routing_env_pointer/`.
- Updated benchmark plan.

## Commands run

```text
python -m pytest tests/test_routing_bench.py -q
python -m research_gym.scripts.bench_routing --max-per-env 80 --out data/benchmarks/routing_results.json --report reports/routing_bench.md --experiment-dir experiments/routing_env_pointer
python -m pytest tests -q
git rev-parse --show-toplevel
```

## Files changed

```text
research_gym/envs/routing.py
research_gym/benchmarks/routing_bench.py
research_gym/scripts/bench_routing.py
tests/test_routing_bench.py
docs/benchmark_plan.md
reports/routing_bench.md
reports/checkpoint_routing_bench.md
data/benchmarks/routing_results.json
experiments/routing_env_pointer/results.json
experiments/routing_env_pointer/training_notes.md
```

## Tests

```text
63 passed
```

## Results

```text
train examples: 402
test examples: 173
ldt: 0.538 accuracy, 93/173 correct, 15 abstained
trm: 0.815 accuracy, 141/173 correct
hybrid: 0.815 accuracy, 141/173 correct, avg LDT candidate set 2.09
```

## Decisions made

- Did not import Tesseract's PyTorch/sklearn router directly to keep this repo dependency-light.
- Mirrored the Tesseract objective: classify `state_prompt` into an environment ID.
- Treated LDT token candidates as model/experience-derived routing evidence, not environment-sound eliminations.
- Hybrid therefore uses LDT candidates as soft guidance and does not hard-eliminate the TRM top route when LDT disagrees.
- Saved model training notes locally in `experiments/routing_env_pointer/training_notes.md`.

## Commit status

No commit was created because `C:/projects/HybridTRMLDT` is not currently inside a git repository:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If commits are required, initialize or point me to the intended git repo root.

## Open questions

- Whether to run a real Tesseract `train_router_trm.py` job and import its checkpoint outputs into this benchmark.
- Whether the next hybrid should add a margin rule: use LDT candidates only when TRM confidence is low.

## Recommended next step

Proceed to storyworld playing benchmark, or initialize a git repo first if every turn must end in a commit.
