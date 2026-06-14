# Checkpoint INTELLECT-3 Logic Env

## Completed

- Located the real local Prime Intellect logic environment fork.
- Removed the incorrect synthetic `int_tasks.py` scaffold.
- Added a dependency-light integration descriptor for `logic-env`.
- Added static verifier-task discovery from `logic_env/task2verifier.py`.
- Added an inspection CLI and generated report.
- Added tests that do not require HuggingFace, `verifiers`, or network access.
- Updated the benchmark plan to point `int-3` at the real `logic-env` integration.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.inspect_intellect_logic --out reports/intellect_logic_env.md --json-out data/benchmarks/intellect_logic_env.json
uv run vf-eval --env logic-env -n1 -r1 -d -v
```

## Files changed

```text
research_gym/envs/int_tasks.py
research_gym/adapters/intellect_logic.py
research_gym/adapters/__init__.py
research_gym/scripts/inspect_intellect_logic.py
tests/test_intellect_logic_adapter.py
docs/benchmark_plan.md
reports/intellect_logic_env.md
reports/checkpoint_intellect_logic.md
data/benchmarks/intellect_logic_env.json
```

## Tests

```text
58 passed
```

## Environment Inspection

```text
env_path: C:/projects/prime_intellect_research_environments/environments/logic_env
env_id: logic-env
dataset: PrimeIntellect/INTELLECT-3-RL
subset: logic
verifier_tasks: 37
```

## Smoke Command Result

The canonical command failed on Windows before environment loading:

```text
uv run vf-eval --env logic-env -n1 -r1 -d -v
ModuleNotFoundError: No module named 'fcntl'
```

The failure comes from `prime_tunnel`, which imports Unix-only `fcntl`. This is a platform/tooling blocker, not a benchmark adapter failure.

## Decisions made

- Do not model `int-3` as a synthetic integer task.
- Keep the integration static and dependency-light in this repo.
- Use the real fork path and verifier task map as the source of truth.
- Defer live `vf-eval` rollouts until WSL/Linux or a Windows-safe upstream dependency path is available.

## Open questions

- Whether to run `logic-env` through WSL for live smoke rollouts.
- Which subset of the 37 verifier tasks should become the first LDT/TRM/hybrid comparison slice.

## Recommended next step

Proceed down the benchmark list to routing, while leaving live INTELLECT-3 rollouts gated on Linux/WSL execution.
