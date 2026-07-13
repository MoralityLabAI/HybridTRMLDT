# Checkpoint: TheySing Native Adapter

## Completed

- Added a process-safe adapter for the compiled TheySing headless harness.
- Used native HTTP session, run, and manual-turn endpoints.
- Ran matched soft, hard, and graduated enforcement structures across three game scenarios and two seeds.
- Added a native bilateral non-aggression pressure probe to distinguish graduated from hard enforcement.
- Saved aggregate metrics, per-scenario metrics, experiment notes, and 24 losslessly compressed native traces.

## Commands run

```powershell
python -m research_gym.scripts.bench_theysing_native --seeds 400,401 --turns 2
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests -q
```

## Files changed

- `research_gym/adapters/theysing.py`
- `research_gym/benchmarks/theysing_native_bench.py`
- `research_gym/scripts/bench_theysing_native.py`
- `tests/test_theysing_native_bench.py`
- `docs/theysing_native_adapter.md`
- native experiment and report artifacts

## Tests

- Native trace scorer focused test passed.
- Full suite: 84 passed.

## Decisions made

- The adapter calls native mechanics rather than recreating TheySing state transitions.
- Full traces are retained as gzip-compressed JSONL to avoid unnecessary repository growth.
- Safety, order acceptance, and sanctions are reported separately.
- A forced pressure probe is explicitly labeled and not mixed up with heuristic game behavior.
- VPD is not part of this experiment.

## Open questions

- Does graduated enforcement improve long-horizon strategic outcomes or only preserve short-run action latitude?
- Which pact/action typing produces the best frontier under learned or adversarial agents?
- Can confidence arbitrate within the graduated set without allowing confident institutional breaches?

## Recommended next step

Run longer high-pressure and Babel sessions with a learned proposal policy, preserving the same native trace and
enforcement scorer contract.
