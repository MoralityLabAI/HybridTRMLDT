# Checkpoint: Control Harness and Game Applications

## Completed

- Scanned 130 local project directories with bounded, dependency-tree-safe discovery.
- Produced a broad candidate inventory and a targeted 12-project application shortlist.
- Added nine source-inspired control applications spanning game choice, diplomacy, provenance, protocol control,
  strategic simulation, and skill delegation.
- Implemented TRM, LDT, hard gate, confidence arbitration, typed membrane, typed confidence, counterfactual beam,
  and train-split skill-router structures.
- Selected confidence threshold and beam width on train cases and evaluated on disjoint held-out cases.
- Saved task cards, decisions, aggregate metrics, inventory, experiment notes, and reports.

## Commands run

```powershell
python -m research_gym.scripts.bench_control_harnesses --n-train 48 --n-eval 48 --seed 23
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests -q
```

`make all` could not run because GNU Make is not installed. Its frame generation, MeTTa generation, symbolic
evaluation, task generation, and pytest commands were run directly with the same arguments.

## Files changed

- `research_gym/discovery/project_inventory.py`
- `research_gym/discovery/application_map.py`
- `research_gym/envs/control_tasks.py`
- `research_gym/benchmarks/control_harness_bench.py`
- `research_gym/scripts/bench_control_harnesses.py`
- `tests/test_project_inventory.py`
- `tests/test_control_harness_bench.py`
- `docs/control_harness_experiments.md`
- generated benchmark, report, and experiment artifacts

## Tests

- Focused tests: 4 passed.
- Full suite: 82 passed.
- Existing frame/task generators completed successfully.

## Decisions made

- Neighboring repositories are scanned and cited as application sources, but proxy scores are not presented as
  native repository performance.
- Only environment-derived exclusions receive hard authority.
- Confidence and beam parameters are trained rather than chosen from held-out results.
- Skill routing is treated as part of the hybrid architecture, not merely an evaluation convenience.
- VPD is not introduced into this experiment.

## Open questions

- Does the learned skill split survive native multi-turn traces and distribution shift?
- Can a dual-clock controller reduce exact-check frequency without increasing unsafe actions?
- Does counterfactual beam become more useful on longer strategic horizons despite its current cost?

## Recommended next step

Implement the TheySing bridge-policy adapter first, then compare hard, soft, and graduated treaty scenarios using
the same decision receipt and unsafe-action metrics.
