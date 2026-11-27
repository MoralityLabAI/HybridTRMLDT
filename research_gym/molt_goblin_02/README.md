# MOLT-GOBLIN-02: Mixture-of-Loops architecture discovery

Companion to MOLT-GOBLIN-01 (`LoopedTransformers/experiments/molt_goblin_v1`, DataGoblin
`molt_goblin`/`molt_retention`/`molt_episode`, and this repository's
`docs/molt_prefix_memory_hybrid_v1.md` and `docs/hrm_conductor_architecture.md`).

This is an independent, small, source-inspired PyTorch implementation. It is not the
upstream MoLT code: there is no distinct initial pass and no upstream kernel. The
`prefix`/`prev` arms reproduce the MOLT-GOBLIN-01 fixture block exactly (tested forward
and gradient parity against a vendored copy of `pilot.py`).

## Modules

| File | Role |
| --- | --- |
| `factory.py` | `MoltConfig` restricted schema and `MoltNet`; memory axis (tied, prev, prefix, full, topk, lowrank, retention, confidence bias, window, loop keys) and routing axis (single, cycle/untied, mix, HRM slow/fast manager), halting, update rules, LDT masks, memory interventions |
| `gym.py` | Eight exact task families with independent solvers, sound admissibility masks and `verify_spec` |
| `train.py` | CPU training and evaluation with per-loop trajectories, interventions, halting calibration, routing statistics |
| `interventions.py` | BPD-style loop-update subspace projections with random-subspace and complement controls |
| `panel.py` | Registered development panel, controls, withheld families, baselines, rungs |
| `inky.py` | Free Inkling-2-Mini client (free id only, bounded retries, context guard) |
| `campaign.py` | Overnight discovery loop with successive halving and adversarial task variants |
| `finals.py` | Freeze finalists, hidden-root and withheld-family evaluation, checkpoints |
| `analysis.py` | Paired seed-level comparisons, root bootstrap, Pareto frontier |
| `export.py` | DataGoblin-compatible trajectory records |
| `stage_a.py` | Stage A reproduction summary |
| `calibrate.py` | Competence calibration of panel specs |
| `report.py` | Hypothesis, architecture and adversarial-task archives from a campaign state |
| `reference_pilot.py` | Verbatim MOLT-GOBLIN-01 `pilot.py` for parity tests |

## Commands

```sh
python -m pytest -q research_gym/molt_goblin_02
python -m research_gym.molt_goblin_02.calibrate --panel --steps 300 --arms prefix tied --out <dir>/calib.jsonl
# credential via TINKER_API_KEY or TINKER_API_KEY_FILE (path to a file holding the key)
python -m research_gym.molt_goblin_02.campaign --out <run> --hours 5.5 --threads 2
python -m research_gym.molt_goblin_02.finals --state <run>/state.json --out <finals> --hours 2
python -m research_gym.molt_goblin_02.analysis --runs <finals>/final_runs*.jsonl --out <finals>
python -m research_gym.molt_goblin_02.export --state <run>/state.json --out <run>/datagoblin \
    --analysis <finals>/analysis.json --freeze <finals>/freeze.json
```

See `PREREG.md` for the registered comparison and `RESULTS.md` for what ran.
