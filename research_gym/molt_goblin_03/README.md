# MOLT-GOBLIN-03: where does the four-module mixture advantage come from?

Follows MOLT-GOBLIN-02 (`research_gym/molt_goblin_02`, `adf4e17`). The registered design is in
`PREREG.md`; results are in `RESULTS.md` once finals complete.

| File | Role |
| --- | --- |
| `factory3.py` | Typed-module recurrence: single, cycle, mix, HRM, native sparse (straight-through, anneal), sequential; shared-weight modules; functional module objectives; auxiliary losses; distillation loss; intervention hooks; hard-dispatch inference path |
| `gym3.py` | New families: proof_resolve, experiment_portfolio, revision_retract, cex_repair, research_plan (finals only), local_map3, cumsum3 (and local_map) |
| `arms.py` | Panel, tournament arms A–K, matched-cost controls, loop-forcing factorial |
| `train3.py` | Training, evaluation, measured FLOPs (`FlopCounterMode`), latency, per-run peak RSS, teacher cache |
| `mech.py` | Specialization: activation CKA, gradient similarity, permutation, single-module ablation, routing interventions, effective rank, task-conditional routing |
| `audit.py` | MOLT-02 reproduction, causality, parameter and visit accounting, loop-forcing gradients |
| `campaign3.py` | Inky Mini campaign with canonical dedup, mechanistic receipts, successive halving, parallel workers |
| `finals3.py` | Freeze, 8-seed hidden-root finals, specialization, multitask routing, checkpoints |
| `analysis3.py` / `export3.py` | Registered contrasts, distillation retention, Pareto front; DataGoblin research-distillation export |
| `resources3.py` | Hard 3 GiB job-wide RAM limit covering worker processes |
| `launch.ps1` | Reproducible launcher for every stage |

```powershell
python -m pytest -q research_gym/molt_goblin_03
powershell -File research_gym/molt_goblin_03/launch.ps1 -Stage all
```

The Tinker credential comes from `TINKER_API_KEY` or a file named by `TINKER_API_KEY_FILE`.
