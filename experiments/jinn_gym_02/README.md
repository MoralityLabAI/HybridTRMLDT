# JINN-GYM-02: utility-constrained persona evolution and compositional routing

This experiment continues [JINN-GYM-01](../jinn_gym_01/README.md) from research/jinn-gym-01@5ce0517. It asks whether a population of externally evolving LDT research personas, routed by a compact TRM, improves the verified research competence of frozen Inkling-2-Mini.

On untouched held-out families, it does not: the TRM-routed population scored −0.038 [−0.076, −0.001] in normalized verified utility against neutral Inky. See [RESULTS.md](RESULTS.md).

| file | contents |
|---|---|
| [audit_gym01/AUDIT.md](audit_gym01/AUDIT.md) | independent reconstruction of GYM-01 from raw records, the abstention-repair decomposition and the winner's curse |
| [PREREGISTRATION.md](PREREGISTRATION.md) | design, splits, selection rule (2-point noninferiority margin), arms, hypotheses |
| `registration.json` | launch configuration and code hashes |
| [DEVIATIONS.md](DEVIATIONS.md) | D1 (verifier timeout during BUILD), D1a (CPU cap), D2 (numeric bounds in the patch prompt, partial-round recovery) |
| [RESULTS.md](RESULTS.md) | measured results, null and negative findings, limitations |
| `outputs/run_20261128/` | raw receipts, mutation histories, archive, router, self-models, compactification, DataGoblin `export/` |

Code is in `research_gym/jinn_gym_02/`; tests are in `tests/test_jinn_gym_02.py`. The Lean verifier needs a Lean 4 toolchain under `~/.elan/toolchains`, or `JINN_LEAN=<path to lean>`.

```
python -m pytest -q tests/test_jinn_gym_02.py tests/test_jinn_gym_01.py
python -m research_gym.jinn_gym_02.campaign --out OUT --hours 1 --dry-run --small        # offline, fake model
python -m research_gym.jinn_gym_02.campaign --out OUT --hours 8 --key-file KEYFILE        # live; rerun to resume
python -m research_gym.jinn_gym_02.report OUT
```
