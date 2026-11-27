# JINN-GYM-01: overnight LDT research-persona engineering

Frozen thinkingmachines/Inkling-2-Mini (free tier) does three jobs: it engineers eight research personas as typed specifications, makes every research decision, and proposes every repair. Each persona lives outside the model, in an exact MeTTa-grammar rubric, a learned additive LDT head and a research action policy. Every executed action passes the host-owned Beast boundary, which reuses Dual-LDT-01 `BeastSnapshot` receipts and the CCA-06 `FiniteRetraction`.

* [PREREGISTRATION.md](PREREGISTRATION.md): design, arms, metrics, hypotheses and decision rules, registered before the live run.
* [registration.json](registration.json): configuration and per-file code hashes at launch.
* [DEVIATIONS.md](DEVIATIONS.md): changes made after launch.
* [RESULTS.md](RESULTS.md): measured results, limitations and null results.
* `outputs/run_20261128/`: the live run (see the file list in RESULTS.md).

Code is in `research_gym/jinn_gym_01/`; tests are in `tests/test_jinn_gym_01.py`.

```
python -m pytest -q tests/test_jinn_gym_01.py
python -m research_gym.jinn_gym_01.runner --out OUT --hours 0.3 --dry-run          # offline, fake model
python -m research_gym.jinn_gym_01.runner --out OUT --hours 8 --key-file KEYFILE    # live; rerun to resume
python -m research_gym.jinn_gym_01.report OUT
```

The key is read from `TINKER_API_KEY` or from `--key-file`, and is never logged or written. A `STOP` file in `OUT` stops the run cleanly, and rerunning the same command resumes it with the same deadline.

## Reused implementations

| Source | Used for |
|---|---|
| jinn-or-beast-paper `two_frame_metta_ldt_v1/src/ldt_experiment.py` (vendored verbatim, SHA-256 pinned) | rubric grammar, `parse_policy`, `score_policy`, `AdditiveCandidateHead`, `train_head` |
| jinn-or-beast-paper `ldt_heads_v3_critical_veto` | critical-tag veto semantics (`heads.VetoedHead`) |
| jinn-or-beast-paper `character_ldt_heads_v1` | head keeps unless top lane is reject; every head sees every tag; boundary never trained; shuffled control |
| jinn-or-beast-paper `ldt_titration_v1` | dose-to-lane view of persona strength (fixed strength 2.0 here) |
| HybridTRMLDT `research_gym/core/dual_ldt.py` (Dual-LDT-01, the corrigibility line of dual_ldt_03) | `BeastSnapshot`, receipts, `verify_execution`; `sort_with_jinn` projection (equality tested) |
| HybridTRMLDT `research_gym/core/cca_compositional.py` (CCA-06) | `FiniteRetraction` repair, commutation and composition; `scalar_persona_metrics` Fisher retention |
| HybridTRMLDT CCA-08 | repair cost and information-loss reporting |
| DataGoblin `Agonist/skills/agonist02` | role split (Inky builder, independent verifier, learner), lineage, preserving invalid builder outputs |
| LoopedTransformers `ink_fp_audit_01/tinker_remote.py` | free-tier OpenAI-compatible route, generic User-Agent, no-redirect credential handling |
| LoopedTransformers `ink_fp_audit_02/exports.py` | DataGoblin rlt_gym row shape |
| SelfModels `nafsprobe` (main 8274a45) | memory as stored verified-outcome receipts; arm F against D is the memory ablation; composition reported as adapter-interference risk (`smtk_starter_repo` adapter routing) |
