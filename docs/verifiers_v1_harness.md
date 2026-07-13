# Verifiers v1 Harness Migration

## Target

The local replay environment targets Prime Intellect Verifiers `0.1.14`, the release that introduced the v1
Taskset/Harness API. It is intentionally packaged separately from the dependency-free research gym core.

## Ownership Split

The package at `environments/hybrid_sequencer_v1/` follows the v1 composition model:

- `HybridSequencerTaskset` owns immutable replay tasks, utility reward, and correctness/cost/violation/control-bound
  metrics.
- `HybridSequencerHarness` owns the `global_signed`, `lineage_only`, `fixed_typed`, `control_math`, and
  `local_calibrated` rollout programs.
- `load_taskset`, `load_harness`, and `load_environment(vf.EnvConfig)` provide strict typed loaders.
- `vf.Env` composes the taskset and harness so one task definition can be evaluated under multiple sequencers.

The program is LLM-free: it selects one saved candidate outcome and emits a hashable control receipt. That makes
the v1 package a harness-integration test and portable replay artifact, not a model evaluation.

## Generated Data

`python -m research_gym.scripts.bench_sequencer_control` regenerates
`environments/hybrid_sequencer_v1/data/replay_tasks.jsonl`. Every row includes the candidate outcomes, frozen
sequencer choices, context topology receipt, and expected answer for one evaluation episode.

## Prime Evaluation

Use Linux or WSL with current Prime tooling:

```bash
prime env install hybrid-sequencer-v1
prime eval run hybrid-sequencer-v1 -c configs/eval/hybrid_sequencer_v1.toml
```

The checked-in config runs 50 `story_secret` examples under `control_math`. Change only `eval.harness.sequencer`
to compare policies over the same taskset.

## Local Validation Status

- Source/package contract tests pass against the v1 Taskset/Harness shape from the `0.1.14` tag.
- Replay data contains 667 immutable evaluation tasks and is included by Hatch packaging.
- The installed global Verifiers version is `0.1.11.dev1`, so it cannot import `verifiers.v1`.
- The installed global `uv` is `0.8.17`; the tagged Verifiers project requires `uv>=0.11.1`.
- An isolated `uv 0.11.1` was bootstrapped on `D:` and used for an exact `verifiers==0.1.14` smoke attempt.
- The exact package environment did not finish dependency installation within the bounded three-minute attempt;
  Python execution was never reached, so this is not evidence of an API failure.
- Native Windows `prime` remains blocked because the installed legacy `prime_tunnel` imports Unix-only `fcntl`.

Run the exact-release smoke when package access is available:

```powershell
uv run --isolated --with "verifiers==0.1.14" python scripts/smoke_verifiers_v1.py
```

The repository adapter exposes installed-version, minimum-`uv`, package-completeness, and native-Windows blocker
checks for automation.
