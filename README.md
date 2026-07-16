# LDT/TRM Research Gym

Starter repo for testing a hybrid Lattice Deduction Transformer / Tiny Recursive Model research program.

The gym is designed for small-agent operation. A GPT-5.4-mini-class agent should be able to run the toy tasks, generate frames, write checkpoint docs, and leave clean artifacts without needing the full Research_Engine stack.

## Core thesis

Use two states:

- `a_t`: explicit lattice or abstract deduction state. This is checkable, typed, and can support abstention or conflict labels.
- `h_t`: recurrent latent state. This is the TRM-style private workspace for heuristic memory, correlation tracking, and skill-regime inference.

The first included environment is a coupled-objective storyworld because it gives exact transitions and computable reachability labels. It is a gym substrate, not the whole research program.

## What this repo does now

- Generates typed reachability frames from a toy storyworld.
- Generates local MeTTa-like execution, deduction, repair, and routing frames.
- Labels conflicts as `live`, `env_sound_dead`, or `model_sound_dead`.
- Exports JSONL frames for TRM/LDT/SFT experiments.
- Includes a dependency-free hybrid LDT/TRM membrane stub: latent proposals must be projected into typed lattice refinements.
- Provides agent workorders for literature clearance, formalization, experiment specs, and starter implementation.
- Keeps VPD as instrumentation and decomposition, not as an assumed edit loop.
- Benchmarks RSITopology-aware authorization over matched TRM, LDT, and typed-hybrid skill sequences with paired
  statistics and raw receipts.
- Exports the frozen evaluation as a Verifiers `0.1.14` v1 Taskset/Harness replay package.
- Exports calibration-derived skill rules through the metta-storyworld AIRIS/DAS service contract, with the
  RSITopology membrane retaining final execution authority.
- Induces AIRIS-style rules independently from calibration utility winners and audits intact-rule precision,
  confidence demotion, and typed fallback behavior on held-out episodes.

## Quickstart

```bash
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
```

Optional tests:

```bash
python -m pytest tests
```

Cross-project game/control application scan and hybrid-structure benchmark:

```bash
python -m research_gym.scripts.bench_control_harnesses --projects-root C:\projects
```

This scans local repositories for candidate game/environment interfaces, then evaluates source-inspired
held-out proxy tasks for TRM, LDT, hard gate, confidence, typed, beam, and skill-routed hybrids. The proxy
results are not reported as native performance for the neighboring repositories.

Native TheySing enforcement benchmark:

```bash
python -m research_gym.scripts.bench_theysing_native --repo-root C:\projects\TheySing\TheySing
```

This launches the compiled TheySing headless harness and compares matched `soft`, `hard`, and `graduated`
enforcement sessions. Native traces are retained as compressed JSONL under the experiment directory.

RSITopology-aware HRM review contract:

```bash
python -m research_gym.scripts.review_model_training
```

This exercises target-blind geometry, lineage, holonomy, utility, damage, and resource receipt joins over a
deterministic candidate matrix. It emits authorize, section, audit, or reject routes. It does not train or mutate
a neural model.

Registered sequencer-control benchmark:

```bash
python -m research_gym.scripts.bench_sequencer_control
```

This compares global signed, lineage-only, fixed-typed, full control-math, and local-calibrated sequencing on 667
paired known-task episodes. The benchmark uses calibration-only fitting, equal-family macro weighting,
context-clustered hierarchical bootstrap intervals, clustered sign-flip tests, Holm correction, effect sizes, and exact McNemar audits. It measures control
over fixed deterministic skills, not neural weight infusion or official leaderboard performance.

Verifiers v1 replay contract:

```bash
python scripts/smoke_verifiers_v1.py
prime eval run hybrid-sequencer-v1 -c configs/eval/hybrid_sequencer_v1.toml
```

The smoke requires `verifiers==0.1.14`; use current Linux/WSL Prime tooling for the CLI path. See
`docs/verifiers_v1_harness.md` for local platform status.

AIRIS/DAS bridge and live local-service conformance:

```bash
python -m research_gym.scripts.bench_airis_das_bridge
python -m research_gym.scripts.bench_airis_das_resilience
python scripts/smoke_airis_das_bridge.py
prime eval run hybrid-sequencer-v1 -c configs/eval/hybrid_sequencer_airis_v1.toml
```

The benchmark seals AIRIS forecasts and topology decisions into the Verifiers taskset. The smoke separately
starts the implementation under `C:\projects\metta-storyworld\metta-etc`, verifies HTTP forecast parity, and
tests stale-protocol and altered-rule rejection. The resilience benchmark compares topology-only and
integrity-sealed arbitration over 6,670 paired clean/fault trials. See `docs/airis_das_integration.md`.

Independent calibration-outcome induction:

```bash
python -m research_gym.scripts.bench_airis_induction
python scripts/smoke_airis_das_bridge.py --rules data/airis_das/induced_rules.json --episodes data/benchmarks/sequencer_control_episodes.jsonl --bridge-results data/benchmarks/airis_induction_results.json --out data/airis_das/induced_live_service_smoke.json
```

This learns 28 context rules from 455 calibration episodes and evaluates them on 667 held-out episodes. It is a
negative-result baseline: rule-label accuracy is 94.0%, but every proposal equals `control_math`, so confidence
fallback has zero utility effect. See `docs/airis_induction.md`.

## Repo map

```text
configs/                  Agent and experiment configs
research_gym/core/         Lattice, hybrid membrane, frames, typed soundness primitives
research_gym/envs/         Toy gym environments and synthetic MeTTa directive reader
research_gym/scripts/      CLI scripts for frame generation and reports
tasks/agent_cards/         Operating instructions for small agents
tasks/workorders/          Checkpointed research tasks
docs/                      Design notes and starter paperspecs
data/                      Generated JSONL frames
papers/                    Drop external papers here, not committed by default
```

## Status

This is a scaffold. It is intentionally small and inspectable. The goal is to create a loop where small agents can make progress safely:

1. Generate frames.
2. Run symbolic checks.
3. Write or update one artifact.
4. Stop at checkpoints.
5. Hand off to a stronger model or human review.

## New in v0.0.2 scaffold

- `research_gym/core/hybrid.py`: candidate-set lattice, proposal object, and certifying membrane.
- `research_gym/core/metta_frames.py`: execution, deduction, repair, and routing frame dataclasses.
- `research_gym/scripts/generate_metta_frames.py`: JSONL generator for synthetic MeTTa-like directive files.
- `docs/hybrid_contract.md` and `docs/metta_frames.md`: operator-facing design contracts.

The repo still avoids claiming direct VPD weight editing. VPD is reserved for instrumentation and decomposition over trained components.
