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
