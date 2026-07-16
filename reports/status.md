# Status

Date: `2026-07-16`

## Repo Tree

```text
configs/                       benchmark and Verifiers eval configs
data/
  airis_das/                   generated AIRIS rules and live service receipt
  benchmarks/                  aggregate and per-trial benchmark artifacts
docs/                          architecture and integration contracts
environments/hybrid_sequencer_v1/
experiments/                   mirrored results and training notes
papers/
  overleaf/                    TeX, bibliography, and TikZ figures
reports/                       benchmark summaries and checkpoints
research_gym/
  adapters/                    MeTTa, Intellect, Verifiers, and AIRIS/DAS adapters
  benchmarks/                  task, architecture, sequencer, and resilience studies
  core/                        lattice, membrane, frames, and training review
  envs/                        synthetic and local task environments
  scripts/                     reproducible artifact entry points
scripts/                       integration smoke tests
tests/                         dependency-light regression suite
```

## Commands Run

```text
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
python -m research_gym.scripts.bench_airis_das_bridge
python -m research_gym.scripts.bench_airis_das_resilience
python scripts/smoke_airis_das_bridge.py
python -m research_gym.scripts.bench_airis_induction
python scripts/smoke_airis_das_bridge.py --rules data/airis_das/induced_rules.json --episodes data/benchmarks/sequencer_control_episodes.jsonl --bridge-results data/benchmarks/airis_induction_results.json --out data/airis_das/induced_live_service_smoke.json
python -m pytest -q
```

## Test Results

```text
117 passed in 15.24s
```

The AIRIS/DAS live HTTP smoke passed with 28 rules and 392 indexed facts. Embedded and HTTP forecasts selected
the same rule. Stale protocol and altered rule material both fell back to `control_math`.

The resilience benchmark contains 6,670 paired trials. Integrity sealing retained 100% clean acceptance and
rejected all 6,003 negative controls. Topology-only arbitration accepted 44.4% of negative controls.

The independent learner induced 28 rules from 455 calibration episodes. Held-out episode-winner precision is
94.0%, with all 40 errors in routing and storyworld contexts. Every proposal matches `control_math`, so confidence
demotion changes acceptance but leaves macro utility fixed at `0.900085`. The induced live-service smoke passed
with 28 rules and 420 indexed facts.

## Known Broken Pieces

- GNU Make is not installed on this Windows host, so `make all` cannot be invoked directly. Every target body was
  run through its Python command and passed.
- No local `pdflatex` or `latexmk` executable is installed. TeX citations and figure inputs are mechanically
  complete, but local PDF compilation is not verified.
- The metta-storyworld service reports `in_memory_das_shaped`; native `das`, `das_agent`, `hyperon`, and
  `hyperon_das` packages are absent.
- AIRIS induction is context-majority learning from deterministic calibration utilities. It does not measure
  causal identification, neural AIRIS training, or native distributed DAS.

## Next Recommended Patch

Add state-conditioned AIRIS features for story reachability and routing observations. Compare confidence-only
demotion against a behaviorally distinct typed fallback while retaining the integrity seal and topology authority
boundary.
