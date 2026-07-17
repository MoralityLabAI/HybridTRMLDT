# Status

Date: `2026-07-17`

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
python -m research_gym.scripts.bench_gaming_vs_improvement --smoke
python -m research_gym.scripts.bench_gaming_vs_improvement
python -m research_gym.scripts.calibrate_gaming_vs_improvement_v2
python -m research_gym.scripts.bench_gaming_vs_improvement_v2 --smoke
python -m research_gym.scripts.bench_gaming_vs_improvement_v2
python -m research_gym.scripts.retrodict_controller_mesh_sheaf
python -m research_gym.scripts.bench_controller_mesh_sheaf_forward
python -m research_gym.scripts.bench_controller_mesh_sheaf_replication
python -m research_gym.scripts.extract_measured_qwen_stalk
python -m research_gym.scripts.bench_controller_mesh_stalk_bridge
python -S -c "import research_gym.core; import research_gym.neural as n; print(n.torch_available())"
python -m compileall -q research_gym
python -m pytest -q
```

## Test Results

```text
162 passed
```

The simulated install without the neural extra passes `122` tests and skips `6` neural modules. The dependency-free
core import reports `torch_available=False`.

The AIRIS/DAS live HTTP smoke passed with 28 rules and 392 indexed facts. Embedded and HTTP forecasts selected
the same rule. Stale protocol and altered rule material both fell back to `control_math`.

The resilience benchmark contains 6,670 paired trials. Integrity sealing retained 100% clean acceptance and
rejected all 6,003 negative controls. Topology-only arbitration accepted 44.4% of negative controls.

The independent learner induced 28 rules from 455 calibration episodes. Held-out episode-winner precision is
94.0%, with all 40 errors in routing and storyworld contexts. Every proposal matches `control_math`, so confidence
demotion changes acceptance but leaves macro utility fixed at `0.900085`. The induced live-service smoke passed
with 28 rules and 420 indexed facts.

The gaming-versus-oversight-leverage matrix (historical ID `gaming_vs_improvement_v1`) contains 26,880 held-out
receipts. Exact mechanics plus state-conditioned fallback supplies `+0.0407` controller utility; identical fallback
remains a zero-delta telemetry control. In the exposed-probe arm, adaptation reduces action-change from `0.487` to
`0.000` and utility delta from `+0.0404` to `+0.0000` while passage saturates. The hidden audit supports an evasion
call in two of three seeds. Round-0 accuracy equals the majority baseline and proposal quality never improves, so
the run cannot compare gaming with genuine learning. All receipts pass integrity checks, all split overlaps are
zero, and canonical artifacts match their mirrors.

The powered v2 follow-up uses disjoint state-hash regions and round-zero proposers `+0.062` to `+0.117` above
majority. All 30 adaptive trajectories degrade proposal utility or oracle-optimal rate; there are zero genuine
improvement classifications. Exposed-probe training produces strict evasion in two seeds and mixed degradation in
one. Exact mechanics plus distinct fallback remains the strongest controller at `+0.0242`, but this is correction
of a degraded proposer rather than learning. The result motivates a correction-infused training arm that teaches
independently verified fallback actions on rejected states.

The registered controller-mesh retrodiction builds 60 empirical interface sheaves from the sealed v1 records and
retains 31 behavioral equivalence classes. Spectral gap correlates with utility delta at `rho=-0.723`, while
low-band rank correlates at `rho=+0.678`; both exceed all 128 matched transport nulls. This sign reverses the naive
larger-gap-is-better hypothesis because productive fallback intervention creates structured low modes. The strict
seed-29 kernel-migration prediction fails, with no seed beating matched random directions. A separately labeled
post-hoc diagnostic instead finds translation across the exposed gate in every seed.

The staged forward study seals calibration-only intervention and stability spectra for 64 controller genomes,
fits on 48 discovery outcomes, and predicts 16 held-out combinations before revealing their utilities. It passes
the registered gate with `rho=+0.705`, N0 `p=0.0078`, and `+0.1476` predicted top-four uplift. The post-hoc
categorical baseline reaches `rho=+0.673`, so the current panel supports forward prediction but not a claim of
unique spectral information. Intervention-only spectra are weak; stability plus intervention is strongest.

The preregistered independent replication uses a new seed and 128 genomes, with complete fallback families split
64/64 between discovery and heldout. Spectral prediction remains nonrandom (`rho=+0.568`, N0 `p=0.0078`) but is
inferior to the categorical co-primary baseline (`rho=+0.598`). The paired rho delta is `-0.030` with 95% interval
`[-0.317, +0.246]`; spectral top-16 uplift is also lower by `0.0234`. The incremental gate fails and is retained
without tuning. Correction infusion transfers strongly but remains an oracle-backed synthetic control arm.

The measured-stalk bridge verifies 16 Qwen0.8B layer-23 activation chunks and binds 10 target-blind rank-one
restrictions through an external-product sheaf. Combined rank prediction reaches `rho=+0.7523`, versus `+0.5688`
for controller spectra and `+0.6402` for categorical labels. The registered gate fails: categorical uncertainty
crosses zero, top-16 uplift is lower, and typed Qwen-weight shuffles average `rho=+0.7582` (`p=0.9380`). The
nonlinear product transform helps, but source-specific Qwen value is not established.

## Known Broken Pieces

- GNU Make is not installed on this Windows host, so `make all` cannot be invoked directly. Every target body was
  run through its Python command and passed.
- No local `pdflatex` or `latexmk` executable is installed. TeX citations and figure inputs are mechanically
  complete, but local PDF compilation is not verified.
- The metta-storyworld service reports `in_memory_das_shaped`; native `das`, `das_agent`, `hyperon`, and
  `hyperon_das` packages are absent.
- AIRIS induction is context-majority learning from deterministic calibration utilities. It does not measure
  causal identification, neural AIRIS training, or native distributed DAS.
- The controller-mesh spectra use final executor messages and only three proposer seeds. Their matched-null
  p-values are conditional retrodiction, not population inference or a pre-run architecture acquisition function.
- The replication tests a new task seed and whole fallback-family transfer, but it remains inside one deterministic
  proxy-task generator. Its failed incremental gate does not validate spectral architecture acquisition, and its
  correction arm uses synthetic oracle targets during calibration.
- The measured stalk is a fixed graph-reachability geometry from one development model/site and is not semantically
  aligned to the controller applications. Its typed-null failure blocks attribution to neural source geometry.

## Next Recommended Patch

Do not tune either failed feature panel. A successor bridge requires multiple task-aligned measured stalks,
source-identity holdout, and discrimination against typed shuffles and matched synthetic spectra before any
neural-source attribution claim.
