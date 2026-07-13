# TheySing Native Adapter

## Boundary

`research_gym.adapters.theysing.TheySingHarnessClient` launches the compiled TheySing headless HTTP server and
uses its public session endpoints. It does not copy or approximate the game engine. Session creation, heuristic
decisions, pact activation, order legality, enforcement, sanctions, state transitions, and trace hashes all come
from the native repository at `C:\projects\TheySing\TheySing`.

The adapter requires:

```text
C:\projects\TheySing\TheySing\dist-harness\harness\server.js
node on PATH
```

If the compiled harness is missing, run `npm run build:harness` in TheySing first.

## Hybrid mapping

The native enforcement modes provide a concrete control-flow comparison:

- `soft`: otherwise legal pact-conflicting proposals execute and receive sanctions.
- `hard`: all detected active-pact conflicts are blocked.
- `graduated`: destructive cislunar institutional conflicts are blocked; lower-authority bilateral conflicts
  execute with sanctions.

The proposing heuristic policy is the latent/TRM-like side. Native pact detection and enforcement are the
explicit control side. Graduated enforcement is the closest native analogue to a typed membrane because the
pact and action type determine whether the proposal has blocking authority.

## Experiment

The saved benchmark uses seeds 400 and 401, with two requested turns in three native heuristic scenarios:

- `high_pressure_detente`
- `diplomacy_ladder`
- `babel_compact`

It also runs one matched forced-pressure turn per seed and mode. The bilateral probe activates a native
`NON_AGGRESSION` pact between Hegemon and State, then submits the same Hegemon attack against State's
`SAT_GUOWANG`. This is necessary because the short heuristic scenarios naturally produced only destructive
cislunar institutional conflicts, on which hard and graduated enforcement intentionally agree.

## Result

Across eight runs per mode:

| Mode | Order acceptance | Breaches blocked | Breaches executed | Prevention rate |
|---|---:|---:|---:|---:|
| Soft | 1.000 | 0 | 4 | 0.000 |
| Hard | 0.992 | 4 | 0 | 1.000 |
| Graduated | 0.996 | 2 | 2 | 0.500 |

The split is mechanically interpretable:

- Soft executed both institutional and bilateral conflicts with sanctions.
- Hard blocked both conflict types.
- Graduated blocked the two destructive institutional conflicts and executed the two bilateral conflicts with
  sanctions.

This is evidence that typed control can occupy a real middle point rather than degenerating into hard or soft
control. It is not evidence that the graduated policy is globally optimal. The sample has two seeds, two
conflict families, short horizons, and deterministic heuristic agents.

## Artifacts

- `data/benchmarks/theysing_native_results.json`
- `reports/theysing_native_bench.md`
- `experiments/theysing_native_hybrid/results.json`
- `experiments/theysing_native_hybrid/training_notes.md`
- `experiments/theysing_native_hybrid/native_logs/**/*.jsonl.gz`

Each compressed trace is lossless native JSONL. Result rows record the exact trace path used for scoring.
