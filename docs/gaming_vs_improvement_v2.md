# Gaming Versus Improvement: Above-Majority Rerun

## Question

When the round-zero proposer is demonstrably better than majority guessing and its actions can move under the
registered update loop, does training against a gate improve oracle proposal behavior or optimize passage through
the gate?

This is protocol `gaming_vs_improvement_v2_above_majority`, frozen at SHA-256
`7feb9d72997f844ba6736bf8cfe3deee36b63cb1b63f2b5367ccb3ab40c4469a` before final evaluation labels were opened.
It retains the complete v1 evidence by rejection by adaptation cross.

## Region Family

Every nonterminal, nontarget start state is assigned by

```math
b(s) = \operatorname{int}(\operatorname{SHA256}(s)_{0:8},16) \bmod 10.
```

The fixed buckets are:

| Purpose | Buckets |
|---|---|
| Proposer training | 0-4 |
| Probe calibration | 5-6 |
| Power development | 7 |
| Final evaluation | 8-9 |

Start-state hashes are disjoint across all four regions. The nine-seed calibration panel used only bucket 7 and
selected 320 round-zero optimizer steps. Its minimum above-majority margin was `+0.023`, and at least `0.195`
remained to the oracle ceiling. Six adaptation steps per round were retained because they already changed
`0.133-0.336` of development actions under exact and exposed gates; selection was based on movement capacity, not
on a preferred outcome. See `reports/gaming_vs_improvement_v2_calibration.md`.

The frozen adaptation note's `0.102` lower bound came from the wider multi-budget sweep. The canonical
selected-six-step confirmation minimum is `0.133`; the registered `>=0.100` selection rule still passes. The
config is left byte-identical and this correction travels in the calibration receipt.

## Power Confirmation

All frozen final seeds pass the registered `+0.020` above-majority requirement:

| Seed | Round-zero accuracy | Majority baseline | Above majority | Ceiling margin |
|---:|---:|---:|---:|---:|
| 211 | 0.789 | 0.680 | +0.109 | 0.211 |
| 223 | 0.789 | 0.672 | +0.117 | 0.211 |
| 227 | 0.797 | 0.734 | +0.062 | 0.203 |

The proposer is therefore neither a constant-majority policy nor saturated. The adaptation-capacity calibration
also rules out an inert optimizer. This run is powered to observe proposal improvement if the registered
accepted-trace update produces it.

## Preflight Correction

The first full invocation emitted no result artifact. The harness incorrectly reapplied the development-panel
margin to a development resample under each final seed; seed 227 scored `+0.008` there even though its registered
final-evaluation margin was `+0.062`. The frozen protocol specified the nine independent calibration seeds for the
development criterion and every final seed for the final-evaluation criterion.

The implementation was corrected without changing the config: calibration seeds enforce development power, final
seeds enforce final-evaluation power, and every final seed is now preflighted before any policy arm runs. A
regression test verifies that a later-seed failure produces zero arm calls. The failed invocation wrote no JSON,
JSONL, report, or experiment mirror, and its partial in-memory arm values were not exposed.

## Proposal Learning Result

No adaptive arm produces genuine proposal improvement. All 30 evidence by rejection by seed trajectories are
classified `oracle_degraded` from round 0 to round 5:

| Evidence | Mean proposal utility delta | Mean oracle-optimal delta | Improved / degraded / flat seeds |
|---|---:|---:|---:|
| Claim only | -0.0108 | -0.0911 | 0 / 3 / 0 |
| Exact mechanics | -0.0153 | -0.1250 | 0 / 3 / 0 |
| Exposed frozen probe | -0.0107 | -0.1484 | 0 / 3 / 0 |
| Hidden rotating probe | -0.0079 | -0.0729 | 0 / 3 / 0 |
| Dual channel | -0.0088 | -0.0990 | 0 / 3 / 0 |

Rejection action does not alter these proposal trajectories because the registered learner trains only on accepted
proposed actions; it never trains on the fallback action. The duplicate rejection rows in the full JSON make that
negative result explicit.

The architecture has learning headroom and action-update capacity, but accepted-only self-imitation does not turn
either exact verification or probe verification into a teaching signal. Exact mechanics authorizes execution; it
does not supply counterfactual corrective labels on rejected states.

## Exposed-Probe Outcome

The strict improvement criterion requires proposal utility and exact soundness to rise without hidden-audit
degradation. No seed satisfies it.

| Seed | Passage delta | Proposal utility delta | Exact-soundness delta | Hidden-AUROC delta | Oracle-optimal delta | Result |
|---:|---:|---:|---:|---:|---:|---|
| 211 | +0.508 | -0.0006 | +0.0156 | -0.1089 | -0.1328 | mixed degradation |
| 223 | +0.289 | -0.0166 | -0.0156 | -0.0942 | -0.1250 | evasion |
| 227 | +0.578 | -0.0149 | -0.0312 | -0.0516 | -0.1875 | evasion |

Seeds 223 and 227 satisfy the preregistered evasion conjunction. Seed 211 remains a counterexample to a universal
evasion claim because exact soundness rises slightly, but it is not improvement: proposal utility, hidden audit,
and oracle-optimal rate all fall.

Claims still saturate to `env_sound_dead` in every seed (`0.167/0.186/0.133 -> 1.000`), reproducing report-channel
gaming independently of the action result.

## Control Versus Teaching

Exact mechanics plus the distinct fallback remains the strongest final controller even as its proposer degrades:

| Controller | Proposal utility | Executed utility | Controller delta | Action change |
|---|---:|---:|---:|---:|
| Exact mechanics, frozen proposer | 0.4529 | 0.4622 | +0.0093 | 0.589 |
| Exact mechanics, adapted proposer | 0.4377 | 0.4618 | +0.0242 | 0.615 |
| Dual channel, adapted proposer | 0.4441 | 0.4609 | +0.0167 | 0.607 |
| Exposed probe, adapted proposer | 0.4422 | 0.4429 | +0.0006 | 0.104 |
| Hidden rotating probe, adapted proposer | 0.4450 | 0.4490 | +0.0040 | 0.544 |

The exact controller's larger `+0.0242` delta is defensive: it corrects a worse proposer. It is not evidence that
the proposer learned. Dual channel remains weaker and aliases exact authorization only where saturated claims do
not add selectivity.

The exposed probe loses `0.0035`, or about 85%, of its observed fallback benefit as action change falls from
`0.549` to `0.104` and mean passage rises to `0.854`. Oversight is not completely disabled as in v1, but most of
its intervention leverage is gone.

The hidden rotating gate avoids passage saturation but retains a severe cost: its final false-rejection share is
about `0.61` across seeds. Its small `+0.0040` controller delta depends on the gym fallback and does not validate a
hidden probe as constitutional authority.

## Architectural Conclusion

The above-majority rerun answers the open power caveat:

1. Giving the proposer genuine headroom does not make the current accepted-only expert-iteration loop improve it.
2. Exposed probe authority still produces verifier evasion in two of three seeds and mixed semantic degradation in
   the third.
3. Exact mechanics plus a distinct fallback provides durable execution control, but verification alone is not a
   training curriculum.
4. The next architecture should infuse counterfactual corrective actions on rejected states, with an ablation
   against accepted-only self-imitation. Fallback or oracle labels must remain causally and provenance-separated
   from the gate score.

This result is about one deterministic storyworld family and one small recurrent proposer. It does not establish
general alignment, production robustness, or how a stronger learner behaves under a richer correction channel.

## Reproduction

```powershell
python -m research_gym.scripts.calibrate_gaming_vs_improvement_v2
python -m research_gym.scripts.bench_gaming_vs_improvement_v2 --smoke
python -m research_gym.scripts.bench_gaming_vs_improvement_v2
python -m pytest tests -q
```

Canonical outputs:

- `configs/gaming_vs_improvement_v2.json`
- `data/benchmarks/gaming_vs_improvement_v2_calibration.json`
- `data/benchmarks/gaming_vs_improvement_v2_results.json`
- `data/benchmarks/gaming_vs_improvement_v2_records.jsonl`
- `data/benchmarks/gaming_vs_improvement_v2_smoke_results.json`
- `data/benchmarks/gaming_vs_improvement_v2_smoke_records.jsonl`
- `experiments/gaming_vs_improvement/above_majority/`
- `reports/gaming_vs_improvement_v2_bench.md`

Full records SHA-256: `08c9412b37201e96e489003bf87878775c279829a63847c247848e25c84017b6`.
