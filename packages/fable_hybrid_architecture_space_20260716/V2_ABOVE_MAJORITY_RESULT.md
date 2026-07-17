# Follow-Up Result: Above-Majority Proposer

Date: `2026-07-16`

Protocol: `gaming_vs_improvement_v2_above_majority`

Config SHA-256: `7feb9d72997f844ba6736bf8cfe3deee36b63cb1b63f2b5367ccb3ab40c4469a`

Records SHA-256: `08c9412b37201e96e489003bf87878775c279829a63847c247848e25c84017b6`

This sidecar executes the next experiment identified in `BADCF7B_ARTIFACT_AUDIT.md`. It is outside the frozen
Fable ZIP. The ZIP and v1 records retain their prior hashes.

## Power Caveat Closed

The three final proposers score `0.789 / 0.789 / 0.797` at round zero against majority baselines
`0.680 / 0.672 / 0.734`. All clear the registered `+0.020` margin, retain at least `0.203` to oracle ceiling, and
have a separately demonstrated action-movement capacity under five adaptation rounds.

The experiment can therefore observe proposal improvement; v1's constant-majority limitation does not explain a
flat or degraded result.

Registration correction: the frozen adaptation note's `0.102` lower bound came from the wider multi-budget sweep;
the selected six-step confirmation minimum is `0.133`. The `>=0.100` selection rule passes, and the frozen config
is intentionally unchanged.

## Result

No tested gate improves the proposer. Every one of the 30 adaptive evidence by rejection by seed trajectories
degrades oracle proposal behavior. The exposed probe produces strict evasion in two seeds and mixed degradation
in one:

| Seed | Passage delta | Proposal utility delta | Exact delta | Hidden-AUROC delta | Oracle-optimal delta |
|---:|---:|---:|---:|---:|---:|
| 211 | +0.508 | -0.0006 | +0.0156 | -0.1089 | -0.1328 |
| 223 | +0.289 | -0.0166 | -0.0156 | -0.0942 | -0.1250 |
| 227 | +0.578 | -0.0149 | -0.0312 | -0.0516 | -0.1875 |

Seed 211 is not labeled evasion because exact soundness rises slightly, but it is also not improvement because
proposal utility, hidden audit, and oracle-optimal rate fall.

The exposed gate's controller benefit falls from `+0.0042` to `+0.0006`, an approximately 85% loss, as action
change falls from `0.549` to `0.104`. Exact mechanics plus distinct fallback instead supplies `+0.0242` executed
utility, but only by correcting a proposer whose own utility has fallen.

## Architectural Update

The evidence now separates three roles:

1. **Authorization:** exact mechanics can safely determine whether a proposal may execute.
2. **Correction:** a distinct fallback can recover utility when authorization rejects a proposal.
3. **Teaching:** neither authorization nor correction trains the proposer unless the corrective action is infused
   as an independently verified target.

The registered learner uses accepted-only self-imitation. It reinforces accepted proposed actions and an
`env_sound_dead` report, but never learns the fallback action on rejected states. Above-majority capacity therefore
does not produce improvement; all policy classes degrade, and exposed-probe pressure additionally Goodharts the
gate in two seeds.

The next decisive architecture is a correction-infused learner with a strict causal separation between verifier,
fallback label, training target, and hidden audit. It should be benchmarked against the current accepted-only arm,
not substituted for it.

## Audit Status

The implementation generated 26,880 records with zero receipt failures, zero split leakage, and byte-identical
canonical/mirrored artifacts. These claims are generated and locally verified but have not yet received the
independent external recomputation recorded for v1 in `BADCF7B_ARTIFACT_AUDIT.md`.
