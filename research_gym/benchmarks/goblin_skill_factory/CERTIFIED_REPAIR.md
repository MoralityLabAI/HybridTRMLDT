# GSF-03: train a certified repair skill

Public development campaign using the native CCA-05 Storyworld skill. The supplied skill template is
[CCA-05 SPEC](../../../../experiments/cca05_certified_trm_repair/SPEC.md), including its modeled-rival
assumption. This is a new small trainable policy, not a reproduction of the private Jinn/Beast checkpoints.

## Research contract

The proposer receives a scenario, state, horizon and current candidate mask. Train a tied recurrent MLP
on certified continuation utilities and masked best-action targets. The host invokes
`bounded_recurrent_repair`: nominate, issue an action certificate, verify the witness independently,
and if rejected commit only an independently certified elimination. All accepted actions pass the
native `execute_certified` binding and candidate-domain check. A proposed action never grants itself
authority. Native certificate issuance and independent witness checking share the deployed environment,
and certification applies under the modeled rival. No claim about nondeterministic-rival safety.

Nine policies:

- Direct (one visit), residual (four visits), refresh (four input refreshes), RLT and matched damping.
- Static refresh: same trained refresh checkpoint, freeze its full-mask ranking after the first forward
  call. Reranking compares the same weights against native re-query on the reduced mask.
- Lexical: pick the first remaining action; no model.
- Native LDT scan and exact certified oracle scan: account for prescan calls separately.

The four recurrent fits have equal width, initializer, updates, data order and cell-visit work.
The direct arm uses fewer visits. The same native repair loop is evaluated at nomination budgets
1, 3 and 5. Budget 5 is the size of the initial finite action domain. Zero unsafe execution is
an invariant, not proof that a learned controller has improved. Measure utility including abstention
at zero, oracle regret, viable cases missed, initial certified proposals, recovery, committed
eliminations, model cell visits, prescan checks, issuer checks and independent elimination
admissibility checks. CPU time includes grading and warmed memoization; it is diagnostic only.

## Partition and receipts

Fixed hash partition of the 875 nonterminal Storyworld states. Group state across scenarios,
horizons, alternate masks, seeds and arms. Default: 256 fit roots, 64 each CAL/dev/shift/trace.
Training sees both scenarios and horizons 3 and 6 with full, rejection-reachable and arbitrary
nonempty masks. CAL/dev/trace use horizon 6; shift uses horizon 2. All are public deterministic
development splits, not new task templates or sealed evaluation. Two initialization seeds, 160
updates, width 32, CPU only. Default: 10 neural fits, 18 policy cells and 2,304 traces on 64
dedicated state roots. Traces at budget 3 only; every trace is replayed from the fitted weights,
exact host semantics, and split contract. Equivalent utility-optimal actions receive success
credit even if they differ from the canonical oracle tie-break.

The record includes candidate masks, per-visit model scores, nominations, certificate
witnesses, verifier decisions, committed lattice ranks, regret, exact correction target and
work vector. Keep oracle targets and future observations out of the proposer input. Train
targets may include verified corrections after rejection, not raw rejected actions as successes.
All records are marked `training_ready=false` pending DataGoblin custody checks. This experiment
measures one Storyworld decision with multiple repair attempts; full multi-decision trajectories
and a learned stop/abstention head are successors.

## Reproduce

```sh
python -m research_gym.benchmarks.goblin_skill_factory.certified_skill --selftest
python -m research_gym.benchmarks.goblin_skill_factory.certified_skill --smoke --out /tmp/repair-smoke
python -m research_gym.benchmarks.goblin_skill_factory.certified_skill --audit /tmp/repair-smoke
python -m research_gym.benchmarks.goblin_skill_factory.certified_skill --out /tmp/repair-dev
python -m research_gym.benchmarks.goblin_skill_factory.certified_skill --audit /tmp/repair-dev
```

CPU, one PyTorch thread; use independent machine-specific RAM and wall bounds.
The auditor checks root custody, regenerated public packets, artifact bytes, state-dict digests,
and policy/certificate replay for every trace. Native certificate tests include stale
horizon, foreign action and excluded-domain rejection. A separate independent implementation
of the full game semantics is not supplied; reuse CCA-05 parity tests.

## Next integration

Expose a skill template, bounded fit knobs and CAL summary to the GoblinBench researcher.
Ask it to compare the new repair module with the existing direct/lexical/LDT and frozen
ranking baselines, select a budget from CAL, and explain when an exact scan is preferable.
For RLT meta-repair use matched continuation of an identical checkpoint and retain unsuccessful
interventions as research targets. For GSF-04 use the external persona/LDT policy contract and
existing Jinn/Beast specialist intake separately from the certified execution floor.
