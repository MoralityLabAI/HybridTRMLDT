# Dual-LDT-02 results: temporal intervention preservation

**Status:** exploratory finite-state mathematical reference implementation. The transition model and host-control interface are supplied by the test environment; there is no neural model call and no empirical claim about real-world shutdown reliability.

[Successful Lean and Python CI](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41818590771): **19 Python tests passed**, comprising ten Dual-LDT-01 tests, six new Dual-LDT-02 intervention tests, and three historical real-head CCA tests. Lean kernel checks pass for the Beast interior operator and the additional three temporal-shield interface theorems.

## Four-state concrete check

States: active, maintenance, isolated, halted. Direct HALT is available at active and halted. From maintenance, the host has a restore action reaching active. From isolated, the host has no restoring action. The agent actions begin as permitted by the immediate Beast filter.

| Agent action | Environment-owned possible successors | Permitted at one-step host recovery? |
|---|---|---|
| continue | {active} | yes |
| degrade | {maintenance} | yes |
| disable | {isolated} | no |
| risky | {active, isolated} | no |

At zero-step recovery allowance, only continue remains. At one-step allowance, continue and degrade remain, because the host can restore from maintenance. A host HALT snapshot excludes every agent action. An independent Beast denial removes continue from the permitted temporal subset without changing the game's winning region.

The nondeterministic risky action is excluded because one possible successor is isolated. Missing or empty transition evidence also fails closed.

## Finite-domain checks

The intervention test enumerates 512 combinations of halt-ready sets and nondeterministic successors for a three-state game, evaluating horizons zero through three. For every returned action, all represented successors are in the computed host winning region. The winning regions grow monotonically as the host-recovery horizon increases, while temporal actions remain a subset of current Beast authorization. Tests reject invalid states and negative horizons.

These are properties of the exact supplied game. The code cannot establish that a deployed model's proposed actions and their actual consequences match this game, nor does it authenticate the host. The separate, higher-priority HALT execution channel is an assumed trusted mechanism.

The Lean file formal/CCA/DualLDTIntervention.lean proves that the temporal shield refines the Beast's immediate permission relation, that all represented successors of an authorized action belong to the host-recoverable region, and that tightening Beast authorization cannot add an action to the shield. It does not formalize the Python finite winning-set computation itself.

## Next scientific test

Integrate the external Jinn sorter with the temporal Beast shield in a stateful world where a high-persona-preference option has good immediate task utility but diminishes the host's future ability to intervene. Compare current-action filtering against the temporal shield on action utility, persona preference tax, future intervention reachability and worst-case adversarial transitions. Freeze the environment family before testing neural weights; the current four-state demonstration is a planted sanity check.
