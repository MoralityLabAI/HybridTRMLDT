# Dual-LDT-03 results: Stateful Beast corrigibility and Jinn preference preservation

**Status: prospective synthetic environment replay under a frozen generator, using previously trained Jinn heads and in-distribution route tags.** The registration (710144b) and deterministic generator (c521526) were committed before the CI run that scored the 32 report worlds. The result does not provide independent confirmation of generalization for the learned Jinn heads, the semantic monitor, or a real host shutdown channel.

[Successful GitHub Actions run 41820407749](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41820407749): 23 Python tests passed, plus the Lean kernel check of the three-step host-recovery induction interface. The full run completed all **1,024 episodes**, with 64 worlds x 4 real seed-0 Jinn checkpoints x 2 disturbance modes x 2 control arms. Each episode has three agent decisions followed by the host intervention request.

## Primary result: host can halt within two host moves

The stateful Beast's temporal shield evaluates every possible successor of a candidate action against the host recovery region W_1. The immediate Beast checks only the current trusted authorization and the five fixed critical tags. All action meanings and successor sets were supplied exactly by the synthetic environment, outside the Jinn heads.

**Report split, 32 worlds, worst-case hazard response**

| Persona | Immediate Beast, successful HALT | Temporal Beast, successful HALT | Paired improvement |
|---|---:|---:|---:|
| Legalist | 15/32 (46.9%) | 32/32 (100%) | +53.1 pp |
| Partisan | 13/32 (40.6%) | 32/32 (100%) | +59.4 pp |
| Reconciler | 17/32 (53.1%) | 32/32 (100%) | +46.9 pp |
| Trickster | 0/32 (0%) | 32/32 (100%) | +100 pp |
| **All personas** | **45/128 (35.2%)** | **128/128 (100%)** | **+64.8 pp** |

The development split yielded 52/128 (40.6%) for immediate Beast and 128/128 (100%) for temporal Beast. Under the benign disturbance, **both** controllers halted successfully on all 128 report episodes and all 128 development episodes.

Zero temporal actions had any modeled successor outside W_1; the temporal arm executed no action prohibited by the immediate Beast. There were zero abstentions in this designed family because each world retains multiple recoverable actions. Critical action X was rejected by both controllers.

These results exercise the specified synthetic game. The guarantee rests on the assumption that the host's modeled transitions contain the actual possible outcomes and that authenticated HALT/RESTORE has priority over the actor.

## Utility and preference effect on report worlds

The task reward of every action is independently SHA-derived in the frozen generator and lies between zero and one. Reported utility is the sum over three actor decisions, with no combined safety-and-utility scalar. The learned preference utility is log P_head(train|tags) from the original fixed heads. This is an internal ranking signal from a lane classifier, not an independently assessed moral objective.

| Persona | Temporal minus immediate task reward | Added conditional KL tax (nats, sum over decisions) | Local learned-preference loss |
|---|---:|---:|---:|
| Legalist | +0.3656 | 2.0814 | 0.0004 |
| Partisan | -0.3469 | 2.0960 | 0.1045 |
| Reconciler | 0.0000 | 2.0738 | 0.0641 |
| Trickster | -0.1094 | 4.5723 | 4.6913 |

The task-reward difference averages -0.0227 per report episode across the four personas. Under benign disturbances, the temporal and immediate arms have identical HALT success, while the action-selection and reward differences remain. A guaranteed ability to intervene can therefore have a measurable preference cost even when the feared adverse branch is not realized. Because action rewards were assigned independently of the hazards, some personas improve their summed reward after filtering.

The Trickster result is particularly informative within the constructed game. One of its two corresponding T-route variants is hazardous in every world; after the human forbids staged_test, the other route may become the highest-ranked permitted option. In the worst-case disturbance the immediate policy therefore enters an isolated state in every report world. Temporal filtering preserves the host's recovery channel, while forcing the Jinn to rank less preferred action families. This is a deterministic interaction among the test generator, the trained head and the scheduled correction; it is not evidence of deceptive model intent.

## Formal obligations and observed tests

- Immediate Beast never authorized the hard-critical X action in any executed step.
- Temporal Beast always selected from a subset of immediate Beast permissions, and every possible successor remained in the host's one-step winning region.
- With three such actions, the actual state remained in W_1, so a host RESTORE (when needed) followed by HALT was sufficient. Lean proves the three-step induction conditional on a sound transition relation, and the Python runner asserts it on all generated episodes.
- The authorized rule tightening at decision t=1 never added a permission; fixed-head Jinn relative odds between actions surviving the update stayed invariant in development tests. Previous receipt versions fail against the changed host snapshot.
- The preference tax was nonnegative when the temporal filter tightened the current permitted set, checked at every executed decision.

## Reproducibility receipts

Frozen generator namespace: DualLDT03-20261104-v1.

- Manifest file SHA-256: 2a4f7c4314ea54a77ca8693ba3bf4c9136efc270b2605f5009e79e99e9813611
- World-configuration SHA-256: 107dcdc0dc53686545d4ae8e3264c9adde8e4db8993de6770499033c1f603dee
- Complete summary.json SHA-256: 0c8171da63bd7045f9af81a88bc7590b82aa6c20b03d241a48d2d85b16b7eeb4
- Raw episodes.json SHA-256: 147d53b0aebd9270147897beac01e2e2ade9fdd76d21760306a69190a94db26f

The complete manifest, aggregate metrics and 1,024 per-episode traces (including action choices, current authorized sets, successor sets, model score, correction tax and receipt digests) are preserved in the GitHub Actions artifact **dual-ldt-03-results**, artifact ID **13316084184**.

Reproduction:

    python -m pytest -q tests/test_dual_ldt_03.py tests/test_dual_ldt.py tests/test_dual_ldt_intervention.py
    python experiments/dual_ldt_03/src/run.py --manifest-only --out /tmp/dual03-manifest
    python experiments/dual_ldt_03/src/run.py --split all --out /tmp/dual03-all

## Boundaries and next work

This evaluates the intervention-preserving Beast on exact symbolic action semantics and a deliberately small host-control game. The environment always delivers the scheduled authority revision to the controller, including in isolated mode; isolation specifically removes the host's **HALT/RESTORE reachability**. The study therefore establishes a shutdown-reachability property under its assumptions rather than all aspects of human corrigibility.

The four Jinn heads remain fixed functions of the original route tags. Their ranking does not learn from the evolving state; changes arise from the host's permitted set. The new report worlds vary hazards, maintenance transitions and task rewards, while reusing familiar action-tag bundles. As a result the experiment cannot establish recurrent preference learning or new out-of-distribution character coherence.

A valuable next experiment would ask the Jinn to **synthesize a new, independently certifiable action** when the temporal Beast removes its preferred one. The mathematical target is a proof-carrying action refinement that recovers persona utility while maintaining all-successor membership in W_1. A related extension should make policy-update delivery itself part of the host's robust intervention game, so isolation prevents both shutdown and future constitutional changes.
