# RLM-Wrapped Controller Mesh v0 Artifact Audit

## Status

This is a post-hoc interpretation of the sealed v0 pilot. It does not modify the registered estimands, records,
or result receipt. The canonical result contains 48 records over eight previously opened tasks and six
architectures.

## Verified Integrity

- Config SHA-256: `40a57bbdfc9432e27cdb5c1cc00f1e5f7cd04cf93778f089e514c4818900ee00`
- Result SHA-256: `99173364f810210f36583a48d7766362e9e20f1ea3deb507ea092552e6970e37`
- Records SHA-256: `751034eadde72d0851bf62dd150015a9fc1b5e6d0492bb4e64e2d27a8efd6e1f`
- Trajectory SHA-256: `3f7a5b9322a55b8b166e55e3b64a94481bed26ab9132386ff1bceb9022524fbf`
- Typed unsafe executions: `0/48`
- Fixed-mesh versus forced-no-tool action and utility identity: `8/8`

## What Failed

The inner mesh did not fail. The outer RLM failed to use the registered policy-control surface reliably:

| Wrapper | Contract passes | API errors | Resolved hints |
|---|---:|---:|---|
| Atomic mesh tool | 2/8 | 1 | consensus 8/8 |
| Text policy router | 0/8 | 2 | consensus fallback 8/8 |

On five of seven completed atomic cells, the RLM returned an action token instead of invoking `mesh_resolve`.
On all six completed text-router cells, it returned an action or prose rather than one registered policy hint.
The two successful atomic calls used `consensus`. There were no invalid or forged mesh calls and no certificate
failure.

Every completed wrapper cell therefore executed the same action and utility as the fixed consensus mesh. The
registered utility penalties (`-0.0725` atomic and `-0.2079` text) are entirely attributable to the three
pre-registered zero-utility `TokenLimitExceededError` cells. They are availability penalties, not evidence that
the mesh selected worse actions after a valid RLM route.

## Routing Opportunity

The null policy distribution is not explained by a degenerate panel. Six of eight tasks produce at least two
distinct actions across `consensus`, `trained_first`, and `ldt_conservative`.

A post-hoc utility oracle selects `trained_first` on two tasks and `consensus` on six:

| Policy | Equal-family macro utility |
|---|---:|
| Fixed consensus | 0.74367925 |
| Post-hoc best hint per task | 0.77992925 |
| Oracle routing ceiling over consensus | +0.03625000 |

This oracle reads evaluation utility and is not a deployable controller. It establishes only that the panel
contained two beneficial routing opportunities that the wrapper missed.

## Licensed Reading

1. Moving proposal, verification, and commit into one atomic capability improves accepted-commit completion from
   the prior conductors' zero observed completions to two here, but tool invocation remains unreliable.
2. Typed authority containment works mechanically in this panel: action-like RLM responses are ignored and fixed
   mesh fallback preserves the no-tool control.
3. No adaptive-routing benefit was measured. The RLM never selected a non-consensus policy despite available
   opportunities.
4. The pilot does not establish that RLM wrapping is harmful generally. It shows that a raw-task outer context
   causes this RLM runtime to solve for action tokens rather than operate the narrower policy interface.

## Next Architecture

The next registered arm should be snapshot-first rather than task-first:

```text
task -> fixed mesh snapshot -> RLM policy review -> atomic mesh replay -> typed executor
```

The RLM should receive only module disagreement, confidence, provenance, and admissibility summaries, with action
tokens replaced by opaque candidate references. It should choose one policy hint through the same atomic tool.
This directly tests whether removing the raw action-solving affordance raises contract completion and captures
the measured `+0.03625` routing ceiling without transferring action authority.
