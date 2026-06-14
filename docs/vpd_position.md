# VPD Position

Do not claim that VPD edits TRMs directly.

Current safe claim:

```text
VPD labels or decomposes weights, adapters, components, or heads by behavior.
```

Open bridge:

```text
Can those labels guide useful training interventions?
```

Possible bridges, ordered from least speculative to most speculative:

1. Data routing.
2. Adapter assignment.
3. Loss weighting.
4. Regularization.
5. Patch proposal.
6. Programmatic weight hill climb.

In this repo, VPD enters only after a model or head exists to instrument.
