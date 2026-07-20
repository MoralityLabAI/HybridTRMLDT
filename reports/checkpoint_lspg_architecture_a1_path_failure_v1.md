# Checkpoint: LSPG A1 Path Construction Failure v1

## Result

The sealed A1 stage runner exited before starting a training child. It passed an
absolute campaign output path to the PowerShell cell wrapper, which attempted to
join that rooted path to the repository root. The path error occurred before the
wrapper's receipt helper was defined, so no cell resource receipt was created.

The stage runner correctly emitted a failed A1 stage receipt with zero completed
cells, zero campaign elapsed seconds, and failure
`missing resource receipt for LSAD-B1-K2L6-01-A1-S0-s401`. No model was
constructed and no task outcome was observed.

## Correction

The wrapper now resolves rooted paths directly and joins only relative paths to
the repository root. This is an orchestration correction; the sealed A1 cells,
precision, budgets, data, and selection rules are unchanged.

## Next Step

Commit the failed stage receipt and rooted-path correction, validate the wrapper
with an absolute dry output, then relaunch the unchanged A1 manifest.
