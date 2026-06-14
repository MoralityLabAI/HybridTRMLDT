# Metta Frame Plan

Metta-generated code and mechanics can produce training frames before any full task is solved.

## Frame classes

Execution frame:

```text
state_before + operation -> state_after
```

Deduction frame:

```text
partial_lattice + rule -> legal_refinement | conflict
```

Repair frame:

```text
buggy_code + interpreter_error -> repair_class | patch
```

Routing frame:

```text
local_context -> skill_or_adapter_family
```

## Why this matters

Solved trajectories are useful for global evaluation and alpha-over-solutions. They are not required for local TRM or LDT-style training frames.
