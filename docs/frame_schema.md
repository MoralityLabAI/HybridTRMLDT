# Frame Schema

The repo keeps specialized dataclasses for local mechanics, but every frame can export to a common JSONL shape.

## Common Frame

```json
{
  "id": "frame-id",
  "family": "reachability | execution | deduction | repair | routing",
  "source": "generator-or-file",
  "input_state": {},
  "operation": {},
  "output_state": {},
  "soundness_type": "env_sound_dead | model_sound_dead | experience_sound_dead | live | unknown",
  "label": "task-specific-target",
  "metadata": {}
}
```

## Field Semantics

- `id`: stable frame identifier.
- `family`: coarse frame family used for routing and evaluation.
- `source`: generator, environment, or directive file that produced the frame.
- `input_state`: state before the local operation or decision.
- `operation`: mechanic, rule, query, repair diagnostic, or routing candidates.
- `output_state`: state after the operation or expected decision output.
- `soundness_type`: typed provenance of the label.
- `label`: task-specific target string.
- `metadata`: auxiliary details that should not be required to parse the core contract.

## Specialized Mappings

Reachability frames map state and horizon into:

```text
family: reachability
input_state: story state
operation: horizon and target predicate
output_state: reachability booleans
soundness_type: reachability label
label: reachability label string
```

MeTTa-like frames map as follows:

```text
execution:
  input_state: before
  operation: rule name and operation
  output_state: after
  soundness_type: unknown
  label: passed or failed

deduction:
  input_state: before lattice
  operation: rule name and eliminated candidates
  output_state: after lattice
  soundness_type: deduction provenance
  label: conflict or refinement

repair:
  input_state: buggy code
  operation: diagnostic
  output_state: repaired code
  soundness_type: unknown
  label: passed or failed

routing:
  input_state: task signature
  operation: candidate skills
  output_state: chosen skill
  soundness_type: unknown
  label: chosen skill
```

## Outputs

Legacy files remain available:

```text
data/frames.jsonl
data/metta_frames.jsonl
```

Common-schema sidecars are emitted by the generators:

```text
data/generated/reachability_frames.jsonl
data/generated/metta_frames.jsonl
```
