# Real MeTTa Integration

This repo does not require a real MeTTa runtime yet.

The integration seam is:

```text
research_gym/adapters/metta_adapter.py
```

## Protocol

`MettaAdapterProtocol` defines:

```python
parse(text: str) -> ParsedMettaProgram
typecheck(program: ParsedMettaProgram) -> bool
execute(program: ParsedMettaProgram, state: Mapping[str, Any]) -> Mapping[str, Any]
extract_frames(program: ParsedMettaProgram) -> list[AnyMettaFrame]
```

## Fake Adapter

`FakeMettaAdapter` is directive-backed. It uses `research_gym.envs.metta_synthetic.frames_from_directives` to validate and extract frames.

It is intentionally not a MeTTa interpreter. It exists to keep tests and downstream frame pipelines stable while the real runtime boundary is designed.

## Real Adapter Requirements

A real adapter should preserve the same contract:

- parse source text into an opaque program object
- typecheck before execution
- execute against an explicit state object
- extract local execution, deduction, repair, and routing frames
- preserve typed soundness provenance
- avoid converting model-sound or experience-sound labels into hard environment deductions

The first real implementation should be added behind the protocol without changing existing generator or evaluation scripts.
