# RLM-Wrapped Controller Mesh v0 Protocol

## Question

Can an official RLM add useful task-conditioned routing around a typed controller mesh without acquiring action
or schedule-authoring authority?

## Architecture

```text
task context -> outer RLM -> policy hint -> atomic typed mesh -> certificate -> executor
                                  |                                 ^
                                  +---- no direct action path -------+
```

The practical mesh contains proxy-TRM, trained-ControlTRM, exact-LDT, deterministic arbitration, and a
certificate-bound executor. The outer RLM can choose only `consensus`, `trained_first`, or `ldt_conservative`.
The atomic tool resolves the entire inner graph in one host call. Missing, repeated, malformed, or uncertified
output executes the fixed consensus mesh.

This deliberately differs from the prior conductor arms. Those arms asked the RLM to sequence proposal,
verification, and commit capabilities and completed no accepted typed commit. Here the sequence is inside the
typed capability; the RLM has policy-selection authority but no schedule or action authority.

## Controls

- `mesh_fixed_consensus`: the inner mesh without an RLM.
- `mesh_forced_no_tool_fallback`: simulates wrapper failure and must reproduce fixed-mesh action and utility.
- `rlm_mesh_text_router`: separates policy-selection value from custom-tool compliance.
- `ldt_only` and `trained_trm_ldt_fixed`: inherited local controls.

## Scope

Eight evaluation tasks are selected deterministically from the already opened v1 suite, two per family. This is
a calibration-informed architecture-extension diagnostic, not a fresh held-out efficacy claim. API errors remain
zero-utility outcomes. No result is pooled across task families before the equal-family macro is computed.

## Run

The config and registration must be committed before any provider outcome:

```powershell
$env:OPENAI_API_KEY = (Get-Content -Raw "$HOME\OneDrive\Desktop\GPTAPI.txt").Trim()
powershell -ExecutionPolicy Bypass -File scripts\run_lsa_kappa_surface_phase.ps1 `
  -Phase validate `
  -Config configs\rlm_wrapped_controller_mesh_v0.json `
  -Output experiments\rlm_wrapped_controller_mesh_v0 `
  -Module research_gym.scripts.bench_rlm_wrapped_mesh `
  -PythonExe C:\projects\rlm-official\.venv\Scripts\python.exe
```
