# RLM-Wrapped Typed Controller Mesh v0

## Construction

An official RLM wraps one atomic `mesh_resolve(policy_hint)` capability. The inner mesh contains proxy-TRM,
trained-ControlTRM, exact-LDT, deterministic arbitration, and a certificate-bound executor. The RLM can choose
only `consensus`, `trained_first`, or `ldt_conservative`; it cannot submit an action or manufacture a commit.
Missing or invalid wrapper output executes the fixed consensus mesh.

This calibration-informed diagnostic uses 8 hash-selected tasks from the previously opened
v1 evaluation suite. It is an architecture-extension pilot, not a fresh held-out efficacy test.

## Results

| Architecture | Utility | Accuracy | Unsafe | Contract | Fallback | Errors | Tokens | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `trained_trm_ldt_fixed` | 0.7505 | 0.8750 | 0 | - | 0.1250 | 0 | 0 | 0.0 |
| `mesh_fixed_consensus` | 0.7437 | 0.7500 | 0 | - | 0.0000 | 0 | 0 | 0.0 |
| `mesh_forced_no_tool_fallback` | 0.7437 | 0.7500 | 0 | 0.0000 | 1.0000 | 0 | 0 | 0.0 |
| `ldt_only` | 0.6848 | 0.3750 | 0 | - | 0.0000 | 0 | 0 | 0.0 |
| `rlm_mesh_atomic_tool` | 0.6712 | 0.7500 | 0 | 0.2500 | 0.7500 | 1 | 58,642 | 75.4 |
| `rlm_mesh_text_router` | 0.5358 | 0.5000 | 0 | 0.0000 | 1.0000 | 2 | 27,697 | 63.6 |

Typed unsafe executions: `0`. Cell errors remain zero-utility outcomes:
`3`. The forced no-tool control reproduced fixed-mesh actions and utility:
`true`.

## Matched Contrasts

- `rlm_mesh_atomic_tool` vs `mesh_fixed_consensus`: -0.0725; family deltas latest_rule_action=+0.0000, multi_hop_reachability=+0.0000, provenance_gate=+0.0000, storyworld_control=-0.2900.
- `rlm_mesh_text_router` vs `mesh_fixed_consensus`: -0.2079; family deltas latest_rule_action=-0.3917, multi_hop_reachability=+0.0000, provenance_gate=-0.4400, storyworld_control=+0.0000.
- `rlm_mesh_atomic_tool` vs `rlm_mesh_text_router`: +0.1354; family deltas latest_rule_action=+0.3917, multi_hop_reachability=+0.0000, provenance_gate=+0.4400, storyworld_control=-0.2900.
- `mesh_fixed_consensus` vs `trained_trm_ldt_fixed`: -0.0068; family deltas latest_rule_action=+0.1177, multi_hop_reachability=+0.0000, provenance_gate=-0.0600, storyworld_control=-0.0850.

## Boundary

This calibration-informed eight-task pilot evaluates one practical atomic typed controller mesh wrapped by the official RLM control flow. It reuses previously opened synthetic tasks, uses the gym ControlTRM rather than official TinyRecursiveModels, and does not test sheaf-spectral predictions, general orchestrator safety, neural alignment, or model superiority.

`Manipulation` is not an endpoint here. The atomic-tool contract measures capability invocation and certificate
completion only. The practical action mesh is topology-compatible with the controller-mesh program, but this
pilot does not compute a sheaf spectrum or test a spectral prediction.

## Integrity

- Result SHA-256: `99173364f810210f36583a48d7766362e9e20f1ea3deb507ea092552e6970e37`
- Records SHA-256: `751034eadde72d0851bf62dd150015a9fc1b5e6d0492bb4e64e2d27a8efd6e1f`
- Trajectory manifest SHA-256: `3f7a5b9322a55b8b166e55e3b64a94481bed26ab9132386ff1bceb9022524fbf`
- Config SHA-256: `40a57bbdfc9432e27cdb5c1cc00f1e5f7cd04cf93778f089e514c4818900ee00`
- Mesh topology SHA-256: `e1bb3a22a3d022b654322c2f14ddc3ddbb7fd7250b750b2ec30bf2a28b60e1d7`
