# MCP-08: Context Economics and Autonomous World Construction

Implementation of **Stage MCP-08A** (CPU fixture construction, strict witness isolation, and deterministic independent verifiers) for *The Cost of Context II: Hardware-Aware Recurrent Memory for Autonomous Local Agents*.

## Contents
- [SPEC.md](SPEC.md) — Formal experimental specification (4 environments, 5 arms, 224 episodes, hardware-aware sweep, 8 storyworld builds, $\eta$ metric, 5 preregistered gates, stages 08A–08F).
- [manifest.json](manifest.json) — Machine-readable preregistration manifest.
- [fixtures.py](fixtures.py) — Deterministic generator for `dev` (8 roots), `pilot` (8 roots), and held-out `eval` (32 roots, 8 per environment) with `assert_zero_witness_leakage`.
- [verifiers.py](verifiers.py) — Independent deterministic verifiers for *The Flooded City*, *Quantum Labyrinth* (complex Hilbert replay), *Arena Architect* (isolated Node.js VM + Canvas draw invariants on unseen maps), *The Broken Archive*, plus native verifiers for *The Clockmaker's Republic* and *The Garden of Unmeasured Things*, $\eta$, paired bootstrap CIs, and gate checks.
- [test_mcp08.py](test_mcp08.py) — Full CPU verification suite.

## Run MCP-08A Verification
```bash
python -m unittest MCP/mcp08/test_mcp08.py -v
```
