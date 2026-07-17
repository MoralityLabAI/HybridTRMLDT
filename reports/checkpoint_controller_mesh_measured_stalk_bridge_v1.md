# Checkpoint: Controller-Mesh Measured-Stalk Bridge v1

Date: `2026-07-17`

## Completed

- verified and sealed 16 Qwen layer-23 activation chunk hashes;
- extracted 10 typed rank-one restrictions without copying raw tensors;
- constructed the controller/Qwen external-product Laplacian;
- froze combined, controller-only, and categorical predictors;
- used a fresh task seed and complete fallback-family holdout;
- ran 128 within-type measured-weight shuffles and paired bootstraps;
- retained the failed bridge gate without tuning;
- serialized all null spectra and rhos for independent receipt reconstruction.

## Commands Run

```text
python -m research_gym.scripts.extract_measured_qwen_stalk
python -m research_gym.scripts.bench_controller_mesh_stalk_bridge
python -m pytest -q tests/test_measured_stalk_bridge.py
```

## Registered Result

- combined / controller / categorical rho: `+0.7523 / +0.5688 / +0.6402`
- combined minus categorical rho: `+0.1121`, 95% interval `[-0.0855, +0.3121]`
- combined minus controller rho: `+0.1834`, 95% interval `[+0.0075, +0.3755]`
- typed N0 mean rho / p: `+0.7582 / 0.9380`
- top-16 uplift delta versus best baseline: `-0.0112`
- registered bridge gate: failed

The external-product transform beats the six controller summaries, but exact measured Qwen restrictions do not
beat typed shuffles and top-set acquisition degrades. Source-specific neural value is not established.

## Files Changed

- `configs/qwen08_l23_measured_stalk_source_v1.json`
- `configs/controller_mesh_measured_stalk_bridge_v1.json`
- `data/bridge/qwen08_l23_measured_stalk_v1.json`
- `research_gym/analysis/measured_stalk_bridge.py`
- `research_gym/analysis/controller_mesh_stalk_bridge.py`
- `research_gym/scripts/extract_measured_qwen_stalk.py`
- `research_gym/scripts/bench_controller_mesh_stalk_bridge.py`
- `tests/test_measured_stalk_bridge.py`
- `data/benchmarks/controller_mesh_measured_stalk_bridge_v1.json`
- `experiments/controller_mesh_measured_stalk_bridge_v1/`
- `docs/controller_mesh_measured_stalk_bridge.md`

## Tests

- focused bridge suite: `5 passed`
- full suite: `162 passed`
- simulated install without neural extra: `122 passed, 6 skipped`
- dependency-free core import: passed with `torch_available=False`
- deterministic replay: byte-identical
- compileall: passed

## Receipts

- source receipt SHA-256: `61bf8342937b6e83d221aad86874af5f40028aac5ed7925ab1885e3047cb18d0`
- bridge config SHA-256: `bbdeeb3fbdcd13cba087c4fb26d9d39488096743973416010615ff85444ac8b1`
- measured graph SHA-256: `0657e27150353572bc6b1cee9076b06cf2be6ab003be461f2afdb32c8db4704e`
- calibration geometry SHA-256: `338b2afeab2e43f9eb7d3d8d2a0c27be7cdff835ac98480ce73df93e1e1c349f`
- discovery outcomes SHA-256: `feebfeae6ce1ddb6d3bc0d90180d8b03825897062ccbec9fef76f4a02c4170ac`
- triple prediction SHA-256: `35c3babfae7b3949de009851f84bbe5782d69161fa4a24c0f2611969a27e787d`
- held-out outcomes SHA-256: `969fcc3b21fd9557f4300bcc0723041a70cfee34bb870a36498dc74246f79cb9`
- analysis receipt SHA-256: `b87888e0d62f40d18303897340e981a86b1023d32e51e3a37e668b5facaad355`
- result-file SHA-256: `f689608403ff920ee838b8b2034baafe2b5601e3e7784f3b582b52eccb3a2423`

## Decisions Made

- use an external product rather than pairing unrelated Qwen and controller episodes;
- bind the minimum construction/validation lineage for each measured restriction;
- compare against both controller-only and categorical predictors;
- interpret N0 failure as rejection of source-specific attribution;
- preserve the nonlinear-product gain as a separate, narrower observation.

## Open Questions

- Can task-aligned measured stalks discriminate their matching controller families from mismatched stalks?
- Does source identity transfer under a complete model/site holdout?

## Recommended Next Step

Require multiple task-aligned measured stalks and source-identity holdout before another bridge claim. Do not tune
this frozen graph against its revealed outcomes.
