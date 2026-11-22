# SFT export: legacy receipts and leakage-controlled RIVET data

## Use for new Silico training

```bash
python -m research_gym.scripts.export_receipt_sft \
  --source data/generated \
  --out-dir data/sft/rivet_stage1_v1 \
  --holdout-family reachability
```

The new `export_receipt_sft.py` uses an **input allowlist** containing only `family`, `input_state`, and `operation`. The supervision target is `label`, `predicted_soundness_type`, and `proposed_output_state`. The exported `train.jsonl`, `calibration.jsonl`, `eval.jsonl`, and `manifest.json` use deterministic public-state/operation group hashes. A repeated state cannot cross splits; named held-out families are directed to eval.

An exported `predicted_soundness_type` is a model prediction. Durable deductions remain gated by the independent `certify_and_apply` membrane or an environment-owned mechanics verifier. Verifier-informed **REPAIR** training requires genuine rollout receipts and is a later stage.

### Example stage-1 input (model-visible)

```json
{"family":"deduction","input_state":{"slot":["a","b"]},"operation":{"rule":"prune"}}
```

### Example target (supervision only)

```json
{"label":"refinement","predicted_soundness_type":"env_sound_dead","proposed_output_state":{"slot":["a"]}}
```

Inspect the manifest and group/family counts before training. When `--holdout-family` excludes every available frame family, generate additional families first. The default `data/generated` directory is populated by the regular reachability and MeTTa frame generators.

## Historical export and reproducibility

```bash
python -m research_gym.scripts.export_sft_dataset --source data/generated --out data/sft/hybrid_frames.jsonl
```

**This legacy exporter has answer leakage:** it supplies `Frame.to_jsonable()` in the model-visible `input`, which contains `output_state`, `soundness_type`, and `label`. `Frame.metadata` can also contain witness actions and counterfactual outputs. Do **not** use its outputs to report predictive SFT performance. Its original behavior remains unmodified for interpreting historical artifacts.

The existing tests preserve that legacy schema; separate tests in `tests/test_receipt_sft_export.py` assert input-field isolation and disjoint state-group splits.

For the experiment contract see `docs/silico_sft_vpd_protocol.md`, `docs/causal_receipt_recurrence_proposal.md`, and `configs/silico_rivet_v1.json`.
