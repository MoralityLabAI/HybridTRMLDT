# SFT Export

The SFT export converts common-schema frames into JSONL records for supervised instruction tuning.

Run:

```bash
python -m research_gym.scripts.export_sft_dataset --source data/generated --out data/sft/hybrid_frames.jsonl
```

Each output record has:

```json
{
  "instruction": "Given the frame, predict the typed refinement decision.",
  "input": "{...frame json...}",
  "output": "{...decision json...}",
  "metadata": {
    "id": "frame-id",
    "family": "deduction",
    "soundness_type": "env_sound_dead",
    "source": "toy_skills"
  }
}
```

The `output` field is a compact decision target:

```json
{
  "family": "deduction",
  "soundness_type": "env_sound_dead",
  "label": "conflict",
  "output_state": {}
}
```

This is a frame-label export, not a claim that model-sound or experience-sound labels are hard deductions. The typed provenance remains explicit in both output and metadata.
