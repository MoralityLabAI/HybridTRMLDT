# LDT-RED02-A: real Qwen 0.5B in bounded CPU CI

[Protocol](RED02_CI_PROTOCOL.md) · [Prior source-only mock gym](README.md) · [Issue #10](https://github.com/MoralityLabAI/HybridTRMLDT/issues/10)

An actual historical, commit-pinned Qwen2.5-0.5B-Instruct checkpoint produces fictional adversarial action proposals, which the native typed Beast authority membrane adjudicates before any simulated action executes. The pilot uses previously excluded roots 901/902, horizons 2 and 8, and host-owned exact DP welfare/authorization ceilings.

The workflow downloads a pinned model snapshot in an explicit setup stage, hashes the actual weight and tokenizer bytes, then loads the model through the existing **local_files_only** adapter. No model weights are uploaded as artifacts. Raw Qwen completions, malformed output counts and authorization receipts are saved as a small JSON artifact. Runtime limit 30 minutes; no GPU required.

Reproduce locally if weights are already present (after verifying the actual HF revision):

\`\`\`bash
python -m unittest research_gym.benchmarks.ldt_adversarial_depth_gym.test_red02_live_ci -v
python -m research_gym.benchmarks.ldt_adversarial_depth_gym.red02_live_ci \
  --local-model /path/to/pinned/snapshot \
  --metadata /path/to/model_metadata.json \
  --out /tmp/ldt_red02_live.json
\`\`\`

This runs a **real red model** against the native host authorization gate; a neural LDT/BPD steering checkpoint is a separate registered stage. The task is wholly fictional and no real action, account, credential, or tool is exposed.
