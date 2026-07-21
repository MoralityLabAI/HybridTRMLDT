# RLM Architecture Neighborhood v0: Construction Failure

The registered 16-cell run produced no architecture evidence. The configured OpenAI credential was rejected with HTTP `401 invalid_api_key` before inference in every cell. The official client reported zero calls and zero tokens.

The all-zero accuracy and Pareto fields in the machine result are therefore **inadmissible** and must not be interpreted as model performance. The output directory is retained only as a failure receipt and adapter audit. It will not be passed to the finalizer or copied to the canonical benchmark receipt path.

The resource wrapper completed in `15.679 s`, cleanup passed, and no provider usage was reported. All 16 error trajectories, the result, events, records, trajectory manifest, and resource receipt are hash-attested by `construction_failure_receipt.json`.

No alternative OpenRouter, Portkey, Vercel AI Gateway, Prime, Ollama, LM Studio, or vLLM transport was available locally. The authenticated Codex CLI passed a capability probe with its configured full model, but `gpt-5-nano` and `gpt-5.1-codex-mini` are unsupported through the ChatGPT-backed transport. A recursive run on the full Codex model was not launched because every call carries substantial coding-agent context overhead and would consume materially more quota than the registered low-cost API design.

## Recovery Rule

Preserve this attempt byte-for-byte. After a valid low-cost provider credential is available, register a successor that retains the task-suite hash, official RLM commit, architecture hashes, safety restrictions, and endpoint definitions. Add a provider capability gate before the task loop and stop after the first global authentication failure.
