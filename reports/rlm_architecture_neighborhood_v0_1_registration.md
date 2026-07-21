# RLM Architecture Neighborhood v0.1 Registration

This successor was registered after transport qualification and before any successor architecture-neighborhood outcome. It preserves the v0 task-suite hash, four architecture hashes, official RLM commit and source hashes, restricted-REPL policy, endpoint definitions, task order, and architecture order. It does not overwrite or reinterpret the sealed v0 construction failure.

The only scientific transport change is `gpt-5-nano` to `gpt-4.1-mini`: the supplied project credential is valid but returned HTTP 403 for `gpt-5-nano`; `gpt-4.1-mini` passed both a Responses API exact-response probe and an exact-response probe through the official RLM `OpenAIClient` chat-completions path. Those pre-registration probes establish transport availability only and are not benchmark observations.

The run has a second capability gate immediately before the task loop. Its receipt and usage are reported separately and excluded from architecture metrics. A 401, 403, `invalid_api_key`, or `model_not_found` error aborts the run before the first task, or after the first such error if access changes during execution.

## Frozen Scope

- Tasks: exact retrieval, three-hop join, distributed sum, and latest revision.
- Architectures: direct full-context, deterministic four-way map/reduce, RLM depth 1, and RLM depth 2.
- Primary endpoint: exact answer-token match per task and architecture.
- Efficiency endpoints: calls, input/output tokens, and wall time.
- Control-flow endpoints: iterations, code blocks, subcalls, recursive subcalls, observed depth, and REPL errors.
- Summary: unscalarized accuracy/calls/tokens/time Pareto frontier.
- Replication: one run per cell; no stochastic uncertainty claim.

The desktop credential is an execution input. It is not copied, hashed, logged, or committed.
