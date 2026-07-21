# RLM Architecture Neighborhood v0: Registration

## Question

How does the official Recursive Language Model control-flow pattern behave near three adjacent inference architectures under a fixed small long-context task bundle?

The four points are direct full-context inference, deterministic four-part map/reduce, RLM depth 1, and RLM depth 2. They are a control-flow neighborhood, not four trained neural architectures. RLM depth counts REPL-bearing recursive levels; at the depth ceiling, `rlm_query` becomes a plain LM call.

## Tasks

The sealed bundle contains one deterministic instance from each family:

1. Exact needle retrieval.
2. Three-hop relational join.
3. Global numeric aggregation.
4. Latest-version resolution.

Each prompt contains more than 600 lines. Answer-token match is reported per task and architecture. The known answer is held by the scorer and is not separately disclosed to the inference adapter.

## Neighborhood

| Architecture | Externalized context | REPL | Recursive depth | Partition policy |
|---|:---:|:---:|---:|---|
| direct full-context | no | no | 0 | none |
| map/reduce-4 | yes | no | 0 | deterministic four-way |
| RLM depth 1 | yes | yes | 1 | model-adaptive |
| RLM depth 2 | yes | yes | 2 | model-adaptive |

All cells use the pinned official implementation at commit `72d6940142ddfb84ee6be573dc999a37e633e671` and `gpt-5-nano`. The provider has no registered deterministic seed, so this pilot does not estimate stochastic uncertainty.

## Safety

The official local REPL is not used unchanged because it exposes filesystem access and unrestricted imports. The registered adapter stores context directly in memory, disables `open`, blocks OS and subprocess modules, and permits only `re`, `json`, `math`, `statistics`, and `collections`. Generated code and errors remain in sealed per-cell trajectories.

The process runs under the existing Windows Job Object wrapper: 2048 MB process RAM, 50% CPU, sustained 50 MB/s I/O abort, 1500 MB VRAM ceiling, and 1800 seconds for the phase. The external RLM virtual-environment interpreter is explicit in the resource receipt.

## Endpoints

No scalar winner is registered. The report retains:

- per-task answer-token match;
- provider calls and tracked input/output tokens;
- wall time;
- RLM iterations, code blocks, subcalls, recursive subcalls, observed depth, and REPL errors;
- the accuracy/calls/tokens/time Pareto frontier.

The official token ceiling is soft between iterations and may overshoot during the final iteration. Any cell failure remains an incorrect outcome; restrictions and budgets are not relaxed after seeing results.

## Claim Boundary

This is a four-task, one-run-per-cell API pilot. It cannot separate recursion from prompts or provider behavior, estimate uncertainty, establish asymptotic long-context scaling, compare trained model architectures, or support a general RLM-superiority claim.
