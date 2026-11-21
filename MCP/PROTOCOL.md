# MCP evaluation protocol

Protocol: `MCP-01`. Scope: synthetic retrieval, index traversal, checked refinement, and bounded evidence packets. The commands and current output fields are documented in [README.md](README.md).

## Stage A: public smoke

The first run exercises deterministic controllers across `flat`, `search`, `layered`, and `layered_ldt` on paired synthetic episodes. Its receipts establish reproducibility and expose accounting or interface failures. The task generator and policies are public. Tasks supply exact entity keys; the split labels partition generated source groups while retaining the task templates. These runs supply development evidence; they do not estimate generalization by a learned router.

A mocked chat-completions server verifies the local adapter's transport and parser. A live model run requires a local endpoint, a recorded response, and a usage/provenance receipt. The report must name which path ran.

Each report records the configuration and seeds, controller identity, arm, success or failure, and accounting totals. Preserve failed episodes, invalid responses, and budget exhaustion. Changes made after inspecting smoke results belong in the history of this stage.

## Stage B: frozen model comparison

Before a later model comparison, save a run manifest that fixes:

- Source revision, environment version, generator, task splits, and catalog hashes.
- Controller/checkpoint identities, tokenizer revisions, precision, parameter counts, and architecture fields.
- Discovery arms, fixed and adaptive depth settings, step limits, packet budgets, and model decoding settings.
- Training data, fitting budget, stopping rule, and the validation procedure used to choose settings.
- Primary comparisons, quality margin, sample size, confidence interval method, and multiple-comparison treatment.
- Host hardware, inference backend, warmup procedure, cache/reset policy, and usage-accounting source.

Stage B starts only after that manifest is frozen. A later change requires a new version and a fresh held-out set. The present release leaves the Stage B manifest and learned results pending.

Split at task-family or catalog-family level before generating paraphrases. Hold out schema names and catalog versions where feasible. A random row split of related templates is insufficient evidence of transfer. Generated answer labels and terminal judge records remain outside the controller's observations and training prompts for held-out episodes.

Use the same episode across arms, with independent controller state. Fit search and learned baselines on the same permitted split. Include ordinary lexical/embedding retrieval, a simple fitted reranker, and the relevant deterministic controls before attributing value to recurrence. The host-integration stage must include Goose Code Mode or an equivalent on-demand discovery baseline.

## Exact success

The environment checks the final packet against the episode's required result and its evidence contract. The submitted facts, units, source binding, and catalog epoch must agree with the retrieved evidence. A small packet that omits required facts fails. A forged or stale receipt fails even when its text resembles a correct answer.

Keep answer accuracy, evidence validity, and budget adherence as separate fields. Define joint success as all required checks passing. Report coverage and abstention so a policy cannot improve its apparent accuracy by answering only easy tasks. A final submission ends the episode; judge feedback cannot become an uncharged answer-search interface.

The environment owns the ground truth. LDT refinement checks use the task's stated predicates and exposed evidence. They must not access the private expected answer to choose a winning route for the controller. Record verification work separately if a new checker performs extra retrieval or computation.

## Accounting

Maintain separate ledgers for local exploration and the packet handed to the next agent. Count every model call, including calls whose output fails JSON validation. Failed tool attempts consume the relevant step/call budget. The current counters measure serialized JSON payloads for schemas, model messages, tool results, and handoff packets. They exclude transport envelope overhead. A later wire-traffic counter must identify its boundary. Token estimates require an explicitly named tokenizer or estimation method; this release reports exact bytes separately from server-reported tokens.

| Quantity | Meaning |
| --- | --- |
| Local model usage | Prompt and generated tokens returned by the local model server, when available |
| Model-request exposure | Serialized task, history, schemas, and observations included in a model request |
| Tool traffic | Tool attempts and serialized result bytes; additional wire counters must state their scope |
| Final packet size | Exact serialized UTF-8 bytes for the submitted packet |
| Runtime | Wall time, inference calls, tool calls, and registered depth allocation |
| Provenance | Observed server identity plus declared or independently verified checkpoint details |

Missing provider usage remains unknown. A usage estimate is reported alongside its estimator and cannot be relabeled as provider measurement. Missing parameter count or checkpoint identity also remains unknown. A response that reports token usage still leaves architecture provenance open.

For paired runs, compute total cost over every assigned episode, including failures. A cost-per-success metric uses that full numerator divided by joint successes. Zero successes yields an undefined/infinite cost-per-success value, with the failure count retained. A successful-episodes-only average can be supplemental if labeled explicitly.

Do not equate unique bytes discovered with tokens processed by a model. Repeated history can be charged on each inference call. Prompt-cache accounting and any local prefill reuse require their own observed fields. Frontier token or dollar savings require an actual downstream run or a clearly labeled calculation with explicit assumptions.

## Comparison criteria

The later study's quality criterion is joint success under the frozen evidence contract. Its efficiency result must accompany that quality result, with paired confidence intervals where the sample supports them. Report median and upper-tail latency; preserve the full episode distribution for audit.

Record the smallest baseline that matches the learned policy's quality and cost. Compare additional recurrent passes at a fixed parameter count, then compare model sizes at a fixed budget. Any claim that LDT adds value requires a measured behavioral difference from `layered` under the same information and budget. Count rejected refinements and lost valid candidates as well as accepted progress.

A learned route must beat the registered search/reranking controls on held-out data before it is promoted into the Data Goblin path. Follow that with a downstream task test in which the larger agent receives only the bounded packet and its available resource handles.

## Required coverage

The test suite should exercise the following failure paths before interpreting a model run:

- Controller observations exclude expected answers and private judge state; terminal feedback cannot reveal the target for a retry.
- Invalid action JSON, extra keys, missing required arguments, unknown tools, and malformed model usage remain visible failures.
- Budget accounting includes unsuccessful attempts, and a failed final packet cannot report joint success.
- LDT checks reject unsupported elimination, widening, forged provenance, and stale evidence. Each removal reduces rank; an unchanged page is recorded as a no-op. A later persistent-state controller must require strict progress for durable refinements.
- An empty or insufficient candidate set produces an explicit failure or abstention. The controller cannot recover a discarded valid candidate by silently widening a certified state.
- Stdio initialization, notification handling, request IDs, protocol errors, and clean shutdown work in a subprocess.
- The local-model endpoint restriction and request timeout are tested, while model identity and token usage remain separate receipt fields.

This list is a review gate. The smoke report identifies which cases the executable tests currently cover.

## MCP wire contract

The server implements a bounded subset of MCP `2025-06-18` over stdio: `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`, plus any methods explicitly listed by the server. It advertises only implemented capabilities. Public task metadata in the initialization result is a gym extension.

Messages use UTF-8 JSON-RPC and one physical line per message. Standard output contains protocol messages only; logs go to standard error. Notifications receive no response. Request IDs are preserved, including integer zero. Unknown methods produce a JSON-RPC method error. Invalid arguments or an unknown tool produce a protocol error; a recognized tool's execution failure uses `isError: true` in its result.

The client negotiates the pinned protocol and sends `notifications/initialized` before tool operations. The stdio implementation has bounded messages and request timeouts. Closing input starts shutdown, followed by process termination if needed. This scope supplies a callable gym and does not claim complete MCP client/server conformance.

Primary specifications:

- [Lifecycle, MCP 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).
- [Transports, MCP 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports).
- [Tools, MCP 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).
