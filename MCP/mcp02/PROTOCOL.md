# MCP-02 protocol

MCP-02 extends the [MCP-01 gym](../README.md) from explicit-key lookups to compositional retrieval across several MCP servers, and replaces the single deterministic controller with interchangeable routing policies. MCP-01's code and its recorded receipts under [`MCP/results/mcp01`](../results/mcp01/README.md) stay unchanged as the reference baseline.

## Questions

1. Which router family finds the evidence for a task with the fewest tool calls and the least local compute: lexical BM25, a trained dense dual encoder, a feed-forward Transformer, a shared-weight looped Transformer, a TRM with recurrent latent refinement, a TRM whose explicit candidate sets are refined through certified LDT exclusions, or an HRM?
2. At matched FLOPs per decision, does recurrence with fewer parameters match distinct layers? At matched parameters, does added recurrent depth help, including depth added only at test time?
3. Does a deeper index (server, collection, capability, provenance, name bucket, entity) save more tool-result traffic than its extra traversal costs?
4. What is the total cost per accepted task once local compute, tool calls, verification and the downstream handoff are all counted?

## Environment

A case is one synthetic organisation served by seven MCP servers with overlapping capabilities: catalog, org, finance, procurement, observability, a replica mirror and an archive. A task names a service by display name and region, asks for a value reached through zero to three relation hops (owner team, cost center, vendor, nested budget section), and may combine two chains by sum or difference.

| Family | Main difficulty |
| --- | --- |
| `alias_lookup` | A twin service has the same display name in another region; region appears only after a read |
| `join_chain` | Service, team, cost center, then a nested budget section linked from the cost center |
| `join_arith` | Two independent chains combined by sum or difference |
| `stale_moved` | The team record moved; the old copy is superseded and may have unknown registry status |
| `epoch_roll` | The catalog epoch advances after a hidden number of calls; the vendor record is replaced and old index node IDs expire |
| `conflict_authority` | Replica and archive copies disagree; a replica title imitates the request's wording |
| `missing` | The system of record reports no value while a replica does; the correct packet abstains and cites the records |
| `paginated` | The needed record sits late in a long collection and is excluded from every branch sketch |

Every world also contains confusable names that share a word with the anchor, replica and archive copies, regional budget sections in another unit, registry entries with unknown status, scope or unit (about 12%), and unrelated distractor entities up to the requested catalog size.

**Ground truth.** Expected answers and required documents live only in the case's `oracle` and `hidden` fields. The environment returns neither. Its judge requires the exact answer, the abstention flag, and a citation of every required document by a ref issued in this episode for the document's current revision. Extra, stale, forged or unread refs fail.

**Grouped splits.** Entity and server names come from three disjoint word pools, one per split. Request phrasing (intro and closing templates) is assigned to exactly one split: four template groups for `train`, one for `dev`, one for `test`. Field synonyms and server role descriptions are shared language. Only `train` worlds produce training data, and `dev` worlds give a held-out ranking check. `test` worlds appear only in evaluation, and `train.py` refuses them.

## Discovery arms

| Arm | Tools | Notes |
| --- | --- | --- |
| `flat` | `catalog_list`, `resource_read` | One call returns every descriptor |
| `search` | `catalog_search`, `resource_read` | BM25 over descriptors, paginated |
| `layered` | `index_children`, `resource_read` | Facet hierarchy given by the layout string |
| `layered_ldt` | `layered` plus `candidate_refine` | Explicit candidate set per node, certified exclusions |

Layouts are strings over facets: `S` server, `C` collection, `K` record kind (capability), `P` registered provenance (scope and status), `B` name bucket with a partial six-name sketch, `E` entity. Branch rows show a facet label, a sketch and a record count. Node IDs are bound to the catalog epoch.

**Certified exclusions.** In `layered_ldt` a policy proposes `(child, predicate)` pairs with predicates `kind`, `status`, `scope`, `unit` or `any`. The environment evaluates each proposal separately through `certify_and_apply_verified(require_progress=True)`. Its verifier accepts only when every record below the child violates the predicate for every record kind the task can cite, using registered metadata. Unknown metadata never proves a violation. Candidate state is bound to the catalog epoch and digest; an epoch change starts a new state. Model scores only order candidates. The private counter `certified_exclusions_containing_required` must stay zero.

## Shared observation and action

Every router sees the same observation:

- the public task;
- the open evidence needs from the shared plan tracker (`Composer`), which reads only the public plan and read results;
- a frontier of candidates, each one expandable by exactly one MCP call.

Candidates are index branches, result pages, resources, links (nested sections, replacements, revalidations after an epoch change) and search queries. When more than 32 non-link candidates are eligible, a BM25 prefilter keeps 32 of them, and this applies to every policy. A policy returns one score per candidate. The loop expands the best unvisited one. The only other action is an exclusion proposal in `layered_ldt`.

Learned routers encode each candidate identically: hashed word tokens `[CLS] needs [SEP] candidate` (4096 buckets, at most 74 positions) plus a nine-value structural vector (item type, depth, page cursor, log record count). No BM25 or match features are given, so matching must be learned.

## Policies

| Policy | Family | Notes |
| --- | --- | --- |
| `deterministic_rules` | Control | Hand-written public rules: identity match, record kind, provenance, role words |
| `lexical_bm25` | Lexical | BM25 of needs and request against candidate text |
| `sham_random` | Sham | Random scores with the same loop and candidate pipeline |
| `sham_shuffled_*` | Sham | A Transformer trained on labels shuffled within each state |
| `oracle_upper_bound` | Bound | Reads private labels; reported only as a ceiling |
| `dense_*` | Dense | Trained dual encoder: mean-pooled embeddings, MLP towers |
| `transformer_l{1,2,4}_*` | Transformer | Distinct pre-LN blocks, cross-encoder |
| `looped_p4_*` | Looped | One shared block, four passes with pass embeddings |
| `trm_c2i1_*` | TRM | Shared block; per cycle one latent update from (x, y, z), then the answer update from (y, z); two cycles |
| `trm_ldt_c2i1_*` | TRM+LDT | TRM plus an exclusion head; its proposals go through certification |
| `hrm_h2l1_*` | HRM | Fast block once per slow update, two slow cycles |

In `layered_ldt`, policies without an exclusion head use the symbolic proposer, which reads public registry metadata and facet labels. `trm_ldt` uses its learned head.

**Training.** Imitation of the oracle labels from `train` worlds, with behaviour policies equal to the oracle with epsilon 0 and 0.35 random exploration. The loss is a listwise softmax over each state's candidates, with every positive counted. The exclusion head adds a cross-entropy term on certifiable predicates. Epochs, learning rate and seed are fixed in the config; there is no early stopping.

## Measurement

| Quantity | Source |
| --- | --- |
| Parameters | Counted from the model, total and non-embedding |
| Recurrent depth | Block applications per forward, from the config and any test-time override |
| FLOPs | `torch.utils.flop_counter.FlopCounterMode` on one candidate at the fixed length, times candidates scored |
| Latency | Wall clock per router decision and per episode on the recorded host |
| Memory | Weight bytes, plus peak working set around a 32-candidate forward in a fresh process (`run.py memprobe`); CUDA peak when on GPU |
| Tool traffic | Tool calls, refine calls, LDT checks and metadata checks, exact result bytes |
| Handoff | Exact UTF-8 bytes of the packet; 2048-byte budget |
| Outcome | Accepted (answer, citations, abstention, budget), evidence recall, failures by type, retries |

Cost per accepted task divides the summed cost of every assigned episode, failures included, by the accepted count. Prices are labelled assumptions in [`configs/economics.json`](configs/economics.json), and four scenarios show sensitivity. The handoff token figure uses a 4-bytes-per-token estimator; exact bytes are reported beside it.

## Stages

- **A. CPU smoke** ([`configs/smoke_cpu.json`](configs/smoke_cpu.json)): tiny data, one epoch, every architecture at width 32. This stage checks plumbing only.
- **B. CPU pilot** ([`configs/pilot_cpu.json`](configs/pilot_cpu.json)): width 48 and 18.7k training decision states over two epochs. Evaluation uses 32 test worlds per cell at catalog sizes 256 and 1024 in all five arm configurations, plus depth, deliberation and layout sweeps.
- **C. GPU pilot** ([`configs/pilot_gpu.json`](configs/pilot_gpu.json)): width 128, three times the data, and the LLM controller. It runs on the RTX 5080 when the card is free, and stays separate from CI.

## History

Before any learned router was trained, control-policy smokes exposed three environment defects, which were fixed:

1. Pagination in `layered_ldt` indexed the filtered child list, so exclusions could hide unseen children. Cursors now index the unfiltered order.
2. The epoch roll fired after the read that crossed the threshold, so an agent could finish without seeing the change. The roll now happens before that call executes.
3. The plan tracker ignored reads made while revalidating. It now reuses current-epoch reads.

Branch labels also gained collection kind words, and buckets shrank from 24 to 12 records with six-name sketches, after the deterministic rules solved under half of the layered episodes. The learned-model configs were written before any learned router was evaluated.

**Encoder amendment.** The CPU pilot's Transformer-L2 (tokens-only encoder) reached a dev top-1 of 0.62. Decision traces on dev worlds showed it read many wrong resources. It had not learned to recognise that a candidate's entity id equals the needed id from hashed tokens alone. At that point no learned router had been evaluated on test worlds. An encoder option `match` was then added: tokens present in both the needs and the candidate get segment ids 2 and 3, and a fixed list of template glue words never counts. The option supplies no BM25 score or count. On the smoke data it raised the same Transformer-L2's dev top-1 from 0.40 to 0.77.

The amendment was applied as follows:

- **CPU pilot:** stays on the registered tokens-only encoder, plus an ablation pair (`transformer_l2_d48_match`, `trm_c2i1_d48_match`).
- **GPU pilot:** uses `match` for every router except `transformer_l2_d128_tokens`, the tokens-only anchor. That checkpoint was trained before the switch and renamed. Two other d=128 tokens-only checkpoints (dense, Transformer-L1) were set aside unevaluated, and one in-progress fit (Transformer-L4) was stopped.

**Run logistics, recorded as they happened.**
- Both pilots ran on a CPU and GPU shared with other sessions' jobs. Evaluations were split into processes by policy subset or instance shard and then merged; resumable runners keep one record per job and case, so splitting changes no outcome.
- The GPU pilot was paused after its MCP-02 grid so the MCP-03 routers could train inside a single VRAM lease, and it resumed afterwards.
- The LLM-controller stage was stopped after six Qwen2.5-0.5B episodes because its server outgrew the job's VRAM lease on a nearly full card. The planned Qwen3-1.7B run did not execute.
- Training receipts before the Windows peak-RSS fix carry CUDA peaks only. MCP-03 training memory comes from fixed-step `trainprobe` runs in fresh processes.

## CI

[`.github/workflows/mcp-gym.yml`](../../.github/workflows/mcp-gym.yml) runs the MCP-01 and MCP-02 contract tests without torch, a deterministic MCP-02 control smoke with summary and replay, and a torch-free replay of the committed pilot tapes. Learned-router training and inference run locally.
