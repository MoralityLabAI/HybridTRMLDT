# MCP-RU02: matched-state utility supervision ablation

**Result:** Utility-target supervision remains useful after holding training states,
candidate slates, optimizer steps and public horizon features fixed. The separate
`expand` / `read_symbol` heads have not shown a reliable advantage over a
single shared utility readout. These are *generated Python repository localization*
results, with no downstream SWE-bench patch grading.

## Pinned evidence

- Source head tested: `2a2017f8c8a1c4205d85e9717cd25bab906b2e7e`
- [GitHub Actions 44926236587](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/44926236587), both matrix seeds passed
- Seed 211 receipt ZIP: workflow artifact `mcp-ru02-matched-211` (14715721451)
- Seed 223 receipt ZIP: workflow artifact `mcp-ru02-matched-223` (14715787307)
- 24 train, 8 dev and 16 held-out test Git roots **per seed**. Held-out names
  changed via `--holdout-symbol-names`. The original eight issue-defect
  semantics and source layouts remain shared across splits.
- Four learned arms fit from **one exact `train.json.gz` slate set** per seed.
  Seed 211: 1115 slates, 140 optimizer steps/model, peak RSS 860,188,672 B.
  Seed 223: 1140 slates, 144 steps/model, peak RSS 935,358,464 B.
  They use identical optimizer seeds, batch sizes and schedules. The shuffled
  target model uses an independent permutation RNG that preserves training
  state order.
- Two legacy heuristic arms: lexical BM25 and identifier match, evaluated on
  the same roots and 24 MCP call / 1,024 packet-*estimated*-token limits.
  Packet counts still use `ceil(UTF-8 bytes / 4)`; exact model tokenizer
  accounting remains outstanding.

## Aggregate across all 32 held-out roots

| Router | Mean gold-symbol recall | Mean gold-file recall | Mean final packet estimate | Mean symbol reads |
|---|---:|---:|---:|---:|
| Binary labels, matched slates | 0.2667 | 0.7813 | 297 | 2.59 |
| Shared utility head | 0.4938 | 0.7813 | 359 | 4.06 |
| Dual expand/read utility heads | **0.4953** | **0.8021** | 473 | 5.75 |
| Dual heads, permuted utility targets | 0.1854 | 0.4167 | 575 | 9.97 |
| BM25 lexical | **0.6760** | **0.8021** | 1000 | 18.84 |
| Identifier rules | 0.6542 | 0.7604 | 1002 | 18.84 |

Paired root-bootstrap intervals (5,000 samples, seed 412):

| Dual utility minus control | Symbol-recall delta | 95% percentile CI |
|---|---:|---|
| Binary on identical slates | **+0.2286** | [+0.0995, +0.3568] |
| Single shared utility head | **+0.0016** | [-0.0813, +0.0943] |
| Dual with shuffled utility labels | **+0.3099** | [+0.2005, +0.4135] |
| BM25 lexical | **-0.1807** | [-0.2661, -0.0990] |

Seed-specific matched dual vs binary was +0.3188 in seed 211,
but +0.1385 in seed 223. The second seed's individual CI included zero.
Across both seeds there are 18 improved, 11 tied, 3 regressed roots.

## Interpretation and next pre-registration

The useful signal comes from supervising **marginal value of actual symbol-body
admission**, rather than training on a binary label that rewards every ancestor
query as though it were a symbol read. A simpler shared utility head learns
about as well as the two-headed readout on this sample. The dual model used
larger packets and more symbol reads, with roughly equal average recall.

The matched experiment establishes a gain over the specific binary baseline
and over shuffled-target supervision. It does not establish superiority to
BM25 on symbol recall, nor validate the proposed SetTRM-LDT architecture.
The original RU01 across non-matched training data showed larger gains and
must retain its documented slate/update-count confound.

A further CPU experiment could compare a deterministic **last-eight-calls
read deadline** with the trained utility arms, under matched calls and packet
budgets, while measuring file preservation and symbol gains separately.
The next architectural upgrade is an explicitly coupled, permutation-equivariant
candidate frontier with persistent typed state, in which LDT's hard exclusions
remain host-certified (`test_path`, `doc_path`).

The 27B campaign should focus on MCP-04's typed history and context preflight,
using the native doc-edit and event-recon graders. It should keep its result
separate from these CPU-only source-localization estimates.
