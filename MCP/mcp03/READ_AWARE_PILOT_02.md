# MCP-03-RU01: multi-defect platformer retrieval, two-seed CPU confirmation

**Status:** positive *generated-code retrieval* result, pending independent
repository families, a stricter tokenizer, matched-label ablations and native
SWE patch grading. This is not a SWE-bench resolution result and is not a
trained SetTRM. The first eight-root easy pilot was saturated and retained
separately in `READ_AWARE_PILOT_01.md`.

## Source and receipts

- PR: https://github.com/MoralityLabAI/HybridTRMLDT/pull/17
- Generator: `MCP/mcp03/platformer_gen.py` (`platformer-gen-v3`).
- Confirmed run: https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/44918324126
- Seed 131 artifact: `mcp03-readaware-cpu-receipts-131` (ID `14712975504`).
- Seed 149 artifact: `mcp03-readaware-cpu-receipts-149` (ID `14713155742`).
- Source commit `6441e0a5cf0634b408be76607f7c10d5d254a5b2`.
- Each seed: 24 train / 8 development / 16 held-out Git repository roots;
  three planted defects in each generated test case, 72 misleading
  lookalike modules, failing original native regression test, passing reference
  patch. Training and test roots are disjoint. The generator was adjusted to
  freeze Git author/committer timestamps, because the Git commit SHA enters the
  policy's episode key for epsilon-oracle rollout sampling. Repeating a seed
  now reproduces the commit SHA; there is a corresponding test.
- Packet budget: 1,024 **byte-based estimated** tokens. Both learned routers
  receive 24 total counted MCP calls; token accounting is not yet tokenizer-exact.
  No downstream patch generator is used here.
- Three controls: BM25 lexical; identifier rules; private-gold-assisted
  `oracle_upper_bound` (diagnostic ceiling only). No gold in runtime policy
  observations.

## Results at the matched call and packet limits

| Seed | Router | Gold-symbol recall | Gold-file recall | Mean packet tokens (estimate) | Mean symbol-read actions |
| --- | --- | ---: | ---: | ---: | ---: |
| 131 | Binary looped P4-D64 | 0.1385 | 0.9375 | 253.9 | 2.188 |
| 131 | Read-aware looped P4-D64 | **0.7031** | 0.9375 | 524.2 | 7.812 |
| 131 | BM25 | 0.6958 | 0.8125 | 984.4 | 18.625 |
| 131 | Private oracle | 0.7927 | 0.8750 | 977.1 | 18.375 |
| 149 | Binary looped P4-D64 | 0.4458 | 0.8125 | 346.1 | 4.250 |
| 149 | Read-aware looped P4-D64 | **0.7125** | 0.8958 | 429.6 | 6.188 |
| 149 | BM25 | 0.6896 | 0.8125 | 977.6 | 18.562 |
| 149 | Private oracle | 0.7625 | 0.8125 | 973.6 | 18.812 |

Mean over all 32 independently generated test roots:

| Metric | Binary looped | Read-aware looped | BM25 |
| --- | ---: | ---: | ---: |
| Gold-symbol recall | 0.2922 | **0.7078** | 0.6927 |
| Mean packet tokens (byte estimate) | 300.0 | **476.9** | 981.0 |

The paired read-aware gain against binary is **+0.415625** symbol recall,
with **30 positive, 2 tied, 0 regressed roots** and a 5,000-sample
seed-412 bootstrap 95% percentile interval approximately **[+0.3365,+0.4943]**.
Mean file-recall difference is **+0.041667**. Both seeds independently meet
the predeclared positive paired-symbol-gain and file-retention gates.

The read-aware model's +0.0151 symbol-recall advantage over BM25 at half
the packet estimate requires a stronger independent benchmark before claims
of general ranking superiority. In these synthetic tasks, every family and
core source layout is derived from the same deterministic templates, while
the corpus contains only eight defect families. Repository-root splitting
alone cannot establish transfer to novel codebases.

The oracle comparison is diagnostic and gold-assisted. Its result need
not be a global optimal packet upper bound, because the current oracle
scores one-step transitive binary reachability and is constrained by the same
packet composer and 24 calls.

## Runtime / training disclosures

- `looped_p4_d64_readaware`: **320,578 total parameters**, 57,922
  non-embedding parameters. Training used CPU only, four PyTorch threads.
- Seed 131: probe peak RSS **811,081,728 bytes**; full fit max reported
  RSS **960,229,376 bytes** for binary, **919,666,688** read-aware.
  156 optimizer steps for binary, 140 for read-aware.
- Seed 149: probe peak RSS **728,805,376 bytes**; full fit max reported
  RSS **944,463,872** binary, **919,740,416** read-aware.
  142 optimizer steps binary, 124 read-aware.
- All four trained-model episodes have full fixed test denominators and
  native packet replay; the original two-seed run completed both CI jobs
  successfully. Checkpoint SHA-256 values appear in artifact receipts.

**Important confound:** binary and read-aware training gather different
numbers of slates and prioritize different positives. The candidate encoder
is identical, but the supervision and slate sampling are both altered.
The next ablation must match rollout states, candidate sets, update count,
and optimization schedule before attributing gain to dual heads or
packet-marginal targets alone.

## Next gates

1. Replay identical seeds with fixed source SHA and compare checkpoint/episode
   digests to validate strict reproducing state beyond Git timestamp stability.
2. Add matched-slate binary, read-aware, utility-shuffle and fixed-prior
   ablations, including symbol exposure and read-call attribution.
3. Use new synthetically generated **structurally different** platformers
   with held-out function names, modules, and defect families.
4. Implement cached per-URI encodings, frontier set coupling and persistent
   typed latent/read state in the actual SetTRM-LDT architecture, retaining
   certified host-only pruning for `test_path` and `doc_path`.
5. Validate exact tokenizer accounting, then test against a fresh external
   SWE task cohort with the native grader; report resolved/assigned and
   total model tokens per successful repair.

Run-131 and run-149 receipt archives include the complete paired
`eval/episodes.jsonl`, `report.json`, CPU probe, manifest, checkpoint
receipts, source config and host provenance. Earlier `platformer-gen-v2`
run `44917953072` also showed a positive signal before timestamp pinning;
this v3 report replaces it as the reproducibility-aware development result.
