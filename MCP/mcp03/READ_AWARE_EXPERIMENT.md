# MCP-03-RU01: separate expand and read_symbol utility heads

**Status:** experimental training implementation. No new retrieval improvement or SWE-bench
resolution result is claimed until the paired campaign has executed.

## Implementation and provenance

- `MCP/mcp03/utility.py` computes private, offline labels for *newly admitted symbol
  bodies* at one fixed packet budget. For a `read_symbol` candidate it calls the
  existing packet composer with and without the read and scores its incremental
  gold-file and gold-symbol recall, rescaled by the number of targets and the
  exact symbol-snippet token count. It gives zero credit for a previously read
  symbol or a body that does not fit in the packet.
- Expansion targets use a **0.30 discounted reachability proxy**, conditional on at
  least two calls remaining and uncovered gold symbols. This is intentionally
  labeled as a proxy, since true expansion value requires on-policy
  counterfactual child rollouts. A follow-up DAgger experiment should replace
  it with multi-horizon value targets.
- `run_code_episode` exposes the remaining call count as a **public** observation
  field, while placing private `utility_values` only in the dataset recording
  callback. Live `Policy.score` sees neither gold labels nor the judge.
- `MCP/mcp02/models.py` supports `"utility_heads": "expand_read"` for transformer,
  looped, TRM, TRM+LDT and HRM architectures. The shared encoder feeds
  separately calibrated expand and read heads; a one-token public horizon
  marker distinguishes early, late, and final-call regimes.
- `MCP/mcp02/train.py` uses a normalized utility-weighted listwise objective
  with a 0.2-weight smooth-L1 calibration penalty. Old binary-listwise
  checkpoints follow the unchanged path. Existing trained checkpoints are
  not replaced.

## CPU development reproduction

This configuration names the *previously inspected* django/sympy source cohort.
It is useful for code and regression debugging, and is not an independent result.

```bash
CFG=MCP/mcp03/configs/read_aware_cpu.json
mkdir -p MCP/runs/readaware/checkpoints
python -m MCP.mcp03.run dataset --config "$CFG" --split train --out MCP/runs/readaware/train.json.gz
python -m MCP.mcp03.run dataset --config "$CFG" --split dev --out MCP/runs/readaware/dev.json.gz

for model in transformer_l2_d64_readaware looped_p4_d64_readaware trm_ldt_c2i1_d64_readaware; do
  python -m MCP.mcp02.train trainprobe --config "$CFG" --model "$model" \
      --train MCP/runs/readaware/train.json.gz --steps 40 --device cpu
  python -m MCP.mcp02.train fit --config "$CFG" --model "$model" \
      --train MCP/runs/readaware/train.json.gz --dev MCP/runs/readaware/dev.json.gz \
      --out "MCP/runs/readaware/checkpoints/$model.pt" --device cpu
done
```

The `work_dir` in this configuration is inherited from the original MCP-03
Windows setup and must refer to a local bare-repository snapshot directory.
Only run the trainprobe first if memory pressure is present. Record peak
process RSS; impose a 3 GiB RAM cap on the experiment. CI runs the stdlib
contracts independently of checkpoint generation.

## Fresh experiment and promotion gates

1. **Freeze the historical baselines.** The 109 already inspected SWE-bench
   Lite instances and the official 26-task end-to-end campaign are development
   evidence. Do not tune on their native grader outcomes. Preserve the 26
   assigned denominator and the separately audited gold-valid denominator.
2. **Generate new platformer defects.** Draw seeded, repository-root-disjoint
   projects with collision, scrolling, state and sprite defects, mixing
   misleading filenames and multiple viable symbol bodies. Save generator
   version/seed, commit and issue hash before any evaluation; split by root and
   family, not just by instance. Run native unit tests before and after each
   reference patch. Keep gold labels inside the training/judge process.
3. **Match the retrieval budgets.** Evaluate BM25+map, old looped, independent
   dual-head looped, and TRM+LDT with the same **24 MCP calls** and a
   **1,024-token final context packet**. Report file recall, symbol recall,
   all-files-hit, admitted symbol bodies, repeat reads, router FLOPs, wall
   time and **total prompt tokens**. Distinguish any BM25 baseline that uses an
   additional counted `file_search` call.
4. **Read utility ablations.** Compare direct binary labels, read-aware listwise,
   read-aware plus calibrated regression, fixed expand/read priors, and
   per-action value shuffled. Count candidate-type selections. Acceptance
   requires improved symbol recall with a paired confidence interval whose
   lower bound exceeds zero and file recall within 0.02 absolute of the
   matched old-looped baseline.
5. **Escalate architecture only after admission works.** Next replace
   independent candidate passes with a bounded permutation-equivariant
   SetTRM frontier, carry typed `z_t`/read ledger through turns, and place
   certified LDT projections inside the recurrence. Host-owned verifiers
   currently certify `test_path` and `doc_path` only. Coverage and stop
   hints stay soft until they have sound exact certificates.
6. **Independent SWE evaluation.** Select fresh externally curated repair
   tasks before reading their gold patches. Use the native grader and keep
   equal model, patch-generation policy and total retrieval budgets. Report
   resolved/assigned and **model tokens per successful repair**, including
   the failed runs in the aggregate cost denominator.

## Known limitations

The current expand labels are discounted reachability proxies. The current
model evaluates candidate pairs independently, even though it has separate
heads. It is a small controlled step toward SetTRM-LDT, not the claimed
frontier-coupled architecture. This branch collects labels using the private
oracle only for training; runtime policy features are public. No new trained
weights, CPU RSS receipts, sealed-root comparisons or patch-resolution gains
have yet been produced by this change alone.

Related: [MCP parent PR #14](https://github.com/MoralityLabAI/HybridTRMLDT/pull/14),
[MCP-04 issue #15](https://github.com/MoralityLabAI/HybridTRMLDT/issues/15),
[MCP/TRAINING_STRATEGIES.md](../TRAINING_STRATEGIES.md).
