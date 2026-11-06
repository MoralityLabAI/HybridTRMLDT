# Kimilings: KIMILINGS-01

Evaluation contracts for hybrid LDT/TRM routing over the verified Kimi-K3.1-Pro expert-index slices.

**Stage:** CPU-only preparation. The available pieces are weight bundles and an H100 synthetic-input instrument canary. There is no runnable 16-expert reduced Kimi, task-grounded VPD, or measured whole-model reasoning result here.

## Inputs and scope

- Original Kimi: https://huggingface.co/moonshotai/Kimi-K3.1-Pro . Its exact checkpoint revision must be pinned when the full-model capture service exists.
- Published 16-index slices: https://huggingface.co/AlephFunk/Kimi-K3.1-Pro-expert-slices-0-15 at commit 387322684297ff5f014d772f4ac6549c6d94a77d, extracted in Silico Task #1. The reported artifact is 8,832 tensors across 92 MoE layers, approximately 25.83 GB for all 16 stacks. Each stack is a collection of layer-local experts. The shared Kimi backbone remains a separate memory dependency.
- Prime Intellect logic: logic-env 0.1.2, dataset PrimeIntellect/INTELLECT-3-RL logic subset, with upstream task-specific verifier. The repository's research_gym/adapters/intellect_logic.py documents the existing local fork. Capture the exact fork SHA and verifiers snapshot before any live eval.
- ARC-AGI-2: https://github.com/arcprize/ARC-AGI-2 at commit f3283f727488ad98fe575ea6a5ac981e4a188e49. Public training has 1,000 tasks; public evaluation has 120. The evaluation IDs and hashes are reserved in the index. Puzzles remain unopened by the indexer.
- The existing research_gym ARC-2 code composes two toy grid rules. It is a useful smoke test and has its own distinct benchmark name.

Use the logic and ARC-AGI-2 task sources under their own terms. The repo deliberately ships task IDs, hashes, adapters, and tiny artificial tests rather than repackaged task content or extracted weights.

## CPU commands

Run all tests from repository root:

~~~bash
python -m unittest discover -s Kimilings/tests -p 'test_*.py' -v
python -m compileall -q Kimilings
python -m research_gym.scripts.bench_arc2
~~~

Prepare a dataset manifest with disjoint train, calibration, validation, and sealed final partitions:

~~~bash
python -m Kimilings.cli index \
  --arc-training /data/ARC-AGI-2/data/training \
  --arc-evaluation /data/ARC-AGI-2/data/evaluation \
  --logic-metadata /data/logic_task_metadata.jsonl \
  --out /tmp/kimilings/task_index.json
python -m Kimilings.cli audit \
  --manifest /tmp/kimilings/task_index.json \
  --out /tmp/kimilings/task_audit.json
~~~

The logic metadata exporter must emit one JSON object per line, containing task_id, group_id, content_sha256, and optional task_family. group_id must bind known generator families and prompt variants. The inventory splits by stable group hash with seed 11: 65% VPD training, 15% calibration, 20% validation. Exact duplicate file/content hashes and repeated group IDs across splits are rejected. Near-duplicate generated logic puzzles may still cross splits; audit source-family overlap separately before making generalization claims.

Normalize the Silico extraction receipt to a JSON document with hf_repo, hf_commit, tensor_count, and bundles. Each of the 16 bundles has expert_index, path, size_bytes, and sha256. The preflight checks metadata alone or checks actual local bytes if the root is provided:

~~~bash
python -m Kimilings.slices \
  --receipt /data/normalized_silico_receipt.json \
  --root /data/downloaded-kimi-slices \
  --out /tmp/kimilings/slice_preflight.json
~~~

Metadata-only inspection reports its limitations explicitly. The original Silico verification should be retained alongside this independent local check.

For captured real Kimi activations, prepare a JSONL file with task_id, split, activation_source=routed_kimi, source_model_revision, capture_id, layer, expert_index, token_count, activation_sha256, router_receipt, and reference_forward_receipt. The tensor payload is referenced by content hash and stored outside Git:

~~~bash
python -m Kimilings.capture \
  --manifest /tmp/kimilings/task_index.json \
  --captures /data/routed_kimi_capture_index.jsonl \
  --min-tokens 100 \
  --out /tmp/kimilings/candidate_sites.json
~~~

The capture audit ranks layer/expert sites by actual routed inputs across both logic and ARC-AGI-2 training partitions. With missing captures, the ready-site count is zero.

## Offline routing, scoring and VPD

Router plans accept one JSONL object per task with task_id, candidate_indices (subset of 0..15), bank_size, top_k_per_token, and trm_scores indexed by stringified expert ID. Optional fields include ldt_allowed, ldt_verification, verifier_receipt, seed and VPD scores. Each policy preserves a fixed requested bank size; a verified filter with too few candidates abstains. A hard filter requires an environment-owned verifier passed to the Python route() API. The standalone CLI deliberately has no authority to certify a model-authored receipt.

~~~bash
python -m Kimilings.cli route \
  --records Kimilings/fixtures/router_plan.jsonl \
  --policy trm \
  --out /tmp/kimilings/trm_plan.json
~~~

The CLI emits a plan. It measures neither Kimi throughput nor task accuracy. For a full experiment, execute every plan in the same frozen inference harness, log actual token top-k and memory/offload costs, then grade predictions using the external task verifiers.

Normalized logic verdicts need task_id, system, seed, correct (boolean), verifier_id, verifier_revision, receipt_sha256, cost and cost_unit. The verdicts must come from the trusted logic-env task verifier, with receipts retained separately. The code aggregates these externally supplied verdicts; it does not authenticate their signatures or rerun logic verifiers.

~~~bash
python -m Kimilings.cli logic \
  --manifest /tmp/kimilings/task_index.json \
  --verdicts /data/logic_verdicts.jsonl \
  --split validation \
  --out /tmp/kimilings/logic_scores.json
~~~

ARC predictions have system, task_id, seed, test_index, attempt (1 or 2), and grid (rectangular cells 0..9). On ARC, a task passes only if every test input's correct output appears within two attempts. Missing tasks score zero. Validation is scored from the training directory:

~~~bash
python -m Kimilings.cli arc \
  --manifest /tmp/kimilings/task_index.json \
  --task-dir /data/ARC-AGI-2/data/training \
  --predictions /data/arc_validation_predictions.jsonl \
  --split validation \
  --out /tmp/kimilings/arc_validation.json
~~~

The public final set is available only with an explicit unlock, a full 120-task index, and an exclusive ledger file. Its consumption is one-shot by contract and the ledger should be preserved. Preserve the source task directory and answer outputs outside source control.

~~~bash
python -m Kimilings.cli arc \
  --manifest /tmp/kimilings/task_index.json \
  --task-dir /data/ARC-AGI-2/data/evaluation \
  --predictions /data/frozen_arc_final_predictions.jsonl \
  --split final --unlock-final \
  --ledger /data/audits/arc_final_consumed.json \
  --out /data/audits/arc_final_scores.json
~~~

VPD records require target_id, task_family, activation_source, seed, step, mask_budget, method, active_count, total_components, parameter_relative_error, all_on_mse, importance_mse, adversarial_mse, all_off_mse. Routed Kimi records additionally require activation_receipt, source_model_revision, reference_forward_receipt, and router_trace_receipt. Report active-L0 versus fresh adversarial reconstruction error, with importance-mask reconstruction as an unoptimized monitor.

~~~bash
python -m Kimilings.cli vpd \
  --records /data/kimi_vpd_sweeps.jsonl \
  --out /tmp/kimilings/vpd_pareto_inputs.json
python -m Kimilings.cli transfer \
  --records /data/kimi_matched_interventions.jsonl \
  --out /tmp/kimilings/causal_transfer_cells.json
~~~

Transfer records name component_family, test_family, task_id, seed, mask_budget, loss_unmasked, loss_ablated, loss_control, control_method, matched_budget=true, activation_source=routed_kimi, activation_receipt, and reference_forward_receipt. Allowed controls are random-rank-matched, magnitude, architectural-unit and shuffled-importance. The report gives paired differential deltas at matched budgets. Final statistical inference, bootstrap intervals, Holm correction, and GPU execution remain downstream.

See REGISTERED_STUDY.md for preregistered comparisons, readout definitions, and execution gates.
