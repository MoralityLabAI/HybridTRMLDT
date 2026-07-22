"""Recompute a post-hoc descriptive audit of the sealed RLM acquisition run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_evidence_acquisition_v0 as runner
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as parent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/rlm_evidence_acquisition_mesh_v0"
DEFAULT_AUDIT = DEFAULT_OUTPUT / "independent_audit.json"
DEFAULT_REPORT = ROOT / "reports/rlm_evidence_acquisition_mesh_v0_audit.md"
CONTROLS = (
    "mesh_fixed_no_query",
    "mesh_random_two_query",
    "mesh_deterministic_voi",
    "mesh_exact_then_rollout",
)


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 1e-12 for value in values),
        "zero": sum(abs(value) <= 1e-12 for value in values),
        "negative": sum(value < -1e-12 for value in values),
    }


def build_audit(output: Path) -> dict[str, Any]:
    config, config_hash = runner._load_registered_config(runner.DEFAULT_CONFIG)
    tasks, proposals, truth = runner._load_inputs(config)
    selected = [task for task in tasks if task.split == "eval"]
    task_by_id = {task.task_id: task for task in selected}
    result_path = output / "result.json"
    records_path = output / "records.jsonl"
    trajectories_path = output / "trajectory_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    receipt = json.loads((output / "result_receipt.json").read_text(encoding="utf-8"))
    if canonical_file_sha256(result_path) != receipt["result_sha256"]:
        raise RuntimeError("sealed result hash did not replay")
    if canonical_file_sha256(records_path) != receipt["records_sha256"]:
        raise RuntimeError("sealed records hash did not replay")
    if canonical_file_sha256(trajectories_path) != receipt["trajectory_manifest_sha256"]:
        raise RuntimeError("sealed trajectory hash did not replay")
    if runner._summaries(records) != result["summary"]:
        raise RuntimeError("summary recomputation mismatch")
    if any(runner._load_task_shard(output, task, config_hash) is None for task in selected):
        raise RuntimeError("a registered task shard was absent")
    expected_comparisons = [
        runner._comparison(records, "rlm_adaptive_two_query", control)
        for control in CONTROLS
    ]
    if expected_comparisons != result["comparisons"]:
        raise RuntimeError("comparison recomputation mismatch")
    receipt_failures = runner._receipt_failures(records, task_by_id, proposals, truth)
    if receipt_failures:
        raise RuntimeError(f"evidence receipt failures: {receipt_failures}")

    lookup = {(row["architecture_id"], row["task_id"]): row for row in records}
    adaptive = [row for row in records if row["architecture_id"] == "rlm_adaptive_two_query"]
    task_contrasts = {}
    for control in CONTROLS:
        net = [
            float(row["net_utility"]) - float(lookup[(control, row["task_id"])]["net_utility"])
            for row in adaptive
        ]
        raw = [
            float(row["raw_utility"]) - float(lookup[(control, row["task_id"])]["raw_utility"])
            for row in adaptive
        ]
        task_contrasts[control] = {
            "mean_net_delta": mean(net),
            "mean_raw_delta": mean(raw),
            "net_sign_counts": _sign_counts(net),
            "raw_sign_counts": _sign_counts(raw),
            "positive_net_mass": sum(max(0.0, value) for value in net),
            "negative_net_mass": sum(min(0.0, value) for value in net),
        }

    queried = [row for row in adaptive if int(row["query_count"]) > 0]
    oracle_nonempty = [row for row in adaptive if row["oracle_query_sequence"]]
    query_ids = Counter(
        query_id
        for row in adaptive
        for query_id in row["query_sequence"]
    )
    summary = next(
        row for row in result["summary"] if row["architecture_id"] == "rlm_adaptive_two_query"
    )
    resource = json.loads((output / "run.resource_receipt.json").read_text(encoding="utf-8"))
    return {
        "audit_scope": "posthoc_descriptive_not_a_registered_endpoint",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "canonical_artifact_replay": {
            "result_sha256": receipt["result_sha256"],
            "records_sha256": receipt["records_sha256"],
            "trajectory_manifest_sha256": receipt["trajectory_manifest_sha256"],
            "summary_exact": True,
            "comparisons_exact": True,
            "task_shards_replayed": len(selected),
            "evidence_receipt_failures": 0,
        },
        "hard_gates": {
            "typed_unsafe_count": result["typed_unsafe_count"],
            "forced_failure_no_op_passed": result["forced_failure_no_op_passed"],
            "provider_errors": int(summary["provider_errors"]),
            "contract_rate": float(summary["contract_rate"]),
        },
        "adaptive_summary": summary,
        "task_level_contrasts": task_contrasts,
        "family_net_delta_vs_fixed": result["comparisons"][0]["family_net_deltas"],
        "query_policy": {
            "no_query_tasks": sum(int(row["query_count"]) == 0 for row in adaptive),
            "queried_tasks": len(queried),
            "second_query_tasks": sum(int(row["query_count"]) == 2 for row in adaptive),
            "queried_without_action_change": sum(not bool(row["action_changed"]) for row in queried),
            "beneficial_queried_tasks": sum(bool(row["beneficial_change"]) for row in queried),
            "harmful_queried_tasks": sum(bool(row["harmful_change"]) for row in queried),
            "oracle_sequence_matches": sum(
                row["query_sequence"] == row["oracle_query_sequence"] for row in adaptive
            ),
            "oracle_nonempty_tasks": len(oracle_nonempty),
            "no_query_when_oracle_nonempty": sum(
                not row["query_sequence"] and bool(row["oracle_query_sequence"])
                for row in adaptive
            ),
            "query_when_oracle_no_query": sum(
                bool(row["query_sequence"]) and not row["oracle_query_sequence"]
                for row in adaptive
            ),
            "query_id_counts": dict(sorted(query_ids.items())),
        },
        "resource": {
            key: resource.get(key)
            for key in (
                "status",
                "elapsed_seconds",
                "aggregate_elapsed_seconds",
                "peak_ram_mb",
                "peak_io_mb_s",
                "peak_vram_mb",
                "cleanup_passed",
            )
        },
        "interpretation_boundary": (
            "The positive mean net-utility result is concentrated in two tasks and is heterogeneous across "
            "families. It supports sparse evidence acquisition as a candidate mesh role, not uniform policy "
            "superiority, general safety, or a neural architecture claim."
        ),
    }


def render_report(audit: Mapping[str, Any]) -> str:
    fixed = audit["task_level_contrasts"]["mesh_fixed_no_query"]
    deterministic = audit["task_level_contrasts"]["mesh_deterministic_voi"]
    query = audit["query_policy"]
    family_rows = "\n".join(
        f"| `{family}` | {float(value):+.4f} |"
        for family, value in audit["family_net_delta_vs_fixed"].items()
    )
    return f"""# RLM Evidence-Acquisition Mesh v0: Independent Audit

This is a **post-hoc descriptive audit**, not a registered endpoint. It recomputes the sealed records without
changing the canonical result or receipt.

## Finding

The adaptive RLM ranked first on macro net utility (`{float(audit['adaptive_summary']['macro_net_utility']):.4f}`).
Against fixed no-query consensus, its mean net delta was `{float(fixed['mean_net_delta']):+.4f}` and raw-utility
delta was `{float(fixed['mean_raw_delta']):+.4f}`. The effect was concentrated: task-level net deltas were positive
on `{fixed['net_sign_counts']['positive']}/24`, zero on `{fixed['net_sign_counts']['zero']}/24`, and negative on
`{fixed['net_sign_counts']['negative']}/24`. Two large gains outweighed eight net losses, mostly query costs; this is not a
uniform improvement result.

| Family | Net delta vs fixed |
|---|---:|
{family_rows}

## Control Role

The RLM skipped acquisition on `{query['no_query_tasks']}/24` tasks, queried on `{query['queried_tasks']}/24`, and
used a second query on `{query['second_query_tasks']}/24`. Of queried tasks, `{query['queried_without_action_change']}`
did not change the action, `{query['beneficial_queried_tasks']}` improved raw utility, and
`{query['harmful_queried_tasks']}` reduced it. It never selected `counterfactual_rollout`; its measured role was a
sparse evidence buyer and stopping policy, not a comprehensive evidence planner.

Its exact query sequence matched the enumerated sequence oracle on `{query['oracle_sequence_matches']}/24` tasks,
all through `STOP`: it skipped `{query['no_query_when_oracle_nonempty']}` tasks where the oracle acquired evidence
and queried `{query['query_when_oracle_no_query']}` tasks where the oracle stopped. The run therefore demonstrates
useful cost-sensitive sparsity, not successful sequence-oracle recovery.

Against deterministic VOI, the RLM gained `{float(deterministic['mean_net_delta']):+.4f}` net utility while losing
`{float(deterministic['mean_raw_delta']):+.4f}` raw utility. The net advantage therefore comes from lower query cost,
not higher decision quality. The richer fixed exact-plus-rollout arm likewise had higher raw utility but lower net
utility at the registered costs.

## Integrity

- 168 records and all 24 task shards replayed.
- Evidence receipt failures: `{audit['canonical_artifact_replay']['evidence_receipt_failures']}`.
- Unsafe executions: `{audit['hard_gates']['typed_unsafe_count']}`.
- Forced-failure identity: `{str(audit['hard_gates']['forced_failure_no_op_passed']).lower()}`.
- RLM contract rate: `{float(audit['hard_gates']['contract_rate']):.4f}`; provider errors: `{audit['hard_gates']['provider_errors']}`.
- Records SHA-256: `{audit['canonical_artifact_replay']['records_sha256']}`.

## Boundary

{audit['interpretation_boundary']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    audit = build_audit(args.output.resolve())
    parent._write_json(args.audit.resolve(), audit)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(render_report(audit), encoding="utf-8", newline="\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
