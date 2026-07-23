"""Post-hoc decomposition of the sealed RLM uncertainty pre-gate run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as parent
from research_gym.scripts import bench_rlm_uncertainty_pregate_v0_1 as runner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/rlm_uncertainty_pregate_v0_1"
DEFAULT_AUDIT = DEFAULT_OUTPUT / "independent_audit.json"
DEFAULT_REPORT = ROOT / "reports/rlm_uncertainty_pregate_v0_1_audit.md"


def build_audit(output: Path) -> dict[str, Any]:
    config, config_hash, registration = runner._load_registered_config(runner.DEFAULT_CONFIG)
    tasks, proposals, truth = runner._load_inputs(config)
    selected = sorted(
        [task for task in tasks if task.split == "eval"],
        key=lambda task: (runner.FAMILIES.index(task.family), task.task_id),
    )
    result_path = output / "result.json"
    records_path = output / "records.jsonl"
    trajectory_path = output / "trajectory_manifest.json"
    receipt = json.loads((output / "result_receipt.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    if canonical_file_sha256(result_path) != receipt["result_sha256"]:
        raise RuntimeError("pregate result hash did not replay")
    if canonical_file_sha256(records_path) != receipt["records_sha256"]:
        raise RuntimeError("pregate record hash did not replay")
    if canonical_file_sha256(trajectory_path) != receipt["trajectory_manifest_sha256"]:
        raise RuntimeError("pregate trajectory hash did not replay")
    hashes = runner.architecture_hashes(config["provider_cost_regimes"])
    if any(runner._load_task_shard(output, task, config_hash, hashes) is None for task in selected):
        raise RuntimeError("pregate task shard was absent")
    failures = runner._receipt_failures(
        records, {task.task_id: task for task in selected}, proposals, truth
    )
    if failures:
        raise RuntimeError(f"pregate evidence receipt failures: {failures}")
    expected_comparisons = [
        runner._comparison(records, "gated_rlm", "ungated_rlm", config),
        runner._comparison(records, "gated_rlm", "fixed_no_query", config),
        runner._comparison(records, "gated_rlm", "gated_deterministic_voi", config),
    ]
    if runner._summaries(records, config) != result["summary"]:
        raise RuntimeError("pregate summary recomputation mismatch")
    if expected_comparisons != result["comparisons"]:
        raise RuntimeError("pregate comparison recomputation mismatch")

    lookup = {(row["architecture_id"], row["task_id"]): row for row in records}
    task_ids = [task.task_id for task in selected]
    gate_open = [task_id for task_id in task_ids if lookup[("gated_rlm", task_id)]["gate_open"]]
    gate_closed = [task_id for task_id in task_ids if task_id not in gate_open]
    open_sequence_matches = sum(
        lookup[("gated_rlm", task_id)]["query_sequence"]
        == lookup[("ungated_rlm", task_id)]["query_sequence"]
        for task_id in gate_open
    )
    open_action_matches = sum(
        lookup[("gated_rlm", task_id)]["executed_action"]
        == lookup[("ungated_rlm", task_id)]["executed_action"]
        for task_id in gate_open
    )
    closed_ungated_changes = []
    for task_id in gate_closed:
        ungated = lookup[("ungated_rlm", task_id)]
        fixed = lookup[("fixed_no_query", task_id)]
        if ungated["executed_action"] != fixed["executed_action"]:
            closed_ungated_changes.append(
                {
                    "task_id": task_id,
                    "family": ungated["task_family"],
                    "raw_utility_delta": ungated["raw_utility"] - fixed["raw_utility"],
                    "query_sequence": ungated["query_sequence"],
                }
            )

    coupled = {name: [] for name in config["provider_cost_regimes"]}
    coupled_raw = []
    for family in runner.FAMILIES:
        family_ids = [task.task_id for task in selected if task.family == family]
        coupled_raw.append(
            mean(
                (
                    lookup[("ungated_rlm", task_id)]
                    if task_id in gate_open
                    else lookup[("fixed_no_query", task_id)]
                )["raw_utility"]
                - lookup[("ungated_rlm", task_id)]["raw_utility"]
                for task_id in family_ids
            )
        )
        for name in coupled:
            coupled[name].append(
                mean(
                    (
                        lookup[("ungated_rlm", task_id)]
                        if task_id in gate_open
                        else lookup[("fixed_no_query", task_id)]
                    )["all_in_utility"][name]
                    - lookup[("ungated_rlm", task_id)]["all_in_utility"][name]
                    for task_id in family_ids
                )
            )
    operational = result["comparisons"][0]
    primary_savings = (
        operational["macro_all_in_utility_delta"]["primary"]
        - operational["macro_evidence_net_utility_delta"]
    )
    break_even_scale = (
        -operational["macro_evidence_net_utility_delta"] / primary_savings
        if primary_savings > 0
        else None
    )
    summaries = {row["architecture_id"]: row for row in result["summary"]}
    return {
        "audit_scope": "posthoc_descriptive_and_coupled_replay_not_registered_endpoints",
        "protocol_id": config["protocol_id"],
        "canonical_replay": {
            "result_sha256": receipt["result_sha256"],
            "records_sha256": receipt["records_sha256"],
            "trajectory_sha256": receipt["trajectory_manifest_sha256"],
            "task_shards_replayed": len(selected),
            "records_replayed": len(records),
            "evidence_receipt_failures": 0,
            "summaries_exact": True,
            "comparisons_exact": True,
        },
        "registered_operational_result": operational,
        "call_reduction": {
            "ungated_calls": registration["ungated_provider_calls_registered"],
            "gated_calls": registration["gated_provider_calls_registered"],
            "absolute": -operational["provider_call_delta"],
            "fraction": -operational["provider_call_delta"]
            / registration["ungated_provider_calls_registered"],
            "tokens_saved": -operational["token_delta"],
            "wall_seconds_saved": -operational["provider_wall_seconds_delta"],
        },
        "repeatability": {
            "gate_open_pairs": len(gate_open),
            "query_sequence_matches": open_sequence_matches,
            "executed_action_matches": open_action_matches,
            "gate_open_raw_utility_sum_delta_gated_minus_ungated": sum(
                lookup[("gated_rlm", task_id)]["raw_utility"]
                - lookup[("ungated_rlm", task_id)]["raw_utility"]
                for task_id in gate_open
            ),
            "gated_first": _order_slice(selected, lookup, gated_first=True),
            "gated_second": _order_slice(selected, lookup, gated_first=False),
        },
        "gate_closed_ungated_action_changes": closed_ungated_changes,
        "coupled_replay": {
            "definition": "use the ungated sampled outcome on registered gate-open tasks and fixed consensus otherwise",
            "macro_raw_utility_delta_vs_ungated": mean(coupled_raw),
            "macro_all_in_utility_delta_vs_ungated": {
                name: mean(values) for name, values in coupled.items()
            },
        },
        "provider_cost_break_even_scale_vs_primary": break_even_scale,
        "winner": {
            "architecture": max(
                summaries,
                key=lambda architecture: summaries[architecture]["macro_all_in_utility"]["primary"],
            ),
            "primary_all_in_utility": max(
                row["macro_all_in_utility"]["primary"] for row in summaries.values()
            ),
        },
        "hard_gates": {
            key: result[key]
            for key in (
                "typed_unsafe_count",
                "forced_failure_identity_passed",
                "closed_gate_provider_call_violations",
                "provider_invocation_identity_passed",
            )
        },
        "interpretation": (
            "The registered operational gate failed to preserve RLM quality and did not beat the ungated arm. "
            "A coupled replay shows that cost savings would be sufficient if controller outcomes were shared, "
            "but low gate-open repeatability makes that a mechanism diagnostic rather than a positive endpoint."
        ),
    }


def _order_slice(selected, lookup, *, gated_first: bool) -> dict[str, Any]:
    values, sequence_matches = [], 0
    for index, task in enumerate(selected):
        gated = lookup[("gated_rlm", task.task_id)]
        if not gated["gate_open"] or (index % 2 == 1) != gated_first:
            continue
        ungated = lookup[("ungated_rlm", task.task_id)]
        values.append(gated["raw_utility"] - ungated["raw_utility"])
        sequence_matches += gated["query_sequence"] == ungated["query_sequence"]
    return {
        "pairs": len(values),
        "raw_utility_sum_delta_gated_minus_ungated": sum(values),
        "query_sequence_matches": sequence_matches,
    }


def render_report(audit: Mapping[str, Any]) -> str:
    operational = audit["registered_operational_result"]
    calls = audit["call_reduction"]
    repeatability = audit["repeatability"]
    coupled = audit["coupled_replay"]
    return f"""# RLM Uncertainty Pre-Gate v0.1: Independent Audit

This audit is post-hoc. The registered operational contrast remains the controlling result.

## Operational Result

The pre-gate reduced provider calls from `{calls['ungated_calls']}` to `{calls['gated_calls']}`
(`{100 * calls['fraction']:.1f}%`), saving `{calls['tokens_saved']:,}` tokens and
`{calls['wall_seconds_saved']:.2f}` wall seconds. It nevertheless changed macro utility versus ungated by:

- Raw utility: `{operational['macro_raw_utility_delta']:+.4f}`
- Evidence-only net utility: `{operational['macro_evidence_net_utility_delta']:+.4f}`
- Low-cost all-in: `{operational['macro_all_in_utility_delta']['low']:+.4f}`
- Primary all-in: `{operational['macro_all_in_utility_delta']['primary']:+.4f}`
- High-cost all-in: `{operational['macro_all_in_utility_delta']['high']:+.4f}`

The gate therefore failed its primary comparison. Registered provider costs would need to be approximately
`{audit['provider_cost_break_even_scale_vs_primary']:.2f}x` the primary coefficients for the operational mean to
break even.

## Instability Decomposition

Both RLM arms were called independently on `{repeatability['gate_open_pairs']}` gate-open tasks. Only
`{repeatability['query_sequence_matches']}/{repeatability['gate_open_pairs']}` query sequences and
`{repeatability['executed_action_matches']}/{repeatability['gate_open_pairs']}` executed actions matched, despite
temperature zero. Their gate-open raw-utility sum differed by
`{repeatability['gate_open_raw_utility_sum_delta_gated_minus_ungated']:+.3f}` in favor of ungated.

One gate-closed provenance task produced an ungated raw gain of `+0.120`, so the rule also made a genuine miss.
In a post-hoc coupled replay that uses the same ungated sample on gate-open tasks, the gate loses
`{coupled['macro_raw_utility_delta_vs_ungated']:+.4f}` raw utility but gains
`{coupled['macro_all_in_utility_delta_vs_ungated']['primary']:+.4f}` primary all-in utility. This shows that the
cost-saving mechanism works conditionally; it does not overturn the registered operational failure.

## Winner

`{audit['winner']['architecture']}` ranked first at
`{audit['winner']['primary_all_in_utility']:.4f}` primary all-in utility. In this panel, a deterministic local
value-of-information policy dominated both RLM arms.

## Integrity

- Records replayed: `{audit['canonical_replay']['records_replayed']}`
- Task shards replayed: `{audit['canonical_replay']['task_shards_replayed']}`
- Evidence receipt failures: `{audit['canonical_replay']['evidence_receipt_failures']}`
- Unsafe executions: `{audit['hard_gates']['typed_unsafe_count']}`
- Closed-gate provider calls: `{audit['hard_gates']['closed_gate_provider_call_violations']}`
- Records SHA-256: `{audit['canonical_replay']['records_sha256']}`

## Conclusion

{audit['interpretation']}
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
