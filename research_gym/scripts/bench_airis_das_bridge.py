from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from research_gym.adapters.airis_das import (
    apply_airis_bridge,
    bridge_markdown,
    build_airis_ruleset,
    dump_jsonl,
    load_jsonl,
)
from research_gym.benchmarks.sequencer_control_bench import v1_replay_rows


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_jsonl(rows), encoding="utf-8")


def training_notes(result: dict[str, object]) -> str:
    summary = result["summary"]
    service = result["service_contract"]
    assert isinstance(summary, dict) and isinstance(service, dict)
    return "\n".join(
        [
            "# AIRIS/DAS Bridge Experiment Notes",
            "",
            "## Design",
            "",
            "- One AIRIS-compatible rule is exported per calibration-derived context plan.",
            "- Evaluation outcomes are not used to construct rules, confidence, support, or routing.",
            "- DAS performs rule persistence and forecast retrieval; it does not receive execution authority.",
            "- The topology membrane accepts only exact, protocol-bound, context-bound, supported, route-consistent,",
            "  orientation-safe proposals. Every failed check falls back to `control_math`.",
            "- Verifiers v1 replays sealed forecasts and decisions, so benchmark evaluation has no mutable service",
            "  dependency. The actual local HTTP service is exercised by a separate smoke test.",
            "",
            "## Quantitative Replay",
            "",
            f"- episodes: `{summary['episode_count']}`",
            f"- rules: `{summary['rule_count']}`",
            f"- forecast coverage: `{summary['forecast_coverage']:.6f}`",
            f"- topology acceptance: `{summary['topology_acceptance_rate']:.6f}`",
            f"- fail-closed fallback: `{summary['fallback_rate']:.6f}`",
            f"- control-math parity: `{summary['control_parity_rate']:.6f}`",
            f"- macro utility delta: `{summary['macro_utility_delta']:+.6f}`",
            "",
            "## Service Contract",
            "",
            f"- expected service schema: `{service['summary_schema']}`",
            f"- expected forecast schema: `{service['forecast_schema']}`",
            f"- backend exercised separately: `{service['backend_mode']}`",
            "",
            "## Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge frozen hybrid plans into AIRIS/DAS replay rules.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_results.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_episodes.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/airis_das_bridge_results.json"),
    )
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=Path("data/benchmarks/airis_das_bridge_episodes.jsonl"),
    )
    parser.add_argument(
        "--rules-out",
        type=Path,
        default=Path("data/airis_das/sequencer_rules.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/airis_das_bridge.md"))
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("experiments/airis_das_bridge")
    )
    parser.add_argument(
        "--v1-taskset-out",
        type=Path,
        default=Path("environments/hybrid_sequencer_v1/data/replay_tasks.jsonl"),
    )
    parser.add_argument(
        "--v1-rules-out",
        type=Path,
        default=Path("environments/hybrid_sequencer_v1/data/airis_rules.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    rows = load_jsonl(str(args.episodes))
    ruleset = build_airis_ruleset(payload)
    summary, bridged_rows = apply_airis_bridge(payload, rows, ruleset)
    result = {
        "schema": "hybrid_airis_das_bridge_result_v1",
        "summary": summary,
        "source": {
            "benchmark": str(args.benchmark),
            "episodes": str(args.episodes),
            "protocol_sha256": payload["protocol_sha256"],
            "eval_sha256": payload["eval_sha256"],
        },
        "artifacts": {
            "rules_sha256": _canonical_hash(ruleset),
            "bridge_rows_sha256": _canonical_hash(bridged_rows),
        },
        "service_contract": {
            "summary_schema": "airis_das_service_v1",
            "forecast_schema": "airis_das_forecast_v1",
            "backend_mode": "in_memory_das_shaped",
            "forecast_endpoint": "POST /api/forecast",
        },
    }

    _write_json(result, args.out)
    _write_jsonl(bridged_rows, args.episodes_out)
    _write_json(ruleset, args.rules_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(bridge_markdown(summary), encoding="utf-8")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, args.experiment_dir / "results.json")
    _write_jsonl(bridged_rows, args.experiment_dir / "episodes.jsonl")
    (args.experiment_dir / "training_notes.md").write_text(
        training_notes(result), encoding="utf-8"
    )

    _write_json(ruleset, args.v1_rules_out)
    _write_jsonl(v1_replay_rows(bridged_rows), args.v1_taskset_out)
    print(bridge_markdown(summary))


if __name__ == "__main__":
    main()
