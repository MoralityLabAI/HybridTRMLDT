from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from research_gym.adapters.airis_das import dump_jsonl, load_jsonl
from research_gym.benchmarks.airis_induction_bench import (
    episodes_sha256,
    evaluate_induced_airis,
    induce_airis_ruleset,
    induction_markdown,
)
from research_gym.benchmarks.sequencer_control_bench import (
    SequencerBenchmarkConfig,
    build_benchmark_episodes,
)
from research_gym.envs.routing import (
    DEFAULT_ROUTING_ENVS,
    DEFAULT_TESSERACT_DATA_ROOT,
    load_tesseract_routing_examples,
)


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
    raw = result["raw_airis"]
    guarded = result["guarded_airis"]
    return "\n".join(
        [
            "# Independently Induced AIRIS Rule Notes",
            "",
            "## Design",
            "",
            "- Rules are induced from calibration episode utility winners, not topology-plan sequence labels.",
            "- Topology plans contribute route, orientation, protocol, and risk features only for authorization.",
            "- Rule support is the calibration winner count; disagreement episodes are retained as counterexamples.",
            "- Rule confidence is support divided by context calibration count.",
            "- Integrity digests are frozen before held-out evaluation.",
            "- The primary confidence threshold is fixed at 0.5; the remaining thresholds are sensitivity analysis.",
            "",
            "## Results",
            "",
            f"- calibration episodes: `{result['source']['calibration_episode_count']}`",
            f"- held-out episodes: `{result['eval_episode_count']}`",
            f"- rules: `{result['rule_count']}`",
            f"- raw held-out oracle-label accuracy: `{raw['oracle_label_accuracy']:.6f}`",
            f"- raw proposal/control parity: `{raw['control_parity_rate']:.6f}`",
            f"- raw macro utility: `{raw['macro_utility']:.6f}`",
            f"- guarded acceptance: `{guarded['acceptance_rate']:.6f}`",
            f"- guarded accepted-label accuracy: `{guarded['accepted_oracle_label_accuracy']:.6f}`",
            f"- intact-but-wrong held-out rules: `{guarded['intact_wrong_rule_count']}`",
            f"- guarded harmful changes: `{guarded['harmful_change_count']}`",
            f"- raw proposals below control: `{guarded['raw_below_control_count']}`",
            f"- fallback save rate: `{guarded['fallback_save_rate']:.6f}`",
            f"- guarded macro utility delta vs control: `{result['primary_macro_utility_delta_vs_control']:+.6f}`",
            f"- calibration-confidence Brier score: `{result['calibration_confidence_brier']:.6f}`",
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce AIRIS rules from calibration outcomes.")
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/airis_induction_v1.json")
    )
    parser.add_argument(
        "--sequencer-config",
        type=Path,
        default=Path("configs/sequencer_control_v1.json"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_results.json"),
    )
    parser.add_argument(
        "--eval-episodes",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_episodes.jsonl"),
    )
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_TESSERACT_DATA_ROOT)
    parser.add_argument(
        "--rules-out", type=Path, default=Path("data/airis_das/induced_rules.json")
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=Path("data/airis_das/induction_calibration_episodes.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/airis_induction_results.json"),
    )
    parser.add_argument(
        "--records-out",
        type=Path,
        default=Path("data/benchmarks/airis_induction_records.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/airis_induction_bench.md")
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("experiments/airis_induction")
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    registration = json.loads(args.sequencer_config.read_text(encoding="utf-8"))
    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    config = SequencerBenchmarkConfig(**registration["benchmark"])
    routing_examples = load_tesseract_routing_examples(
        args.routing_root,
        DEFAULT_ROUTING_ENVS,
        max_per_env=config.routing_max_per_env,
    )
    episodes = build_benchmark_episodes(config, routing_examples)
    calibration = [episode for episode in episodes if episode.split == "calibration"]
    evaluation = [episode for episode in episodes if episode.split == "eval"]
    if episodes_sha256(evaluation) != payload.get("eval_sha256"):
        raise ValueError("regenerated evaluation episodes do not match the frozen benchmark")
    eval_rows = load_jsonl(str(args.eval_episodes))
    if len(eval_rows) != len(evaluation):
        raise ValueError("saved evaluation row count does not match regenerated episodes")

    ruleset = induce_airis_ruleset(payload, calibration, protocol)
    result, records = evaluate_induced_airis(payload, eval_rows, ruleset, protocol)
    result["protocol"] = protocol
    result["source"] = {
        "protocol": str(args.protocol),
        "sequencer_config": str(args.sequencer_config),
        "benchmark": str(args.benchmark),
        "eval_episodes": str(args.eval_episodes),
        "routing_root": str(args.routing_root),
        "routing_environments": list(DEFAULT_ROUTING_ENVS),
        "routing_example_count": len(routing_examples),
        "calibration_episode_count": len(calibration),
    }
    result["artifacts"] = {
        "rules_sha256": _canonical_hash(ruleset),
        "records_sha256": _canonical_hash(records),
        "protocol_sha256": _canonical_hash(protocol),
    }

    calibration_rows = [episode.to_jsonable() for episode in calibration]
    _write_json(ruleset, args.rules_out)
    _write_jsonl(calibration_rows, args.calibration_out)
    _write_json(result, args.out)
    _write_jsonl(records, args.records_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(induction_markdown(result), encoding="utf-8")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ruleset, args.experiment_dir / "rules.json")
    _write_jsonl(calibration_rows, args.experiment_dir / "calibration_episodes.jsonl")
    _write_json(result, args.experiment_dir / "results.json")
    _write_jsonl(records, args.experiment_dir / "evaluation_records.jsonl")
    (args.experiment_dir / "training_notes.md").write_text(
        training_notes(result), encoding="utf-8"
    )
    print(induction_markdown(result))


if __name__ == "__main__":
    main()
