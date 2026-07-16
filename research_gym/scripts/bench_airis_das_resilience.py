from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from research_gym.adapters.airis_das import dump_jsonl, load_jsonl
from research_gym.benchmarks.airis_resilience_bench import (
    resilience_markdown,
    run_airis_resilience_benchmark,
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
    topology = result["controllers"]["topology_only"]
    sealed = result["controllers"]["integrity_sealed"]
    effect = result["integrity_effect"]
    return "\n".join(
        [
            "# AIRIS/DAS Forecast Integrity Experiment Notes",
            "",
            "## Design",
            "",
            "- Paired deterministic fault injection: every evaluation episode receives the same ten conditions.",
            "- Clean retrieval is the positive control; nine protocol, context, receipt, and outage conditions are",
            "  negative controls.",
            "- `topology_only` preserves the prior protocol/context/route/orientation membrane without a rule digest.",
            "- `integrity_sealed` additionally binds rule ID, confidence, support, counterexample count, preconditions,",
            "  and predicted outcome to the frozen calibration-derived rule registry.",
            "- Candidate outcomes are held fixed. No model, solver, or task policy is retrained.",
            "",
            "## Results",
            "",
            f"- episodes: `{result['episode_count']}`",
            f"- total paired trials: `{result['trial_count']}`",
            f"- attack trials: `{result['attack_trial_count']}`",
            f"- topology-only clean acceptance: `{topology['clean_acceptance_rate']:.6f}`",
            f"- topology-only attack acceptance: `{topology['attack_acceptance_rate']:.6f}`",
            f"- integrity-sealed clean acceptance: `{sealed['clean_acceptance_rate']:.6f}`",
            f"- integrity-sealed attack acceptance: `{sealed['attack_acceptance_rate']:.6f}`",
            f"- integrity-sealed attack fallback: `{sealed['attack_fallback_rate']:.6f}`",
            f"- attack acceptance delta: `{effect['attack_acceptance_rate_delta']:+.6f}`",
            "",
            "## Interpretation",
            "",
            "Topology checks reject semantic route, authority, protocol, context, and orientation conflicts, but they",
            "cannot identify a syntactically valid forecast whose rule material changed in transit. The digest seal",
            "closes that transport-integrity gap while preserving clean acceptance.",
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AIRIS/DAS forecast integrity under paired faults.")
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
        "--rules", type=Path, default=Path("data/airis_das/sequencer_rules.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/airis_das_resilience_results.json"),
    )
    parser.add_argument(
        "--trials-out",
        type=Path,
        default=Path("data/benchmarks/airis_das_resilience_trials.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/airis_das_resilience.md")
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("experiments/airis_das_resilience")
    )
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    rows = load_jsonl(str(args.episodes))
    ruleset = json.loads(args.rules.read_text(encoding="utf-8"))
    result, trials = run_airis_resilience_benchmark(payload, rows, ruleset)
    result["source"] = {
        "benchmark": str(args.benchmark),
        "episodes": str(args.episodes),
        "rules": str(args.rules),
        "eval_sha256": payload.get("eval_sha256"),
    }
    result["artifacts"] = {
        "rules_sha256": _canonical_hash(ruleset),
        "trials_sha256": _canonical_hash(trials),
    }

    _write_json(result, args.out)
    _write_jsonl(trials, args.trials_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(resilience_markdown(result), encoding="utf-8")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, args.experiment_dir / "results.json")
    _write_jsonl(trials, args.experiment_dir / "trials.jsonl")
    (args.experiment_dir / "training_notes.md").write_text(
        training_notes(result), encoding="utf-8"
    )
    print(resilience_markdown(result))


if __name__ == "__main__":
    main()
