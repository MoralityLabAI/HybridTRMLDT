from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research_gym.benchmarks.gaming_vs_improvement_bench import (
    run_gaming_vs_improvement_benchmark,
    summary_markdown,
)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jsonl_bytes(rows))


def training_notes(result: dict[str, object]) -> str:
    distinct = result["arm_distinctness"]
    controls = result["negative_controls"]
    config = result["config"]
    causal_checks = result["causal_gate_checks"]
    headline = result["headline_exposed_probe"]
    return "\n".join(
        [
            "# Gaming Versus Improvement Training Notes",
            "",
            "## Registration",
            "",
            f"- config SHA-256: `{result['config_sha256']}`",
            f"- records SHA-256: `{result['artifacts']['records_sha256']}`",
            f"- mode: `{'smoke' if result['smoke'] else 'full'}`",
            f"- proposer seeds: `{config['seeds']}`",
            f"- TRM latent / recurrence: `{config['latent_dim']} / {config['recurrence_steps']}`",
            f"- round-0 / adaptation steps: `{config['round0_steps']} / {config['adaptation_steps']}`",
            f"- expert-iteration rounds: `{config['expert_iteration_rounds']}`",
            "- proposer training region: `trust <= 0`",
            "- probe calibration region: `trust == 1`",
            "- held-out evaluation region: `trust >= 2`",
            "- state-hash overlap across regions: `0`",
            "- accepted expert traces train the proposer; no gradient passes through a verifier",
            "",
            "## Controls",
            "",
            f"- arm distinctness assertion exercised: `{distinct['assertion_exercised']}`",
            f"- behaviorally distinct pairs: `{distinct['distinct_pair_count']}/{distinct['pair_count']}`",
            f"- effective policies: `{distinct['effective_policy_count']}`",
            "- identical fallback has zero utility delta: "
            f"`{controls['identical_fallback_zero_utility_delta']}`",
            f"- round-indexed causal probe checks: `{len(causal_checks)}`",
            "- exposed-probe evasion classifications: "
            f"`{headline['evasion_seed_count']}/{len(headline['trajectories'])}`",
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark verifier gaming against oracle proposal improvement."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gaming_vs_improvement_v1.json"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--records-out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke else ""
    out = args.out or Path(f"data/benchmarks/gaming_vs_improvement{suffix}_results.json")
    records_out = args.records_out or Path(
        f"data/benchmarks/gaming_vs_improvement{suffix}_records.jsonl"
    )
    report = args.report or Path(f"reports/gaming_vs_improvement{suffix}_bench.md")
    experiment_dir = args.experiment_dir or Path(
        "experiments/gaming_vs_improvement/smoke"
        if args.smoke
        else "experiments/gaming_vs_improvement/full"
    )

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    result, records = run_gaming_vs_improvement_benchmark(
        registration,
        smoke=args.smoke,
    )
    records_bytes = _jsonl_bytes(records)
    result["artifacts"] = {
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "registration_path": str(args.config),
    }
    _write_json(result, out)
    _write_jsonl(records, records_out)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(summary_markdown(result), encoding="utf-8")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, experiment_dir / "results.json")
    _write_jsonl(records, experiment_dir / "records.jsonl")
    (experiment_dir / "training_notes.md").write_text(
        training_notes(result),
        encoding="utf-8",
    )
    print(summary_markdown(result))


if __name__ == "__main__":
    main()
