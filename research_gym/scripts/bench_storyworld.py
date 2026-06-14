from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.storyworld_bench import run_storyworld_benchmark, summary_markdown, write_experiment_bundle


def training_notes(payload: dict[str, object]) -> str:
    lines = [
        "# Storyworld Playing Training Notes",
        "",
        "Task: play the coupled storyworld to reach the secret ending predicate.",
        "",
        "Models:",
        "",
        "- `ldt`: exact finite-horizon reachability planner under the modeled rival policy.",
        "- `trm`: heuristic policy over persistent local deficits: heat, evidence, trust, scene.",
        "- `hybrid`: TRM proposes actions; LDT checks modeled reachability and overrides unsafe proposals.",
        "",
        "Data:",
        "",
        f"- sampled viable starts: `{payload['n']}`",
        f"- horizon: `{payload['horizon']}`",
        f"- seed: `{payload['seed']}`",
        "",
        "This run uses exact environment transitions and does not involve VPD.",
        "",
    ]
    summary = payload["summary"]
    assert isinstance(summary, dict)
    for player, metrics in summary.items():
        assert isinstance(metrics, dict)
        lines.append(
            f"- `{player}` success_rate={float(metrics['success_rate']):.3f} "
            f"successes={metrics['successes']}/{metrics['episodes']} "
            f"avg_steps={float(metrics['avg_steps']):.2f} overrides={metrics['overrides']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LDT/TRM/hybrid storyworld playing benchmark.")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/storyworld_results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/storyworld_bench.md"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/storyworld_playing"))
    args = parser.parse_args()

    payload = run_storyworld_benchmark(n=args.n, horizon=args.horizon, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = summary_markdown(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_experiment_bundle(payload, args.experiment_dir, notes=training_notes(payload))
    print(report)


if __name__ == "__main__":
    main()
