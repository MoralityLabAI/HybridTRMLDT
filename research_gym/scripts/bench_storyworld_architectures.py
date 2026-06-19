from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.storyworld_architecture_bench import (
    run_storyworld_architecture_benchmark,
    summary_markdown,
    write_experiment_bundle,
)


def training_notes(payload: dict[str, object]) -> str:
    lines = [
        "# Storyworld Architecture Training Notes",
        "",
        "Task: compare typed membrane, hard gate, and confidence arbitration on storyworld control.",
        "",
        "Scenarios:",
        "",
        "- `secret_ending`: exact secret-ending reachability is environment-sound.",
        "- `moral_optimization`: terminal moral score is a soft preference surface, not a hard ending gate.",
        "",
        "Policies:",
        "",
        "- `trm`: greedy local heuristic.",
        "- `typed_membrane`: TRM proposal plus typed reachability override.",
        "- `hard_gate`: always use LDT reachability action when available.",
        "- `confidence_arbitration`: use soft moral-score action when margin is high; otherwise defer to LDT.",
        "",
        f"- episodes per scenario/policy: `{payload['n']}`",
        f"- horizon: `{payload['horizon']}`",
        f"- seed: `{payload['seed']}`",
        f"- confidence gamma: `{float(payload['gamma']):.2f}`",
        "",
    ]
    summary = payload["summary"]
    assert isinstance(summary, dict)
    for scenario, rows in summary.items():
        lines.append(f"## {scenario}")
        assert isinstance(rows, dict)
        for policy, metrics in rows.items():
            assert isinstance(metrics, dict)
            lines.append(
                f"- `{policy}` success_rate={float(metrics['success_rate']):.3f} "
                f"avg_score={float(metrics['avg_score']):.2f} overrides={int(metrics['overrides'])} "
                f"avg_margin={float(metrics['avg_margin']):.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run storyworld architecture control benchmark.")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/storyworld_architecture_results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/storyworld_architecture_bench.md"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/storyworld_architectures"))
    args = parser.parse_args()

    payload = run_storyworld_architecture_benchmark(n=args.n, horizon=args.horizon, seed=args.seed, gamma=args.gamma)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = summary_markdown(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_experiment_bundle(payload, args.experiment_dir, notes=training_notes(payload))
    print(report)


if __name__ == "__main__":
    main()
