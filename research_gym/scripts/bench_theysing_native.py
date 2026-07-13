from __future__ import annotations

import argparse
from pathlib import Path

from research_gym.benchmarks.theysing_native_bench import (
    run_theysing_native_benchmark,
    summary_markdown,
    write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched native TheySing enforcement-mode sessions.")
    parser.add_argument("--repo-root", type=Path, default=Path(r"C:\projects\TheySing\TheySing"))
    parser.add_argument("--seeds", default="400,401")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/theysing_native_hybrid"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/theysing_native_results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/theysing_native_bench.md"),
    )
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())

    payload = run_theysing_native_benchmark(
        args.repo_root,
        args.experiment_dir,
        seeds=seeds,
        turns=args.turns,
    )
    write_results(payload, args.out)
    write_results(payload, args.experiment_dir / "results.json")
    report = summary_markdown(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    (args.experiment_dir / "training_notes.md").write_text(
        report
        + "\n## Notes\n\n"
        + "- Game-scenario agents used TheySing's deterministic native heuristic policy.\n"
        + "- `bilateral_probe` used one matched native manual turn to force a non-aggression conflict.\n"
        + "- Seeds, scenarios, and requested turns are matched across enforcement modes.\n"
        + "- Raw native JSONL traces are retained losslessly as `*.jsonl.gz` under `native_logs/`.\n"
        + "- Safety and usefulness remain separate; prevention is not treated as free utility.\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
