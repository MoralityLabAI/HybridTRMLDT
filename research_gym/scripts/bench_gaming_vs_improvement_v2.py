from __future__ import annotations

import argparse
from pathlib import Path

from research_gym.benchmarks.gaming_vs_improvement_bench import summary_markdown
from research_gym.scripts.bench_gaming_vs_improvement import run_and_write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen above-majority gaming-versus-improvement protocol."
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke else ""
    mode = "smoke" if args.smoke else "full"
    result = run_and_write(
        config_path=Path("configs/gaming_vs_improvement_v2.json"),
        smoke=args.smoke,
        out=Path(f"data/benchmarks/gaming_vs_improvement_v2{suffix}_results.json"),
        records_out=Path(
            f"data/benchmarks/gaming_vs_improvement_v2{suffix}_records.jsonl"
        ),
        report=Path(f"reports/gaming_vs_improvement_v2{suffix}_bench.md"),
        experiment_dir=Path("experiments/gaming_vs_improvement/above_majority")
        / mode,
    )
    print(summary_markdown(result))


if __name__ == "__main__":
    main()
