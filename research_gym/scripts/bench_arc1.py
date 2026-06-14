from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.arc_bench import run_arc1_benchmark, summary_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARC-1 LDT/TRM/hybrid micro-benchmark.")
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/arc1_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/arc1_bench.md"))
    args = parser.parse_args()

    results = run_arc1_benchmark()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_jsonable(), sort_keys=True) + "\n")

    report = summary_markdown(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
