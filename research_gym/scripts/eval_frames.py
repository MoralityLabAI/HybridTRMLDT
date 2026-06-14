from __future__ import annotations

import argparse
from pathlib import Path

from research_gym.core.frames import Frame, read_frame_jsonl
from research_gym.eval.baselines import metrics_to_markdown, summarize_common_frames


def read_sources(source: Path) -> list[Frame]:
    if source.is_file():
        return read_frame_jsonl(source)

    frames: list[Frame] = []
    for path in sorted(source.glob("*.jsonl")):
        frames.extend(read_frame_jsonl(path))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate common-schema frame baselines.")
    parser.add_argument("--source", type=Path, default=Path("data/generated"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_eval.md"))
    args = parser.parse_args()

    frames = read_sources(args.source)
    metrics = summarize_common_frames(frames)
    report = metrics_to_markdown(metrics, title="Baseline Frame Evaluation")
    report += f"\nFrames evaluated: {len(frames)}\n"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
