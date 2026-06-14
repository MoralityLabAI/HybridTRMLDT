from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from research_gym.core.frames import ReachabilityFrame


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize typed reachability frames.")
    parser.add_argument("--frames", type=Path, default=Path("data/frames.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/symbolic_report.md"))
    args = parser.parse_args()

    frames = []
    with args.frames.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(ReachabilityFrame.from_jsonable(json.loads(line)))

    counts = Counter(frame.label.value for frame in frames)
    total = len(frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Symbolic Frame Report", "", f"Total frames: {total}", "", "## Label counts", ""]
    for label, count in sorted(counts.items()):
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"- `{label}`: {count} ({pct:.1f}%)")

    lines += ["", "## Sample frames", ""]
    for frame in frames[:5]:
        lines.append(f"- `{frame.frame_id}` {frame.label.value} state={dict(frame.state)} witness={frame.witness_actions[:3]}")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
