from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_gym.core.frames import Frame, read_frame_jsonl


INSTRUCTION = "Given the frame, predict the typed refinement decision."


def read_common_frames(source: Path) -> list[Frame]:
    if source.is_file():
        return read_frame_jsonl(source)

    frames: list[Frame] = []
    for path in sorted(source.glob("*.jsonl")):
        frames.extend(read_frame_jsonl(path))
    return frames


def decision_for_frame(frame: Frame) -> dict[str, Any]:
    return {
        "family": frame.family,
        "soundness_type": frame.soundness_type.value,
        "label": frame.label,
        "output_state": dict(frame.output_state),
    }


def sft_record_for_frame(frame: Frame) -> dict[str, Any]:
    return {
        "instruction": INSTRUCTION,
        "input": json.dumps(frame.to_jsonable(), sort_keys=True),
        "output": json.dumps(decision_for_frame(frame), sort_keys=True),
        "metadata": {
            "id": frame.id,
            "family": frame.family,
            "soundness_type": frame.soundness_type.value,
            "source": frame.source,
        },
    }


def write_sft_dataset(frames: list[Frame], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(sft_record_for_frame(frame), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export common frames as an SFT JSONL dataset.")
    parser.add_argument("--source", type=Path, default=Path("data/generated"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/hybrid_frames.jsonl"))
    args = parser.parse_args()

    frames = read_common_frames(args.source)
    write_sft_dataset(frames, args.out)
    print(f"wrote {len(frames)} SFT records to {args.out}")


if __name__ == "__main__":
    main()
