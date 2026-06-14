from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.core.frames import write_frame_jsonl
from research_gym.core.metta_frames import count_by_kind, write_jsonl
from research_gym.envs.metta_synthetic import default_frames, frames_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local MeTTa-like training frames.")
    parser.add_argument("--out", type=Path, default=Path("data/metta_frames.jsonl"))
    parser.add_argument("--source", type=Path, default=None, help="Optional synthetic directive file.")
    parser.add_argument("--common-out", type=Path, default=Path("data/generated/metta_frames.jsonl"))
    args = parser.parse_args()

    if args.source is None:
        frames = default_frames()
        source_msg = "built-in directives"
    else:
        frames = frames_from_file(args.source)
        source_msg = str(args.source)

    write_jsonl(args.out, frames)
    write_frame_jsonl(args.common_out, (frame.to_frame() for frame in frames))
    print(f"wrote {len(frames)} frames to {args.out} from {source_msg}")
    print(f"wrote {len(frames)} common frames to {args.common_out}")
    print(json.dumps(count_by_kind(frames), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
