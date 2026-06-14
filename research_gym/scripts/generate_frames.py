from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from research_gym.core.frames import write_frame_jsonl
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate typed reachability frames.")
    parser.add_argument("--out", type=Path, default=Path("data/frames.jsonl"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-out", type=Path, default=Path("data/generated/reachability_frames.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = CoupledStoryworldEnv()
    states = list(env.all_states())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    frames = []
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            state = rng.choice(states)
            frame = env.label_frame(state, args.horizon, frame_id=f"toy-story-{args.seed}-{i:05d}")
            frames.append(frame)
            counts[frame.label.value] = counts.get(frame.label.value, 0) + 1
            f.write(json.dumps(frame.to_jsonable(), sort_keys=True) + "\n")
    write_frame_jsonl(args.common_out, (frame.to_frame() for frame in frames))

    print(f"wrote {args.n} frames to {args.out}")
    print(f"wrote {args.n} common frames to {args.common_out}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
