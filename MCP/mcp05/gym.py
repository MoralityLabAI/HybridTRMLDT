"""Generate and stage root-disjoint, natively verified repair cases."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from MCP.mcp03.platformer_gen import build_case

ROOT = Path(__file__).resolve().parents[2]
GUARD = Path(__file__).with_name("verifier_guard.py")
VERIFIER = ["-B", str(GUARD), "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]


def dump(path: Path, obj) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate(out: Path, seed: int, train: int, test: int) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    train_families = list(range(6))
    rng.shuffle(train_families)
    manifest = {"seed": seed, "train": [], "test": []}
    for split, count in (("train", train), ("test", test)):
        for index in range(count):
            if split == "train":
                family = train_families[index % len(train_families)]
                cohort = (family, train_families[(index + 1) % len(train_families)]) if index % 3 == 2 else (family,)
            else:
                family = 6 + index % 2
                cohort = (family, 13 - family) if index % 3 == 2 else (family,)
            case = build_case(out, split, index, seed=seed, family=family,
                              family_group=cohort, difficulty="hard" if len(cohort) > 1 else "easy",
                              surface_shift=split == "test", layout_shift=split == "test",
                              export_bare=False)
            manifest[split].append(case)
    dump(out / "sealed.json", manifest)
    return manifest


def clone(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    # Copy both the frozen worktree and its local Git object store.
    shutil.copytree(source, destination)


def source_root(out: Path, case: dict) -> Path:
    return out / "sources" / case["repo"].split("/", 1)[1]


def verifier_manifest(python: str = sys.executable) -> dict:
    return {"argv": [python, *VERIFIER], "editable": ["engine/"], "timeout_seconds": 30}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--train", type=int, default=18)
    parser.add_argument("--test", type=int, default=6)
    args = parser.parse_args()
    manifest = generate(args.out.resolve(), args.seed, args.train, args.test)
    print(json.dumps({k: len(manifest[k]) for k in ("train", "test")}))


if __name__ == "__main__":
    main()
