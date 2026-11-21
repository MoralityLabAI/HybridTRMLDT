"""Generate fresh, root-disjoint Python platformer defects for MCP-03 retrieval.

Generated gold patches and expected values live in private experiment metadata,
outside the public MCP task/observation. Each repository is created at a frozen
base commit and its bare clone is available to MCP-03's SnapshotStore.

    python -m MCP.mcp03.platformer_gen --out MCP/runs/platformer --seed 117 \
        --train 12 --dev 4 --test 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from MCP.mcp02.textutil import digest
from .repo import SnapshotStore, gold_symbols, snapshot_from_dir


def run(*argv: str, cwd: Path | None = None, check: bool = True,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                          check=check, env=env)


# Path, failing expression, repaired expression, asserted behavior, problem statement.
DEFECTS = (
    ("engine/physics.py", "next_speed = speed - gravity",
     "next_speed = speed + gravity",
     "self.assertEqual(advance_vertical(10.0, 0.0, 0.5), (10.5, 0.5))",
     "The player keeps rising after a jump. Vertical gravity should move the avatar down over later frames; inspect advance_vertical."),
    ("engine/camera.py", "return min(max(0, world_x - viewport // 2), world_width)",
     "return min(max(0, world_x - viewport // 2), max(0, world_width - viewport))",
     "self.assertEqual(camera_target(200, 120, 40), 80)",
     "The side-scrolling camera exposes black space past the level's right boundary. Locate the clamp in camera_target."),
    ("engine/collision.py", "if y < 0 or y > len(grid) or x < 0 or x > len(grid[0]):",
     "if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):",
     "self.assertTrue(blocking([[0, 1], [0, 0]], 2, 1))",
     "The tile collision check reads beyond the edge of a platform grid. Fix blocking for cells just outside the map."),
    ("engine/animation.py", "return (tick // hold) % (frames + 1)",
     "return (tick // hold) % frames",
     "self.assertEqual(frame_index(12, 3, 4), 0)",
     "A running sprite sometimes selects a nonexistent animation frame. Correct frame_index's wraparound."),
    ("engine/inventory.py", "return total - gained",
     "return total + gained",
     "self.assertEqual(award_pickup(4, 3), 7)",
     "Collecting a coin reduces the scoreboard total. Fix the logic in award_pickup."),
    ("engine/motion.py", "return speed - drag",
     "return max(0.0, speed - drag)",
     "self.assertEqual(friction_step(0.2, 0.7), 0.0)",
     "When the avatar nearly stops, ground friction reverses its horizontal movement. Inspect friction_step."),
    ("engine/jump.py", "return impulse if grounded else speed",
     "return -impulse if grounded else speed",
     "self.assertEqual(jump_when_grounded(0.0, True, 7.0), -7.0)",
     "The jump button pushes the character into the floor. Correct grounded jump impulse direction."),
    ("engine/health.py", "return health - max_damage",
     "return max(0, health - max_damage)",
     "self.assertEqual(apply_damage(3, 9), 0)",
     "An enemy hit drives player health below zero. Clamp the hit points in apply_damage."),
)

MODULES = {
    "engine/physics.py": '''"""Vertical motion integrator for player movement."""


def advance_vertical(position: float, speed: float, gravity: float) -> tuple[float, float]:
    """Integrate gravity across a single frame and return position and velocity."""
    next_speed = speed + gravity
    return position + next_speed, next_speed


def terminal_speed(speed: float, maximum: float) -> float:
    return min(speed, maximum)
''',
    "engine/camera.py": '''"""Scrolling camera frame and viewport tracking."""


def camera_target(world_x: int, world_width: int, viewport: int) -> int:
    """Keep horizontal focus within the visible world boundaries."""
    return min(max(0, world_x - viewport // 2), max(0, world_width - viewport))


def centered(point: int, viewport: int) -> int:
    return point - viewport // 2
''',
    "engine/collision.py": '''"""Tile collision at grid edges."""


def blocking(grid: list[list[int]], x: int, y: int) -> bool:
    """Treat any out-of-map tile as a solid collision boundary."""
    if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
        return True
    return bool(grid[y][x])


def floor_contact(y: float, floor_y: float) -> bool:
    return y >= floor_y
''',
    "engine/animation.py": '''"""Sprite sheet animation timing."""


def frame_index(tick: int, hold: int, frames: int) -> int:
    """Choose one of the available frames in a repeating sequence."""
    return (tick // hold) % frames


def sprite_interval(ticks: int, frames: int) -> float:
    return ticks / frames
''',
    "engine/inventory.py": '''"""Reward inventory and coin pickups."""


def award_pickup(total: int, gained: int) -> int:
    """Accumulate collected coins in the HUD counter."""
    return total + gained


def score_multiplier(score: int, multiplier: int) -> int:
    return score * multiplier
''',
    "engine/motion.py": '''"""Horizontal avatar movement on a platform."""


def friction_step(speed: float, drag: float) -> float:
    """Apply nonnegative ground friction to a rightward velocity."""
    return max(0.0, speed - drag)


def walk_acceleration(speed: float, force: float) -> float:
    return speed + force
''',
    "engine/jump.py": '''"""Avatar jump input and grounded state."""


def jump_when_grounded(speed: float, grounded: bool, impulse: float) -> float:
    """Apply upward impulse using negative vertical velocity."""
    return -impulse if grounded else speed


def can_double_jump(grounded: bool, charges: int) -> bool:
    return grounded or charges > 0
''',
    "engine/health.py": '''"""Avatar hit points and combat damage."""


def apply_damage(health: int, max_damage: int) -> int:
    """Remove damage without reducing health below zero."""
    return max(0, health - max_damage)


def respawn_health(max_health: int) -> int:
    return max_health
''',
}

DISTRACTOR = '''"""Platformer support system: {topic}. Catalog tag {tag}."""


def {name}_preview(value: int, amount: int) -> int:
    """Return a harmless preview for {topic} in a menu."""
    return value + amount


def {name}_bounds(value: int, cap: int) -> int:
    """Clamp a {topic} preview for the editor."""
    return min(max(value, 0), cap)
'''

TOPICS = ("camera", "physics", "collision", "sprite", "inventory", "movement",
          "jump", "health", "level", "viewport", "tiles", "camera_follow")


def put(root: Path, path: str, value: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def check_suite(root: Path, *, expected_success: bool) -> None:
    # Same-length edits within the filesystem timestamp granularity can reuse
    # an old Python .pyc. A fixture must execute the exact current source.
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache)
    proc = run(sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
               "-p", "test_*.py", cwd=root, check=False)
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache)
    if (proc.returncode == 0) != expected_success:
        raise AssertionError(f"Unexpected test outcome: {root}\n{proc.stdout}\n{proc.stderr}")


HOLDOUT_SYMBOLS = {
    "advance_vertical": "integrate_fall",
    "camera_target": "focus_viewport",
    "blocking": "tile_is_solid",
    "frame_index": "sprite_frame_at",
    "award_pickup": "collect_reward",
    "friction_step": "slow_on_ground",
    "jump_when_grounded": "begin_ground_jump",
    "apply_damage": "take_player_hit",
}


def shifted_source(text: str) -> str:
    """Apply a consistent holdout naming change without modifying behavior."""
    for old, new in HOLDOUT_SYMBOLS.items():
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    return text


HARD_SYMPTOMS = (
    "A character jumps but falls in the wrong direction during frame integration.",
    "The scrolling viewport shows empty tiles after reaching the far right of a level.",
    "Touching a boundary tile can crash the world collision query.",
    "The running sprite occasionally looks up a frame past its animation sheet.",
    "Coins collected from a platform change the HUD total in the wrong direction.",
    "Ground drag changes the avatar's horizontal direction at low speed.",
    "Jumping from a solid floor produces a downward kick.",
    "An enemy strike makes the player health counter negative.",
)


def build_case(work: Path, split: str, index: int, *, seed: int, family: int,
               difficulty: str = "easy", surface_shift: bool = False,
               family_group: tuple[int, ...] | None = None, layout_shift: bool = False,
               export_bare: bool = True) -> dict:
    repo = f"platformer-seeded/{split}-{seed}-{index:03d}"
    root = work / "sources" / f"{split}-{seed}-{index:03d}"
    root.mkdir(parents=True, exist_ok=False)
    if difficulty not in ("easy", "hard"):
        raise ValueError("difficulty must be easy or hard")
    families = [family] if difficulty == "easy" else [
        family, (family + 3) % len(DEFECTS), (family + 5) % len(DEFECTS)]
    families = list(family_group) if family_group is not None else families
    paths = {name: name.replace("engine/", "engine/runtime/", 1) if layout_shift else name
             for name in MODULES}
    repairs = {paths[DEFECTS[f][0]]: (DEFECTS[f][1], DEFECTS[f][2]) for f in families}
    primary_issue = DEFECTS[family][4]
    issue = (primary_issue if difficulty == "easy" else
             primary_issue + " Two other related core engine failures are observed: " +
             " ".join(HARD_SYMPTOMS[f] for f in families[1:]))
    if surface_shift:
        issue = shifted_source(issue)

    put(root, "engine/__init__.py", "")
    if layout_shift:
        put(root, "engine/runtime/__init__.py", "")
    for name, value in sorted(MODULES.items()):
        if surface_shift:
            value = shifted_source(value)
        if paths[name] in repairs:
            broken, repaired = repairs[paths[name]]
            assert value.count(repaired) == 1, (name, repaired)
            value = value.replace(repaired, broken)
        put(root, paths[name], value)
    put(root, "tools/__init__.py", "")
    n_variants = 1 if difficulty == "easy" else 6
    for i in range(n_variants):
        for topic in TOPICS:
            tag = hashlib.sha256(f"{seed}/{split}/{index}/{topic}/{i}".encode()).hexdigest()[:8]
            folder = "tools" if difficulty == "easy" else ("tools", "engine/plugins", "samples")[i % 3]
            put(root, f"{folder}/{topic}_guide_{i}.py",
                DISTRACTOR.format(topic=f"{topic} camera physics collision player sprite world",
                                  tag=tag, name=f"{topic}_variant_{i}"))
    put(root, "docs/level_notes.py", '''"""Platformer tutorial notes about gravity collision cameras and sprite animations."""


def describe_level() -> str:
    return "Platformer guides, not active gameplay code"
''')
    imports = []
    asserts = []
    for f in families:
        file_path, _broken, _repaired, assertion, _issue = DEFECTS[f]
        func = MODULES[file_path].split("def ")[1].split("(")[0]
        if surface_shift:
            func = HOLDOUT_SYMBOLS.get(func, func)
        imports.append(f"from {paths[file_path][:-3].replace('/', '.')} import " + func)
        asserts.append(f"    def test_regression_{f}(self):\n        "
                       + (shifted_source(assertion) if surface_shift else assertion))
    put(root, "tests/test_gameplay.py",
        "import unittest\n" + "\n".join(imports) + "\n\n"
        + "class GameplayRegression(unittest.TestCase):\n"
        + "\n".join(asserts) + "\n")
    put(root, "README.md",
        f"# Generated platformer engine\n\nScenario {split}-{index:03d}.\n"
        "Python source under engine/, UI helper modules under tools/.\n")

    check_suite(root, expected_success=False)
    run("git", "init", "-q", cwd=root)
    run("git", "add", "-A", cwd=root)
    # Snapshot commit SHA enters episode_key for epsilon-oracle rollouts.
    # Freeze author/committer clocks so a repeated seed reproduces training
    # trajectories, not merely the visible patch and gold digest.
    commit_env = dict(os.environ, GIT_AUTHOR_DATE="2001-01-01T00:00:00+0000",
                      GIT_COMMITTER_DATE="2001-01-01T00:00:00+0000")
    run("git", "-c", "user.name=MCP Fixture", "-c", "user.email=mcp@example.invalid",
        "commit", "-q", "-m", "generated platformer base", cwd=root, env=commit_env)
    commit = run("git", "rev-parse", "HEAD", cwd=root).stdout.strip()

    originals = {}
    for name, (broken, repaired) in sorted(repairs.items()):
        base = (root / name).read_text(encoding="utf-8")
        assert base.count(broken) == 1, (name, broken)
        originals[name] = base
        put(root, name, base.replace(broken, repaired))
    check_suite(root, expected_success=True)
    patch = run("git", "diff", "--", *sorted(repairs), cwd=root).stdout
    assert patch.count("diff --git") == len(families)
    for name, base in originals.items():
        put(root, name, base)
    if export_bare:
        clone_path = work / "repos" / repo.replace("/", "__")
        run("git", "clone", "-q", "--bare", str(root), str(clone_path))
        snapshot = SnapshotStore(work / "repos", work / "cache").snapshot(repo, commit)
    else:
        snapshot = snapshot_from_dir(root)
    gold = gold_symbols(snapshot, patch)
    if gold["files"] != sorted(repairs) or len(gold["symbols"]) < len(families):
        raise AssertionError(f"Invalid generated gold targets: {gold}")
    instance_id = f"platformer-{split}-{seed}-{index:03d}"
    return {"instance_id": instance_id, "repo": repo, "base_commit": commit,
            "created_at": f"2026-11-21T12:{index // 60:02d}:{index % 60:02d}Z",
            "problem_statement": issue + " Make the smallest source-level correction.",
            "patch": patch,
            "_receipt": {"instance_id": instance_id, "root": repo, "family": family,
                         "base_commit": commit, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                         "difficulty": difficulty, "families": families,
                         "surface_shift": surface_shift, "layout_shift": layout_shift,
                         "gold_digest": digest(gold), "gold_symbols": gold["symbols"]}}


def generate(out: Path, *, seed: int = 117, train: int = 12,
             dev: int = 4, test: int = 6, difficulty: str = "easy",
             holdout_symbol_names: bool = False) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be a new empty directory: {out}")
    for name in ("data", "repos", "sources", "cache"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {}
    receipts: dict[str, list[dict]] = {}
    for split, count in (("train", train), ("dev", dev), ("test", test)):
        rows = []
        receipts[split] = []
        for index in range(count):
            family = (index + rng.randrange(len(DEFECTS))) % len(DEFECTS)
            case = build_case(out, split, index, seed=seed, family=family, difficulty=difficulty,
                              surface_shift=holdout_symbol_names and split == "test")
            receipts[split].append(case.pop("_receipt"))
            rows.append(case)
        with (out / "data" / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        splits[split] = rows

    from pathlib import Path
    base_cfg = json.loads((Path(__file__).resolve().parent / "configs" / "read_aware_cpu.json")
                          .read_text(encoding="utf-8"))
    base_cfg["work_dir"] = str(out.resolve())
    base_cfg["splits"] = {
        split: {"source": f"data/{split}.jsonl", "repos": [r["repo"] for r in rows]}
        for split, rows in splits.items()
    }
    base_cfg["dataset"].update({"arms": ["search", "layered_ldt"],
                                "epsilons": [0.0, 0.35], "max_candidates": 12})
    controls = ["lexical_bm25", "identifier_rules"]
    if difficulty == "hard":
        controls.append("oracle_upper_bound")  # private-gold-assisted diagnostic ceiling only
    base_cfg["evaluation"].update({"arms": ["search"], "controls": controls,
                                   "max_tool_calls": [24], "budgets": [1024],
                                   "bm25_file_budgets": [1024], "prefilter_k": 32,
                                   "composed_packets": False})
    base_cfg["training"].update({"epochs": 2, "batch_states": 16, "threads": 4})
    old_config = json.loads(json.dumps(base_cfg))
    old_config["dataset"]["utility_targets"] = False
    old_config["models"] = {"looped_p4_d64_binary": {
        "config": {"arch": "looped", "d": 64, "heads": 4, "passes": 4, "encoder": "match"}}}
    read_config = json.loads(json.dumps(base_cfg))
    read_config["models"] = {"looped_p4_d64_readaware": base_cfg["models"]["looped_p4_d64_readaware"]}
    eval_config = json.loads(json.dumps(read_config))
    eval_config["models"].update(old_config["models"])
    # RU02 matched-state controls. All four models receive *exactly* the
    # same encoded training slates from read_train.json.gz, including the
    # public remaining-call horizon marker. Only the head and loss differ.
    matched_config = json.loads(json.dumps(base_cfg))
    core = {"arch": "looped", "d": 64, "heads": 4, "passes": 4, "encoder": "match"}
    matched_config["models"] = {
        "looped_binary_matched": {"config": dict(core, use_horizon=True)},
        "looped_shared_utility": {"config": dict(core, single_utility=True)},
        "looped_dual_utility": {"config": dict(core, utility_heads="expand_read")},
        "looped_dual_shuffled": {"config": dict(core, utility_heads="expand_read"),
                                 "shuffle_utilities_only": True},
    }
    matched_config["evaluation"]["controls"] = ["lexical_bm25", "identifier_rules"]
    for name, cfg in (("readaware", read_config), ("binary", old_config),
                      ("eval", eval_config), ("matched", matched_config)):
        put(out, f"config_{name}.json", json.dumps(cfg, indent=2) + "\n")
    summary = {"generator": "platformer-gen-v4", "difficulty": difficulty, "seed": seed,
               "holdout_symbol_names": holdout_symbol_names,
               "cases": {k: len(v) for k, v in splits.items()},
               "roots": {k: [r["repo"] for r in v] for k, v in splits.items()},
               "gold_receipts": receipts,
               "condition": "train/dev/test roots disjoint; public observations exclude patch"}
    put(out, "manifest.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--difficulty", choices=("easy", "hard"), default="easy")
    parser.add_argument("--holdout-symbol-names", action="store_true")
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--dev", type=int, default=4)
    parser.add_argument("--test", type=int, default=6)
    args = parser.parse_args(argv)
    if min(args.train, args.dev, args.test) < 1:
        parser.error("every split needs at least one root")
    manifest = generate(args.out, seed=args.seed, train=args.train, dev=args.dev, test=args.test,
                        difficulty=args.difficulty, holdout_symbol_names=args.holdout_symbol_names)
    print(json.dumps({k: manifest[k] for k in ("generator", "difficulty", "seed", "cases", "condition")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
