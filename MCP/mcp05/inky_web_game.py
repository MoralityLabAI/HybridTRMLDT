"""Run and retain exact Goose GDK provider output for original standalone games."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .creative_bench import save

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".agents/skills/inky-web-game/SKILL.md"
MODEL = "inky-mini-varloop"
BRIEFS = {
    "rpg": """Write a new, complete small action RPG set in a ruined place with a striking, original palette
and a clear objective. Draw the scenery and characters procedurally in Canvas. Build an attractive interface
with a large arena, title, visible health and controls. A player uses WASD/arrows to move and Space to fight.
Begin in play. Include enemies that move or react, combat feedback, a win after fighting and reaching the goal,
a loss at zero health, and a restart button with id restart. Start with one foe within one sword swing so an early
Space hit can be verified, with at least two more to encounter. Presses and releasing keys should work on Windows.
Implement window.gameState() returning at least {x,y,hp,hits,kills,status}, with status one of playing, won or lost.
hits must count damage the player dealt, and kills must count defeated foes. Do not emit gameplay classes, design,
or output from any previous game. Return one complete HTML document only. Compact source is fine. Finish the
closing html tag.""",
    "story": """Write a new, complete, original interactive story about a fantastical place and one meaningful
choice. Draw visually distinct environments in Canvas by code, with a strong composed palette and a readable
interface. Use at least four distinct scenes including a good and a bad ending. Start at 'start'. Give the player
a visible Study choice with attribute data-choice=study that finds a clue and goes to an intermediate scene.
A Cross choice with data-choice=cross reaches the good ending only after the clue. Show that choice disabled
before study. A Rush choice with data-choice=rush reaches the bad ending without a clue. Give ending shortcuts
a real difference in prose and visuals. Use a restart button with id restart and expose a read-only
window.gameState() returning {scene,clue,ending}, with ending null, 'good', or 'bad'. Invent all story text,
layout, CSS and visuals. Return one complete HTML document only, including the closing html tag.""",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(wire: Path, target: Path) -> dict:
    if not wire.exists():
        return {"status": "no-provider-exchange"}
    events = [json.loads(line) for line in wire.read_bytes().splitlines() if line]
    if [event["event"] for event in events] != ["request", "response"] or events[1]["status"] != 200:
        return {"status": "provider-error"}
    request = json.loads(events[0]["body"])
    response = json.loads(events[1]["body"])
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""
    raw = content.encode("utf-8")
    match = re.search(rb"<!doctype html\s*>[\s\S]+?</html\s*>", raw, flags=re.IGNORECASE)
    receipt = {"status": "incomplete-document", "model": response["model"], "requested_model": request["model"],
               "finish_reason": choice["finish_reason"], "usage": response.get("usage"), "timings": response.get("timings"),
               "reply_sha256": sha(raw), "wire_sha256": sha(wire.read_bytes())}
    if match:
        target.write_bytes(match.group())
        receipt.update(status="extracted", source_sha256=sha(match.group()), source_bytes=len(match.group()),
                       response_utf8_start=match.start(), response_utf8_end=match.end())
    return receipt


def run(game, output, binary, feedback):
    folder = output / game
    folder.mkdir(exist_ok=True)
    attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:02d}"
    attempt.mkdir()
    empty = attempt / "empty-workspace"
    empty.mkdir()
    skill = SKILL.read_bytes()
    (attempt / "skill.md").write_bytes(skill)
    task = BRIEFS[game]
    if feedback:
        task += "\n\nRevise the previous attempt using this independent browser feedback:\n" + feedback
    (attempt / "brief.txt").write_text(task, encoding="utf-8", newline="\n")
    wire = attempt / "inky-wire.jsonl"
    env = {**os.environ, "MCP05_FRESH_PROMPT": "1", "MCP05_WIRE_LOG": str(wire),
           "MCP05_SERVER_WIRE_LOG": str(attempt / "mcp-wire.jsonl")}
    for key in ("MCP05_EPISODE_PREFIX_PATH", "MCP05_FEDERATION", "MCP05_INDEX_ROUTER"):
        env.pop(key, None)
    command = [
        str(binary), "--repo", str(empty), "--goblin-cwd", str(ROOT), "--python", sys.executable,
        "--task-file", str(attempt / "brief.txt"), "--evidence-file", str(attempt / "skill.md"),
        "--session-file", str(attempt / "session.json"), "--base-url", "http://127.0.0.1:18097/v1/",
        "--model", MODEL, "--context-window", "100096", "--max-tokens", "5120",
        "--max-agent-iterations", "1", "--max-tool-calls", "4", "--budget", "1024", "--no-packet",
        "--memory-mode", "raw", "--answer-only", "--timeout-secs", "840",
    ]
    save(attempt / "invocation.json", {"argv": command, "skill_sha256": sha(skill), "workspace_count": 0})
    started = time.perf_counter()
    with (attempt / "gdk.json").open("wb") as stdout, (attempt / "gdk.stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=900, check=False)
    receipt = extract(wire, attempt / "index.html")
    receipt.update(game=game, attempt=attempt.relative_to(output).as_posix(),
                   gdk_exit=result.returncode, wall_seconds=time.perf_counter()-started, skill_sha256=sha(skill))
    save(attempt / "source-receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", choices=sorted(BRIEFS), nargs="+", default=list(BRIEFS))
    parser.add_argument("--feedback", type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feedback = args.feedback.read_text(encoding="utf-8") if args.feedback else None
    for game in args.games:
        run(game, output, args.gdk.resolve(strict=True), feedback)


if __name__ == "__main__":
    main()
