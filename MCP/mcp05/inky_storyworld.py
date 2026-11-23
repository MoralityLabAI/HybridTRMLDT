"""Run small-storyworld-builder phases through the local Inkling Goose GDK."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "MCP.mcp05"

from .context import sha
from .creative_bench import save
from .inky_goose_game import MODEL, ROOT, git, read_json, replay_writes
from .storyworld_swmd import parse_scene, profile_pair, reachable, rehearse, render_card, sweepweave_document, swmd_document, world_ids

CARD_BRIEF = """Use create_file to write world-card.json. Plan an original illustrated interactive fable
about a cat and an unusual hat. Invent your own characters and visual identity. This turn is just the
compact planning card for later SWMD authorship. Keep it below 1,400 characters, with no scene list.

Use this JSON structure:
{title, premise, cast:[{id,name,voice,visual}], stats:{trust,nerve,care},
acts:[string,string,string,string,string], endings:[{id,type,condition,expected_critic_score}], balance}
Use 2-4 characters with IDs like char_cat and one-sentence motives/visual signatures. Stats are numeric
initial values in [-1,1]. Give five very short act summaries. The 20 encounters will be written later.
Plan 3 or 4 terminal scenes, and let ordinary encounters fill the remainder, with three meaningful options
and two reactions per option. Define terminal IDs in endings with conditions and numeric expected_critic_score
values. Use a meaningful fallback when the others do not apply. State the tradeoffs and safety
bounds for later reaction scoring and effects in balance. The card guides your later authoring.
Write the JSON file via one tool action, then call the verifier. You own all creative decisions."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_card(path: Path) -> dict:
    card = read_json(path)
    require(all(isinstance(card[key], str) and card[key].strip() for key in ("title", "premise")),
            "title and premise must be nonempty strings")
    require(isinstance(card["balance"], (str, dict)) and card["balance"], "describe the balance model")
    cast = card["cast"]
    require(2 <= len(cast) <= 4 and len({c["id"] for c in cast}) == len(cast), "provide 2–4 unique cast IDs")
    require(all(c["id"].startswith("char_") and all(c[key].strip() for key in ("name", "voice", "visual"))
                for c in cast), "give each character a named voice and visual identity")
    stats = card["stats"]
    require(set(stats) == {"trust", "nerve", "care"}
            and all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1 for x in stats.values()),
            "stats must initialize trust, nerve and care in [-1,1]")
    require(len(card["acts"]) == 5 and all(isinstance(a, str) and a.strip() for a in card["acts"]),
            "provide a five-act outline")
    endings = card["endings"]
    require(3 <= len(endings) <= 4 and len(set(world_ids(card))) == 20 and all(
            re.fullmatch(r"[a-z][a-z0-9_]*", e["id"]) and e["type"] and e["condition"] and
            type(e["expected_critic_score"]) in (float, int) and math.isfinite(e["expected_critic_score"])
            for e in endings), "provide 3–4 distinct ending IDs, types, conditions, and finite authored scores")
    return card


def check_profiles(source: dict) -> dict:
    profiles = source["profiles"]
    require(isinstance(profiles, dict) and 6 <= len(profiles) <= 9, "provide six to nine named profile pairs")
    fingerprints = set()
    accepted = {}
    for name, profile in profiles.items():
        require(re.fullmatch(r"[a-z][a-z0-9_]{1,22}", name), f"invalid profile name: {name}")
        try:
            if isinstance(profile, dict):
                profile = [[profile[arm]["score"], profile[arm]["delta"]] for arm in ("a", "b")]
            require(len(profile) == 2,
                    'use {"a":{"score":[b,t,n,c],"delta":[t,n,c]},"b":{"score":[b,t,n,c],"delta":[t,n,c]}} '
                    "for each named profile, with your actual numbers in each of the four arrays")
            fingerprints.add(profile_pair(profile))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{name}: {error}") from error
        accepted[name] = profile
    require(len(fingerprints) == len(profiles), "each profile needs a different pair of effect vectors")
    return accepted


def with_profiles(card: dict, root: Path) -> dict:
    model = root / "00-math/workspace/reaction-model.json"
    return {**card, "profiles": check_profiles(read_json(model))} if model.exists() else card


def scene_path(scene_id):
    return f"{scene_id}.card.json"


def groups(card):
    playable = 20 - len(card["endings"])
    return [(start, min(start + 3, playable)) for start in range(1, playable + 1, 4)] + [(playable + 1, 20)]


def check_scenes(card: dict, workspace: Path, ids: list[str]) -> list[dict]:
    scenes, errors = [], []
    for scene_id in ids:
        try:
            block = render_card(read_json(workspace / scene_path(scene_id)), card, scene_id)
            scenes.append(parse_scene(block, card, scene_id))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{scene_id}: {error}")
    require(not errors, "\n".join(errors))
    return scenes


def scene_brief(card, ids, recent):
    names = ", ".join(scene_path(id_) for id_ in ids)
    terminal_ids = [e["id"] for e in card["endings"]]
    roster = world_ids(card)
    playable = 20 - len(terminal_ids)
    act_indices = {(roster.index(id_)) // 4 for id_ in ids}
    act = " / ".join(card["acts"][i] for i in sorted(act_indices))
    cast = " / ".join(f"{c['name']}: {c['voice'][:110]}" for c in card["cast"])
    brief = (f"Use create_file, one JSON encounter card per tool call, in order: {names}.\n"
             f"World: {card['title']}. {card['premise'][:240]}\nCast: {cast}\nAct goal: {act}\n"
             f"Prior scene: {recent or 'Opening'}\n"
             "Each card is a JSON object with title, text, art and options. Write 2–4 vivid sentences of scene prose, "
             "a distinct drawable visual cue, dialogue and a consequence that fits your story. Your cards compile to SWMD. "
             "Call run_verifier after all requested files exist.\n")
    if ids[0] in terminal_ids:
        endings = " / ".join(f"{e['id']}: {e['condition']}" for e in card["endings"])
        return brief + f"All are terminal. Set options to []. Pay off these endings: {endings}"
    routes = []
    for id_ in ids:
        i = roster.index(id_)
        choices = roster[i + 1:min(i + 3, playable)] if i < playable - 1 else terminal_ids
        routes.append(f"{id_} -> {', '.join(choices)}")
    if profiles := card.get("profiles"):
        return brief + """Each ordinary card has three distinct options. Atomically write exactly one card.
An option is {"label":"YOUR CHOICE","model":"PROFILE_NAME","reactions":[{"next":"ENCOUNTER_ID","reply":"YOUR REPLY"},{"next":"ENCOUNTER_ID","reply":"ANOTHER REPLY"}]}
Give each option a different profile from your authored rulebook. A reaction's next value is an actual
encounter ID. Use the first permitted target at least once per scene; the finale uses every listed target.
Keep encounter prose, artwork directions, and replies vivid and concise.
Your named profiles: """ + ", ".join(profiles) + "\nPermitted routes: " + " / ".join(routes)
    return brief + """Use objects with named fields. The JSON shape for a single option is:
{"label":"YOUR CHOICE","reactions":[{"next":"ENCOUNTER_ID","reply":"YOUR RESPONSE","score":[0,0,0,0],"delta":[0,0,0]}, {"next":"ENCOUNTER_ID","reply":"A DIFFERENT RESPONSE","score":[0,0,0,0],"delta":[0,0,0]}]}
Write exactly 3 options. Replace every placeholder, including all-zero vectors. The `next` field is an
encounter ID from the route list, never a speaker. `score` is four integer hundredths [bias,trust,nerve,care],
and `delta` is three integer hundredths [trust,nerve,care]. Score abs total <=100. Use opposite nonzero
slopes and small biases in each reaction pair. The sum of the absolute differences between the three slopes
must exceed the absolute bias difference. Change 2–3 stats per reaction within -24 to 24. Use distinct tradeoffs.
The first listed target for each regular scene must occur at least once; the finale must use all listed targets.
Use exactly these permitted targets: """ + " / ".join(routes)


def run_phase(root: Path, gdk: Path, brief: str, paths: list[str], verifier: list[str], *,
              max_tokens: int = 2304, iterations: int = 7, context: int = 8192, validate_create: list[str] | None = None) -> dict:
    root.mkdir(parents=True)
    workspace = root / "workspace"
    workspace.mkdir()
    git(workspace, "init", "--quiet")
    git(workspace, "-c", "user.name=Goose workspace", "-c", "user.email=goose@example.invalid",
        "commit", "--quiet", "--allow-empty", "-m", "Begin with an empty authoring workspace")
    task = root / "brief.txt"
    task.write_text(brief + "\n", encoding="utf-8", newline="\n")
    manifest = root / "verifier.json"
    save(manifest, {"argv": verifier, "editable": paths, "creatable": paths, "timeout_seconds": 45,
                    **({"validate_create_argv": validate_create} if validate_create else {})})
    command = [str(gdk), "--repo", str(workspace), "--goblin-cwd", str(ROOT), "--python", sys.executable,
               "--task-file", str(task), "--verifier", str(manifest), "--session-file", str(root / "session.json"),
               "--base-url", "http://127.0.0.1:18097/v1/", "--model", MODEL, "--context-window", str(context),
               "--max-tokens", str(max_tokens), "--max-agent-iterations", str(iterations), "--max-tool-calls", "24",
               "--budget", "1024", "--no-packet", "--memory-mode", "raw", "--allow-create", "--json-actions",
               "--timeout-secs", "840"]
    invocation = {"argv": command, "base_commit": git(workspace, "rev-parse", "HEAD"),
                  "verifier_sha256": sha(manifest.read_bytes()),
                  "brief_sha256": sha(task.read_bytes()), "gdk_sha256": sha(gdk.read_bytes())}
    if runtime_file := os.environ.get("INKY_STORYWORLD_RUNTIME"):
        invocation["model_runtime"] = read_json(Path(runtime_file))
    save(root / "invocation.json", invocation)
    env = {**os.environ, "MCP05_WIRE_LOG": str(root / "inky-wire.jsonl"),
           "MCP05_SERVER_WIRE_LOG": str(root / "mcp-wire.jsonl")}
    if validate_create:
        env["MCP05_COMPACT_CREATE_HISTORY"] = "1"
    for key in ("MCP05_FRESH_PROMPT", "MCP05_EPISODE_PREFIX_PATH", "MCP05_FEDERATION", "MCP05_INDEX_ROUTER"):
        env.pop(key, None)
    print(f"Inkling phase {root.name}", flush=True)
    with (root / "gdk.json").open("wb") as stdout, (root / "gdk.stderr.log").open("wb") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    return read_json(root / "gdk.json")


def export(root: Path, card_file: Path, gptstoryworld: Path) -> dict:
    card = with_profiles(check_card(card_file), root)
    roster = world_ids(card)
    ifid = "SW-INKY-CAT-HAT-20261123"
    (root / "export").mkdir(exist_ok=True)
    out = root / "export"
    receipts, blocks, scenes = {}, [], []
    card_phase = card_file.parents[1]
    receipts[card_phase.name] = replay_writes(card_phase, [card_file.name], require_verified=False)
    if card.get("profiles"):
        receipts["00-math"] = replay_writes(root / "00-math", ["reaction-model.json"])
    for start, end in groups(card):
        phase = root / f"{start:02}-{end:02}-encounters"
        ids = roster[start - 1:end]
        paths = [scene_path(id_) for id_ in ids]
        receipts[phase.name] = replay_writes(phase, paths)
        scenes.extend(check_scenes(card, phase / "workspace", ids))
        blocks.extend(render_card(read_json(phase / "workspace" / name), card, id_) for name, id_ in zip(paths, ids, strict=True))
    require(reachable(scenes) == set(roster), "encounter graph has unreachable scenes")
    require(len({scene["art"] for scene in scenes}) == 20, "every encounter must have distinct art direction")
    require(len({scene["text"] for scene in scenes}) == 20, "every encounter must contain original scene prose")
    receipts_path = root / "authoring-receipts.json"
    save(receipts_path, receipts)
    swmd = out / "world.swmd.min.md"
    swmd.write_text(swmd_document(card, ifid, blocks), encoding="utf-8", newline="\n")
    sources = gptstoryworld.resolve(strict=True)
    parser_path = sources / "mcp-storyworld-encounter/swmd_store.py"
    native = runpy.run_path(str(parser_path))
    require(native["parse_swmd_min"](swmd).encounter_order == roster, "GPTStoryworld SWMD index lost an encounter")
    world = out / "storyworld.json"
    save(world, sweepweave_document(card, ifid, scenes))
    validate_path = sources / "codex-skills/storyworld-building/scripts/sweepweave_validator.py"
    errors = runpy.run_path(str(validate_path))["validate_storyworld"](str(world))
    require(not errors, "\n".join(errors))
    balance = rehearse(scenes, card["stats"])
    save(out / "balance.json", balance)
    data = {"title": card["title"], "premise": card["premise"], "cast": card["cast"], "stats": card["stats"],
            "endings": card["endings"], "scenes": scenes}
    (out / "world-data.js").write_text("window.WORLD = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                      encoding="utf-8", newline="\n")
    save(out / "manifest.json", {"source_commit": git(sources, "rev-parse", "HEAD"),
                                 "skill_sha256": sha((sources / "codex-skills/small-storyworld-builder/SKILL.md").read_bytes()),
                                 "validator_sha256": sha(validate_path.read_bytes()), "swmd_index_sha256": sha(parser_path.read_bytes()),
                                 "artifacts": {file.name: sha(file.read_bytes()) for file in (swmd, world, out / "balance.json", out / "world-data.js")}})
    print(f"Validated SWMD and SweepWeave export for {card['title']}. Ending counts: {balance['endings']}", flush=True)
    return data


def play_brief(data):
    art = "\n".join(f"{item['id']} {item['art']}" for item in data["scenes"])
    cast = "\n".join(f"{item['name']}: {item['visual']}" for item in data["cast"])
    return f"""Build the original {data['title']} browser edition. Use Goose create_file for index.html first,
then for art.js. The world-data.js export is provided at runtime. You author all player code, Canvas
art, colors, typography, layout, and interaction. Use inline CSS and no external resources.

index.html must load world-data.js and art.js with script src. window.WORLD contains title, premise,
cast:[{{id,name,voice,visual}}], stats:{{trust,nerve,care}}, endings:[{{id,type,condition,expected_critic_score}}],
scenes:[{{id,title,text,art,options:[{{id,text,reactions:[{{id,to,text,effects,score}}]}}]}}].
Start on the first scene. For each visible choice, use button[data-option=OPTION_ID]. On its click
select the reaction with maximum score b+t*trust+n*nerve+c*care, array order [b,t,n,c], first wins ties.
Apply its effects map by addition, clamped to [-1,1]. Show the reaction prose and effects before Continue.
Continue button id continue goes to reaction.to. Restart id restart resets the world. Terminal scenes show
no active choice. Expose a read-only window.gameState() with keys scene, stats (actual unrounded values),
reaction (ID or null), and ending (terminal scene ID or null). Call window.drawScene(canvas,scene,WORLD.cast)
for the first scene and after navigation. Use accessible, legible controls and readable scene prose.

In art.js define window.drawScene(canvas,scene,cast). Draw your own character portraits using paths,
fill/stroke, and a coherent palette. Draw setting-specific composition and props for each of these
20 scenes through Canvas code. Give landmark and ending illustrations distinct silhouettes and geometry.
Use the prose cues below as your art direction. Keep each file within one action. Call run_verifier after both.

Your character designs:\n{cast}\n\nYour scene cues:\n{art}"""


def play(root: Path, gdk: Path, data: dict, node: Path, chrome: Path) -> None:
    out = root / "export"
    target = root / "21-player"
    captures = target / "browser"
    files = ["index.html", "art.js"]
    browser_check = [str(node), str(ROOT / "MCP/demos/browser-smoke.mjs"), str(chrome), str(captures),
                     "storyworld", str(target / "workspace"), str(out / "world-data.js"), str(out / "balance.json")]
    if not target.exists():
        result = run_phase(target, gdk, play_brief(data), files, browser_check,
                           max_tokens=4608, context=16384, iterations=10)
        require(result["state"].get("verified"), f"Inkling's browser verification failed: {target}")
    receipt = replay_writes(target, files)
    for name in files:
        shutil.copyfile(target / "workspace" / name, out / name)
    save(target / "source-receipt.json", receipt)
    manifest = read_json(out / "manifest.json")
    manifest["artifacts"].update({name: sha((out / name).read_bytes()) for name in files})
    manifest["browser_check"] = read_json(captures / "browser-receipt.json")
    save(out / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check-card", "math", "check-math", "check-proposed-math",
                        "scenes", "check-scenes", "check-proposed", "export", "play"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--gptstoryworld", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--chrome", type=Path)
    args = parser.parse_args()
    if args.command in ("check-math", "check-proposed-math"):
        profiles = check_profiles(read_json(args.path) if args.command == "check-math" else json.loads(sys.stdin.buffer.read()))
        print(f"Validated {len(profiles)} state-dependent math profiles.")
        return 0
    if args.command == "check-proposed":
        filename = args.path.name
        id_ = filename.removesuffix(".card.json")
        card = with_profiles(check_card(args.card), args.card.resolve().parents[2])
        require(filename == scene_path(id_) and id_ in world_ids(card), "write an authorized encounter file")
        source = json.loads(sys.stdin.buffer.read())
        parse_scene(render_card(source, card, id_), card, id_)
        return 0
    if args.command == "check-card":
        card = check_card(args.path)
        print(f"Validated five-act plan for {card['title']} with {len(card['endings'])} ending cards.")
        return 0
    if args.command == "check-scenes":
        items = check_scenes(with_profiles(check_card(args.card), args.card.resolve().parents[2]), args.path, args.ids)
        print("Verified SWMD encounter blocks: " + ", ".join(scene["id"] for scene in items))
        return 0
    if args.command == "export":
        if args.gptstoryworld is None:
            parser.error("export requires --gptstoryworld")
        root = args.path.resolve()
        export(root, args.card.resolve() if args.card else root / "00-world/workspace/world-card.json", args.gptstoryworld)
        return 0
    if args.gdk is None:
        parser.error("a rollout requires --gdk")
    gdk = args.gdk.resolve(strict=True)
    phase = args.path.resolve()
    if args.command == "math":
        card = check_card(phase / "00-world/workspace/world-card.json")
        target = phase / "00-math"
        script = str(Path(__file__).resolve())
        math_brief = f"""Write reaction-model.json with create_file for your cat-and-hat story {card['title']}.
Define exactly six named profiles in the top-level profiles object. Use this shape for each named profile:
{{"a":{{"score":[BIAS,TRUST,NERVE,CARE],"delta":[TRUST,NERVE,CARE]}},"b":{{"score":[BIAS,TRUST,NERVE,CARE],"delta":[TRUST,NERVE,CARE]}}}}.
Replace all placeholders with numbers. Use lowercase descriptive names. Each score is four
integer hundredths [bias, trustWeight, nerveWeight, careWeight]. Make both biases zero and the two
weight vectors opposite, with restrained magnitudes, so either reaction can win across character states.
Each delta is three integer hundredths [trustDelta, nerveDelta, careDelta]. Use distinct tradeoffs around
2–12 hundredths, change two or three stats per response, and include both gains and costs. Every profile
must use a different pair of effect vectors. Author all the numbers. Call run_verifier after the write."""
        result = run_phase(target, gdk, math_brief, ["reaction-model.json"],
                           [sys.executable, script, "check-math", str(target / "workspace/reaction-model.json")],
                           validate_create=[sys.executable, script, "check-proposed-math"], max_tokens=1024)
        return 0 if result["state"].get("verified") else 1
    if args.command == "play":
        if not (args.gptstoryworld and args.node and args.chrome):
            parser.error("play requires --gptstoryworld, --node, and --chrome")
        card_file = args.card.resolve() if args.card else phase / "00-world/workspace/world-card.json"
        data = export(phase, card_file, args.gptstoryworld)
        play(phase, gdk, data, args.node.resolve(strict=True), args.chrome.resolve(strict=True))
        return 0
    if args.command == "scenes":
        card_path = args.card.resolve() if args.card else phase / "00-world/workspace/world-card.json"
        card = with_profiles(check_card(card_path), phase)
        preceding = []
        roster = world_ids(card)
        for start, end in groups(card):
            ids = roster[start - 1:end]
            target = phase / f"{start:02}-{end:02}-encounters"
            if target.exists():
                result = read_json(target / "gdk.json")
            else:
                recent = f"{preceding[-1]['title']}: {preceding[-1]['text'][:135]}" if preceding else ""
                result = run_phase(target, gdk, scene_brief(card, ids, recent), [scene_path(id_) for id_ in ids],
                                   [sys.executable, str(Path(__file__).resolve()), "check-scenes", str(target / "workspace"),
                                    "--card", str(card_path), "--ids", *ids], max_tokens=1408, iterations=14,
                                   validate_create=[sys.executable, str(Path(__file__).resolve()), "check-proposed", "--card", str(card_path)])
            require(result["state"].get("verified"), f"Goose did not finish verification in {target}")
            preceding.extend(check_scenes(card, target / "workspace", ids))
        save(phase / "parsed-scenes.json", preceding)
        return 0
    target = phase / "workspace/world-card.json"
    result = run_phase(phase, gdk, CARD_BRIEF, [target.name],
                       [sys.executable, str(Path(__file__).resolve()), "check-card", str(target)])
    return 0 if result["state"].get("verified") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Storyworld check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
