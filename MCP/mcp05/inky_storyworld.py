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
from .inky_goose_game import MODEL, ROOT, events, git, read_json, replay_writes
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
        require(isinstance(name, str) and name.strip() and len(name) <= 64 and name.isprintable(),
                f"invalid profile name: {name}")
        try:
            if isinstance(profile, dict) and set(profile) == {"weights", "effectsA", "effectsB"}:
                profile = [profile["weights"], profile["effectsA"], profile["effectsB"]]
            elif isinstance(profile, dict):
                profile = [[profile[arm]["score"], profile[arm]["delta"]] for arm in ("a", "b")]
            if len(profile) == 3 and all(isinstance(row, list) and len(row) == 3 for row in profile):
                weights, delta_a, delta_b = profile
                profile = [[[0, *weights], delta_a], [[0, *(-x for x in weights)], delta_b]]
            require(len(profile) == 2,
                    'use {"a":{"score":[b,t,n,c],"delta":[t,n,c]},"b":{"score":[b,t,n,c],"delta":[t,n,c]}} '
                    "for each named profile, with your actual numbers in each of the four arrays")
            fingerprints.add(profile_pair(profile))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{name}: {error}") from error
        accepted[name] = profile
    require(len(fingerprints) == len(profiles), "each profile needs a different pair of effect vectors")
    return accepted


def with_authoring_cards(card: dict, root: Path) -> dict:
    model = root / "00-math/workspace/reaction-model.json"
    outline = root / "00-outline/workspace/outline.json"
    if model.exists():
        card = {**card, "profiles": check_profiles(read_json(model))}
    if outline.exists():
        card = {**card, "outline": check_outline(read_json(outline), card)}
    return card


def check_outline(source, card):
    rows = source.get("beats") if isinstance(source, dict) else source
    roster = world_ids(card)
    if isinstance(rows, dict) and set(rows) == set(roster):
        rows = [[id_, *rows[id_]] for id_ in roster]
    require(isinstance(rows, list) and len(rows) == 20 and all(isinstance(row, list) and len(row) == 3
            and all(isinstance(cell, str) and cell.strip() for cell in row) for row in rows),
            "write 20 rows of [encounter ID, unique title, event beat]")
    require([row[0] for row in rows] == roster, "the outline must include the 20 supplied IDs in order")
    require(len({row[1].casefold() for row in rows}) == len({row[2].casefold() for row in rows}) == 20,
            "each title and event beat must be distinct")
    return {id_: (title, beat) for id_, title, beat in rows}


def duplicated(scene, earlier) -> str | None:
    for other in earlier:
        if any(" ".join(scene[key].casefold().split()) == " ".join(other[key].casefold().split()) for key in ("text", "art")):
            return other["id"]
    return None


def written_encounter_files(root: Path, id_):
    for phase in sorted(root.glob("??-??-encounters*")):
        if re.fullmatch(r"\d{2}-\d{2}-encounters(?:-retry-\d+)?", phase.name):
            path = phase / "workspace" / scene_path(id_)
            if path.is_file():
                yield path


def scene_path(scene_id):
    return f"{scene_id}.card.json"


def encounter_card(source) -> dict:
    if isinstance(source, dict):
        title = source["title"]
        text = source.get("text", source.get("sceneText"))
        art = source.get("art", source.get("artCue"))
        rows = source.get("options", [])
    else:
        require(isinstance(source, list) and len(source) == 4,
                "write [title, sceneText, artCue, options] as one JSON array")
        title, text, art, rows = source
    require(isinstance(rows, list), "options must be a list")
    options = []
    for row in rows:
        if isinstance(row, dict):
            options.append(row)
            continue
        require(isinstance(row, list) and len(row) == 6,
                "each option must be [label, profile, nextA, replyA, nextB, replyB]")
        label, model, next_a, reply_a, next_b, reply_b = row
        options.append({"label": label, "model": model, "reactions": [
            {"next": next_a, "reply": reply_a}, {"next": next_b, "reply": reply_b}]})
    return {"title": title, "text": text, "art": art, "options": options}


def groups(card):
    playable = 20 - len(card["endings"])
    return ([(start, min(start + 3, playable)) for start in range(1, playable + 1, 4)]
            + [(end, end) for end in range(playable + 1, 21)])


def collect_scenes(root: Path, card: dict, card_path: Path, gdk: Path, first_reply_from: Path | None = None) -> list[dict]:
    roster = world_ids(card)
    prior, sources = [], {}
    for start, end in groups(card):
        ids = roster[start - 1:end]
        prefix = f"{start:02}-{end:02}-encounters"
        attempts = [root / prefix, *sorted(root.glob(f"{prefix}-retry-*"))]
        ready = {}
        for id_ in ids:
            for phase in attempts:
                if (phase / "workspace" / scene_path(id_)).is_file():
                    candidate = check_scenes(card, phase / "workspace", [id_])[0]
                    if not duplicated(candidate, [*prior, *(entry[0] for entry in ready.values())]):
                        ready[id_] = (candidate, phase.name)
                        break
        stalled = 0
        while len(ready) < len(ids):
            require(stalled < 2, f"no new valid encounter files in two attempts for {prefix}")
            previous_count = len(ready)
            missing = [id_ for id_ in ids if id_ not in ready]
            phase = root / (prefix if not attempts[0].exists() else f"{prefix}-retry-{len(attempts):02}")
            previous = ready.get(roster[roster.index(missing[0]) - 1]) if missing[0] != roster[0] else None
            last = previous[0] if previous else (prior[-1] if prior else None)
            recent = f"{last['title']}: {last['text'][:135]}" if last else ""
            run_phase(phase, gdk, scene_brief(card, missing, recent), [scene_path(id_) for id_ in missing],
                      [sys.executable, str(Path(__file__).resolve()), "check-scenes", str(phase / "workspace"),
                       "--card", str(card_path), "--ids", *missing], max_tokens=640 if start > 20-len(card["endings"]) else 1408,
                      iterations=len(missing),
                      validate_create=[sys.executable, str(Path(__file__).resolve()), "check-proposed", "--card", str(card_path)],
                      allow_basename=True, first_reply_from=first_reply_from)
            first_reply_from = None
            if phase != attempts[0]:
                attempts.append(phase)
            for id_ in missing:
                if (phase / "workspace" / scene_path(id_)).is_file():
                    candidate = check_scenes(card, phase / "workspace", [id_])[0]
                    if not duplicated(candidate, [*prior, *(entry[0] for entry in ready.values())]):
                        ready[id_] = (candidate, phase.name)
            stalled = stalled + 1 if len(ready) == previous_count else 0
        for id_ in ids:
            scene, sources[id_] = ready[id_]
            prior.append(scene)
        save(root / "scene-sources.json", sources)
    return prior


def check_scenes(card: dict, workspace: Path, ids: list[str]) -> list[dict]:
    scenes, errors = [], []
    for scene_id in ids:
        try:
            block = render_card(encounter_card(read_json(workspace / scene_path(scene_id))), card, scene_id)
            scenes.append(parse_scene(block, card, scene_id))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{scene_id}: {error}")
    require(not errors, "\n".join(errors))
    return scenes


def scene_brief(card, ids, recent):
    names = ", ".join(scene_path(id_) for id_ in ids)
    terminal_ids = [e["id"] for e in card["endings"]]
    if ids[0] in terminal_ids:
        ending = next(e for e in card["endings"] if e["id"] == ids[0])
        return (f"Call create_file now with exact path {names}. Write only this ending for your story {card['title']}. "
                f"The authored ending condition is {ending['condition']}. "
                "Put four values in the JSON content array: [title, sceneText, artCue, []]. "
                "Write two vivid sentences that resolve what happened. Include a short drawable visual motif. "
                f"Use the exact authorized filename {names}, and close the tool call.")
    roster = world_ids(card)
    playable = 20 - len(terminal_ids)
    act_indices = {(roster.index(id_)) // 4 for id_ in ids}
    act = " / ".join(card["acts"][i] for i in sorted(act_indices))
    cast = " / ".join(f"{c['name']}: {c['voice'][:110]}" for c in card["cast"])
    brief = (f"Use create_file, one JSON encounter card per tool call, in order: {names}.\n"
             f"World: {card['title']}. {card['premise'][:240]}\nCast: {cast}\nAct goal: {act}\n"
             f"Prior scene: {recent or 'Opening'}\n"
             "Write two concise sentences of scene prose, a distinctive drawable art cue, and consequential responses. "
             "Your card compiles to SWMD. "
             "An empty content field in earlier tool history means those bytes were archived. Always write fresh JSON. "
             "Advance the story at each new encounter. Never copy prior scene prose or illustration cues. "
             "Each write receives an immediate structural and math check.\n")
    if outline := card.get("outline"):
        brief += "Your planned events to develop now: " + " / ".join(f"{id_}: {outline[id_][0]} — {outline[id_][1]}" for id_ in ids) + "\n"
    routes = []
    for id_ in ids:
        i = roster.index(id_)
        if i < playable - 1:
            routes.append(f"{id_} -> {id_} or {roster[i+1]} through {roster[playable-1]}; include {roster[i+1]}")
        else:
            routes.append(f"{id_} -> {', '.join(terminal_ids)}")
    if profiles := card.get("profiles"):
        return brief + """Write minified JSON with no keys. The entire card is [title, sceneText, artCue, options].
Each option is [label, profile, nextA, replyA, nextB, replyB]. Use three distinct options and three distinct profiles.
Keep each reply to one short line. Use real encounter IDs as both targets. Include the required route for each scene.
The finale uses every listed ending target. Finish one card per tool call.
Your named profiles: """ + json.dumps(list(profiles), ensure_ascii=False) + "\nPermitted routes: " + " / ".join(routes)
    return brief + """Use a JSON object with title, text, art, and options. The shape for a single option is:
{"label":"YOUR CHOICE","reactions":[{"next":"ENCOUNTER_ID","reply":"YOUR RESPONSE","score":[0,0,0,0],"delta":[0,0,0]}, {"next":"ENCOUNTER_ID","reply":"A DIFFERENT RESPONSE","score":[0,0,0,0],"delta":[0,0,0]}]}
Write exactly 3 options. Replace every placeholder, including all-zero vectors. The `next` field is an
encounter ID from the route list, never a speaker. `score` is four integer hundredths [bias,trust,nerve,care],
and `delta` is three integer hundredths [trust,nerve,care]. Score abs total <=100. Use opposite nonzero
slopes and small biases in each reaction pair. The sum of the absolute differences between the three slopes
must exceed the absolute bias difference. Change 2–3 stats per reaction within -24 to 24. Use distinct tradeoffs.
The first listed target for each regular scene must occur at least once; the finale must use all listed targets.
Use exactly these permitted targets: """ + " / ".join(routes)


def run_phase(root: Path, gdk: Path, brief: str, paths: list[str], verifier: list[str], *,
              max_tokens: int = 2304, iterations: int = 7, context: int = 8192,
              validate_create: list[str] | None = None, first_reply_from: Path | None = None,
              allow_basename: bool = False) -> dict:
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
                    **({"allow_create_basename": True} if allow_basename else {}),
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
    if first_reply_from:
        provider_events = events(first_reply_from / "inky-wire.jsonl")
        first = events(first_reply_from / "session.preflight.jsonl")[0]
        require(provider_events[0]["event"] == "request" and provider_events[1]["event"] == "response"
                and provider_events[1]["status"] == 200, "replay needs a successful initial exchange")
        key = sha(first["preflight"]["formatted_prompt_sha256"].encode() + b"\0" + provider_events[0]["body"].encode())
        prefix = root / "replayed-prefix.json"
        save(prefix, [{"request_key": key, "response": json.loads(provider_events[1]["body"]),
                       "provider_seconds": first["provider_seconds"]}])
        env["MCP05_EPISODE_PREFIX_PATH"] = str(prefix)
        invocation["replayed_first_reply"] = {"trace_sha256": sha((first_reply_from / "inky-wire.jsonl").read_bytes()),
                                             "request_key": key}
        save(root / "invocation.json", invocation)
    print(f"Inkling phase {root.name}", flush=True)
    with (root / "gdk.json").open("wb") as stdout, (root / "gdk.stderr.log").open("wb") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    return read_json(root / "gdk.json")


def export(root: Path, card_file: Path, gptstoryworld: Path) -> dict:
    card = with_authoring_cards(check_card(card_file), root)
    roster = world_ids(card)
    ifid = "SW-INKY-CAT-HAT-20261123"
    (root / "export").mkdir(exist_ok=True)
    out = root / "export"
    receipts = {}
    card_phase = card_file.parents[1]
    receipts[card_phase.name] = replay_writes(card_phase, [card_file.name], require_verified=False)
    if card.get("profiles"):
        receipts["00-math"] = replay_writes(root / "00-math", ["reaction-model.json"])
    if card.get("outline"):
        receipts["00-outline"] = replay_writes(root / "00-outline", ["outline.json"])
    catalog = read_json(root / "scene-sources.json")
    route_patch = root / "balance-revisions/workspace/route-revisions.json"
    targets = checked_targets(root, card, read_json(route_patch)) if route_patch.exists() else {}
    blocks, scenes = compile_routes(root, card, targets)
    if route_patch.exists():
        receipts["balance-revisions"] = replay_writes(route_patch.parents[1], [route_patch.name], require_verified=False)
    for id_ in roster:
        phase = root / catalog[id_]
        if phase.name not in receipts:
            paths = [name.name for name in sorted((phase / "workspace").glob("*.card.json"))]
            receipts[phase.name] = replay_writes(phase, paths, require_verified=False)
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
    require(balance["timeouts"] == 0, "a local retry route stalled in seeded rehearsal")
    missing = sorted(set(roster) - set(balance["encounter_visits"]))
    require(not missing, f"seeded rehearsal did not reach {', '.join(missing)}")
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


def compile_routes(root, card, targets):
    catalog = read_json(root / "scene-sources.json")
    ids = world_ids(card)
    blocks = [render_card(encounter_card(read_json(root / catalog[id_] / "workspace" / scene_path(id_))),
                          card, id_, targets) for id_ in ids]
    return blocks, [parse_scene(block, card, id_) for block, id_ in zip(blocks, ids, strict=True)]


def checked_targets(root, card, source):
    require(isinstance(source, dict), "write an object mapping reaction IDs to targets")
    targets = source.get("targets", source)
    _, old = compile_routes(root, card, {})
    ids = {r["id"] for scene in old for option in scene["options"] for r in option["reactions"]}
    require(isinstance(targets, dict) and targets and set(targets) <= ids, "supply known reaction IDs in targets")
    _, scenes = compile_routes(root, card, targets)
    require(reachable(scenes) == set(world_ids(card)), "patch disconnects one or more encounters")
    result = rehearse(scenes, card["stats"])
    missing = sorted(set(world_ids(card)) - set(result["encounter_visits"]))
    require(not missing and not result["timeouts"],
            f"route rehearsal failed: missing={missing}; stalls={result['timeout_scenes']}; endings={result['endings']}")
    return targets


def balance_brief(root, card):
    _, scenes = compile_routes(root, card, {})
    result = rehearse(scenes, card["stats"])
    rows = []
    for scene in scenes:
        if not scene["options"]:
            continue
        retries = [r["id"] for option in scene["options"] for r in option["reactions"] if r["to"] == scene["id"]]
        if retries:
            rows.append(f"{scene['id']} {scene['title']}: {', '.join(retries)}")
    finale = scenes[20 - len(card['endings']) - 1]
    last_choices = [f"{option['text']}: " + ", ".join(f"{r['id']} -> {r['to']} ({result['reaction_hits'].get(r['id'], 0)} visits)"
                    for r in option["reactions"]) for option in finale["options"]]
    return ("Author route-revisions.json via create_file with path and content. Its JSON contains a targets object "
            "mapping existing reaction IDs to their revised encounter IDs. Keep your existing story, art, options, "
            "scores, and effects. You are adjusting transitions only. Remove local repetition that can trap Pip, "
            "using forward encounter IDs; keep every scene reachable. Make every ending reachable, especially the missing "
            "fallback, while honoring your endings and three finale choices. Choose exact destination IDs yourself.\n"
            f"Seed 523, 5,000 trial results: timeout scenes {result['timeout_scenes']}, ending counts {result['endings']}.\n"
            f"Your retry reaction IDs: {' / '.join(rows)}.\n"
            f"Finale diagnostics: {' / '.join(last_choices)}. Endings: {', '.join(e['id'] for e in card['endings'])}.\n"
            "Submit the revised targets together. Goose will rehearse your patch before accepting it.")


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
                        "outline", "check-outline", "check-proposed-outline", "balance", "check-balance", "check-proposed-balance",
                        "scenes", "check-scenes", "check-proposed", "export", "play"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--gptstoryworld", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--first-reply-from", type=Path)
    parser.add_argument("--world", type=Path)
    args = parser.parse_args()
    if args.command in ("check-balance", "check-proposed-balance"):
        root = args.world.resolve()
        card = with_authoring_cards(check_card(root / "00-world/workspace/world-card.json"), root)
        source = read_json(args.path) if args.command == "check-balance" else json.loads(sys.stdin.buffer.read())
        targets = checked_targets(root, card, source)
        print(f"Validated {len(targets)} authored route revisions over 5,000 complete rehearsals")
        return 0
    if args.command in ("check-outline", "check-proposed-outline"):
        source = read_json(args.path) if args.command == "check-outline" else json.loads(sys.stdin.buffer.read())
        check_outline(source, check_card(args.card))
        print("Validated 20 distinct outline beats")
        return 0
    if args.command in ("check-math", "check-proposed-math"):
        profiles = check_profiles(read_json(args.path) if args.command == "check-math" else json.loads(sys.stdin.buffer.read()))
        print(f"Validated {len(profiles)} state-dependent math profiles.")
        return 0
    if args.command == "check-proposed":
        filename = args.path.name
        id_ = filename.removesuffix(".card.json")
        card = with_authoring_cards(check_card(args.card), args.card.resolve().parents[2])
        roster = world_ids(card)
        require(filename == scene_path(id_) and id_ in roster, "write an authorized encounter file")
        raw = sys.stdin.buffer.read()
        require(not raw.startswith(b"[Prior proposal archived;"),
                "that text is a history placeholder. Author fresh JSON for this encounter's actual content")
        source = encounter_card(json.loads(raw))
        scene = parse_scene(render_card(source, card, id_), card, id_)
        earlier = roster[:roster.index(id_)]
        for prior_id in earlier:
            for file in written_encounter_files(args.card.resolve().parents[2], prior_id):
                prior = {"id": prior_id, **encounter_card(read_json(file))}
                require(not duplicated(scene, [prior]), f"scene text or art repeats {prior_id}; author the planned new event and a different illustration")
        print(f"{id_} accepted. Continuity card: {scene['title'][:72]} / {scene['text'][:170]}")
        return 0
    if args.command == "check-card":
        card = check_card(args.path)
        print(f"Validated five-act plan for {card['title']} with {len(card['endings'])} ending cards.")
        return 0
    if args.command == "check-scenes":
        items = check_scenes(with_authoring_cards(check_card(args.card), args.card.resolve().parents[2]), args.path, args.ids)
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
    if args.command == "balance":
        card = with_authoring_cards(check_card(phase / "00-world/workspace/world-card.json"), phase)
        script = str(Path(__file__).resolve())
        root = phase / "balance-revisions"
        verifier = [sys.executable, script, "check-proposed-balance", "--world", str(phase)]
        result = run_phase(root, gdk, balance_brief(phase, card), ["route-revisions.json"],
                           [sys.executable, script, "check-balance", str(root / "workspace/route-revisions.json"),
                            "--world", str(phase)], validate_create=verifier,
                           max_tokens=1792, iterations=5, allow_basename=True)
        return 0 if (root / "workspace/route-revisions.json").is_file() else 1
    if args.command == "outline":
        card_file = phase / "00-world/workspace/world-card.json"
        card = check_card(card_file)
        target = phase / "00-outline"
        script = str(Path(__file__).resolve())
        outline_brief = (f"Use create_file: path outline.json, content a single JSON array of 20 brief rows.\n"
                         f"Your story is {card['title']}. {card['premise']}.\n"
                         f"Cast: {', '.join(c['name'] for c in card['cast'])}. Acts: {' / '.join(card['acts'])}.\n"
                         f"Endings: {' / '.join(e['condition'] for e in card['endings'])}.\n"
                         f"IDs in this order: {', '.join(world_ids(card))}.\n"
                         "Each row is [encounter ID, short unique title, 8-to-12-word event]. Use all IDs once. "
                         "Move the cat and the hat through genuinely different places, discoveries, decisions, and costs. "
                         "Carry consequences into the ending beats. Keep the full action compact. Then call run_verifier.")
        result = run_phase(target, gdk, outline_brief, ["outline.json"],
                           [sys.executable, script, "check-outline", str(target / "workspace/outline.json"), "--card", str(card_file)],
                           validate_create=[sys.executable, script, "check-proposed-outline", "--card", str(card_file)],
                           max_tokens=1792, allow_basename=True)
        return 0 if result["state"].get("verified") else 1
    if args.command == "math":
        card = check_card(phase / "00-world/workspace/world-card.json")
        target = phase / "00-math"
        script = str(Path(__file__).resolve())
        math_brief = f"""Create reaction-model.json for your fable {card['title']}. The tool arguments are path and content.
Content is one JSON object with a profiles object containing six short named entries.
Each name maps to three arrays of three signed integers: [weights, effectsA, effectsB].
The columns are trust, nerve, care. Scores are the dot product and its negative, in hundredths.
Choose nonzero weights with total absolute value <= 100. The effects arrays change two or three stats each,
with signed changes under 24 hundredths. Include a tradeoff within each pair. Use six different pairs of effects.
Author your own mechanics. Close the tool call. Then call run_verifier with empty arguments."""
        result = run_phase(target, gdk, math_brief, ["reaction-model.json"],
                           [sys.executable, script, "check-math", str(target / "workspace/reaction-model.json")],
                           validate_create=[sys.executable, script, "check-proposed-math"], max_tokens=1024,
                           first_reply_from=args.first_reply_from, allow_basename=True)
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
        card = with_authoring_cards(check_card(card_path), phase)
        save(phase / "parsed-scenes.json", collect_scenes(phase, card, card_path, gdk, args.first_reply_from))
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
