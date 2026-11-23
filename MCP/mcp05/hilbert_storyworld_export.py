"""Export the authored Hilbert storyworld and start its numerical player."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys

from .context import sha
from .creative_bench import save
from .hilbert_storyworld import BASIS, START, classify, transition
from .storyworld_swmd import require, sweepweave_document


def scene_graph(scenes, scene_order):
    positions = {id_: i for i, id_ in enumerate(scene_order)}
    records = []
    for scene in scenes:
        index = positions[scene["id"]]
        options = []
        for slot, (label, reply, operation, clue, advance) in enumerate(scene["options"], 1):
            i = index + advance
            target = scene_order[i] if i < len(scene_order) else "enc_finale"
            options.append({"id": f"opt_{index + 1:02d}_{slot}", "text": label, "operation": operation, "clue": clue,
                            "reactions": [{"id": f"rxn_{index + 1:02d}_{slot}", "text": reply, "to": target,
                                           "score": [1, 0, 0, 0], "effects": {}}]})
        records.append({"id": scene["id"], "title": scene["title"], "text": scene["text"], "options": options})
    return records


def topology(candidate):
    scenes = candidate["scenes"]
    records = scene_graph(scenes, [scene["id"] for scene in scenes])
    exits = {row[0]: (row[1], row[2]) for row in candidate["endings"]}
    terminal_ids = list(exits)
    options = [{"id": f"opt_finale_{i}", "text": exits[end][0], "secret_gate": end if end in candidate["math"]["secrets"] else None,
                "reactions": [{"id": f"rxn_finale_{i}", "text": "", "to": end, "score": [1, 0, 0, 0], "effects": {}}]}
               for i, end in enumerate(terminal_ids, 1)]
    records.append({"id": "enc_finale", "title": candidate["plan"]["acts"][-1][0], "text": "", "options": options})
    records.extend({"id": end, "title": title, "text": text, "options": []}
                   for end, (title, text) in exits.items())
    return records


def swmd_blocks(records, secrets):
    rows = []
    for scene in records:
        rows.extend((f"ENC {scene['id']} turn=0..999", f"# TITLE: {scene['title']}"))
        rows.extend(f"# T: {line}" for line in scene["text"].splitlines())
        for option in scene["options"]:
            reaction = option["reactions"][0]
            label = " ".join(option["text"].splitlines()).replace("|", "¦")
            rows.append(f"ORX {option['id']}/{reaction['id']} -> {reaction['to']} | O:{label} | D:C(1)")
            rows.extend(f"# R {reaction['id']}: {line}" for line in reaction["text"].splitlines())
            if "operation" in option:
                rows.append(f"# DIRAC {option['id']}: {json.dumps([option['operation'], option['clue']])}")
            if option.get("secret_gate"):
                rows.append(f"# SECRET {option['id']}: {json.dumps(secrets[option['secret_gate']])}")
        rows.append("")
    return rows


def export(candidate, out, skill, target):
    out.mkdir(exist_ok=True)
    plan = candidate["plan"]
    records = topology(candidate)
    ifid = f"SW-INKY-HILBERT-{target}-20261124"
    model = candidate["math"]
    variables = [f"{part}_{bit}" for part in ("real", "imag") for bit in BASIS] + list(model["clues"])
    title = json.dumps(plan["title"], ensure_ascii=False)
    blocks = ["---", f"title: {title}", f"storyworld_id: {ifid}", "source_format: SWMD-0-MIN",
              "hilbert_player_data: storyworld.hilbert.json", f"state_variables: {json.dumps(variables)}",
              "endings:"]
    for end, secret, promise, clues in plan["endings"]:
        blocks.extend((f"  - id: {end}", f"    type: {'secret' if secret else 'public'}",
                       f"    condition: {json.dumps(promise, ensure_ascii=False)}",
                       f"    clue_chain: {json.dumps(clues, ensure_ascii=False)}"))
    blocks.extend(("---", "", "# SWMD-0-MIN", f"id: {ifid}", f"title: {plan['title']}", ""))
    blocks.extend(swmd_blocks(records, model["secrets"]))
    swmd = out / "world.swmd.min.md"
    swmd.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    repository = skill.parents[2]
    parser = repository / "mcp-storyworld-encounter/swmd_store.py"
    actual = runpy.run_path(str(parser))["parse_swmd_min"](swmd).encounter_order
    require(actual == [scene["id"] for scene in records], "native SWMD parser lost or repeated an encounter")

    card = {"title": plan["title"], "premise": plan["premise"], "stats": {},
            "cast": [{"id": row[0], "name": row[1], "voice": row[3]} for row in plan["cast"]]}
    graph = sweepweave_document(card, ifid, records)
    graph["hilbert_player_data"] = "storyworld.hilbert.json"
    graph["topology_only"] = True
    for scene in graph["encounters"]:
        if scene["id"] in model["secrets"]:
            scene["is_secret_ending"] = True
    graph_path = out / "sweepweave-topology.json"
    save(graph_path, graph)
    validator = repository / "codex-skills/storyworld-building/scripts/sweepweave_validator.py"
    errors = runpy.run_path(str(validator))["validate_storyworld"](str(graph_path))
    require(not errors, "\n".join(errors))
    pathing = repository / "hermes-skills/storyworld-conveyor/scripts/build_hilbert_pathing_packet.py"
    subprocess.run([sys.executable, "-B", str(pathing), "--world-json", str(graph_path),
                    "--out-dir", str(out / "topology-audit"), "--target-turns-min", "25", "--target-turns-max", "31"],
                   check=True, timeout=90, capture_output=True, text=True)
    save(out / "storyworld.hilbert.json", candidate)
    save(out / "manifest.json", {"input_target": target, "source_tools": {
        "small_storyworld_skill_sha256": sha(skill.read_bytes()), "swmd_parser_sha256": sha(parser.read_bytes()),
        "sweepweave_validator_sha256": sha(validator.read_bytes()), "pathing_audit_sha256": sha(pathing.read_bytes()),
    }, "native_encounter_count": len(actual), "artifacts": {
        file.relative_to(out).as_posix(): sha(file.read_bytes()) for file in sorted(out.rglob("*"))
        if file.is_file() and file.name != "manifest.json"}})


def choose(labels):
    for number, label in enumerate(labels, 1):
        print(f"{number}. {label}")
    while True:
        answer = input("> ").strip()
        if answer.isdecimal() and 1 <= int(answer) <= len(labels):
            return int(answer) - 1


def play(candidate):
    scenes, model = candidate["scenes"], candidate["math"]
    print(f"\n{candidate['plan']['title']}\n")
    i, state, clues = 0, START, frozenset()
    while i < len(scenes):
        scene = scenes[i]
        print(f"\n{scene['title']}\n\n{scene['text']}\n")
        selected = choose([option[0] for option in scene["options"]])
        option = scene["options"][selected]
        print(f"\n{option[1]}")
        advance, state, clues = transition(option, model, state, clues)
        i += advance
    secrets = classify(model, state, clues)
    eligible = [row for row in candidate["endings"] if row[0] in secrets or row[0] not in model["secrets"]]
    selected = choose([row[1] for row in eligible])
    print(f"\n{eligible[selected][1]}\n\n{eligible[selected][2]}")
