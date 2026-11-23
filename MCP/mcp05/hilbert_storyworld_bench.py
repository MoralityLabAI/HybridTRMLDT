"""Run matched, staged Inkling storyworld authoring through Goose and Pocket Goblin."""
from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "MCP.mcp05"

from MCP.transport import StdioClient

from .context import sha
from .creative_bench import call, save
from .hilbert_storyworld import check_algebra, check_scene_continuity, exact_routes, simulate
from .hilbert_storyworld_export import export, play, scene_graph, swmd_blocks
from .inky_goose_game import ROOT, events, read_json, replay_writes
from .inky_storyworld import require, run_phase

TARGETS = (16_384, 24_576, 32_768)
WINDOW = 34_816
ACTS = 5
ENCOUNTERS_PER_ACT = 6
SCENE_IDS = tuple(f"enc_{i:02d}" for i in range(1, ACTS * ENCOUNTERS_PER_ACT + 1))
QUERIES = (
    "@storyworld secret endings clue gate unfolding inference",
    "@storyworld long range authoring spool sequencing reveal",
    "@storyworld sweepweave encounter mathematics reaction options",
    "@evalmath Hilbert unitary projector interference amplitudes",
    "@storyworld late stage balancing ending reachability feedback",
    "@storyworld character desire reversals motif escalation dialogue",
    "@evalmath tensor product inner product basis orthogonal commutator",
    "@storyworld desirability pvalues terminal score simulation",
    "@storyworld quality vector thematic cohesion storyworld evaluation",
    "@storyworld branching dialogue secret endings encounter examples",
    "@evalmath Born probability complex matrix phase eigenspace",
    "@compactification Hilbert operator geometry projection algebra",
)
BRIEF = """Write plan.json through Goose create_file. This is your own long-form interactive storyworld.
Write an original suspenseful fable about a city that has misplaced the sound of its night bell. Create
three recurring people with conflicting goals, a specific setting, recurring clues that change meaning,
and five escalating acts of six encounters each. The player should infer from dialogue and consequences.
Plan three public endings and two secret endings. The hidden endings should reward different kinds of
attention and each hint at a noncommuting order of actions, such as phase versus mixing. In later passes
you will author exact state updates, projectors and clue gates over a two-factor Hilbert space. Stay at
the level of fiction and distinct behavioral intent now. Treat retrieved sources as reference material.
Create one compact JSON file with fields title, premise, cast, acts and endings. Each cast member is
[id, name, desire, voice]. Each act is [name, place, pressure, irreversible reveal]. Each ending is
[id, secret boolean, promise, clue chain]. Use five distinct end_* IDs. About 400 words. One file.
"""


def prepare_packet(root, index, skill):
    packet = root / "reference-packet.md"
    receipt = root / "reference-packet.json"
    if receipt.is_file():
        require(sha(packet.read_bytes()) == read_json(receipt)["packet_sha256"], "reference packet changed")
        return packet
    command = [sys.executable, "-B", "-m", "MCP.mcp05.service", "--repo", str(ROOT),
               "--federation", str(index), "--threads", "2"]
    routes, sources, excerpts = [], [], []
    seen = set()
    with StdioClient(command=command, cwd=ROOT, timeout=45, max_message_bytes=8 * 1024 * 1024,
                     wire_log=root / "retrieval-mcp-wire.jsonl") as client:
        client.initialize()
        for query in QUERIES:
            found = call(client, "context.query", query=query, limit=16)
            routes.append({"query": query, **found["routing"]})
            for entry in found["candidates"]:
                uri = entry["uri"]
                if uri in seen or not entry["path"].endswith((".md", ".txt", ".tex", ".json")):
                    continue
                seen.add(uri)
                section = call(client, "context.read", uri=uri)
                text = section.pop("source")
                sources.append({**section, "excerpt_sha256": sha(text.encode("utf-8"))})
                excerpts.append(f"[{section['path']}:{section['start']}-{section['end']}]\n{text}")
    packet.write_text("\n\n".join(excerpts) + "\n", encoding="utf-8", newline="\n")
    require(packet.stat().st_size >= 135_000, "Pocket Goblin returned too little reference material for 32K")
    save(receipt, {"skill_sha256": sha(skill.read_bytes()), "packet_sha256": sha(packet.read_bytes()),
                   "routes": routes, "sources": sources, "wire_sha256": sha((root / "retrieval-mcp-wire.jsonl").read_bytes())})
    return packet


def check_plan(plan):
    require(set(plan) == {"title", "premise", "cast", "acts", "endings"}, "missing plan fields")
    require(all(isinstance(plan[k], str) and plan[k].strip() for k in ("title", "premise")), "empty premise or title")
    premise = f"{plan['title']} {plan['premise']}".casefold()
    require("bell" in premise and any(cue in premise for cue in ("silent", "silenc", "missing", "misplac", "lost",
                                                                   "stolen", "vanish", "no longer", "disappear",
                                                                   "soundless", "swallowed", "muffled", "cease", "can't hear")),
            "the premise needs the missing sound of the night bell")
    for key, count in (("cast", 3), ("acts", ACTS), ("endings", 5)):
        items = plan[key]
        require(len(items) == count and all(isinstance(row, list) and len(row) == 4 for row in items),
                f"{key} needs {count} four-field records")
        require(len({row[0] for row in items}) == count, f"{key} IDs must be unique")
        for row in items:
            for column, value in enumerate(row):
                if key == "endings" and column == 1:
                    require(type(value) is bool, "ending visibility must be boolean")
                else:
                    require(isinstance(value, str) and value.strip(), f"empty {key} field")
    require(all(row[0].startswith("end_") for row in plan["endings"]), "expected end_* IDs")
    require(sum(row[1] for row in plan["endings"]) == 2, "plan needs two secret endings")
    return plan


def next_attempt(arm, stage):
    return arm / f"{stage}-attempt-{1 + len(list(arm.glob(f'{stage}-attempt-*'))):02d}"


def collect_plan(phase, target):
    report = phase / "gdk.json"
    trace = phase / "session.preflight.jsonl"
    if not report.is_file() or not report.stat().st_size or not trace.is_file():
        return None
    rows = trace.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        return None
    measured = json.loads(rows[0])
    if measured.get("provider_preflight_parity") is not True or measured.get("provider_status") != 200:
        return None
    require(abs(measured["provider_prompt_tokens"] - target) <= 8, "planning arm is outside its input condition")
    authored = phase / "workspace/plan.json"
    result = {"target": target, "phase": phase.name, "measurement": measured}
    if authored.is_file():
        result["authorship"] = replay_writes(phase, [authored.name], require_verified=False)
        try:
            result["plan"] = check_plan(read_json(authored))
        except (KeyError, TypeError, ValueError) as error:
            result["status"], result["diagnostic"] = "invalid-plan", str(error)
    else:
        result["status"] = "no-authored-plan"
    return result


def plan_arm(root, gdk, packet, target):
    arm = root / f"{target // 1024:02d}k"
    arm.mkdir(exist_ok=True)
    receipt = arm / "plan.json"
    if receipt.is_file():
        return read_json(receipt)
    for previous in sorted(arm.glob("plan-attempt-*")):
        result = collect_plan(previous, target)
        if result is not None:
            save(receipt, result)
            return result
    phase = next_attempt(arm, "plan")
    authored = phase / "workspace/plan.json"
    verifier = [sys.executable, "-B", str(Path(__file__).resolve()), "check-plan", str(authored)]
    run_phase(phase, gdk, BRIEF, [authored.name], verifier, max_tokens=1024, iterations=1,
              context=WINDOW, allow_basename=True, reference_prompt=(packet, target))
    result = collect_plan(phase, target)
    require(result is not None, "planning arm lacks a successful provider token count")
    save(receipt, result)
    return result


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def measured_reference(arm, stage):
    for trace in arm.glob(f"{stage}-attempt-*/session.preflight.jsonl"):
        for line in trace.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("target_input_tokens") and row.get("provider_preflight_parity"):
                return True
    return False


def author(arm, gdk, stage, brief, validate, *, max_tokens=1024, reference=None):
    receipt = arm / f"{stage}.json"
    if receipt.is_file():
        return read_json(receipt)["value"]
    feedback = ""
    for _ in range(3):
        phase = next_attempt(arm, stage)
        path = phase / "workspace/result.json"
        check = [sys.executable, "-m", "json.tool", str(path)]
        exposure = reference if reference and not measured_reference(arm, stage) else None
        run_phase(phase, gdk, brief + feedback, [path.name], check, max_tokens=max_tokens,
                  iterations=1, context=WINDOW if exposure else 8192, allow_basename=True, reference_prompt=exposure)
        try:
            require(path.is_file(), "Goose did not receive a complete create_file action")
            result = read_json(path)
            validate(result)
            proof = replay_writes(phase, [path.name], require_verified=False)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            save(phase / "validation-error.json", {"error": str(error)})
            feedback = f"\nEarlier result rejected: {error}. Create one small, valid result.json in one action."
            continue
        save(receipt, {"phase": phase.name, "value": result, "authorship": proof})
        return result
    raise ValueError(f"{arm.name}/{stage}: {feedback.strip()}")


def text_rows(rows, keys, width):
    require(isinstance(rows, list) and [row[0] for row in rows] == list(keys), "rows need the provided IDs in order")
    require(all(isinstance(row, list) and len(row) == width for row in rows), f"every row needs {width} fields")
    require(all(isinstance(field, str) and field.strip() for row in rows for field in row), "each field needs text")
    return rows


def continuity_card(arm, parser, parser_sha, scenes, target_scene):
    recent = scenes[-2:]
    expected = [scene["id"] for scene in recent]
    blocks = ["# SWMD-0-MIN", "", *swmd_blocks(scene_graph(recent, SCENE_IDS), {})]
    blocks.extend((f"ENC {target_scene} turn=0..999", ""))
    source = arm / f"continuity-{target_scene}.swmd.min.md"
    source.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    card = parser(path=source, encounter_id=target_scene, neighbor_hops=2, context_budget_tokens=8192,
                  reserve_output_tokens=1800, planning_card_tokens=900, include_poetics=False)
    require({neighbor["encounter_id"] for neighbor in card["neighbors"]} == set(expected), "missing continuity neighbor")
    save(source.with_suffix(".json"), {"parser_sha256": parser_sha, "source_sha256": sha(source.read_bytes()), "card": card})
    return [neighbor["block"] for neighbor in reversed(card["neighbors"])]


def develop(arm, gdk, packet, target, skill):
    planning = read_json(arm / "plan.json")
    try:
        require("plan" in planning, planning.get("diagnostic", "no complete long-context plan"))
        plan = check_plan(planning["plan"])
    except (KeyError, TypeError, ValueError) as error:
        save(arm / "plan-review.json", {"initial_phase": planning["phase"], "accepted": False, "reason": str(error)})
        plan = author(arm, gdk, "plan-repair", BRIEF.replace("plan.json", "result.json") +
                      "\nUse short actor, act and ending arrays. Keep the plan focused on the missing night bell.",
                      check_plan)
    else:
        save(arm / "plan-review.json", {"initial_phase": planning["phase"], "accepted": True})
    cast_ids = [row[0] for row in plan["cast"]]
    endings = [row[0] for row in plan["endings"]]
    public = [row[0] for row in plan["endings"] if not row[1]]
    voices = author(arm, gdk, "characterize", "Create result.json with one array row per cast member, in order. "
                    "Each is [id, private contradiction, original line of dialogue, what would break their resolve]. "
                    "Make their speech rhythms and bargains recognizably different.\n" + encode(plan),
                    lambda rows: text_rows(rows, cast_ids, 4), max_tokens=640)
    model_brief = """Create result.json as the first algebra draft for your own night-bell story. Choose meanings
for two binary state variables and map their joint states as basis {00:...,01:...,10:...,11:...}. State starts
at |00>. Provide clues {c0:...,c1:...,c2:...,c3:...,c4:...,c5:...}; descriptions are evidence found in fiction.
In operators, define 6 to 8 short names mapping to distinct lists of 1–4 primitive gates, executed in list
order. Allowed I, H0, H1, X0, X1, Z0, Z1, S0, S1, CX01, CX10. Qubit 0 is left. Include mixing and phase.
The secrets object keys are your two hidden end IDs. Each gate is {requires:[two to four clue IDs],
projectors:[[P,low,high],[P,low,high]]}. P can be 00, 01, 10, 11, or a Pauli sign like X0+, Y1-, Z0+.
Intervals lie inside [0,1]. At least one X or Y test per ending. Choose different clue sets and physically
possible joint conditions. The player can see the ending option only if all intervals and clues pass.
Author JSON fields basis, clues, operators, secrets. Your story:\n"""
    model = author(arm, gdk, "math-draft", model_brief + encode(plan),
                   lambda content: check_algebra(content, plan), reference=(packet, target))
    scenes, act_cards = [], []
    store = skill.parents[2] / "mcp-storyworld-encounter/swmd_store.py"
    parse_card = runpy.run_path(str(store))["iteration_packet"]
    store_sha = sha(store.read_bytes())
    for act_i, act in enumerate(plan["acts"]):
        for segment in range(2):
            index = act_i * ENCOUNTERS_PER_ACT + segment * 3
            ids = SCENE_IDS[index:index + 3]
            brief = """Create result.json, a JSON array of three distinct scenes with the given IDs in order.
Each {id,title,text,options} includes original dialogue in two to four compact paragraphs and three distinct
options. Option array [choice label, voiced response, named operator, clue ID or null, advance]. Advance
1 goes to the next scene, 2 bypasses one. Use advance 2 only for an explicitly listed shortcut scene.
The listed shortcut scene needs both advance 1 and advance 2. Use route consequences
that make sense even when the preceding scene was bypassed. Use clue-bearing choices sparingly and make
their discovery legible. Put the tone of the nearby characters into the responses. One tool action.
"""
            brief += encode({"ids": ids, "shortcut_scenes": [id_ for id_ in ids if int(id_[4:]) % 6 == 3],
                             "title": plan["title"], "premise": plan["premise"], "act": act,
                             "cast": voices, "prior_act": act_cards[-1:],
                             "previous_scenes_swmd": continuity_card(arm, parse_card, store_sha, scenes, ids[0]),
                             "clues": model["clues"], "operators": model["operators"]})
            batch = author(arm, gdk, f"encounter-{index + 1:02d}", brief,
                           lambda data: check_scene_continuity(data, ids, model, scenes), max_tokens=1800)
            scenes.extend(batch)
        summary_brief = """Create result.json, one compact array [act ID, an established fact, a character cost,
an unresolved clue, a different next-act constraint]. Use only this act's authored encounters and the intended
next act. The note will guide the next encounter block. One file.\n"""
        act_id = f"act_{act_i + 1}"
        recent = scenes[-ENCOUNTERS_PER_ACT:]
        card = author(arm, gdk, f"act-complete-{act_i + 1}", summary_brief + encode({"act_id": act_id,
                      "scenes": recent, "next": plan["acts"][act_i + 1:act_i + 2]}),
                      lambda row: text_rows([row], [act_id], 5), max_tokens=400)
        act_cards.append(card)
    revised_cast = author(arm, gdk, "recharacterize", """Create result.json with the three actor IDs in order.
Each row is [id, what this person now believes, a late-scene line, what loss is visible in the endings].
Base the changes on the actual act receipts. Give each a distinct late voice.\n""" + encode({"before": voices,
                           "acts": act_cards}), lambda rows: text_rows(rows, cast_ids, 4), max_tokens=640)
    terminals = author(arm, gdk, "endings", """Create result.json as five rows [ending id, title, final prose].
Use the given IDs in order. Write two short paragraphs per ending. The two secret endings should pay off
different clue chains and expose different truths about the night bell. Write scenes with consequences,
in the cast's late voices. One file.\n""" + encode({"plan": plan["endings"], "cast": revised_cast,
                       "acts": act_cards, "math": model}), lambda rows: text_rows(rows, endings, 3), max_tokens=1400)
    draft = simulate(scenes, model, public)
    draft_exact = exact_routes(scenes, model)
    save(arm / "draft-rehearsal.json", {"simulation": draft, "exact": draft_exact})
    tune = """Create result.json as just the revised secrets object, keyed by the two hidden endings. Keep choice
effects and the named gate operators fixed. Improve the requirements and intervals based on actual rehearsal.
A secret should need two to four distinct clues plus two to three meaningful projector intervals, including
an X or Y test. Aim for both to be attainable without becoming default options. Preserve their different
fictional payoffs. At least one meaningful adjacent operation swap should close each gate. One file.\n"""
    evidence = {key: value for key, value in draft.items() if key not in ("witnesses", "encounter_visits")}
    revised = author(arm, gdk, "late-stage-holistic", tune + encode({"model": model, "rehearsal": evidence, "exact": draft_exact,
                     "acts": act_cards, "endings": plan["endings"]}),
                     lambda secrets: check_algebra({**model, "secrets": secrets}, plan), reference=(packet, target))
    final_model = {**model, "secrets": revised}
    revised_exact = exact_routes(scenes, final_model)
    save(arm / "holistic-rehearsal.json", revised_exact)
    for attempt in range(1, 4):
        incomplete = [end for end, report in revised_exact["secret_gates"].items()
                      if not report["order_witness"] or not 0 < report["eligible_probability"] < 1]
        if not incomplete:
            break
        revised = author(arm, gdk, f"gate-repair-{attempt}", tune + encode({"check_failed": incomplete,
                         "model": final_model, "exact": revised_exact, "draft_exact": draft_exact,
                         "endings": plan["endings"]}), lambda secrets: check_algebra({**model, "secrets": secrets}, plan),
                         max_tokens=720)
        final_model = {**model, "secrets": revised}
        revised_exact = exact_routes(scenes, final_model)
        save(arm / f"gate-repair-{attempt}-rehearsal.json", revised_exact)
    balance = simulate(scenes, final_model, public)
    result = {"plan": plan, "cast": revised_cast, "scenes": scenes, "endings": terminals,
              "math": final_model, "algebra": check_algebra(final_model, plan), "balance": balance, "exact": revised_exact}
    save(arm / "candidate.json", result)
    return result


def describe(arm, candidate, target):
    traces = []
    for path in sorted(arm.glob("*/session.preflight.jsonl")):
        wire = path.with_name("inky-wire.jsonl")
        responses = iter(json.loads(frame["body"]) for frame in events(wire)
                         if frame["event"] == "response" and frame["status"] == 200) if wire.is_file() else iter(())
        files = sorted((path.parent / "workspace").glob("*.json"))
        for turn, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            response = next(responses, {}) if row.get("provider_status") == 200 else {}
            traces.append({"stage": path.parent.name, "turn": turn, **{
                k: row[k] for k in ("provider_status", "provider_seconds", "provider_prompt_tokens",
                                    "provider_completion_tokens", "provider_preflight_parity", "target_input_tokens")
                if k in row}, "timings": response.get("timings"),
                "finish_reason": next(iter(response.get("choices", [])), {}).get("finish_reason"),
                "authored_file": bool(files), "validator_error": (path.parent / "validation-error.json").is_file()})
    eligible = candidate["exact"]["secret_gates"]
    draft = read_json(arm / "draft-rehearsal.json")["exact"]["secret_gates"]
    return {
        "input_target": target, "title": candidate["plan"]["title"], "encounters": len(candidate["scenes"]),
        "initial_plan_review": read_json(arm / "plan-review.json"),
        "scene_words": sum(len(scene["text"].split()) for scene in candidate["scenes"]),
        "unique_scene_texts": len({scene["text"].strip() for scene in candidate["scenes"]}),
        "unique_choice_labels": len({option[0].lower() for scene in candidate["scenes"] for option in scene["options"]}),
        "options": sum(len(scene["options"]) for scene in candidate["scenes"]),
        "route_lengths": candidate["balance"]["route_lengths"],
        "secrets": {end: {"draft_probability": draft[end]["eligible_probability"],
                          "final_probability": row["eligible_probability"],
                          "order_sensitive": bool(row["order_witness"]), "clue_probability": row["clue_probability"]}
                    for end, row in eligible.items()},
        "rehearsed_endings": candidate["balance"]["endings"],
        "traces": traces, "manifest_sha256": sha((arm / "export/manifest.json").read_bytes()),
    }


def write_report(root, arms, errors):
    lines = ["# Inky Mini Hilbert Storyworld Benchmark", "",
             "Each condition uses the same staged task, checkpoint, two loops, and output budgets. The long",
             "turns use increasingly long prefixes of the same Pocket Goblin reference packet. The comparison",
             "therefore measures the effect of supplying more retrieved material in this run. After a measured",
             "long turn fails to author a valid file, the repair uses a bounded working card.", "",
             "The model owns the story and the choice rules. The numerical replay treats the amplitude vector",
             "as a pure state. It applies authored sequences in order and reads Born weights at the ending.",
             "The route probabilities below assume uniformly random available choices.", "",
             "## Sized prompts", "", "| Arm | Stage | Actual input | Output | Provider seconds |",
             "|---|---|---:|---:|---:|"]
    for arm in arms:
        for trace in arm["traces"]:
            if trace.get("target_input_tokens") and trace.get("provider_preflight_parity"):
                lines.append(f"| {arm['input_target']:,} | {trace['stage']} | {trace['provider_prompt_tokens']:,} | "
                             f"{trace['provider_completion_tokens']:,} | {trace['provider_seconds']:.2f} |")
    lines.extend(("", "## Story and routes", "", "| Arm | Encounters | Scene words | Distinct choice labels | Decisions per route |",
                  "|---|---:|---:|---:|---:|"))
    for arm in arms:
        lengths = sorted(map(int, arm["route_lengths"]))
        lines.append(f"| {arm['input_target']:,} | {arm['encounters']} | {arm['scene_words']:,} | "
                     f"{arm['unique_choice_labels']} | {lengths[0]}–{lengths[-1]} |")
    lines.extend(("", "## Secret gates", "", "| Arm | Ending | Draft access | Final access | Swap closed gate |",
                  "|---|---|---:|---:|---|"))
    for arm in arms:
        for end, row in arm["secrets"].items():
            lines.append(f"| {arm['input_target']:,} | `{end}` | {row['draft_probability']:.3%} | "
                         f"{row['final_probability']:.3%} | {'Yes' if row['order_sensitive'] else 'Unobserved'} |")
    if errors:
        lines.extend(("", "## Unfinished phases", ""))
        for item in errors:
            lines.append(f"- {item['input_target']:,} tokens, {item['stage']}. {item['error']}")
    (root / "benchmark-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "develop", "all", "check-plan", "play"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--target", type=int, choices=TARGETS, action="append")
    args = parser.parse_args()
    if args.command == "check-plan":
        check_plan(read_json(args.path))
        return
    if args.command == "play":
        play(read_json(args.path))
        return
    if not all((args.gdk, args.index, args.skill)):
        parser.error("authoring requires --gdk, --index and --skill")
    root = args.path.resolve(strict=True)
    skill = args.skill.resolve(strict=True)
    packet = prepare_packet(root, args.index.resolve(strict=True), skill)
    targets = args.target or TARGETS
    gdk = args.gdk.resolve(strict=True)
    errors = []
    if args.command in ("plan", "all"):
        for target in targets:
            try:
                result = plan_arm(root, gdk, packet, target)
                measured = result["measurement"]
                print(f"{target}: {measured['provider_prompt_tokens']} input tokens, "
                      f"{measured['provider_seconds']:.1f}s provider time, authored plan={bool(result.get('plan'))}", flush=True)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                errors.append({"input_target": target, "stage": "plan", "error": str(error)})
                print(f"{target}: {error}", file=sys.stderr, flush=True)
    if args.command in ("develop", "all"):
        for target in targets:
            arm = root / f"{target // 1024:02d}k"
            if (arm / "result.json").is_file():
                continue
            try:
                candidate = develop(arm, gdk, packet, target, skill)
                export(candidate, arm / "export", skill, target)
                save(arm / "result.json", describe(arm, candidate, target))
                print(f"{target}: {len(candidate['scenes'])} scenes, "
                      f"secret order witnesses={list(candidate['balance']['counterfactuals'])}", flush=True)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                errors.append({"input_target": target, "stage": "develop", "error": str(error)})
                print(f"{target}: {error}", file=sys.stderr, flush=True)
        rows = [read_json(root / f"{target // 1024:02d}k/result.json") for target in TARGETS
                if (root / f"{target // 1024:02d}k/result.json").is_file()]
        save(root / "benchmark-results.json", {"targets": TARGETS, "arms": rows, "errors": errors,
             "design": "Matched authoring stages at one seed. Long prompts use increasing prefixes of one retrieved reference packet."})
        write_report(root, rows, errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
