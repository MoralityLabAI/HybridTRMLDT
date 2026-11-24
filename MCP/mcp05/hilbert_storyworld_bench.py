"""Run matched, staged Inkling storyworld authoring through Goose and Pocket Goblin."""
from __future__ import annotations

import argparse
import json
import os
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
from .hilbert_canvas_bench import author_canvas_edition
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


def author(arm, gdk, stage, brief, validate, *, max_tokens=1800, reference=None):
    receipt = arm / f"{stage}.json"
    if receipt.is_file():
        return read_json(receipt)["value"]
    for previous in sorted(arm.glob(f"{stage}-attempt-*")):
        prev_path = previous / "workspace/result.json"
        if not prev_path.is_file():
            continue
        try:
            raw = read_json(prev_path)
            checked = validate(raw)
            result = raw if checked is None else checked
            proof = replay_writes(previous, [prev_path.name], require_verified=False)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        save(receipt, {"phase": previous.name, "value": result, "authorship": proof})
        return result
    feedback = ""
    prior_attempts = sorted(arm.glob(f"{stage}-attempt-*"))
    if prior_attempts and (prior_attempts[-1] / "validation-error.json").is_file():
        last_err = read_json(prior_attempts[-1] / "validation-error.json").get("error", "")
        if last_err:
            feedback = f"\nEarlier result rejected: {last_err}. Create one valid compact JSON result.json in one action."
    for _ in range(4):
        phase = next_attempt(arm, stage)
        path = phase / "workspace/result.json"
        check = [sys.executable, "-m", "json.tool", str(path)]
        exposure = reference if reference and not measured_reference(arm, stage) else None
        phase_max = 1024 if exposure else max_tokens
        phase_ctx = WINDOW if exposure else max(12288, phase_max + 6144)
        run_phase(phase, gdk, brief + feedback, [path.name], check, max_tokens=phase_max,
                  iterations=1, context=phase_ctx, allow_basename=True, reference_prompt=exposure)
        try:
            require(path.is_file(), "Goose did not receive a complete create_file action")
            raw = read_json(path)
            checked = validate(raw)
            result = raw if checked is None else checked
            proof = replay_writes(phase, [path.name], require_verified=False)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            save(phase / "validation-error.json", {"error": str(error)})
            feedback = f"\nEarlier result rejected: {error}. Create one valid compact JSON result.json in one action."
            continue
        save(receipt, {"phase": phase.name, "value": result, "authorship": proof})
        return result
    raise ValueError(f"{arm.name}/{stage}: {feedback.strip()}")


def text_rows(rows, keys, width):
    require(isinstance(rows, list) and [row[0] for row in rows] == list(keys), "rows need the provided IDs in order")
    require(all(isinstance(row, list) and len(row) == width for row in rows), f"every row needs {width} fields")
    require(all(isinstance(field, str) and field.strip() for row in rows for field in row), "each field needs text")
    return rows


def normalize_rows(source, keys, width, *, wrapper_keys=(), field_groups=()):
    rows = source
    if isinstance(rows, dict):
        for wk in (*wrapper_keys, "rows", "items", "result", "data"):
            if wk in rows and isinstance(rows[wk], list):
                rows = rows[wk]
                break
        else:
            if set(rows) == set(keys):
                rows = [[k, *rows[k]] if isinstance(rows[k], list) else {"id": k, **rows[k]} for k in keys]
            elif len(keys) == 1:
                rows = [rows]
    if len(keys) == 1 and isinstance(rows, list) and len(rows) == width and all(isinstance(x, str) for x in rows):
        rows = [rows]
    require(isinstance(rows, list) and len(rows) == len(keys), f"expected {len(keys)} rows in order {list(keys)}")
    normalized = []
    for idx, item in enumerate(rows):
        if isinstance(item, dict):
            row_id = item.get("id") or item.get("act_id") or item.get("ending_id") or keys[idx]
            cells = [row_id]
            for group in field_groups:
                val = next((item[g] for g in group if g in item and isinstance(item[g], str) and item[g].strip()), None)
                if val is not None:
                    cells.append(val)
            if len(cells) != width:
                used = {"id", "act_id", "ending_id", "name", "voice", "want", "desire"}
                extra = [v for k, v in item.items() if k not in used and isinstance(v, str) and v.strip()]
                if 1 + len(extra) == width:
                    cells = [row_id, *extra]
            normalized.append(cells)
        else:
            normalized.append(item)
    return text_rows(normalized, keys, width)


def normalize_plan(source):
    if isinstance(source, dict):
        out = dict(source)
        if isinstance(out.get("cast"), list):
            out["cast"] = [
                [r.get("id"), r.get("name"), r.get("desire") or r.get("want") or r.get("goal"), r.get("voice")]
                if isinstance(r, dict) else r
                for r in out["cast"]
            ]
        if isinstance(out.get("acts"), list):
            out["acts"] = [
                [r.get("name") or r.get("id"), r.get("place") or r.get("setting"),
                 r.get("pressure"), r.get("irreversible_reveal") or r.get("reveal")]
                if isinstance(r, dict) else r
                for r in out["acts"]
            ]
        if isinstance(out.get("endings"), list):
            out["endings"] = [
                [r.get("id"), r.get("secret"), r.get("promise") or r.get("summary"),
                 r.get("clue_chain") or r.get("clues")]
                if isinstance(r, dict) else r
                for r in out["endings"]
            ]
        source = out
    return check_plan(source)


def _to_num(val):
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return val
    return val


def _clean_secret_map(secrets):
    if not isinstance(secrets, dict):
        return secrets
    cleaned = {}
    for end_id, rule in secrets.items():
        if not isinstance(rule, dict):
            cleaned[end_id] = rule
            continue
        projs = rule.get("projectors")
        if isinstance(projs, list):
            norm_projs = []
            for p in projs:
                if isinstance(p, dict):
                    p = [p.get("projector") or p.get("P"), p.get("low"), p.get("high")]
                if isinstance(p, list) and len(p) == 3:
                    p = [p[0], _to_num(p[1]), _to_num(p[2])]
                norm_projs.append(p)
            projs = norm_projs
        cleaned[end_id] = {k: (projs if k == "projectors" else v) for k, v in rule.items()}
    return cleaned


def _algebra_diagnostics(model, plan):
    errs = []
    basis = model.get("basis")
    if not isinstance(basis, dict) or set(basis) != {"00", "01", "10", "11"} or len({str(v).strip() for v in basis.values()}) != 4:
        errs.append("basis must map exact keys '00','01','10','11' to 4 distinct non-empty strings")
    clues = model.get("clues")
    if not isinstance(clues, dict) or set(clues) != {f"c{i}" for i in range(6)}:
        errs.append("clues must map exact keys 'c0'..'c5' to 6 non-empty strings")
    ops = model.get("operators")
    if not isinstance(ops, dict) or not (6 <= len(ops) <= 8):
        errs.append(f"operators must have 6 to 8 named gate sequences (got {len(ops) if isinstance(ops, dict) else 'non-dict'})")
    hidden = [row[0] for row in plan["endings"] if row[1]]
    secrets = model.get("secrets")
    if not isinstance(secrets, dict) or set(secrets) != set(hidden):
        errs.append(f"secrets must be an object with exact hidden ending keys {hidden}")
    else:
        for hid in hidden:
            rule = secrets[hid]
            if not isinstance(rule, dict) or set(rule) != {"requires", "projectors"}:
                errs.append(f"secrets[{hid!r}] must be {{'requires':[2..4 clue IDs],'projectors':[[P,low,high],...]}}")
            else:
                projs = rule.get("projectors") or []
                if not any(isinstance(p, list) and p and str(p[0]).startswith(("X", "Y")) for p in projs):
                    errs.append(
                        f"secrets[{hid!r}].projectors MUST include at least one Pauli X or Y projector "
                        f"('X0+','X0-','Y0+','Y0-','X1+','X1-','Y1+','Y1-'), not only '00'..'11' or 'Z'"
                    )
    return errs


def normalize_algebra(source, plan):
    if isinstance(source, dict) and "math" in source and isinstance(source["math"], dict) and "operators" not in source:
        source = source["math"]
    require(isinstance(source, dict), "algebra draft must be a JSON object")
    model = {k: source[k] for k in ("basis", "clues", "operators", "secrets") if k in source}
    if isinstance(model.get("basis"), dict):
        b = model["basis"]
        if isinstance(b.get("states"), dict):
            b = b["states"]
        elif {"00", "01", "10", "11"} <= set(b):
            b = {k: b[k] for k in ("00", "01", "10", "11")}
        model["basis"] = b
    if isinstance(model.get("clues"), list) and len(model["clues"]) == 6 and all(isinstance(x, str) for x in model["clues"]):
        model["clues"] = {f"c{i}": text for i, text in enumerate(model["clues"])}
    if "secrets" in model:
        model["secrets"] = _clean_secret_map(model["secrets"])
    errs = _algebra_diagnostics(model, plan)
    require(not errs, "; ".join(errs))
    check_algebra(model, plan)
    return model


def normalize_secrets(source, base_model, plan):
    if isinstance(source, dict):
        if set(source) == {"secrets"} and isinstance(source["secrets"], dict):
            source = source["secrets"]
        elif "secrets" in source and isinstance(source["secrets"], dict) and (
            "operators" in source or "basis" in source or "clues" in source
        ):
            source = source["secrets"]
    secrets = _clean_secret_map(source)
    candidate_model = {**base_model, "secrets": secrets}
    errs = _algebra_diagnostics(candidate_model, plan)
    require(not errs, "; ".join(errs))
    check_algebra(candidate_model, plan)
    return secrets


def compact_exact(exact_obj):
    gates = {}
    for end_id, info in exact_obj["secret_gates"].items():
        gates[end_id] = {
            "eligible_probability": round(info["eligible_probability"], 6),
            "clue_probability": round(info["clue_probability"], 6),
            "has_order_witness": bool(info["order_witness"]),
            "top_projector_values": [
                {"values": row["values"], "probability": round(row["probability"], 6)}
                for row in info["projector_distribution"][:10]
            ],
        }
    return {"secret_gates": gates}


def normalize_outline(source, existing_by_id):
    rows = source
    if isinstance(rows, dict):
        for wk in ("outline", "beats", "scenes", "encounters", "rows"):
            if wk in rows and isinstance(rows[wk], list):
                rows = rows[wk]
                break
        else:
            if set(rows) == set(SCENE_IDS):
                rows = [[sid, *(rows[sid] if isinstance(rows[sid], list) else [rows[sid].get("title"), rows[sid].get("beat")])]
                        for sid in SCENE_IDS]
    require(isinstance(rows, list) and len(rows) == len(SCENE_IDS), f"outline needs {len(SCENE_IDS)} rows for enc_01..enc_30")
    normalized = []
    for idx, item in enumerate(rows):
        if isinstance(item, dict):
            sid = item.get("id") or item.get("scene_id") or SCENE_IDS[idx]
            title = item.get("title") or item.get("unique_title") or item.get("name")
            beat = item.get("beat") or item.get("event_beat") or item.get("summary") or item.get("event")
        else:
            require(isinstance(item, list) and len(item) == 3, "each outline row must be [scene_id, unique_title, event_beat]")
            sid, title, beat = item
        if sid in existing_by_id:
            title = existing_by_id[sid]
        require(sid == SCENE_IDS[idx] and isinstance(title, str) and title.strip() and isinstance(beat, str) and beat.strip(),
                f"invalid outline row at {SCENE_IDS[idx]}")
        normalized.append([sid, title.strip(), beat.strip()])
    seen_titles = set()
    dups = []
    for _, title, _ in normalized:
        tc = title.casefold()
        if tc in seen_titles:
            dups.append(title)
        seen_titles.add(tc)
    require(not dups, f"outline has duplicate scene titles {dups} — all 30 titles must be distinct")
    return {sid: [title, beat] for sid, title, beat in normalized}


def normalize_encounters(source, ids, model, earlier, outline=None):
    scenes = source
    if isinstance(scenes, dict):
        for wk in ("scenes", "encounters", "result", "items"):
            if wk in scenes and isinstance(scenes[wk], list):
                scenes = scenes[wk]
                break
        else:
            if set(scenes) == set(ids):
                scenes = [{"id": sid, **scenes[sid]} if isinstance(scenes[sid], dict) else scenes[sid] for sid in ids]
    require(isinstance(scenes, list), "encounters batch must be a JSON list of 3 scene objects")
    norm_scenes = []
    for scene in scenes:
        require(isinstance(scene, dict), "each encounter must be a JSON object")
        opts = scene.get("options")
        require(isinstance(opts, list), f"{scene.get('id')}: options must be a list")
        norm_opts = []
        for opt in opts:
            if isinstance(opt, dict):
                opt = [
                    opt.get("label") or opt.get("choice"),
                    opt.get("reply") or opt.get("response"),
                    opt.get("operator") or opt.get("op"),
                    opt.get("clue"),
                    opt.get("advance", 1),
                ]
            if isinstance(opt, list) and len(opt) == 5:
                choice, reply, op, clue, adv = opt
                if clue in ("", "null", "None", "none", False):
                    clue = None
                if isinstance(adv, str) and adv.strip() in ("1", "2"):
                    adv = int(adv.strip())
                elif isinstance(adv, float) and adv in (1.0, 2.0):
                    adv = int(adv)
                opt = [choice, reply, op, clue, adv]
            require(isinstance(opt, list) and len(opt) == 5, f"{scene.get('id')}: each option must have 5 items [label, reply, operator, clue, advance]")
            c_lbl, c_rep, c_op, c_clue, c_adv = opt
            require(c_op in model["operators"], f"{scene.get('id')}: unknown operator {c_op!r}; must be in {list(model['operators'])}")
            require(c_clue is None or c_clue in model["clues"], f"{scene.get('id')}: invalid clue {c_clue!r}; must be null or c0..c5")
            require(type(c_adv) is int and c_adv in (1, 2), f"{scene.get('id')}: advance must be integer 1 or 2, got {c_adv!r}")
            norm_opts.append(opt)
        norm_scenes.append({"id": scene.get("id"), "title": scene.get("title"), "text": scene.get("text"), "options": norm_opts})
    prior_titles = {s["title"].strip().casefold(): s["title"] for s in earlier}
    prior_texts = {s["text"].strip().casefold(): s["id"] for s in earlier}
    seen_batch_titles = set()
    seen_batch_texts = {}
    dup_titles = []
    dup_texts = []
    shortcut_ids = [sid for sid in ids if int(sid[4:]) % 6 == 3]
    for s in norm_scenes:
        sid = s.get("id")
        t = (s.get("title") or "").strip()
        tc = t.casefold()
        if (not t or tc in prior_titles or tc in seen_batch_titles) and outline and sid in outline:
            planned_t = outline[sid][0].strip()
            if planned_t.casefold() not in prior_titles and planned_t.casefold() not in seen_batch_titles:
                s["title"] = t = planned_t
                tc = t.casefold()
        if tc in prior_titles or tc in seen_batch_titles:
            dup_titles.append(t)
        seen_batch_titles.add(tc)
        tx = (s.get("text") or "").strip().casefold()
        if tx in prior_texts:
            dup_texts.append(f"{sid} repeats text of {prior_texts[tx]}")
        elif tx in seen_batch_texts:
            dup_texts.append(f"{sid} repeats text of {seen_batch_texts[tx]}")
        else:
            seen_batch_texts[tx] = sid
        steps = {opt[4] for opt in s["options"]}
        if sid in shortcut_ids:
            require(steps == {1, 2}, f"{sid} IS the shortcut scene in this batch and MUST have at least one option with advance=1 AND at least one option with advance=2 (got {sorted(steps)})")
        else:
            require(steps == {1}, f"{sid} is NOT a shortcut scene so ALL 3 of its options MUST have advance=1 (got {sorted(steps)}; only {shortcut_ids} uses advance=2)")
    require(not dup_titles, f"repeated encounter title(s) {dup_titles} — use the exact planned unique titles in target_batch_outline")
    require(not dup_texts, f"repeated encounter prose ({'; '.join(dup_texts)}) — write distinct scene prose for each encounter")
    check_scene_continuity(norm_scenes, ids, model, earlier)
    return norm_scenes


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
        plan = normalize_plan(planning["plan"])
    except (KeyError, TypeError, ValueError) as error:
        save(arm / "plan-review.json", {"initial_phase": planning["phase"], "accepted": False, "reason": str(error)})
        plan = author(arm, gdk, "plan-repair", BRIEF.replace("plan.json", "result.json") +
                      "\nUse short actor, act and ending arrays. Keep the plan focused on the missing night bell.",
                      normalize_plan, max_tokens=1800)
    else:
        save(arm / "plan-review.json", {"initial_phase": planning["phase"], "accepted": True})
    cast_ids = [row[0] for row in plan["cast"]]
    endings = [row[0] for row in plan["endings"]]
    public = [row[0] for row in plan["endings"] if not row[1]]
    voices = author(arm, gdk, "characterize", "Call create_file now with path result.json. Write a top-level JSON array with one 4-string row per cast member, in order: "
                    "[[id, private contradiction, original line of dialogue, what would break their resolve], ...]. "
                    "Do not wrap in an object and do not repeat name or voice. Make their speech rhythms and bargains recognizably different.\n" + encode(plan),
                    lambda rows: normalize_rows(
                        rows, cast_ids, 4, wrapper_keys=("cast", "voices", "characters"),
                        field_groups=(("private_contradiction", "contradiction"),
                                      ("original_line", "line", "dialogue"),
                                      ("what_would_break_their_resolve", "break_resolve", "break")),
                    ), max_tokens=1800)
    hidden = [row[0] for row in plan["endings"] if row[1]]
    model_brief = (
        "Call create_file now with path result.json as the Hilbert algebra draft for your night-bell story. "
        "Output ONLY a JSON object with exact top-level keys basis, clues, operators, secrets "
        "(do NOT repeat title, premise, cast, acts, or endings).\n"
        "- basis: {\"00\":\"...\",\"01\":\"...\",\"10\":\"...\",\"11\":\"...\"} with 4 distinct joint-state meanings (state starts at |00>).\n"
        "- clues: {\"c0\":\"...\",\"c1\":\"...\",\"c2\":\"...\",\"c3\":\"...\",\"c4\":\"...\",\"c5\":\"...\"} with 6 concrete story evidence descriptions.\n"
        "- operators: JSON object {\"op_name_1\":[\"H0\",\"CX01\"],\"op_name_2\":[\"S0\",\"H0\"], ...} with 6 to 8 distinct identifier keys (never a list of names!), where each value is a unique array of 1 to 4 primitive gates from "
        "[\"I\",\"H0\",\"H1\",\"X0\",\"X1\",\"Z0\",\"Z1\",\"S0\",\"S1\",\"CX01\",\"CX10\"] (qubit 0 is left; include mixing H, phase S/Z, and entanglement so at least one pair does not commute).\n"
        f"- secrets: object with exact hidden ending keys {json.dumps(hidden)}. Example shape per secret: "
        "{\"requires\":[\"c0\",\"c2\"],\"projectors\":[[\"X0+\",0.25,0.95],[\"01\",0.1,0.8]]}. "
        "CRITICAL: BOTH secrets MUST include at least one Pauli X or Y projector from "
        "[\"X0+\",\"X0-\",\"Y0+\",\"Y0-\",\"X1+\",\"X1-\",\"Y1+\",\"Y1-\"] (never use only \"00\"..\"11\"!), "
        "2 or 3 distinct projectors per secret with 0 <= low < high <= 1 and high - low < 1, "
        "and 2 to 4 clues from c0..c5 with different clue sets between the two secrets.\n"
        "Your story:\n"
    )
    model = author(arm, gdk, "math-draft", model_brief + encode(plan),
                   lambda content: normalize_algebra(content, plan), max_tokens=2048, reference=(packet, target))
    existing_by_id = {
        s["id"]: s["title"]
        for p in sorted(arm.glob("encounter-??.json"))
        for s in read_json(p)["value"]
    }
    outline_brief = (
        "Call create_file now with path result.json. Write a top-level JSON array of 30 rows [scene_id, unique_title, event_beat] in exact order enc_01..enc_30 "
        "(6 encounters per act across your 5 acts; do NOT wrap in an object).\n"
        "Every single one of the 30 scene titles MUST be distinct (never reuse an act name or an earlier scene title!), "
        "and each event_beat is one concrete sentence advancing your night-bell mystery and clue discoveries.\n"
    )
    if existing_by_id:
        outline_brief += f"Keep these already-authored titles fixed and make all remaining titles distinct from them: {encode(existing_by_id)}\n"
    outline = author(
        arm, gdk, "outline",
        outline_brief + encode({"ids": SCENE_IDS, "title": plan["title"], "acts": plan["acts"],
                                "cast": voices, "clues": model["clues"]}),
        lambda data: normalize_outline(data, existing_by_id), max_tokens=2800,
    )
    scenes, act_cards = [], []
    store = skill.parents[2] / "mcp-storyworld-encounter/swmd_store.py"
    parse_card = runpy.run_path(str(store))["iteration_packet"]
    store_sha = sha(store.read_bytes())
    for act_i, act in enumerate(plan["acts"]):
        for segment in range(2):
            index = act_i * ENCOUNTERS_PER_ACT + segment * 3
            ids = SCENE_IDS[index:index + 3]
            used_clues = sorted({opt[3] for s in scenes for opt in s["options"] if opt[3]})
            unplaced_clues = sorted(set(model["clues"]) - set(used_clues))
            adv_rules = {
                sid: "MUST include both advance 1 and advance 2 across its 3 options"
                if int(sid[4:]) % 6 == 3 else "all 3 options MUST have advance 1"
                for sid in ids
            }
            planned_batch = [
                [
                    sid,
                    outline[sid][0],
                    outline[sid][1],
                    "options advance MUST mix 1 and 2 (e.g. 1, 1, 2)"
                    if int(sid[4:]) % 6 == 3 else "ALL 3 options MUST have advance = 1 (1, 1, 1)",
                ]
                for sid in ids
            ]
            swmd_neighbors = [blk[:240] for blk in continuity_card(arm, parse_card, store_sha, scenes, ids[0])]
            brief = """Call create_file now with path result.json. Write a top-level JSON array of 3 distinct scene objects in exact ids order (do NOT repeat plan, cast, clues, or operators):
[{"id":"enc_XX","title":"Exact Planned Title","text":"Two to four compact paragraphs with original dialogue...","options":[["Choice label","Voiced response","operator_name","c0 or null",1], ...]}, ...]
Rules:
- For each scene ID in ids, use its exact planned title, develop its planned beat, and obey its exact advance rule in target_batch_outline at the end of this prompt! Write brand new scene prose for every encounter.
- Each scene needs 3 options with 3 distinct choice labels.
- Each option is a 5-item array: [choice_label, voiced_response, operator_name, clue_id_or_null, advance_int].
- operator_name MUST be one of allowed_operators; clue_id_or_null MUST be null or one of c0..c5.
- Ensure unplaced_clues are woven into choices across acts so all c0..c5 become reachable.
- Put the nearby characters' voices into the responses. Close the create_file tool call cleanly.
"""
            brief += encode({"ids": ids,
                             "advance_rules": adv_rules,
                             "allowed_operators": list(model["operators"]),
                             "operators": model["operators"],
                             "clues": model["clues"],
                             "unplaced_clues": unplaced_clues,
                             "title": plan["title"], "premise": plan["premise"], "act": act,
                             "cast": voices, "prior_act": act_cards[-1:],
                             "previous_scenes_swmd": swmd_neighbors,
                             "target_batch_outline": planned_batch})
            batch = author(arm, gdk, f"encounter-{index + 1:02d}", brief,
                           lambda data, _ids=ids: normalize_encounters(data, _ids, model, scenes, outline=outline),
                           max_tokens=3400)
            scenes.extend(batch)
        summary_brief = """Call create_file now with exact path result.json. Put ONE 5-string JSON array in content: [act_id, established_fact, character_cost, unresolved_clue, next_act_constraint] (do NOT repeat scenes). Use only this act's authored encounters and the intended next act, then close the tool call.\n"""
        act_id = f"act_{act_i + 1}"
        recent = [
            {
                "id": s["id"],
                "title": s["title"],
                "excerpt": s["text"][:260],
                "clues": [o[3] for o in s["options"] if o[3]],
            }
            for s in scenes[-ENCOUNTERS_PER_ACT:]
        ]
        card = author(arm, gdk, f"act-complete-{act_i + 1}", summary_brief + encode({"act_id": act_id,
                      "scenes": recent, "next": plan["acts"][act_i + 1:act_i + 2]}),
                      lambda row, _aid=act_id: normalize_rows(
                          row, [_aid], 5, wrapper_keys=("act", "summary"),
                          field_groups=(("established_fact", "fact"), ("character_cost", "cost"),
                                        ("unresolved_clue", "clue"), ("next_act_constraint", "constraint")),
                      )[0], max_tokens=1200)
        act_cards.append(card)
    revised_cast = author(arm, gdk, "recharacterize", """Call create_file now with exact path result.json. Write a top-level JSON array with the three actor IDs in order (do NOT repeat before or acts):
[[id, what this person now believes, a late-scene line, what loss is visible in the endings], ...].
Base the changes on the actual act receipts. Give each a distinct late voice.\n""" + encode({"before": voices,
                           "acts": act_cards}),
                          lambda rows: normalize_rows(
                              rows, cast_ids, 4, wrapper_keys=("cast", "characters", "after"),
                              field_groups=(("belief", "now_believes", "what_this_person_now_believes"),
                                            ("late_line", "line", "late_scene_line"),
                                            ("loss", "visible_loss", "what_loss_is_visible_in_the_endings")),
                          ), max_tokens=1800)
    terminals = author(arm, gdk, "endings", """Call create_file now with exact path result.json. Write a top-level JSON array of five 3-string rows (do NOT repeat plan, cast, acts, or math):
[[ending_id, title, final_prose], ...].
Use the given IDs in order. Write two short paragraphs per ending. The two secret endings should pay off
different clue chains and expose different truths about the night bell. Write scenes with consequences,
in the cast's late voices. Close the tool call cleanly.\n""" + encode({"plan": plan["endings"], "cast": revised_cast,
                       "acts": act_cards, "math": model}),
                       lambda rows: normalize_rows(
                           rows, endings, 3, wrapper_keys=("endings", "terminals"),
                           field_groups=(("title",), ("prose", "final_prose", "text")),
                       ), max_tokens=2800)
    draft = simulate(scenes, model, public)
    draft_exact = exact_routes(scenes, model)
    save(arm / "draft-rehearsal.json", {"simulation": draft, "exact": draft_exact})
    def _rehearsal_summary(ex_obj, cur_secrets):
        lines = []
        for hid in hidden:
            rep = ex_obj["secret_gates"][hid]
            top_vals = ", ".join(
                f"{list(r['values'])} (p={r['probability']:.3f})"
                for r in rep["projector_distribution"][:6]
            )
            lines.append(
                f"- {hid}: current={encode(cur_secrets[hid])}; "
                f"eligible_prob={rep['eligible_probability']:.4f}; clue_prob={rep['clue_probability']:.4f}; "
                f"order_sensitive={bool(rep['order_witness'])}; top_projector_tuples=[{top_vals}]"
            )
        return "\n".join(lines)

    tune_header = (
        f"Call create_file now with exact path result.json. Write ONLY a small JSON object with exact top-level keys {json.dumps(hidden)} "
        "(around 150 tokens total; NEVER include 'model', 'basis', 'operators', 'rehearsal', or 'exact' keys!).\n"
        "Each of the two keys maps to {\"requires\":[2 to 4 clue IDs from c0..c5],\"projectors\":[[P,low,high],...]} "
        "with 2 to 3 distinct projectors (at least one Pauli X or Y projector 'X0+','X0-','Y0+','Y0-','X1+','X1-','Y1+','Y1-' in BOTH secrets, "
        "0 <= low < high <= 1, high - low < 1, and different clue sets).\n"
        "Tighten or tune intervals using the exact Hilbert rehearsal numbers below so both secrets stay selective (0 < eligible_prob < 0.35) "
        "and order_sensitive=True:\n"
    )
    holistic_ctx = (
        f"Monte Carlo ending counts (5000 runs): {encode(draft['endings'])}\n"
        f"Clue IDs: {list(model['clues'])}\n"
        f"Exact Hilbert gate rehearsal:\n{_rehearsal_summary(draft_exact, model['secrets'])}\n"
        f"Write ONLY the revised two-key secrets JSON object based on: {encode(model['secrets'])}"
    )
    revised = author(arm, gdk, "late-stage-holistic", tune_header + holistic_ctx,
                     lambda secrets: normalize_secrets(secrets, model, plan), max_tokens=1200, reference=(packet, target))
    final_model = {**model, "secrets": revised}
    revised_exact = exact_routes(scenes, final_model)
    save(arm / "holistic-rehearsal.json", revised_exact)
    for attempt in range(1, 4):
        incomplete = [end for end, report in revised_exact["secret_gates"].items()
                      if not report["order_witness"] or not 0 < report["eligible_probability"] < 1]
        if not incomplete:
            break
        repair_ctx = (
            f"Gates needing adjustment (must have 0 < eligible_prob < 1 and order_sensitive=True): {incomplete}\n"
            f"Current rehearsal:\n{_rehearsal_summary(revised_exact, final_model['secrets'])}\n"
            f"Initial draft rehearsal:\n{_rehearsal_summary(draft_exact, model['secrets'])}\n"
            f"Write ONLY the fixed two-key secrets JSON object based on: {encode(final_model['secrets'])}"
        )
        revised = author(arm, gdk, f"gate-repair-{attempt}", tune_header + repair_ctx,
                         lambda secrets: normalize_secrets(secrets, model, plan),
                         max_tokens=1200)
        final_model = {**model, "secrets": revised}
        revised_exact = exact_routes(scenes, final_model)
        save(arm / f"gate-repair-{attempt}-rehearsal.json", revised_exact)
    balance = simulate(scenes, final_model, public)
    result = {"plan": plan, "cast": revised_cast, "scenes": scenes, "endings": terminals,
              "math": final_model, "algebra": check_algebra(final_model, plan), "balance": balance, "exact": revised_exact}
    save(arm / "candidate.json", result)
    return result


def describe(arm, candidate, target, *, canvas=None):
    traces = []
    for path in sorted(arm.glob("*/session.preflight.jsonl")):
        wire = path.with_name("inky-wire.jsonl")
        responses = iter(json.loads(frame["body"]) for frame in events(wire)
                         if frame["event"] == "response" and frame["status"] == 200) if wire.is_file() else iter(())
        ws = path.parent / "workspace"
        files = sorted(p for p in ws.iterdir() if p.is_file() and not p.name.startswith(".")) if ws.is_dir() else []
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
    if canvas is None and (arm / "canvas-edition.json").is_file():
        canvas = read_json(arm / "canvas-edition.json")
    summary = {
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
    if canvas is not None:
        summary["canvas"] = canvas
    return summary


def write_report(root, arms, errors, *, model_label="Inky Mini"):
    lines = [f"# {model_label} Hilbert Storyworld + Canvas Art Benchmark", "",
             "Each condition uses the same staged task, checkpoint, deterministic seed, and output budgets. The long",
             "turns use increasingly long prefixes of the same Pocket Goblin reference packet. The comparison",
             "therefore measures the effect of supplying more retrieved material in this run. After a measured",
             "long turn fails to author a valid file, the repair uses a bounded working card.", "",
             "The model owns the story, choice rules, 35-scene Canvas art map, HTML layout, procedural 2D Canvas",
             "renderer, and Hilbert quantum browser player. The numerical replay treats the amplitude vector",
             "as a pure state, applies authored sequences in order, and reads Born weights at the ending.",
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
    canvas_arms = [arm for arm in arms if arm.get("canvas")]
    if canvas_arms:
        lines.extend((
            "",
            "## Canvas art and browser verification",
            "",
            "| Arm | Landmarks | Palettes | Renderer lines | Visited scenes | Distinct hashes | Mean colors | Edge density | Mean RGB L1 | Verified |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ))
        for arm in canvas_arms:
            c = arm["canvas"]
            am = c["art_map_stats"]
            rs = c["renderer_stats"]
            br = c["browser"]
            vm = br["visual_metrics"]
            vis = br["visited_scenes"] if isinstance(br["visited_scenes"], int) else len(br["visited_scenes"])
            ok = br.get("status") in ("ok", "verified")
            lines.append(
                f"| {arm['input_target']:,} | {am['distinct_landmarks']} | {am['distinct_palettes']} | "
                f"{rs['lines']} | {vis} | {br['distinct_canvas_hashes']} | "
                f"{vm['mean_quantized_colors']:.1f} | {vm['mean_edge_density']:.4f} | "
                f"{vm['mean_pairwise_rgb_l1']:.2f} | {'Yes' if ok else 'No'} |"
            )
    if errors:
        lines.extend(("", "## Unfinished phases", ""))
        for item in errors:
            lines.append(f"- {item['input_target']:,} tokens, {item['stage']}. {item['error']}")
    (root / "benchmark-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "develop", "canvas", "all", "check-plan", "play"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--target", type=int, choices=TARGETS, action="append")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-label", type=str, default="Inky Mini")
    parser.add_argument("--node", type=Path, default=None)
    parser.add_argument("--chrome", type=Path, default=None)
    parser.add_argument("--no-canvas", action="store_true")
    args = parser.parse_args()
    if args.command == "check-plan":
        check_plan(read_json(args.path))
        return
    if args.command == "play":
        play(read_json(args.path))
        return
    if not all((args.gdk, args.index, args.skill)):
        parser.error("authoring requires --gdk, --index and --skill")
    if args.base_url:
        os.environ["MCP05_BASE_URL"] = args.base_url
    if args.model:
        os.environ["MCP05_MODEL"] = args.model
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
    if args.command in ("develop", "canvas", "all"):
        for target in targets:
            arm = root / f"{target // 1024:02d}k"
            if args.command != "canvas" and (arm / "result.json").is_file() and (
                args.no_canvas or (arm / "canvas-edition.json").is_file()
            ):
                continue
            try:
                if (arm / "candidate.json").is_file():
                    candidate = read_json(arm / "candidate.json")
                else:
                    require(args.command != "canvas", f"{arm.name} lacks candidate.json for canvas stage")
                    candidate = develop(arm, gdk, packet, target, skill)
                if not (arm / "export/manifest.json").is_file():
                    export(candidate, arm / "export", skill, target)
                canvas = None if args.no_canvas else author_canvas_edition(
                    arm, candidate, gdk, node=args.node, chrome=args.chrome
                )
                save(arm / "result.json", describe(arm, candidate, target, canvas=canvas))
                print(f"{target}: {len(candidate['scenes'])} scenes, "
                      f"secret order witnesses={list(candidate['balance']['counterfactuals'])}, "
                      f"canvas_verified={bool(canvas and canvas.get('status') == 'verified')}", flush=True)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                errors.append({"input_target": target, "stage": args.command, "error": str(error)})
                print(f"{target}: {error}", file=sys.stderr, flush=True)
        rows = [read_json(root / f"{target // 1024:02d}k/result.json") for target in TARGETS
                if (root / f"{target // 1024:02d}k/result.json").is_file()]
        save(root / "benchmark-results.json", {
            "model_label": args.model_label,
            "targets": TARGETS,
            "arms": rows,
            "errors": errors,
            "design": "Matched authoring + Canvas art + headless Chrome Hilbert verification stages at one seed. Long prompts use increasing prefixes of one retrieved reference packet.",
        })
        write_report(root, rows, errors, model_label=args.model_label)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
