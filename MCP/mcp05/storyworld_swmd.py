"""Compile a bounded SWMD-0-MIN authoring subset without rewriting authored prose."""
from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from random import Random

STATS = ("trust", "nerve", "care")
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
ORX = re.compile(r"ORX (opt_\w+)/(rxn_\w+) -> ([a-z][a-z0-9_]*)\s*\| O:(.*?)\s*\| E:(.*?)\s*\| D:(.*)")
RXN_TEXT = re.compile(r"# R (rxn_\w+):\s*(.+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def world_ids(card: dict) -> list[str]:
    playable = 20 - len(card["endings"])
    return [f"enc_{i:02}" for i in range(1, playable + 1)] + [e["id"] for e in card["endings"]]


def score(expression: str, character: str) -> list[float]:
    """Parse affine ADD/C/P/MUL expressions to [constant, trust, nerve, care]."""
    exp = re.sub(r"\s+", "", expression)
    terms = exp[4:-1] if exp.startswith("ADD(") and exp.endswith(")") else exp
    term = re.compile(rf"C\(({NUMBER})\)|MUL\(P\({re.escape(character)}\.({'|'.join(STATS)})\),({NUMBER})\)")
    out = [0.0] * 4
    while terms:
        match = term.match(terms)
        require(match is not None, f"unsupported desirability expression {expression}")
        const, stat, weight = match.groups()
        if stat:
            out[1 + STATS.index(stat)] += float(weight)
        else:
            out[0] += float(const)
        terms = terms[match.end():]
        if terms:
            require(terms.startswith(",") and len(terms) > 1 and exp.startswith("ADD("),
                    f"invalid term delimiter in {expression}")
            terms = terms[1:]
    require(any(out[1:]) and sum(map(abs, out)) <= 1, "score must be state dependent and bounded in [-1,1]")
    return out


def reaction_effects(text: str, character: str) -> dict[str, float]:
    effects = {}
    for raw in text.split(";"):
        match = re.fullmatch(rf"SET\s+{re.escape(character)}\.({'|'.join(STATS)})\s*\+=\s*({NUMBER})", raw.strip())
        require(match is not None, f"invalid bounded delta {raw}")
        stat, delta = match.groups()
        delta = float(delta)
        require(stat not in effects and 0.01 <= abs(delta) <= 0.24, "use unique nonzero deltas with magnitude .01–.24")
        effects[stat] = delta
    require(len(effects) >= 2, "give each reaction at least two distinct effects")
    return effects


def render_card(source: dict, card: dict, scene_id: str) -> str:
    """Expand an Inkling-authored compact card into canonical ORX and prose records."""
    def text(value):
        require(isinstance(value, str) and value.strip() and "\n" not in value and "\r" not in value and "|" not in value,
                "use a nonempty single-line string without pipes for each prose field")
        return value

    hero = card["cast"][0]["id"]
    ordinal = world_ids(card).index(scene_id) + 1
    rows = [f"ENC {scene_id} turn=0..999", f"# T: {text(source['text'])}",
            f"# TITLE: {text(source['title'])}", f"# ART: {text(source['art'])}"]
    for op_i, option in enumerate(source["options"], 1):
        if isinstance(option, dict):
            label, reactions = option["label"], option["reactions"]
        else:
            label, *reactions = option
        require(len(reactions) == 2, "each option row must contain a label and exactly two reactions")
        for rx_i, reaction in enumerate(reactions, 1):
            if isinstance(reaction, dict):
                target, reply, scores, deltas = (reaction[key] for key in ("next", "reply", "score", "delta"))
            else:
                require(isinstance(reaction, list) and len(reaction) == 4,
                        "a reaction is an object with next, reply, score and delta")
                target, reply, scores, deltas = reaction
            require(len(scores) == 4 and len(deltas) == 3 and all(type(x) is int for x in [*scores, *deltas]),
                    "reaction scores and deltas must be four and three integer hundredths respectively")
            require(sum(map(abs, scores)) <= 100 and all(abs(delta) <= 24 for delta in deltas),
                    "keep score coefficient mass within 100 and deltas within 24 hundredths")
            opt, rxn = f"opt_{ordinal:02}_{op_i}", f"rxn_{ordinal:02}_{op_i}_{rx_i}"
            effects = "; ".join(f"SET {hero}.{stat} += {delta / 100:g}" for stat, delta in zip(STATS, deltas) if delta)
            terms = [f"C({scores[0] / 100:g})"] if scores[0] else []
            terms.extend(f"MUL(P({hero}.{stat}),{weight / 100:g})" for stat, weight in zip(STATS, scores[1:]) if weight)
            desirability = f"ADD({','.join(terms)})" if len(terms) > 1 else (terms[0] if terms else "C(0)")
            rows.extend([f"ORX {opt}/{rxn} -> {target} | O:{text(label)} | E:{effects} | D:{desirability}",
                         f"# R {rxn}: {text(reply)}"])
    return "\n".join(rows) + "\n"


def parse_scene(text: str, card: dict, scene_id: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(lines and lines[0] == f"ENC {scene_id} turn=0..999", f"expected ENC {scene_id} turn=0..999")
    require(len(lines) > 1 and lines[1].startswith("# T:") and len(lines[1].split()) >= 10,
            "write a substantive # T: paragraph on line 2")
    require(len(lines) >= 4 and lines[2].startswith("# TITLE:") and lines[3].startswith("# ART:"),
            "add # TITLE: and # ART: as lines 3 and 4")
    scene = {"id": scene_id, "text": lines[1][4:].strip(), "title": lines[2][8:].strip(),
             "art": lines[3][6:].strip(), "options": []}
    require(scene["title"] and scene["art"], "both title and visual cue are required")
    ids = world_ids(card)
    idx = ids.index(scene_id) + 1
    playable = 20 - len(card["endings"])
    if idx > playable:
        require(len(lines) == 4, "a terminal scene has only ENC, # T, # TITLE, and # ART lines")
        return scene
    character = card["cast"][0]["id"]
    require(len(lines) == 16, "each ordinary encounter needs six ORX lines, each followed by its # R line")
    opts, reaction_ids = {}, set()
    for raw, comment in zip(lines[4::2], lines[5::2], strict=True):
        match, reaction_text = ORX.fullmatch(raw), RXN_TEXT.fullmatch(comment)
        require(match is not None and reaction_text is not None, f"invalid ORX or # R syntax in {scene_id}")
        opt, rxn, target, choice, effects, desirability = match.groups()
        require(opt.startswith(f"opt_{idx:02}_") and rxn.startswith(f"rxn_{idx:02}_")
                and rxn not in reaction_ids and reaction_text[1] == rxn, "encounter-specific option/reaction IDs must match")
        reaction_ids.add(rxn)
        allowed = ids[idx:min(idx + 2, playable)] if idx < playable else ids[playable:]
        require(target in allowed, f"invalid forward target {target}, expected one of {allowed}")
        row = opts.setdefault(opt, {"id": opt, "text": choice, "reactions": []})
        require(choice.strip() and row["text"] == choice, "repeat the same nonempty option text on both ORX lines")
        row["reactions"].append({"id": rxn, "to": target, "text": reaction_text[2],
                                 "effects": reaction_effects(effects, character), "score": score(desirability, character)})
    require(len(opts) == 3 and len({o["text"] for o in opts.values()}) == 3, "write three distinct player options")
    vectors = set()
    for option in opts.values():
        pair = option["reactions"]
        require(len(pair) == 2 and pair[0]["text"] != pair[1]["text"], "options must have two distinct responses")
        require(pair[0]["effects"] != pair[1]["effects"], "each response needs a distinct effect vector")
        diff = [a - b for a, b in zip(pair[0]["score"], pair[1]["score"], strict=True)]
        require(sum(map(abs, diff[1:])) > abs(diff[0]), "response ordering must reverse for some bounded states")
        vectors.add(tuple(sorted(tuple(sorted(r["effects"].items())) for r in pair)))
    require(len(vectors) == 3, "player choices need different effect pairs")
    scene["options"] = list(opts.values())
    targets = {r["to"] for option in scene["options"] for r in option["reactions"]}
    needed = {ids[idx]} if idx < playable else set(ids[playable:])
    require(needed <= targets, f"required route target missing: {', '.join(sorted(needed - targets))}")
    return scene


def evaluate(weights, state):
    return weights[0] + sum(weights[i + 1] * state[key] for i, key in enumerate(STATS))


def rehearse(scenes: list[dict], initial: dict, *, runs: int = 5000, seed: int = 523) -> dict:
    by_id = {s["id"]: s for s in scenes}
    endings, visits, hits, steps = Counter(), Counter(), Counter(), Counter()
    saturated = 0
    rng = Random(seed)
    witnesses, scene_witnesses = {}, {}
    for _ in range(runs):
        state = dict(initial)
        node = scenes[0]
        turn = 0
        path = []
        while node["options"]:
            visits[node["id"]] += 1
            scene_witnesses.setdefault(node["id"], path.copy())
            option = rng.choice(node["options"])
            reply = max(option["reactions"], key=lambda r: evaluate(r["score"], state))
            path.append({"scene": node["id"], "option": option["id"], "reaction": reply["id"]})
            hits[reply["id"]] += 1
            for key, delta in reply["effects"].items():
                x = state[key] + delta
                saturated += not -1 <= x <= 1
                state[key] = max(-1.0, min(1.0, x))
            path[-1]["after"] = dict(state)
            node = by_id[reply["to"]]
            turn += 1
        visits[node["id"]] += 1
        scene_witnesses.setdefault(node["id"], path)
        endings[node["id"]] += 1
        witnesses.setdefault(node["id"], path)
        steps[turn] += 1
    return {"runs": runs, "seed": seed, "endings": dict(sorted(endings.items())),
            "encounter_visits": dict(sorted(visits.items())), "reaction_hits": dict(sorted(hits.items())),
            "lengths": dict(sorted(steps.items())), "clamped_effects": saturated, "witnesses": witnesses,
            "scene_witnesses": scene_witnesses}


def reachable(scenes: list[dict]) -> set[str]:
    seen = {scenes[0]["id"]}
    for scene in scenes:
        if scene["id"] in seen:
            seen.update(r["to"] for r in itertools.chain.from_iterable(o["reactions"] for o in scene["options"]))
    return seen


def swmd_document(card: dict, ifid: str, blocks: list[str]) -> str:
    scalar = lambda value: json.dumps(value, ensure_ascii=False)
    head = ["---", f"title: {scalar(card['title'])}", f"storyworld_id: {scalar(ifid)}",
            'source_format: "SWMD-0-MIN"', f"state_variables: {scalar(list(STATS))}", "endings:"]
    for ending in card["endings"]:
        head.append(f"  - id: {scalar(ending['id'])}")
        for key in ("type", "condition", "expected_critic_score"):
            head.append(f"    {key}: {scalar(ending[key])}")
    return "\n".join([*head, "---", "", "# SWMD-0-MIN", f"id: {ifid}",
                      f"title: {card['title']}", "", "\n\n".join(block.rstrip() for block in blocks), ""])


def sweepweave_document(card: dict, ifid: str, scenes: list[dict]) -> dict:
    def constant(value, kind="String"):
        return {"script_element_type": "Pointer", "pointer_type": f"{kind} Constant", "value": value}

    def pointer(stat, weight=1):
        return {"script_element_type": "Pointer", "pointer_type": "Bounded Number Pointer",
                "character": hero, "keyring": [stat], "coefficient": weight}

    def op(name, *operands):
        return {"script_element_type": "Operator", "operator_type": name, "operands": operands}

    hero = card["cast"][0]["id"]
    truth = constant(True, "Boolean")
    encs = []
    for index, scene in enumerate(scenes):
        options = []
        for choice in scene["options"]:
            reactions = []
            for reaction in choice["reactions"]:
                bias, *weights = reaction["score"]
                terms = [pointer(stat, weight) for stat, weight in zip(STATS, weights, strict=True) if weight]
                if bias:
                    terms.append(constant(bias, "Bounded Number"))
                effects = [{"effect_type": "Bounded Number Effect", "Set": pointer(stat),
                            "to": op("Nudge", pointer(stat), constant(delta, "Bounded Number"))}
                           for stat, delta in reaction["effects"].items()]
                reactions.append({"id": reaction["id"], "text_script": constant(reaction["text"]),
                                  "consequence_id": reaction["to"], "after_effects": effects,
                                  "desirability_script": op("Addition", *terms) if len(terms) > 1 else terms[0]})
            options.append({"id": choice["id"], "text_script": constant(choice["text"]),
                            "visibility_script": truth, "performability_script": truth, "reactions": reactions})
        encs.append({"id": scene["id"], "title": scene["title"], "earliest_turn": 0, "latest_turn": 999,
                     "creation_index": index, "connected_spools": ["spool_main"], "options": options,
                     "text_script": constant(scene["text"]), "acceptability_script": truth,
                     "desirability_script": constant(1 if index == 0 else 0, "Bounded Number")})
    return {"IFID": ifid, "storyworld_title": card["title"], "storyworld_author": "Inkling",
            "sweepweave_version": "0.1.9", "about_text": constant(card["premise"]), "css_theme": "dark",
            "debug_mode": False, "display_mode": "default", "creation_time": 0, "modified_time": 0,
            "characters": [{"id": c["id"], "name": c["name"], "description": c["voice"],
                            "bnumber_properties": card["stats"] if c["id"] == hero else {}}
                           for c in card["cast"]],
            "authored_properties": [{"id": key, "property_name": key, "property_type": "bounded number",
                                     "depth": 0, "default_value": value, "affected_characters": [hero]}
                                    for key, value in card["stats"].items()],
            "spools": [{"id": "spool_main", "spool_name": card["title"], "starts_active": True,
                        "creation_index": 0, "encounters": [s["id"] for s in scenes]}], "encounters": encs}
