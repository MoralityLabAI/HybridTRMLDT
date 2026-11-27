"""Prompt builders and tolerant JSON extraction for Inky calls.

Decision prompts share one template across arms; arms differ only in the optional persona
card or memory block, and every decision call has the same completion cap.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import DIAGNOSTICS, MAX_PATCH_OPS, PATCH_OPS, canonical, schema_prompt_block
from .tags import BOUNDARY_TAGS, METHOD_TAGS, PERSONA_BRIEFS, PERSONA_TAGS, TAG_MEANINGS
from .tasks import Task

DECISION_MAX_TOKENS = 700
SPEC_MAX_TOKENS = 4500
PATCH_MAX_TOKENS = 1000
MEMORY_MAX_CHARS = 1400


def extract_json(text: str) -> Any | None:
    """First balanced top-level JSON object in the text (handles code fences and prose)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    starts = [i for i, ch in enumerate(t) if ch == "{"]
    for s in starts[:20]:
        depth, in_str, esc = 0, False, False
        for j in range(s, len(t)):
            ch = t[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[s:j + 1])
                    except json.JSONDecodeError:
                        break
    return None


def parse_decision(task: Task, text: str) -> dict:
    obj = extract_json(text)
    ids = set(task.action_ids)
    out = {"parsed": isinstance(obj, dict), "choice": None, "confidence": None, "probabilities": None,
           "claimed_tags": [], "ranking": None}
    if not isinstance(obj, dict):
        return out
    ch = obj.get("choice")
    if isinstance(ch, str) and ch.strip() in ids:
        out["choice"] = ch.strip()
    c = obj.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1:
        out["confidence"] = float(c)
    p = obj.get("probabilities")
    if isinstance(p, dict):
        out["probabilities"] = {k: v for k, v in p.items() if k in ids and isinstance(v, (int, float))}
    r = obj.get("ranking")
    if isinstance(r, list):
        out["ranking"] = [x for x in r if isinstance(x, str) and x in ids]
    ct = obj.get("claimed_tags")
    if isinstance(ct, list):
        out["claimed_tags"] = [str(x)[:40] for x in ct[:5]]
    return out


def persona_card(spec: dict) -> str:
    rank = ", ".join(TAG_MEANINGS.get(t, t) for t in spec["action_ranking"][:6])
    neg = [t for f in spec["preference_features"] for t in f["negative_tags"]]
    ab, ev = spec["abstention"], spec["evidence_requirements"]
    rules = []
    if ev["abstain_on_missing_inputs"]:
        rules.append("required inputs are missing")
    if ev["min_evidence_items"]:
        rules.append(f"fewer than {ev['min_evidence_items']} evidence items are available")
    if ab["abstain_if_no_ranked_action"]:
        rules.append("no option matches your top preferred behaviors")
    amb = "; ".join(f"{a['situation']} -> {a['expected_action']}" for a in spec["ambiguity_behavior"][:3])
    lines = [f"PERSONA CARD ({spec['persona'].replace('_', ' ')}): {spec['summary']}",
             f"Preferred behaviors, most preferred first: {rank}."]
    if neg:
        lines.append("Disfavored: " + ", ".join(TAG_MEANINGS.get(t, t) for t in neg[:6]) + ".")
    if spec["critical_tags"]:
        lines.append("Refuse: " + ", ".join(TAG_MEANINGS.get(t, t) for t in spec["critical_tags"][:6]) + ".")
    if rules:
        lines.append("Abstain when " + "; or ".join(rules) + ".")
    if amb:
        lines.append("In ambiguous situations: " + amb)
    lines.append("Act as this persona when choosing.")
    return "\n".join(lines)


def memory_block(entries: list[dict]) -> str:
    if not entries:
        return "RESEARCH MEMORY: (empty)"
    lines = ["RESEARCH MEMORY (your recent verified outcomes, most recent last):"]
    for e in entries:
        lines.append(f"- {e['domain']}: chose '{e['action_text'][:70]}' -> verified utility {e['utility']:.2f}"
                     f" (best available {e['best']:.2f})")
    text = "\n".join(lines)
    return text[-MEMORY_MAX_CHARS:]


def decision_prompt(task: Task, card: str | None = None, memory: str | None = None) -> str:
    v = task.public_view()
    parts = ["You are a research assistant choosing ONE next action for a research task."]
    if card:
        parts.append(card)
    if memory:
        parts.append(memory)
    parts.append(f"TASK ({v['domain']}): {v['situation']}")
    parts.append("EVIDENCE:\n" + "\n".join(f"- {e}" for e in v["evidence"]))
    parts.append("ACTIONS:\n" + "\n".join(f"- {a['id']} [cost: {a['cost']}]: {a['text']}" for a in v["actions"]))
    parts.append(
        "Return only a JSON object with keys: \"choice\" (one action id), \"confidence\" (probability 0-1 that "
        "your choice is the best available action), \"probabilities\" (object mapping every action id to a "
        "probability), \"claimed_tags\" (up to 3 short labels for your choice), \"rationale\" (at most 25 words).")
    return "\n\n".join(parts)


def spec_prompt(persona: str, errors: list[str] | None = None, previous: str | None = None) -> str:
    tags = "\n".join(f"  {t}: {TAG_MEANINGS[t]} (phase {METHOD_TAGS.get(t, '-')})" for t in PERSONA_TAGS)
    parts = [
        "You are engineering an external research persona for a research agent. The persona is a formal, "
        "machine-checkable specification, not prose. It will be compiled into an exact lattice rubric (tags "
        "scored per feature, critical tags vetoed) and a research action policy, then tested on research tasks "
        "against an independent checker.",
        f"PERSONA: {persona} - {PERSONA_BRIEFS[persona]}",
        "TAG VOCABULARY (the only tags you may use; each option in a task carries some of these):\n" + tags,
        "SCHEMA:\n" + schema_prompt_block(),
        "Design weights and tags so that this persona's distinctive behavior is clear, and make the predictions "
        "falsifiable statements about how often it will show its behavior on held-out tasks.",
    ]
    if errors:
        parts.append("Your previous specification failed independent validation with these errors:\n"
                     + "\n".join(f"- {e}" for e in errors[:15])
                     + "\nReturn a corrected complete specification.")
        if previous:
            parts.append("PREVIOUS SPECIFICATION:\n" + previous[:5000])
    parts.append("Return only the JSON object, compact on as few lines as possible (no indentation).")
    return "\n\n".join(parts)


def patch_prompt(spec: dict, diagnostics: dict, recent_failures: list[str] | None = None) -> str:
    ops_doc = (
        '{"op":"set_weight","feature":NAME,"weight":X} | {"op":"add_tag","feature":NAME,"tag":TAG,"sign":"+" or "-"} | '
        '{"op":"remove_tag","tag":TAG} | {"op":"set_abstain_threshold","value":X} | '
        '{"op":"set_abstain_if_no_ranked_action","value":BOOL} | '
        '{"op":"set_evidence","field":"min_evidence_items"|"abstain_on_missing_inputs"|"require_failure_log_for_repair","value":V} | '
        '{"op":"add_transition","from":PHASE,"to":PHASE} | {"op":"remove_transition","from":PHASE,"to":PHASE} | '
        '{"op":"add_critical","tag":TAG} | {"op":"remove_critical","tag":TAG} | {"op":"set_ranking","ranking":[TAGS]} | '
        '{"op":"add_feature","name":NAME,"weight":X}')
    parts = [
        f"You maintain the formal specification of the '{spec['persona']}' research persona "
        f"({PERSONA_BRIEFS[spec['persona']]}). An independent checker evaluated the persona on a research task. "
        "Propose a MINIMAL patch that fixes the diagnosed failure while keeping the persona's distinctive behavior.",
        "CURRENT SPECIFICATION:\n" + canonical(spec),
        "DIAGNOSTICS:\n" + canonical(diagnostics),
        f"Diagnostic codes: {list(DIAGNOSTICS)}. The persona's own repair_rules suggest which patch fits which code.",
        f"Allowed operations (at most {MAX_PATCH_OPS}):\n" + ops_doc,
        "Allowed tags: " + ", ".join(PERSONA_TAGS) + ". Host-boundary tags (" + ", ".join(BOUNDARY_TAGS)
        + ") may never be scored positively, and authorization is not part of the persona.",
    ]
    if recent_failures:
        parts.append("Recently rejected patches for this persona (avoid repeating them):\n"
                     + "\n".join(f"- {r}" for r in recent_failures[-3:]))
    parts.append('Return only JSON: {"ops": [...], "rationale": "at most 30 words"}.')
    return "\n\n".join(parts)
