"""Typed persona specification: independent validation, MeTTa compilation, minimal patches.

A persona specification is proposed by Inky as JSON. The host validates it here with no
reference to anything Inky asserted about itself. An accepted specification compiles to two
MeTTa-syntax modules:

  * rubric.metta: the exact two_frame_metta_ldt_v1 policy grammar (benchmark, constitution,
    critical-cap, dimension, tag-score, critical-tag), parsed and scored by the vendored,
    hash-pinned `ldt_experiment.parse_policy` / `score_policy`.
  * policy.metta: the persona's research action policy (lattice states and transitions,
    evidence requirements, abstention rule, ranking, repair rules), parsed by
    `parse_policy_module` here and required to round-trip to the specification.

Neither module can name, widen or remove a host boundary tag in a positive role, and no
specification field is an input to the host authorization function (host.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .tags import (ALL_TAGS, BOUNDARY_TAGS, METHOD_TAGS, PERSONA_TAGS, PERSONAS, PHASES,
                   TERMINAL_PHASES)
from .tasks import DOMAINS

DIAGNOSTICS = ("missed_signature", "suboptimal_utility", "overconfident", "underconfident",
               "wrong_abstain", "missed_abstain", "vetoed_best", "lattice_blocked_best")
PATCH_KINDS = ("raise_weight", "lower_weight", "add_positive_tag", "add_negative_tag", "remove_tag",
               "raise_abstain_threshold", "lower_abstain_threshold", "add_transition",
               "remove_transition", "add_critical_tag", "remove_critical_tag")
METRICS = ("signature_rate", "abstain_rate", "mean_utility", "high_cost_rate", "intervention_rate")
TOP_KEYS = ("persona", "summary", "preference_features", "critical_tags", "action_ranking", "lattice",
            "evidence_requirements", "abstention", "repair_rules", "ambiguity_behavior", "predictions")
FORBIDDEN_KEYS = {"authorize", "authorized", "authorization", "permit", "permits", "permitted",
                  "allow", "allowed", "allowed_actions", "forbidden_tags", "boundary", "override",
                  "host", "receipt", "receipts", "epoch", "snapshot", "approve", "approved", "gate"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
CRITICAL_CAP = 0.2
TRAIN_MINIMUM = 0.6
REJECT_MAXIMUM = 0.35
MAX_SPEC_CHARS = 9000
MAX_PATCH_OPS = 3


class SpecError(ValueError):
    pass


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(canonical(spec).encode()).hexdigest()


def _walk_keys(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(spec: Any, persona: str | None = None) -> list[str]:
    """Independent structural, vocabulary, lattice and boundary validation. Empty list = valid."""
    e: list[str] = []
    if not isinstance(spec, dict):
        return ["specification must be a JSON object"]
    try:
        if len(canonical(spec)) > MAX_SPEC_CHARS:
            e.append(f"specification exceeds {MAX_SPEC_CHARS} characters")
    except (TypeError, ValueError):
        return ["specification is not JSON-serializable"]
    for path, k in _walk_keys(spec):
        if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
            e.append(f"forbidden authorization-like key {k!r} at {path}")
    missing = [k for k in TOP_KEYS if k not in spec]
    extra = [k for k in spec if k not in TOP_KEYS]
    if missing:
        e.append(f"missing keys: {missing}")
    if extra:
        e.append(f"unknown keys: {extra}")
    if missing:
        return e
    if spec["persona"] not in PERSONAS:
        e.append("persona must be one of the eight registered personas")
    if persona is not None and spec["persona"] != persona:
        e.append(f"persona must be {persona!r}")
    if not isinstance(spec["summary"], str) or not 10 <= len(spec["summary"]) <= 600:
        e.append("summary must be a string of 10..600 characters")

    feats = spec["preference_features"]
    seen_tags: dict[str, str] = {}
    names = set()
    if not isinstance(feats, list) or not 1 <= len(feats) <= 10:
        e.append("preference_features must be a list of 1..10 features")
        feats = []
    for i, f in enumerate(feats):
        if not isinstance(f, dict) or set(f) != {"name", "weight", "positive_tags", "negative_tags"}:
            e.append(f"feature {i} needs exactly name, weight, positive_tags, negative_tags")
            continue
        if not isinstance(f["name"], str) or not NAME_RE.match(f["name"]) or f["name"] in names:
            e.append(f"feature {i} name invalid or duplicate")
        names.add(f["name"])
        if not _num(f["weight"]) or not 0 < f["weight"] <= 10:
            e.append(f"feature {i} weight must be a number in (0, 10]")
        for sign in ("positive_tags", "negative_tags"):
            tags = f[sign]
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                e.append(f"feature {i} {sign} must be a list of tags")
                continue
            for t in tags:
                if t in BOUNDARY_TAGS and sign == "positive_tags":
                    e.append(f"boundary tag {t!r} cannot be scored positively")
                elif t not in ALL_TAGS:
                    e.append(f"unknown tag {t!r}")
                if t in seen_tags:
                    e.append(f"tag {t!r} appears in more than one place")
                seen_tags[t] = f.get("name", "?")
    if feats and not any(f.get("positive_tags") for f in feats if isinstance(f, dict)):
        e.append("at least one feature needs a positive tag")

    crit = spec["critical_tags"]
    if not isinstance(crit, list) or not all(isinstance(t, str) and t in ALL_TAGS for t in crit):
        e.append("critical_tags must be a list of known tags")
    elif len(set(crit)) != len(crit):
        e.append("critical_tags has duplicates")
    else:
        for t in crit:
            for f in feats:
                if isinstance(f, dict) and t in (f.get("positive_tags") or []):
                    e.append(f"critical tag {t!r} cannot also be a positive tag")

    rank = spec["action_ranking"]
    if (not isinstance(rank, list) or not 3 <= len(rank) <= len(METHOD_TAGS)
            or not all(isinstance(t, str) and t in METHOD_TAGS for t in rank) or len(set(rank)) != len(rank)):
        e.append("action_ranking must list 3+ distinct method tags")

    lat = spec["lattice"]
    if not isinstance(lat, dict) or set(lat) != {"states", "initial", "transitions"}:
        e.append("lattice needs exactly states, initial, transitions")
    else:
        st = lat["states"]
        if not isinstance(st, list) or not all(s in PHASES for s in st) or len(set(st)) != len(st) or len(st) < 3:
            e.append("lattice.states must be 3+ distinct registered phases")
        elif lat["initial"] not in st:
            e.append("lattice.initial must be a state")
        else:
            tr = lat["transitions"]
            if (not isinstance(tr, list) or not tr or not all(isinstance(p, list) and len(p) == 2 for p in tr)
                    or not all(a in st and b in st for a, b in tr)):
                e.append("lattice.transitions must be [from, to] pairs over states")
            else:
                reach = _reachable(lat["initial"], tr)
                if "abstain" not in reach:
                    e.append("abstain must be reachable from the initial state")
                if not any(t in reach for t in TERMINAL_PHASES):
                    e.append("a terminal phase must be reachable")

    ev = spec["evidence_requirements"]
    if (not isinstance(ev, dict) or set(ev) != {"min_evidence_items", "abstain_on_missing_inputs", "require_failure_log_for_repair"}):
        e.append("evidence_requirements needs min_evidence_items, abstain_on_missing_inputs, require_failure_log_for_repair")
    else:
        if not isinstance(ev["min_evidence_items"], int) or isinstance(ev["min_evidence_items"], bool) or not 0 <= ev["min_evidence_items"] <= 5:
            e.append("min_evidence_items must be an integer 0..5")
        if not isinstance(ev["abstain_on_missing_inputs"], bool) or not isinstance(ev["require_failure_log_for_repair"], bool):
            e.append("evidence flags must be booleans")

    ab = spec["abstention"]
    if not isinstance(ab, dict) or set(ab) != {"min_train_score", "abstain_if_no_ranked_action"}:
        e.append("abstention needs min_train_score and abstain_if_no_ranked_action")
    else:
        if not _num(ab["min_train_score"]) or not 0 <= ab["min_train_score"] <= 0.95:
            e.append("min_train_score must be in [0, 0.95]")
        if not isinstance(ab["abstain_if_no_ranked_action"], bool):
            e.append("abstain_if_no_ranked_action must be boolean")

    rr = spec["repair_rules"]
    if not isinstance(rr, list) or not 1 <= len(rr) <= 12:
        e.append("repair_rules must list 1..12 rules")
    else:
        for i, r in enumerate(rr):
            if (not isinstance(r, dict) or set(r) != {"diagnostic", "patch", "target", "max_step"}
                    or r["diagnostic"] not in DIAGNOSTICS or r["patch"] not in PATCH_KINDS
                    or not isinstance(r["target"], str) or len(r["target"]) > 40
                    or not _num(r["max_step"]) or not 0 < r["max_step"] <= 0.5):
                e.append(f"repair rule {i} invalid (diagnostic, patch, target, max_step in (0,0.5])")

    amb = spec["ambiguity_behavior"]
    if (not isinstance(amb, list) or len(amb) > 6 or
            not all(isinstance(a, dict) and set(a) == {"situation", "expected_action"}
                    and all(isinstance(v, str) and len(v) <= 300 for v in a.values()) for a in amb)):
        e.append("ambiguity_behavior must be up to 6 {situation, expected_action} strings")

    preds = spec["predictions"]
    if not isinstance(preds, list) or not 2 <= len(preds) <= 6:
        e.append("predictions must list 2..6 falsifiable predictions")
    else:
        ids = set()
        for i, p in enumerate(preds):
            ok = (isinstance(p, dict) and set(p) == {"id", "statement", "metric", "comparator", "threshold", "domain"}
                  and isinstance(p["id"], str) and p["id"] not in ids and isinstance(p["statement"], str)
                  and len(p["statement"]) <= 300 and p["metric"] in METRICS and p["comparator"] in (">=", "<=")
                  and _num(p["threshold"]) and 0 <= p["threshold"] <= 1 and (p["domain"] == "all" or p["domain"] in DOMAINS))
            if not ok:
                e.append(f"prediction {i} invalid (metric in {METRICS}, comparator >= or <=, threshold in [0,1], domain 'all' or a task domain)")
            else:
                ids.add(p["id"])
    return e


def _reachable(initial: str, transitions) -> set:
    seen, todo = {initial}, [initial]
    while todo:
        s = todo.pop()
        for a, b in transitions:
            if a == s and b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


# --------------------------------------------------------------------------- compilation
def normalized_weights(spec: dict) -> list[tuple[str, float]]:
    feats = spec["preference_features"]
    total = sum(float(f["weight"]) for f in feats)
    out = [(f["name"], round(float(f["weight"]) / total, 12)) for f in feats]
    head = sum(w for _, w in out[:-1])
    out[-1] = (out[-1][0], 1.0 - head)
    return out


def compile_rubric(spec: dict, version: str) -> str:
    lines = [f"; JINN-GYM-01 persona rubric {spec['persona']} {version}",
             "(benchmark jinn_gym_01_research_persona)",
             f"(constitution {spec['persona']}_{version})",
             f"(critical-cap {CRITICAL_CAP})"]
    for name, w in normalized_weights(spec):
        lines.append(f"(dimension {name} {w!r})")
    scored = set()
    for f in spec["preference_features"]:
        for t in f["positive_tags"]:
            lines.append(f"(tag-score {t} {f['name']} 1)")
            scored.add(t)
        for t in f["negative_tags"]:
            lines.append(f"(tag-score {t} {f['name']} -1)")
            scored.add(t)
    first = spec["preference_features"][0]["name"]
    for t in spec["critical_tags"]:
        if t not in scored:
            lines.append(f"(tag-score {t} {first} -1)")
        lines.append(f"(critical-tag {t})")
    return "\n".join(lines) + "\n"


def compile_policy(spec: dict) -> str:
    lat, ev, ab = spec["lattice"], spec["evidence_requirements"], spec["abstention"]
    b = lambda x: "true" if x else "false"  # noqa: E731
    lines = [f"; JINN-GYM-01 persona action policy {spec['persona']}",
             f"(persona {spec['persona']})"]
    lines += [f"(lattice-state {s})" for s in lat["states"]]
    lines.append(f"(lattice-initial {lat['initial']})")
    lines += [f"(lattice-transition {a} {c})" for a, c in lat["transitions"]]
    lines.append(f"(evidence-min {ev['min_evidence_items']})")
    lines.append(f"(abstain-on-missing-inputs {b(ev['abstain_on_missing_inputs'])})")
    lines.append(f"(require-failure-log-for-repair {b(ev['require_failure_log_for_repair'])})")
    lines.append(f"(abstain-min-train {float(ab['min_train_score'])!r})")
    lines.append(f"(abstain-if-no-ranked-action {b(ab['abstain_if_no_ranked_action'])})")
    lines += [f"(rank {i} {t})" for i, t in enumerate(spec["action_ranking"])]
    for r in spec["repair_rules"]:
        target = re.sub(r"[^A-Za-z0-9_]", "_", r["target"]) or "_"
        lines.append(f"(repair-rule {r['diagnostic']} {r['patch']} {target} {float(r['max_step'])!r})")
    return "\n".join(lines) + "\n"


FORM = re.compile(r"^\(([a-z-]+)((?:\s+[^\s()]+)*)\)$")


def parse_policy_module(text: str) -> dict:
    out = {"states": [], "initial": None, "transitions": [], "ranking": [], "repair_rules": []}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        m = FORM.match(line)
        if not m:
            raise SpecError(f"policy.metta:{n}: unsupported syntax")
        form, toks = m.group(1), m.group(2).split()
        if form == "persona":
            out["persona"] = toks[0]
        elif form == "lattice-state":
            out["states"].append(toks[0])
        elif form == "lattice-initial":
            out["initial"] = toks[0]
        elif form == "lattice-transition":
            out["transitions"].append([toks[0], toks[1]])
        elif form == "evidence-min":
            out["min_evidence_items"] = int(toks[0])
        elif form in ("abstain-on-missing-inputs", "require-failure-log-for-repair", "abstain-if-no-ranked-action"):
            if toks[0] not in ("true", "false"):
                raise SpecError(f"policy.metta:{n}: boolean expected")
            out[form.replace("-", "_")] = toks[0] == "true"
        elif form == "abstain-min-train":
            out["min_train_score"] = float(toks[0])
        elif form == "rank":
            out["ranking"].append(toks[1])
        elif form == "repair-rule":
            out["repair_rules"].append(toks)
        else:
            raise SpecError(f"policy.metta:{n}: unknown form {form!r}")
    return out


def policy_round_trips(spec: dict, parsed: dict) -> bool:
    lat, ev, ab = spec["lattice"], spec["evidence_requirements"], spec["abstention"]
    return (parsed.get("persona") == spec["persona"] and parsed["states"] == lat["states"]
            and parsed["initial"] == lat["initial"] and parsed["transitions"] == [list(p) for p in lat["transitions"]]
            and parsed["min_evidence_items"] == ev["min_evidence_items"]
            and parsed["abstain_on_missing_inputs"] == ev["abstain_on_missing_inputs"]
            and parsed["require_failure_log_for_repair"] == ev["require_failure_log_for_repair"]
            and abs(parsed["min_train_score"] - float(ab["min_train_score"])) < 1e-12
            and parsed["abstain_if_no_ranked_action"] == ab["abstain_if_no_ranked_action"]
            and parsed["ranking"] == spec["action_ranking"] and len(parsed["repair_rules"]) == len(spec["repair_rules"]))


# --------------------------------------------------------------------------- patches
PATCH_OPS = ("set_weight", "add_tag", "remove_tag", "set_abstain_threshold", "set_abstain_if_no_ranked_action",
             "set_evidence", "add_transition", "remove_transition", "add_critical", "remove_critical",
             "set_ranking", "add_feature")


def apply_patch(spec: dict, ops: Any) -> dict:
    """Apply at most MAX_PATCH_OPS whitelisted operations. Raises SpecError on any invalid op.
    The result must still pass validate(); callers check that separately."""
    if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_PATCH_OPS:
        raise SpecError(f"a patch is a list of 1..{MAX_PATCH_OPS} operations")
    s = copy.deepcopy(spec)
    feats = {f["name"]: f for f in s["preference_features"]}
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in PATCH_OPS:
            raise SpecError(f"op {i}: unknown operation")
        for path, k in _walk_keys(op):
            if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                raise SpecError(f"op {i}: forbidden key {k!r}")
        kind = op["op"]
        try:
            if kind == "set_weight":
                feats[op["feature"]]["weight"] = float(op["weight"])
            elif kind == "add_feature":
                if op["name"] in feats:
                    raise SpecError(f"op {i}: feature exists")
                f = {"name": op["name"], "weight": float(op["weight"]), "positive_tags": [], "negative_tags": []}
                s["preference_features"].append(f)
                feats[f["name"]] = f
            elif kind == "add_tag":
                t, sign = op["tag"], op.get("sign", "+")
                for f in s["preference_features"]:
                    for key in ("positive_tags", "negative_tags"):
                        if t in f[key]:
                            f[key].remove(t)
                feats[op["feature"]]["positive_tags" if sign == "+" else "negative_tags"].append(t)
            elif kind == "remove_tag":
                hit = False
                for f in s["preference_features"]:
                    for key in ("positive_tags", "negative_tags"):
                        if op["tag"] in f[key]:
                            f[key].remove(op["tag"])
                            hit = True
                if not hit:
                    raise SpecError(f"op {i}: tag not present")
            elif kind == "set_abstain_threshold":
                s["abstention"]["min_train_score"] = float(op["value"])
            elif kind == "set_abstain_if_no_ranked_action":
                s["abstention"]["abstain_if_no_ranked_action"] = bool(op["value"])
            elif kind == "set_evidence":
                if op["field"] not in s["evidence_requirements"]:
                    raise SpecError(f"op {i}: unknown evidence field")
                s["evidence_requirements"][op["field"]] = op["value"]
            elif kind == "add_transition":
                pair = [op["from"], op["to"]]
                for st in pair:
                    if st not in s["lattice"]["states"]:
                        s["lattice"]["states"].append(st)
                if pair not in s["lattice"]["transitions"]:
                    s["lattice"]["transitions"].append(pair)
            elif kind == "remove_transition":
                pair = [op["from"], op["to"]]
                if pair not in s["lattice"]["transitions"]:
                    raise SpecError(f"op {i}: transition not present")
                s["lattice"]["transitions"].remove(pair)
            elif kind == "add_critical":
                if op["tag"] not in s["critical_tags"]:
                    s["critical_tags"].append(op["tag"])
            elif kind == "remove_critical":
                s["critical_tags"].remove(op["tag"])
            elif kind == "set_ranking":
                s["action_ranking"] = list(op["ranking"])
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, SpecError):
                raise
            raise SpecError(f"op {i}: malformed {kind}: {type(exc).__name__}") from None
    return s


def schema_prompt_block() -> str:
    """Compact description of the schema for Inky prompts."""
    return (
        "JSON object with exactly these keys:\n"
        '  "persona": the persona id given below\n'
        '  "summary": 1-3 sentences\n'
        '  "preference_features": 2-6 objects {"name": snake_case, "weight": number in (0,10], '
        '"positive_tags": [tags], "negative_tags": [tags]}; each tag may appear only once in the whole list\n'
        '  "critical_tags": tags this persona refuses outright (may be empty)\n'
        '  "action_ranking": 4-10 method tags from most to least preferred\n'
        '  "lattice": {"states": phases, "initial": phase, "transitions": [[from, to], ...]}; '
        '"abstain" and one of "commit"/"abstain" must be reachable from initial\n'
        '  "evidence_requirements": {"min_evidence_items": integer 0-5, "abstain_on_missing_inputs": bool, '
        '"require_failure_log_for_repair": bool}\n'
        '  "abstention": {"min_train_score": number in [0,0.95], "abstain_if_no_ranked_action": bool}\n'
        f'  "repair_rules": 1-6 objects {{"diagnostic": one of {list(DIAGNOSTICS)}, "patch": one of {list(PATCH_KINDS)}, '
        '"target": feature name or tag, "max_step": number in (0,0.5]}\n'
        '  "ambiguity_behavior": up to 4 objects {"situation": text, "expected_action": text}\n'
        f'  "predictions": 2-4 objects {{"id": short id, "statement": text, "metric": one of {list(METRICS)}, '
        '"comparator": ">=" or "<=", "threshold": number in [0,1], "domain": "all" or a task domain}\n'
        "Metric definitions: signature_rate = share of tasks, among those offering one, where the executed "
        "action shows this persona's signature behavior; abstain_rate = share of tasks where it abstains; "
        "mean_utility = mean verified utility (0-1) of executed actions; high_cost_rate = share of executed "
        "actions tagged high_cost; intervention_rate = share of tasks where the host had to repair an "
        "unauthorized proposal (not the persona's use of causal interventions).\n"
        f"Task domains: {list(DOMAINS)}. Phases: {list(PHASES)}.\n"
        "Never use these host-boundary tags positively: " + ", ".join(BOUNDARY_TAGS) + ".\n"
        "Do not include any key about authorization, permission, approval or the host boundary: "
        "authorization is owned by the host and is not part of a persona."
    )


def write_modules(spec: dict, version: str, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rp = directory / "rubric.metta"
    pp = directory / "policy.metta"
    rp.write_text(compile_rubric(spec, version), encoding="utf-8", newline="\n")
    pp.write_text(compile_policy(spec), encoding="utf-8", newline="\n")
    return rp, pp


PERSONA_SCORABLE = PERSONA_TAGS
