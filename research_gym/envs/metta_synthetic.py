from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from research_gym.core.metta_frames import DeductionFrame, ExecutionFrame, RepairFrame, RoutingFrame, AnyMettaFrame
from research_gym.core.typed_soundness import SoundnessType


DEFAULT_DIRECTIVES = """
# Synthetic MeTTa-like frame directives. These are not a real MeTTa parser.
!exec rule=investigate before=trust:1,evidence:0,heat:1,scene:2 after=trust:1,evidence:1,heat:2,scene:3 invariants=bounds,scene_progress
!exec rule=befriend before=trust:0,evidence:1,heat:2,scene:1 after=trust:1,evidence:1,heat:1,scene:2 invariants=bounds,scene_progress
!exec rule=arith_add_heat before=heat:1,delta:2 after=heat:3,delta:2 invariants=arithmetic,bounds
!exec rule=compare_heat_gate before=heat:4,threshold:3 after=heat:4,threshold:3,gate:1 invariants=comparison,boolean_gate
!exec rule=precondition_fail_defuse before=heat:1,scene:2 after=heat:1,scene:2 invariants=precondition_failed,no_state_change passed=false
!exec rule=postcondition_update_scene before=scene:2,complete:0 after=scene:3,complete:1 invariants=postcondition_update,scene_progress
!deduce rule=type_check_action before=action:{befriend,defuse,42};type:{Action,Number} after=action:{befriend,defuse};type:{Action} eliminated=action:42,type:Number conflict=false soundness=env_sound_dead
!deduce rule=deduction_closure before=route:{ally,solo,rush};heat:{0,1,2,3};safe:{yes,no} after=route:{ally,solo};heat:{0,1,2};safe:{yes} eliminated=route:rush,heat:3,safe:no conflict=false soundness=env_sound_dead
!deduce rule=bottom_conflict before=choice:{left,right};requires:{key};has_key:{no} after=choice:{};requires:{key};has_key:{no} eliminated=choice:left,choice:right conflict=true soundness=env_sound_dead
!deduce rule=heat_conflict before=secret:{possible,dead};heat:{3,4};scene:{4,5} after=secret:{dead};heat:{3,4};scene:{4,5} eliminated=secret:possible conflict=true soundness=env_sound_dead
!deduce rule=replay_prune before=path:{a,b,c};skill:{rush,defuse} after=path:{a,b};skill:{defuse} eliminated=path:c,skill:rush conflict=false soundness=experience_sound_dead
!route task=secret_ending candidates=befriend,investigate,defuse,rush chosen=defuse reason=heat_too_high
!route task=typed_arithmetic candidates=arith_add_heat,compare_heat_gate,type_check_action chosen=arith_add_heat reason=integer_state_update
!route task=failed_precondition candidates=precondition_fail_defuse,postcondition_update_scene,deduction_closure chosen=precondition_fail_defuse reason=guard_blocks_transition
!repair diagnostic=unknown_symbol buggy="(call rush-fast state)" repaired="(call rush state)" passed=true
!repair diagnostic=malformed_rule buggy="(= (can-defuse $s) (and (Heat $s $h) (< $h)))" repaired="(= (can-defuse $s) (and (Heat $s $h) (> $h 1)))" passed=true
""".strip()


def _parse_kv_tokens(line: str) -> Dict[str, str]:
    tokens = shlex.split(line)
    if not tokens or not tokens[0].startswith("!"):
        raise ValueError(f"Directive must start with !kind: {line}")
    data = {"kind": tokens[0][1:]}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"Bad token without '=': {token}")
        key, value = token.split("=", 1)
        data[key] = value
    return data


def _parse_state(text: str) -> Dict[str, int]:
    if not text:
        return {}
    out: Dict[str, int] = {}
    for item in text.split(","):
        key, value = item.split(":", 1)
        out[key] = int(value)
    return out


def _parse_list(text: str) -> List[str]:
    if not text:
        return []
    return [x for x in text.split(",") if x]


def _parse_lattice(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not text:
        return out
    for item in text.split(";"):
        key, rest = item.split(":", 1)
        rest = rest.strip()
        if not (rest.startswith("{") and rest.endswith("}")):
            raise ValueError(f"Expected set syntax key:{{a,b}}, got {item}")
        out[key] = _parse_list(rest[1:-1])
    return out


def _parse_eliminated(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not text:
        return out
    for item in text.split(","):
        key, value = item.split(":", 1)
        out.setdefault(key, []).append(value)
    return out


def frames_from_directives(text: str, *, source: str = "synthetic") -> List[AnyMettaFrame]:
    frames: List[AnyMettaFrame] = []
    n = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        data = _parse_kv_tokens(line)
        kind = data["kind"]
        frame_id = f"{source}-{n:05d}"
        n += 1
        if kind == "exec":
            frames.append(
                ExecutionFrame(
                    frame_id=frame_id,
                    source=source,
                    rule_name=data["rule"],
                    before=_parse_state(data.get("before", "")),
                    after=_parse_state(data.get("after", "")),
                    operation=data["rule"],
                    invariants=_parse_list(data.get("invariants", "")),
                    passed=data.get("passed", "true").lower() == "true",
                )
            )
        elif kind == "deduce":
            frames.append(
                DeductionFrame(
                    frame_id=frame_id,
                    source=source,
                    rule_name=data["rule"],
                    before=_parse_lattice(data.get("before", "")),
                    after=_parse_lattice(data.get("after", "")),
                    eliminated=_parse_eliminated(data.get("eliminated", "")),
                    conflict=data.get("conflict", "false").lower() == "true",
                    soundness=SoundnessType(data.get("soundness", SoundnessType.ENV_SOUND_DEAD.value)),
                )
            )
        elif kind == "route":
            frames.append(
                RoutingFrame(
                    frame_id=frame_id,
                    source=source,
                    task_signature={"task": data["task"]},
                    candidate_skills=_parse_list(data.get("candidates", "")),
                    chosen_skill=data["chosen"],
                    reason=data.get("reason", ""),
                )
            )
        elif kind == "repair":
            frames.append(
                RepairFrame(
                    frame_id=frame_id,
                    source=source,
                    diagnostic=data["diagnostic"],
                    buggy_code=data["buggy"],
                    repaired_code=data["repaired"],
                    passed_after_repair=data.get("passed", "false").lower() == "true",
                )
            )
        else:
            raise ValueError(f"Unknown directive kind: {kind}")
    return frames


def frames_from_file(path: Path) -> List[AnyMettaFrame]:
    return frames_from_directives(path.read_text(encoding="utf-8"), source=path.stem)


def default_frames() -> List[AnyMettaFrame]:
    return frames_from_directives(DEFAULT_DIRECTIVES, source="default_metta")
