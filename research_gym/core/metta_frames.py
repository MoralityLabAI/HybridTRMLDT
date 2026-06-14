from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .frames import Frame
from .typed_soundness import SoundnessType


class MettaFrameKind(str, Enum):
    """Local frame kinds emitted from MeTTa-like code or mechanics.

    These frames are deliberately local. They do not assume solved rollouts.
    """

    EXECUTION = "execution"
    DEDUCTION = "deduction"
    REPAIR = "repair"
    ROUTING = "routing"


@dataclass(frozen=True)
class ExecutionFrame:
    """A local transition frame: code/mechanic plus before/after state."""

    frame_id: str
    source: str
    rule_name: str
    before: Mapping[str, int]
    after: Mapping[str, int]
    operation: str
    invariants: List[str] = field(default_factory=list)
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: MettaFrameKind = MettaFrameKind.EXECUTION

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    def to_frame(self) -> Frame:
        return Frame(
            id=self.frame_id,
            family=self.kind.value,
            source=self.source,
            input_state=dict(self.before),
            operation={"rule_name": self.rule_name, "operation": self.operation},
            output_state=dict(self.after),
            soundness_type=SoundnessType.UNKNOWN,
            label="passed" if self.passed else "failed",
            metadata={"invariants": list(self.invariants), **self.metadata},
        )


@dataclass(frozen=True)
class DeductionFrame:
    """A local monotone-pruning frame for LDT-style heads.

    `before` and `after` are serialized candidate lattices. The expected shape is
    slot -> list[str] or slot -> scalar for simple toy states. This keeps the
    file dependency-free while allowing later replacement by real lattice types.
    """

    frame_id: str
    source: str
    rule_name: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    eliminated: Mapping[str, List[str]] = field(default_factory=dict)
    conflict: bool = False
    soundness: SoundnessType = SoundnessType.ENV_SOUND_DEAD
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: MettaFrameKind = MettaFrameKind.DEDUCTION

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["soundness"] = self.soundness.value
        return data

    def to_frame(self) -> Frame:
        return Frame(
            id=self.frame_id,
            family=self.kind.value,
            source=self.source,
            input_state=dict(self.before),
            operation={"rule_name": self.rule_name, "eliminated": dict(self.eliminated)},
            output_state=dict(self.after),
            soundness_type=self.soundness,
            label="conflict" if self.conflict else "refinement",
            metadata={"conflict": self.conflict, **self.metadata},
        )


@dataclass(frozen=True)
class RepairFrame:
    """A local code-repair frame: diagnostic plus buggy/fixed snippet."""

    frame_id: str
    source: str
    diagnostic: str
    buggy_code: str
    repaired_code: str
    passed_after_repair: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: MettaFrameKind = MettaFrameKind.REPAIR

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    def to_frame(self) -> Frame:
        return Frame(
            id=self.frame_id,
            family=self.kind.value,
            source=self.source,
            input_state={"buggy_code": self.buggy_code},
            operation={"diagnostic": self.diagnostic},
            output_state={"repaired_code": self.repaired_code},
            soundness_type=SoundnessType.UNKNOWN,
            label="passed" if self.passed_after_repair else "failed",
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class RoutingFrame:
    """A skill-routing frame: task signature plus candidate skill choices."""

    frame_id: str
    source: str
    task_signature: Mapping[str, Any]
    candidate_skills: List[str]
    chosen_skill: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    kind: MettaFrameKind = MettaFrameKind.ROUTING

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    def to_frame(self) -> Frame:
        return Frame(
            id=self.frame_id,
            family=self.kind.value,
            source=self.source,
            input_state=dict(self.task_signature),
            operation={"candidate_skills": list(self.candidate_skills)},
            output_state={"chosen_skill": self.chosen_skill},
            soundness_type=SoundnessType.UNKNOWN,
            label=self.chosen_skill,
            metadata={"reason": self.reason, **self.metadata},
        )


AnyMettaFrame = ExecutionFrame | DeductionFrame | RepairFrame | RoutingFrame


def frame_from_jsonable(data: Mapping[str, Any]) -> AnyMettaFrame:
    copied = dict(data)
    kind = MettaFrameKind(copied.pop("kind"))
    if kind == MettaFrameKind.EXECUTION:
        return ExecutionFrame(kind=kind, **copied)
    if kind == MettaFrameKind.DEDUCTION:
        copied["soundness"] = SoundnessType(copied["soundness"])
        return DeductionFrame(kind=kind, **copied)
    if kind == MettaFrameKind.REPAIR:
        return RepairFrame(kind=kind, **copied)
    if kind == MettaFrameKind.ROUTING:
        return RoutingFrame(kind=kind, **copied)
    raise ValueError(f"Unsupported frame kind: {kind}")


def write_jsonl(path: Path, frames: Iterable[AnyMettaFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_jsonable(), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[AnyMettaFrame]:
    frames: List[AnyMettaFrame] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(frame_from_jsonable(json.loads(line)))
    return frames


def count_by_kind(frames: Sequence[AnyMettaFrame]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for frame in frames:
        counts[frame.kind.value] = counts.get(frame.kind.value, 0) + 1
    return counts
