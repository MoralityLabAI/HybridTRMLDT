from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .typed_soundness import SoundnessType


@dataclass(frozen=True)
class Frame:
    """Common JSONL schema for local training and evaluation frames."""

    id: str
    family: str
    source: str
    input_state: Mapping[str, Any]
    operation: Any
    output_state: Mapping[str, Any]
    soundness_type: SoundnessType
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["soundness_type"] = self.soundness_type.value
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "Frame":
        copied = dict(data)
        copied["soundness_type"] = SoundnessType(copied["soundness_type"])
        copied["metadata"] = dict(copied.get("metadata", {}))
        return cls(**copied)


@dataclass(frozen=True)
class ReachabilityFrame:
    """One local training/evaluation frame for typed reachability."""

    frame_id: str
    state: Mapping[str, int]
    horizon: int
    label: SoundnessType
    target_reachable_self_only: bool
    target_reachable_with_other_model: bool
    target_reachable_with_other_frozen: bool
    witness_actions: List[str] = field(default_factory=list)
    counterfactual: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        data = asdict(self)
        data["label"] = self.label.value
        return data

    def to_frame(self) -> Frame:
        return Frame(
            id=self.frame_id,
            family="reachability",
            source="coupled_storyworld",
            input_state=dict(self.state),
            operation={"horizon": self.horizon, "target": "secret_ending"},
            output_state={
                "target_reachable_self_only": self.target_reachable_self_only,
                "target_reachable_with_other_model": self.target_reachable_with_other_model,
                "target_reachable_with_other_frozen": self.target_reachable_with_other_frozen,
            },
            soundness_type=self.label,
            label=self.label.value,
            metadata={
                "witness_actions": list(self.witness_actions),
                "counterfactual": dict(self.counterfactual),
            },
        )

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "ReachabilityFrame":
        copied = dict(data)
        copied["label"] = SoundnessType(copied["label"])
        return cls(**copied)


def write_frame_jsonl(path: Path, frames: Iterable[Frame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_jsonable(), sort_keys=True) + "\n")


def read_frame_jsonl(path: Path) -> List[Frame]:
    frames: List[Frame] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(Frame.from_jsonable(json.loads(line)))
    return frames
