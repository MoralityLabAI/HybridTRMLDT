from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from research_gym.core.metta_frames import AnyMettaFrame
from research_gym.envs.metta_synthetic import frames_from_directives


@dataclass(frozen=True)
class ParsedMettaProgram:
    """Opaque parsed program placeholder for adapter tests."""

    source: str
    text: str


class MettaAdapterProtocol(Protocol):
    """Minimal seam for later real MeTTa integration."""

    def parse(self, text: str) -> ParsedMettaProgram:
        ...

    def typecheck(self, program: ParsedMettaProgram) -> bool:
        ...

    def execute(self, program: ParsedMettaProgram, state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def extract_frames(self, program: ParsedMettaProgram) -> list[AnyMettaFrame]:
        ...


class FakeMettaAdapter:
    """Directive-backed adapter used until a real MeTTa runtime is wired in."""

    def __init__(self, *, source: str = "fake_metta") -> None:
        self.source = source

    def parse(self, text: str) -> ParsedMettaProgram:
        return ParsedMettaProgram(source=self.source, text=text)

    def typecheck(self, program: ParsedMettaProgram) -> bool:
        try:
            frames_from_directives(program.text, source=program.source)
        except ValueError:
            return False
        return True

    def execute(self, program: ParsedMettaProgram, state: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.typecheck(program):
            raise ValueError("Cannot execute a program that failed fake typecheck")
        return {"state": dict(state), "executed": True, "source": program.source}

    def extract_frames(self, program: ParsedMettaProgram) -> list[AnyMettaFrame]:
        return frames_from_directives(program.text, source=program.source)
