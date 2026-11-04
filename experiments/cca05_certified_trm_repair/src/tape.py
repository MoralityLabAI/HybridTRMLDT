"""Proposer tapes: the exact TRM logits every controller consumed (no torch).

A tape is written by trm_live.LiveTRM from a hash-verified checkpoint. Replay
returns the recorded float32 logits; a query that was never recorded raises
KeyError, so a replay cannot substitute invented neural evidence.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from research_gym.envs.coupled_storyworld import StoryState

import semantics as M


def mask_bits(mask: frozenset[str]) -> str:
    return "".join("1" if a in mask else "0" for a in M.ACTIONS)


def input_key(scenario: str, state: StoryState) -> str:
    return f"{scenario}|{state.trust},{state.evidence},{state.heat},{state.scene}"


def query_key(scenario: str, state: StoryState, mask: frozenset[str]) -> str:
    return f"{input_key(scenario, state)}|{mask_bits(mask)}"


class ReplayTRM:
    def __init__(self, tape: dict) -> None:
        self.tape = tape
        self.full = tape["full_mask_passes"]
        self.masked = tape["masked_final"]
        self.queries = 0

    @classmethod
    def load(cls, path: Path) -> "ReplayTRM":
        return cls(read_gz(path))

    def final_logits(self, scenario, state, mask):
        self.queries += 1
        if mask == frozenset(M.ACTIONS):
            return tuple(self.full[input_key(scenario, state)][3])
        return tuple(self.masked[query_key(scenario, state, mask)])

    def pass_logits(self, scenario, state):
        return tuple(tuple(row) for row in self.full[input_key(scenario, state)])


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_tape(tape: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(tape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as fh:
            fh.write(data)
