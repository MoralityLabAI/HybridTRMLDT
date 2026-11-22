"""Experimental ECAN-inspired attention for MCP-04's typed working set.

This is a deterministic retrieval policy, not a neural ECAN implementation.
Every retained atom points to an immutable result handle stored by MCP-04.
Negative observations are pinned as *historical* observations; their presence
never authorizes subtree pruning or overrides a verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Iterable

_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{2,}")
_HINT_FIELDS = ("query", "pattern", "symbol", "name", "q", "search", "uri", "node_id")
_SNAPSHOT_FIELDS = ("snapshot_id", "revision", "commit", "epoch", "version")


def _single_line(value: object, limit: int = 88) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[:limit - 3] + "..."


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hint(args: dict[str, Any]) -> str:
    for field in _HINT_FIELDS:
        value = args.get(field)
        if isinstance(value, (str, int)) and str(value):
            return _single_line(value)
    return "(unspecified)"


def _snapshot(args: dict[str, Any]) -> str:
    for field in _SNAPSHOT_FIELDS:
        value = args.get(field)
        if isinstance(value, (str, int)) and str(value):
            return _single_line(value, 64)
    return "unknown"


def _terms(text: str) -> set[str]:
    return {s.lower() for s in _TERM.findall(text) if len(s) > 2}


@dataclass(frozen=True)
class ReadAtom:
    name: str
    path: str
    hint: str
    args_sha: str
    snapshot: str
    local_epoch: int
    handle: str
    status: str
    is_empty: bool
    last_step: int
    uses: int = 1

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        # A changed tool body is kept even if the query and path are identical.
        return (self.name, self.args_sha, self.snapshot, self.local_epoch, self.handle)

    @property
    def lineage(self) -> str:
        # Repeats of one source/query epoch are one evidential lineage.
        return _digest(self.key)[:16]

    def line(self) -> str:
        prefix = "observed-empty" if self.is_empty else "read"
        return (f"{prefix} {self.name}({self.path}) query={self.hint!r} "
                f"args_sha={self.args_sha[:16]} snapshot={self.snapshot!r} "
                f"local_edit_epoch={self.local_epoch} lineage={self.lineage} "
                f"uses={self.uses} status={self.status} handle={self.handle}")


def observe_read(
    table: dict[tuple[str, str, str, int, str], ReadAtom], *, name: str,
    path: str, args: dict[str, Any], handle: str, status: str,
    is_empty: bool, step: int, local_epoch: int,
) -> None:
    """Add one read observation, coalescing identical repeats only.

    Unknown remote revisions remain explicitly unknown. "local_epoch"
    describes successful edit calls observed in this transcript only.
    """
    atom = ReadAtom(name=name, path=path, hint=_hint(args), args_sha=_digest(args),
                    snapshot=_snapshot(args), local_epoch=local_epoch,
                    handle=handle, status=status, is_empty=is_empty,
                    last_step=step)
    previous = table.get(atom.key)
    if previous is not None:
        atom = replace(atom, uses=previous.uses + 1)
    table[atom.key] = atom


def rank_read_atoms(
    atoms: Iterable[ReadAtom], *, task_text: str, tail_text: str,
    edit_paths: Iterable[str],
) -> tuple[list[ReadAtom], list[ReadAtom]]:
    """Return (pinned historical empties, optional reads), deterministically.

    Short-term importance is task/recency/reuse based. Co-use with edited paths
    provides a small association boost. Long-term storage stays at the result
    handle; immutable edit facts remain pinned by MCP-04 itself.
    """
    atoms = list(atoms)
    edited = set(edit_paths)
    target = _terms(task_text + " " + tail_text)
    newest = max((a.last_step for a in atoms), default=0)

    def score(a: ReadAtom) -> float:
        shared = len(_terms(a.name + " " + a.path + " " + a.hint) & target)
        # Saturation avoids treating repeated identical reads as new evidence.
        reuse = min(a.uses, 4) * 0.35
        recency = 2.0 / (1.0 + newest - a.last_step)
        edit_co_use = 2.0 if a.path in edited else 0.0
        return 3.0 * shared + reuse + recency + edit_co_use

    order = lambda a: (-score(a), -a.last_step, a.path, a.args_sha, a.handle)
    pinned = sorted((a for a in atoms if a.is_empty), key=lambda a: (a.last_step, a.path, a.args_sha))
    optional = sorted((a for a in atoms if not a.is_empty), key=order)
    return pinned, optional
