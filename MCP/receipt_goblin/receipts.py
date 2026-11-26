"""Flatten run receipts into labelled numeric leaves and index them lexically.

A list element that is a dict is labelled by its identity fields (``[arm=layered policy=dense_d64]``)
instead of its position, so a leaf path names what was measured: the file, the cell, the metric.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 4 * 1024 * 1024
CONFIG_KEYS = {"catalog_size", "depth", "recurrent_depth", "budget", "tool_budget", "passes", "seed",
               "width", "layers", "test_depth", "max_tool_calls", "resources", "k", "n_layers", "loops"}
CONFIG_SUFFIXES = ("_size", "_depth", "_budget", "_passes", "_seed", "_width", "_layers")
SKIP_ID_KEYS = {"created_utc", "sha256", "checkpoint", "path", "command", "digest", "hash", "url", "error",
                "note", "notes", "description", "purpose", "scope", "started", "finished", "timestamp"}
HEXISH = re.compile(r"^[0-9a-f]{16,}$")
TOKEN_RE = re.compile(r"[a-z]+|\d+")
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are", "was", "were", "with", "for",
        "by", "from", "as", "it", "its", "this", "that", "than", "per", "each", "every", "all", "not", "no",
        "be", "while", "same", "only", "also", "about", "into", "they", "their", "them", "both", "any"}


@dataclass(frozen=True)
class Leaf:
    lid: int
    file: str
    parts: tuple[str, ...]
    value: float

    @property
    def path(self) -> str:
        return self.file + " :: " + "/".join(self.parts)


def _identity(element: dict) -> str | None:
    """Label a list element by its short string fields, then its configuration integers (at most 8)."""
    names, configs = [], []
    for key, value in element.items():
        if key in SKIP_ID_KEYS or isinstance(value, (dict, list)):
            continue
        if isinstance(value, str) and 0 < len(value) <= 48 and not HEXISH.match(value) and "T" not in value[10:11]:
            names.append(f"{key}={value}")
        elif (value is None or (isinstance(value, int) and not isinstance(value, bool))) and (
                key in CONFIG_KEYS or key.endswith(CONFIG_SUFFIXES)):
            configs.append(f"{key}={value}")
    fields = (names + configs)[:8]
    return "[" + " ".join(fields) + "]" if fields else None


def _echoes_label(parts: list[str]) -> bool:
    """A leaf that repeats a field of the label just above it (``[catalog_size=8]/catalog_size``)."""
    return len(parts) >= 2 and parts[-2].startswith("[") and f"{parts[-1]}=" in parts[-2]


def _walk(obj, parts: list[str], out: list[tuple[tuple[str, ...], float]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk(value, parts + [str(key)], out)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            label = _identity(value) if isinstance(value, dict) else None
            _walk(value, parts + [label or f"[{index}]"], out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
        if not _echoes_label(parts):
            out.append((tuple(parts), float(obj)))


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
    out = []
    for tok in TOKEN_RE.findall(text):
        if tok in STOP:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class ReceiptIndex:
    """All numeric leaves of the ``*.json`` receipts under one root, with a BM25 index over paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.leaves: list[Leaf] = []
        self.files: list[str] = []
        for path in sorted(self.root.rglob("*.json")):
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            rel = path.relative_to(self.root).as_posix()
            found: list[tuple[tuple[str, ...], float]] = []
            _walk(data, [], found)
            self.files.append(rel)
            for parts, value in found:
                self.leaves.append(Leaf(len(self.leaves), rel, parts, value))
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for leaf in self.leaves:
            toks = tokens(leaf.file.replace(".json", "") + " " + " ".join(leaf.parts))
            self._lengths.append(len(toks))
            for tok, tf in Counter(toks).items():
                self._postings[tok].append((leaf.lid, tf))
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        self._by_shape: dict[tuple[str, int], list[int]] = defaultdict(list)
        for leaf in self.leaves:
            self._by_shape[(leaf.file, len(leaf.parts))].append(leaf.lid)

    def search(self, query: str, k: int = 40, k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
        n = len(self.leaves)
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokens(query)):
            posting = self._postings.get(tok)
            if not posting:
                continue
            idf = math.log(1.0 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for lid, tf in posting:
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * self._lengths[lid] / self._avg))
                scores[lid] += idf * norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def lexical_scores(self, query: str, lids: list[int]) -> dict[int, float]:
        wanted = set(lids)
        return {lid: s for lid, s in self.search(query, k=len(self.leaves)) if lid in wanted}

    def siblings(self, lid: int) -> list[tuple[int, int]]:
        """Leaves in the same file whose path differs from ``lid`` in exactly one component.

        Returns (leaf id, shared identity fields) so callers can prefer the closest neighbours.
        """
        leaf = self.leaves[lid]
        out = []
        for other in self._by_shape[(leaf.file, len(leaf.parts))]:
            if other == lid:
                continue
            diff = [i for i, (a, c) in enumerate(zip(leaf.parts, self.leaves[other].parts)) if a != c]
            if len(diff) != 1:
                continue
            a, c = leaf.parts[diff[0]], self.leaves[other].parts[diff[0]]
            shared = len(set(a.strip("[]").split()) & set(c.strip("[]").split()))
            out.append((other, shared))
        return out
