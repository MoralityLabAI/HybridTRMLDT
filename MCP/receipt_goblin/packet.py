"""Compact evidence packets for one claim, and the deterministic certification of a linked leaf.

A packet lists candidate receipt leaves under short ids (L01, L02, ...): the best lexical matches
for the claim's wording plus the leaves whose value reads as the claimed number. In blind mode the
values are withheld, so a model has to link the claim by meaning; certification then compares the
claimed number with the linked leaf's value in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from MCP.mcp03.tokens import count_tokens

from .numbers import Num, matches, render, short
from .receipts import ReceiptIndex

HEADER = ("Receipt leaves (id, file :: path = value). A path segment in [brackets] names the cell "
          "a list element describes.")
HEADER_BLIND = ("Receipt leaves (id, file :: path). Values are withheld. A path segment in [brackets] "
                "names the cell a list element describes.")


@dataclass
class Packet:
    text: str
    ids: dict[str, int]
    tokens: int
    lexical: list[int] = field(default_factory=list)
    numeric: list[int] = field(default_factory=list)


def number_matches(index: ReceiptIndex, target: Num) -> list[int]:
    return [leaf.lid for leaf in index.leaves if matches(target, leaf.value)]


def build(index: ReceiptIndex, query: str, target: Num, budget_tokens: int = 1500, k_lexical: int = 40,
          k_numeric: int = 12, k_neighbours: int = 0, blind: bool = False) -> Packet:
    """Lexical top ``k_lexical`` plus number-matched leaves, optionally with their path neighbours.

    Neighbours of a number-matched leaf are the same metric in the adjacent cells (and adjacent
    metrics of the same cell), ranked by lexical score; they let a model see whether the matched
    value sits in the cell the claim names or one next to it.
    """
    scores = dict(index.search(query, k=len(index.leaves)))
    lexical = sorted(scores, key=lambda lid: (-scores[lid], lid))[:k_lexical]
    numeric = sorted(number_matches(index, target), key=lambda lid: (-scores.get(lid, 0.0), lid))[:k_numeric]
    neighbours: set[int] = set()
    if k_neighbours:
        for lid in numeric[:4]:
            near = sorted((other for other, _ in index.siblings(lid)), key=lambda o: (-scores.get(o, 0.0), o))
            neighbours.update(near[:k_neighbours])
    chosen = sorted(set(lexical) | set(numeric) | neighbours, key=lambda lid: (-scores.get(lid, 0.0), lid))
    lines = [HEADER_BLIND if blind else HEADER]
    used = count_tokens(lines[0])
    ids: dict[str, int] = {}
    for lid in chosen:
        leaf = index.leaves[lid]
        key = f"L{len(ids) + 1:02d}"
        line = f"{key} {leaf.path}" + ("" if blind else f" = {short(leaf.value)}")
        cost = count_tokens(line) + 1
        if used + cost > budget_tokens:
            continue
        lines.append(line)
        used += cost
        ids[key] = lid
    text = "\n".join(lines)
    return Packet(text, ids, count_tokens(text), lexical, numeric)


def certify(index: ReceiptIndex, target: Num, lid: int | None) -> dict:
    if lid is None:
        return {"verdict": "unsupported", "leaf": None, "leaf_value": None, "leaf_rendered": None}
    leaf = index.leaves[lid]
    return {"verdict": "supported" if matches(target, leaf.value) else "contradicted",
            "leaf": leaf.path, "leaf_value": leaf.value, "leaf_rendered": render(leaf.value, target)}
