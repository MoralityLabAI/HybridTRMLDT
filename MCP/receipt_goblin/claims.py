"""Numeric claims in Markdown: one claim per number, with the sentence or table row around it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .numbers import Num, parse

CODE_SPAN = re.compile(r"`[^`]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
EMPHASIS = re.compile(r"\*\*|__")
SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")
MARK_OPEN, MARK_CLOSE = "⟦", "⟧"


@dataclass(frozen=True)
class Claim:
    cid: str
    doc: str
    line: int
    text: str
    target: Num

    def marked(self) -> str:
        t = self.target
        return self.text[:t.start] + MARK_OPEN + self.text[t.start:t.end] + MARK_CLOSE + self.text[t.end:]

    def context(self) -> str:
        """The claim text without its target number, used as the lexical query."""
        return self.text[:self.target.start] + " " + self.text[self.target.end:]


def _clean(text: str) -> str:
    text = CODE_SPAN.sub("‹code›", text)
    text = LINK.sub(r"\1", text)
    text = EMPHASIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_year(num: Num) -> bool:
    return num.decimals == 0 and num.scale == 1.0 and not num.percent and not num.commas and 1990 <= num.mantissa <= 2100


def units(markdown: str) -> list[tuple[int, str, int]]:
    """(line number, text, own-text offset) for every sentence and table row outside code and headings.

    A table row is written as ``header: cell; ...`` after the last sentence before the table, which
    usually says what the table counts; numbers before the offset belong to that lead sentence.
    """
    out: list[tuple[int, str, int]] = []
    header: list[str] | None = None
    lead = ""
    fenced = False
    for number, raw in enumerate(markdown.splitlines(), 1):
        line = raw.rstrip()
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            fenced = not fenced
            continue
        if fenced or not line.strip() or line.lstrip().startswith("#"):
            if not line.strip():
                header = None
            continue
        if line.lstrip().startswith("|"):
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if TABLE_RULE.match(line.strip().strip("|").strip()):
                continue
            if header is None:
                header = cells
                continue
            pairs = [f"{h}: {c}" if h else c for h, c in zip(header + [""] * len(cells), cells)]
            prefix = f"{lead} | " if lead else ""
            out.append((number, prefix + "; ".join(pairs), len(prefix)))
            continue
        header = None
        text = _clean(re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line))
        sentences = [s for s in SENTENCE_SPLIT.split(text) if s]
        for sentence in sentences:
            out.append((number, sentence, 0))
        if sentences:
            lead = text if len(text) <= 320 else text[:320].rsplit(" ", 1)[0] + " ..."
    return out


def _in_range(text: str, num: Num) -> bool:
    """An endpoint of a range such as ``0.90–0.92`` or ``25–37``: an aggregate, not one receipt value."""
    before, after = text[max(0, num.start - 1):num.start], text[num.end:num.end + 1]
    return before in ("–", "—") or after in ("–", "—")


def extract(path: str | Path, doc_name: str | None = None) -> list[Claim]:
    path = Path(path)
    name = doc_name or path.as_posix()
    claims = []
    for line, text, own in units(path.read_text(encoding="utf-8")):
        for num in parse(text):
            if num.start < own or _is_year(num) or _in_range(text, num):
                continue
            claims.append(Claim(f"{name}:{line}:{num.start - own}", name, line, text, num))
    return claims
