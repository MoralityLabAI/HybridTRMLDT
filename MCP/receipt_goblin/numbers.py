"""Numbers as written in documents: parsing, matching against receipt values, re-formatting.

A claim number keeps its written precision, so ``0.86`` matches any receipt value that rounds to
0.86 and ``78.5K`` matches 78,450 to 78,550. Integers written without decimals must equal an
integral receipt value exactly, or round to it when the receipt value is fractional.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# A number token: 13,430 | 0.86 | 78.5K | 86% | 4. Not part of an identifier (Qwen2.5-0.5B, MCP-02,
# d128, p4), a version (5.0.2), a date (2026-11-20) or a hash.
NUMBER_RE = re.compile(
    r"(?<![\w.\-/#`])"
    r"(?P<mant>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?[KM](?![A-Za-z])|%)?"
    r"(?![\w`]|\.\d|-\d|/\d)"
)


@dataclass(frozen=True)
class Num:
    surface: str
    mantissa: float
    decimals: int
    scale: float
    percent: bool
    commas: bool
    start: int = 0
    end: int = 0

    @property
    def value(self) -> float:
        return self.mantissa * self.scale


def parse(text: str) -> list[Num]:
    out = []
    for m in NUMBER_RE.finditer(text):
        mant = m.group("mant")
        suffix = (m.group("suffix") or "").strip()
        digits = mant.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = {"K": 1e3, "M": 1e6}.get(suffix, 1.0)
        out.append(Num(m.group(0).strip(), float(digits), decimals, scale, suffix == "%", "," in mant,
                       m.start(), m.start() + len(m.group(0).rstrip())))
    return out


def parse_one(surface: str) -> Num:
    nums = parse(surface)
    if len(nums) != 1 or nums[0].surface != surface.strip():
        raise ValueError(f"not a single number: {surface!r}")
    return nums[0]


def _close(candidate: float, num: Num) -> bool:
    if num.decimals == 0 and num.scale == 1.0 and float(candidate).is_integer():
        return candidate == num.mantissa
    half = 0.5 * 10.0 ** (-num.decimals)
    return abs(candidate - num.mantissa) <= half + 1e-9 * max(1.0, abs(num.mantissa))


def matches(num: Num, value) -> bool:
    """True when a receipt value, shown at the claim's precision and scale, reads as the claim."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if num.percent:
        return _close(value * 100.0, num) or (value > 1.0 and _close(float(value), num))
    return _close(value / num.scale, num)


def render(value: float, like: Num) -> str:
    """Write ``value`` (in receipt units) in the same style as ``like``."""
    if like.percent:
        shown = value * 100.0 if value <= 1.0 else value
        return f"{shown:.{like.decimals}f}%"
    mant = value / like.scale
    text = f"{mant:,.{like.decimals}f}" if like.commas else f"{mant:.{like.decimals}f}"
    suffix = {1e3: "K", 1e6: "M"}.get(like.scale, "")
    return text + suffix


def short(value) -> str:
    """Compact display of a receipt value for packets."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"
