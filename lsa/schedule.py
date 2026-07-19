"""A small algebra for finite, nested controller schedules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True)
class VisitSymbol:
    """One physical-module application with explicit state interfaces."""

    name: str
    module: str
    reads: frozenset[str]
    writes: str

    def __post_init__(self) -> None:
        if not self.name or not self.module or not self.writes:
            raise ValueError("visit name, module, and write register must be non-empty")


@dataclass(frozen=True)
class ExpandedVisit:
    position: int
    symbol: VisitSymbol
    symbol_round: int
    module_round: int

    @property
    def label(self) -> str:
        return f"{self.symbol.name}{self.symbol_round}"


class Word(ABC):
    @abstractmethod
    def symbols(self) -> tuple[VisitSymbol, ...]:
        """Return the finite visit sequence represented by this word."""

    def __add__(self, other: "Word") -> "Word":
        return Concat((self, other))

    def __pow__(self, count: int) -> "Word":
        return Power(self, count)

    def expand(self) -> tuple[ExpandedVisit, ...]:
        symbol_counts: dict[str, int] = {}
        module_counts: dict[str, int] = {}
        expanded = []
        for position, symbol in enumerate(self.symbols()):
            symbol_counts[symbol.name] = symbol_counts.get(symbol.name, 0) + 1
            module_counts[symbol.module] = module_counts.get(symbol.module, 0) + 1
            expanded.append(
                ExpandedVisit(
                    position=position,
                    symbol=symbol,
                    symbol_round=symbol_counts[symbol.name],
                    module_round=module_counts[symbol.module],
                )
            )
        return tuple(expanded)


@dataclass(frozen=True)
class SymbolWord(Word):
    symbol: VisitSymbol

    def symbols(self) -> tuple[VisitSymbol, ...]:
        return (self.symbol,)


@dataclass(frozen=True)
class Concat(Word):
    parts: tuple[Word, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("concatenation requires at least one word")

    def symbols(self) -> tuple[VisitSymbol, ...]:
        return tuple(symbol for part in self.parts for symbol in part.symbols())


@dataclass(frozen=True)
class Power(Word):
    word: Word
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("word powers must be non-negative")

    def symbols(self) -> tuple[VisitSymbol, ...]:
        return self.word.symbols() * self.count


def atom(symbol: VisitSymbol) -> SymbolWord:
    return SymbolWord(symbol)


_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|\d+|[().^])")


def parse_word(expression: str, symbols: Mapping[str, VisitSymbol]) -> Word:
    """Parse concatenation and integer powers, e.g. ``(L^3.H)^2``."""

    tokens: list[str] = []
    cursor = 0
    while cursor < len(expression):
        match = _TOKEN.match(expression, cursor)
        if match is None:
            raise ValueError(f"invalid schedule syntax at offset {cursor}")
        tokens.append(match.group(1))
        cursor = match.end()
    index = 0

    def parse_sequence() -> Word:
        nonlocal index
        parts: list[Word] = []
        while index < len(tokens) and tokens[index] != ")":
            if tokens[index] == ".":
                index += 1
                continue
            parts.append(parse_term())
        if not parts:
            raise ValueError("empty schedule group")
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def parse_term() -> Word:
        nonlocal index
        token = tokens[index]
        if token == "(":
            index += 1
            value = parse_sequence()
            if index >= len(tokens) or tokens[index] != ")":
                raise ValueError("unclosed schedule group")
            index += 1
        else:
            if token not in symbols:
                raise ValueError(f"unknown visit symbol: {token}")
            value = atom(symbols[token])
            index += 1
        if index < len(tokens) and tokens[index] == "^":
            index += 1
            if index >= len(tokens) or not tokens[index].isdigit():
                raise ValueError("schedule power requires a non-negative integer")
            value = Power(value, int(tokens[index]))
            index += 1
        return value

    if not tokens:
        raise ValueError("schedule expression is empty")
    parsed = parse_sequence()
    if index != len(tokens):
        raise ValueError(f"unexpected token: {tokens[index]}")
    return parsed


class GradientMask(Protocol):
    name: str

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]: ...


@dataclass(frozen=True)
class FullGradient:
    name: str = "full"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        return frozenset(visit.position for visit in visits)


@dataclass(frozen=True)
class LastKGradient:
    k: int
    name: str = "last-k"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        if self.k < 0:
            raise ValueError("k must be non-negative")
        return frozenset(visit.position for visit in visits[-self.k :]) if self.k else frozenset()


@dataclass(frozen=True)
class LastOuterCycleGradient:
    cycle_width: int
    name: str = "one-step"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        if self.cycle_width <= 0:
            raise ValueError("cycle width must be positive")
        return frozenset(visit.position for visit in visits[-self.cycle_width :])


@dataclass(frozen=True)
class ExplicitGradient:
    positions: frozenset[int]
    name: str = "arbitrary"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        valid = {visit.position for visit in visits}
        if not self.positions.issubset(valid):
            raise ValueError("gradient mask contains an out-of-range visit")
        return self.positions


@dataclass(frozen=True)
class ModuleTailGradient:
    """Keep the last ``k`` visits independently for each named symbol."""

    tails: Mapping[str, int]
    name: str = "module-last-k"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        selected: set[int] = set()
        for symbol_name, count in self.tails.items():
            if count < 0:
                raise ValueError("module tail counts must be non-negative")
            candidates = [v.position for v in visits if v.symbol.name == symbol_name]
            selected.update(candidates[-count:] if count else ())
        return frozenset(selected)


@dataclass(frozen=True)
class HRMTextWarmupGradient:
    """The exact H-prioritized ``bp_warmup`` mask in HRM-Text."""

    bp_steps: int
    h_cycles: int
    l_cycles: int
    h_symbol: str = "H"
    l_symbol: str = "L"
    name: str = "hrm-text-bp-warmup"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        if self.bp_steps < 2:
            raise ValueError("HRM-Text reserves at least one visible H and L visit")
        h_steps = min(self.h_cycles, self.bp_steps - 1)
        l_steps = self.bp_steps - h_steps
        if l_steps > self.h_cycles * self.l_cycles:
            raise ValueError("bp_steps exceeds the represented HRM-Text schedule")
        return ModuleTailGradient(
            {self.h_symbol: h_steps, self.l_symbol: l_steps}
        ).select(visits)


def selected_visits(word: Word, mask: GradientMask) -> tuple[ExpandedVisit, ...]:
    visits = word.expand()
    selected = mask.select(visits)
    return tuple(visit for visit in visits if visit.position in selected)


def labels(visits: Iterable[ExpandedVisit]) -> tuple[str, ...]:
    return tuple(visit.label for visit in visits)
