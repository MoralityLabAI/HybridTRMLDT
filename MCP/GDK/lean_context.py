"""Deterministic, tool-free packet construction for a future Goose lean-context adapter.

This module accounts for canonical packet serialization only. The runtime is
responsible for full provider-request tokenization, durable state, evidence
verification, tool authorization, and catalog pinning.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

SCHEMA = "gdk.lean_context.v0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BudgetError(ValueError):
    """The required packet or tokenizer contract exceeds the registered budget."""


class PhaseError(ValueError):
    """A state-machine transition lacks the required host-owned receipt."""


class Phase(str, Enum):
    INSPECT = "INSPECT"
    ACT = "ACT"
    VERIFY = "VERIFY"
    FINAL = "FINAL"


class GateEvent(str, Enum):
    ACTION_SELECTED = "action_selected"
    TOOL_RECEIPT = "tool_receipt"
    VERIFIED = "verified"
    NEEDS_MORE = "needs_more"
    NEW_TASK = "new_task"


@dataclass(frozen=True)
class Budget:
    context_tokens: int = 12_288
    output_reserve_tokens: int = 2_048
    safety_reserve_tokens: int = 2_048

    @property
    def input_capacity(self) -> int:
        if (
            self.context_tokens <= 0
            or self.output_reserve_tokens < 0
            or self.safety_reserve_tokens < 0
            or self.output_reserve_tokens + self.safety_reserve_tokens >= self.context_tokens
        ):
            raise BudgetError("invalid context/output/safety budget")
        return self.context_tokens - self.output_reserve_tokens - self.safety_reserve_tokens


@dataclass(frozen=True)
class TokenCounter:
    name: str
    count: Callable[[str], int]
    exact: bool = False

    def tokens(self, text: str) -> int:
        if not self.name.strip():
            raise BudgetError("tokenizer name is required")
        value = self.count(text)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BudgetError("token counter must return a nonnegative integer")
        return value


@dataclass(frozen=True)
class Evidence:
    ref: str
    catalog_epoch: str
    source_sha256: str
    summary: str
    priority: int = 0
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.ref or not self.catalog_epoch or not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("evidence requires a ref, catalog epoch, and lowercase SHA-256")
        if not isinstance(self.summary, str) or not isinstance(self.priority, int):
            raise TypeError("evidence summary/priority have invalid types")

    def visible(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "source_sha256": self.source_sha256,
            "summary": self.summary,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class PacketReceipt:
    tokenizer_name: str
    counter_exact: bool
    input_capacity_units: int
    external_input_units: int
    packet_budget_units: int
    packet_units: int
    selected_refs: tuple[str, ...]
    omitted_refs: tuple[str, ...]
    stale_refs: tuple[str, ...]
    measurement_scope: str = "canonical_packet_only"


def _serialize(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assemble_packet(
    *,
    task: str,
    phase: Phase,
    catalog_epoch: str,
    allowed_tool_ids: Sequence[str],
    evidence: Sequence[Evidence],
    counter: TokenCounter,
    budget: Budget = Budget(),
    external_input_tokens: int = 0,
    max_packet_tokens: int | None = None,
    require_exact: bool = True,
) -> tuple[str, PacketReceipt]:
    """Build one compact packet, retaining complete evidence records or skipping them.

    The host supplies a named counter and a full-request external-input estimate
    measured at the SAME tokenization boundary. This function cannot enforce the
    complete provider request. A final request-preparer preflight is mandatory.
    """
    if require_exact and not counter.exact:
        raise BudgetError("exact counter required for hard budget")
    if not isinstance(phase, Phase) or not task.strip() or not catalog_epoch.strip():
        raise ValueError("phase, task, and catalog epoch are required")
    if not isinstance(external_input_tokens, int) or external_input_tokens < 0:
        raise BudgetError("external input units must be nonnegative")
    input_capacity = budget.input_capacity
    available = input_capacity - external_input_tokens
    if max_packet_tokens is not None:
        if not isinstance(max_packet_tokens, int) or max_packet_tokens <= 0:
            raise BudgetError("max packet units must be positive")
        available = min(available, max_packet_tokens)
    if available <= 0:
        raise BudgetError("no space left for a packet")
    tool_ids = list(allowed_tool_ids)
    if any(not isinstance(t, str) or not t.strip() for t in tool_ids):
        raise ValueError("allowed tool IDs must be nonempty strings")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "phase": phase.value,
        "task": task,
        "catalog_epoch": catalog_epoch,
        "allowed_tools": sorted(set(tool_ids)),
        "evidence": [],
    }
    packet = _serialize(payload)
    if counter.tokens(packet) > available:
        raise BudgetError("fixed packet fields exceed remaining input allowance")

    refs: set[str] = set()
    for item in evidence:
        if item.ref in refs:
            raise ValueError("duplicate evidence ref in working set")
        refs.add(item.ref)

    selected: list[str] = []
    omitted: list[str] = []
    stale: list[str] = []
    for item in sorted(evidence, key=lambda e: (-int(e.verified), -e.priority, e.ref)):
        if item.catalog_epoch != catalog_epoch:
            stale.append(item.ref)
            continue
        records = payload["evidence"]
        assert isinstance(records, list)
        records.append(item.visible())
        candidate = _serialize(payload)
        if counter.tokens(candidate) <= available:
            selected.append(item.ref)
            packet = candidate
        else:
            records.pop()
            omitted.append(item.ref)

    return packet, PacketReceipt(
        tokenizer_name=counter.name,
        counter_exact=counter.exact,
        input_capacity_units=input_capacity,
        external_input_units=external_input_tokens,
        packet_budget_units=available,
        packet_units=counter.tokens(packet),
        selected_refs=tuple(selected),
        omitted_refs=tuple(omitted),
        stale_refs=tuple(stale),
    )


def advance_phase(
    phase: Phase,
    event: GateEvent,
    *,
    persisted_tool_receipts: Sequence[str] = (),
    trusted_verifier_refs: Sequence[str] = (),
) -> Phase:
    """Apply a phase guard after the HOST authenticates supplied receipt IDs."""
    if phase is Phase.INSPECT and event is GateEvent.ACTION_SELECTED:
        return Phase.ACT
    if phase is Phase.ACT and event is GateEvent.TOOL_RECEIPT and persisted_tool_receipts:
        return Phase.VERIFY
    if phase is Phase.VERIFY and event is GateEvent.VERIFIED and trusted_verifier_refs:
        return Phase.FINAL
    if phase is Phase.VERIFY and event is GateEvent.NEEDS_MORE:
        return Phase.INSPECT
    if phase is Phase.FINAL and event is GateEvent.NEW_TASK:
        return Phase.INSPECT
    raise PhaseError(f"unpermitted transition: {phase} + {event}")


def _demo() -> None:
    from dataclasses import asdict
    # Development-only heuristic; NOT a provider tokenizer.
    estimate = TokenCounter(
        name="estimate:utf8_bytes_div_4",
        count=lambda text: math.ceil(len(text.encode("utf-8")) / 4),
    )
    packet, receipt = assemble_packet(
        task="Locate candidate_refine metadata checks",
        phase=Phase.INSPECT,
        catalog_epoch="sample-epoch",
        allowed_tool_ids=["read_symbol", "code_search"],
        evidence=[
            Evidence(
                ref="symbol://MCP/mcp02/env.py#candidate_refine",
                catalog_epoch="sample-epoch",
                source_sha256="a" * 64,
                summary="Read candidate metadata before certifying exclusions",
            )
        ],
        counter=estimate,
        require_exact=False,
        max_packet_tokens=512,
    )
    print(packet)
    print(json.dumps(asdict(receipt), indent=2))


if __name__ == "__main__":
    _demo()
