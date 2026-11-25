"""Scaffolding for Goose/oi-kit lean context. No runtime hooks are installed here."""

from .lean_context import (
    Budget,
    BudgetError,
    Evidence,
    GateEvent,
    PacketReceipt,
    Phase,
    PhaseError,
    TokenCounter,
    advance_phase,
    assemble_packet,
)

__all__ = [
    "Budget",
    "BudgetError",
    "Evidence",
    "GateEvent",
    "PacketReceipt",
    "Phase",
    "PhaseError",
    "TokenCounter",
    "advance_phase",
    "assemble_packet",
]
