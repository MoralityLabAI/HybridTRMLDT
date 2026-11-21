"""Contract tests; byte units are synthetic, not real model tokens."""
import json
import unittest

from MCP.GDK.lean_context import (
    Budget,
    BudgetError,
    Evidence,
    GateEvent,
    Phase,
    PhaseError,
    TokenCounter,
    advance_phase,
    assemble_packet,
)

BYTE_COUNTER = TokenCounter(
    name="fixture:exact_utf8_bytes",
    count=lambda text: len(text.encode("utf-8")),
    exact=True,
)


def ev(ref, *, epoch="e1", summary="short", priority=0, verified=False):
    return Evidence(
        ref=ref,
        catalog_epoch=epoch,
        source_sha256="a" * 64,
        summary=summary,
        priority=priority,
        verified=verified,
    )


class PacketTests(unittest.TestCase):
    def make(self, **changes):
        args = dict(
            task="locate the symbol",
            phase=Phase.INSPECT,
            catalog_epoch="e1",
            allowed_tool_ids=["read_symbol", "code_search", "read_symbol"],
            evidence=[ev("b"), ev("a", verified=True)],
            counter=BYTE_COUNTER,
            budget=Budget(context_tokens=1200, output_reserve_tokens=200, safety_reserve_tokens=200),
            external_input_tokens=80,
            max_packet_tokens=600,
        )
        args.update(changes)
        return assemble_packet(**args)

    def test_packet_determinism_and_priority(self):
        packet, receipt = self.make()
        obj = json.loads(packet)
        self.assertEqual(obj["allowed_tools"], ["code_search", "read_symbol"])
        self.assertEqual([item["ref"] for item in obj["evidence"]], ["a", "b"])
        self.assertEqual(receipt.selected_refs, ("a", "b"))
        self.assertEqual(receipt.measurement_scope, "canonical_packet_only")
        self.assertEqual(receipt.packet_units, len(packet.encode("utf-8")))
        self.assertEqual(packet, self.make()[0])

    def test_skip_oversize_record_continue(self):
        packet, receipt = self.make(
            evidence=[ev("big", summary="x" * 1500, priority=50), ev("small", summary="fits")],
            max_packet_tokens=490,
        )
        self.assertEqual([item["ref"] for item in json.loads(packet)["evidence"]], ["small"])
        self.assertEqual(receipt.omitted_refs, ("big",))
        self.assertLessEqual(receipt.packet_units, 490)

    def test_epoch_staleness(self):
        packet, receipt = self.make(evidence=[ev("old", epoch="e0"), ev("new")])
        self.assertEqual(receipt.stale_refs, ("old",))
        self.assertEqual([item["ref"] for item in json.loads(packet)["evidence"]], ["new"])

    def test_fixed_fields_and_other_messages_count(self):
        with self.assertRaises(BudgetError):
            self.make(external_input_tokens=795)
        with self.assertRaises(BudgetError):
            self.make(task="z" * 2000)

    def test_estimator_requires_explicit_opt_in(self):
        estimator = TokenCounter(name="estimate", count=lambda text: len(text) // 4)
        with self.assertRaises(BudgetError):
            self.make(counter=estimator)
        _, receipt = self.make(counter=estimator, require_exact=False)
        self.assertFalse(receipt.counter_exact)

    def test_duplicate_and_invalid_evidence(self):
        with self.assertRaises(ValueError):
            self.make(evidence=[ev("same"), ev("same")])
        with self.assertRaises(ValueError):
            Evidence(ref="x", catalog_epoch="e1", source_sha256="bad", summary="x")

    def test_no_partial_evidence_record(self):
        packet, receipt = self.make(evidence=[ev("big", summary="⛄" * 500)])
        self.assertEqual(json.loads(packet)["evidence"], [])
        self.assertEqual(receipt.omitted_refs, ("big",))


class PhaseTests(unittest.TestCase):
    def test_guarded_success_and_retry(self):
        self.assertEqual(advance_phase(Phase.INSPECT, GateEvent.ACTION_SELECTED), Phase.ACT)
        with self.assertRaises(PhaseError):
            advance_phase(Phase.ACT, GateEvent.TOOL_RECEIPT)
        self.assertEqual(
            advance_phase(Phase.ACT, GateEvent.TOOL_RECEIPT, persisted_tool_receipts=["call-1"]),
            Phase.VERIFY,
        )
        self.assertEqual(advance_phase(Phase.VERIFY, GateEvent.NEEDS_MORE), Phase.INSPECT)
        with self.assertRaises(PhaseError):
            advance_phase(Phase.VERIFY, GateEvent.VERIFIED)
        self.assertEqual(
            advance_phase(Phase.VERIFY, GateEvent.VERIFIED, trusted_verifier_refs=["proof-1"]),
            Phase.FINAL,
        )
        self.assertEqual(advance_phase(Phase.FINAL, GateEvent.NEW_TASK), Phase.INSPECT)

    def test_reject_unregistered_transition(self):
        with self.assertRaises(PhaseError):
            advance_phase(Phase.INSPECT, GateEvent.VERIFIED, trusted_verifier_refs=["proof-1"])


if __name__ == "__main__":
    unittest.main()
