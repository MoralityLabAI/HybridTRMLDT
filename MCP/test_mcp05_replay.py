"""No-model unit contracts for MCP-05 paired native replay receipts."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from MCP.hubrun.paired_replay import ARMS, build_report


class ReplayReportTests(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.runs = self.root / "runs"
        self.calls = self.root / "calls.jsonl"

    def fixture(self, mode="smoke", bad_task=None, missing=None, bad_parity=None,
                missing_parity=None):
        records = []
        for taskset in ("docedit", "er-t1"):
            for arm in ARMS:
                tag = f"mcp05-{mode}-{taskset}-{arm}"
                if (taskset, arm) == missing:
                    continue
                directory = self.runs / tag
                directory.mkdir(parents=True)
                (directory / "DONE").write_text("2026-11-22\n")
                task_id = 1 if (taskset, arm) == bad_task else 0
                score = 0.0 if arm == "raw" else 1.0
                if taskset == "docedit":
                    content = {"example_id": task_id, "reward": score, "num_turns": 5}
                    (directory / "results.jsonl").write_text(json.dumps(content) + "\n")
                else:
                    content = {"task": {"data": {"name": f"evt-{task_id}"}}, "errors": [],
                               "traces": [{"ok": True, "errors": [], "stop_condition": "done",
                                           "metrics": {"turns": 5},
                                           "rewards": {"score": {"score": score, "weight": 1.0}}}]}
                    (directory / "traces.jsonl").write_text(json.dumps(content) + "\n")
                record = {"tag": tag, "arm": arm, "status": 200,
                          "prompt_tokens": 350, "completion_tokens": 20,
                          "compaction": {"compacted": arm != "raw",
                                         "pinned_negative_atoms": 2 if arm == "mcp05" else 0}}
                if arm != "raw" and (taskset, arm) != missing_parity:
                    record["preflight_matches_provider"] = (taskset, arm) != bad_parity
                records.append(record)
        self.calls.write_text("\n".join(json.dumps(x) for x in records) + "\n")

    def test_valid_three_arm_pair_reports_native_outcomes(self):
        self.fixture()
        receipt = build_report(self.runs, self.calls, "smoke")
        self.assertTrue(receipt["valid_comparison"])
        self.assertTrue(receipt["promotion_gate"]["task_success"])
        self.assertEqual(receipt["tasksets"]["docedit"]["pairs"]["mcp05_vs_raw"]["better"], 1)
        self.assertEqual(receipt["tasksets"]["er-t1"]["arms"]["mcp05"]["pinned_negative_atoms"], 2)

    def test_mismatched_task_allocations_reject_paired_claim(self):
        self.fixture(bad_task=("docedit", "mcp05"))
        receipt = build_report(self.runs, self.calls, "smoke")
        self.assertFalse(receipt["valid_comparison"])
        self.assertFalse(receipt["tasksets"]["docedit"]["matched"])

    def test_missing_arm_rejects_paired_claim(self):
        self.fixture(missing=("er-t1", "mcp04"))
        receipt = build_report(self.runs, self.calls, "smoke")
        self.assertFalse(receipt["valid_comparison"])
        self.assertTrue(any("mcp04" in error for error in receipt["errors"]))

    def test_provider_token_mismatch_rejects_paired_claim(self):
        self.fixture(bad_parity=("docedit", "mcp05"))
        receipt = build_report(self.runs, self.calls, "smoke")
        self.assertFalse(receipt["valid_comparison"])
        self.assertEqual(receipt["tasksets"]["docedit"]["arms"]["mcp05"]["provider_parity_failures"], 1)

    def test_missing_usage_rejects_paired_claim(self):
        self.fixture(missing_parity=("docedit", "mcp05"))
        receipt = build_report(self.runs, self.calls, "smoke")
        self.assertFalse(receipt["valid_comparison"])
        self.assertEqual(receipt["tasksets"]["docedit"]["arms"]["mcp05"]["provider_parity_missing"], 1)

    def test_allocation_contract_prevents_smoke_claimed_as_full(self):
        self.fixture(mode="full")
        receipt = build_report(self.runs, self.calls, "full")
        self.assertFalse(receipt["valid_comparison"])
        self.assertTrue(any("expected 8" in error for error in receipt["errors"]))


if __name__ == "__main__":
    unittest.main()
