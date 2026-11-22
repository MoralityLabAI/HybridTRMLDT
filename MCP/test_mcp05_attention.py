"""ECAN-style typed working-set regression tests (stdlib, no model inference)."""
from __future__ import annotations

import json
import unittest

from MCP.hubrun.attention import observe_read, rank_read_atoms
from MCP.hubrun.mcp04 import WorkingSetError, typed_packet
from MCP.test_mcp04 import episode


class AttentionUnitTests(unittest.TestCase):
    def test_same_path_distinct_queries_and_responses_are_independent(self):
        table = {}
        for step, query, handle in [(1, "collision", "mcp04:a"),
                                    (2, "bounds", "mcp04:b"),
                                    (3, "collision", "mcp04:c")]:
            observe_read(table, name="search_file", path="engine/collision.py",
                         args={"path": "engine/collision.py", "query": query}, handle=handle,
                         status="returned", is_empty=False, step=step, local_epoch=0)
        _, ranked = rank_read_atoms(table.values(), task_text="fix collision",
                                    tail_text="", edit_paths=())
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].hint, "collision")
        self.assertEqual(len({a.lineage for a in ranked}), 3)

    def test_repeated_identical_read_is_one_evidential_lineage(self):
        table = {}
        for step in (1, 7, 9):
            observe_read(table, name="read_symbol", path="src/a.py",
                         args={"uri": "src/a.py#foo"}, handle="mcp04:sha1",
                         status="returned", is_empty=False, step=step, local_epoch=0)
        self.assertEqual(len(table), 1)
        self.assertEqual(next(iter(table.values())).uses, 3)

    def test_negative_snapshot_and_local_edit_epoch_are_distinct(self):
        table = {}
        for revision, epoch in [("sha123", 0), ("sha124", 0), ("sha124", 1)]:
            observe_read(table, name="search_symbol", path="src/a.py",
                         args={"path": "src/a.py", "query": "missing", "revision": revision},
                         handle="mcp04:empty", status="returned", is_empty=True,
                         step=epoch + 1, local_epoch=epoch)
        pinned, _ = rank_read_atoms(table.values(), task_text="missing", tail_text="",
                                    edit_paths=())
        self.assertEqual(len(pinned), 3)
        self.assertTrue(all(a.line().startswith("observed-empty") for a in pinned))
        self.assertTrue(all("lineage=" in a.line() for a in pinned))

    def test_task_relevance_and_edit_co_use_change_read_order(self):
        table = {}
        observe_read(table, name="read_file", path="noise/log.txt", args={"path":"noise/log.txt"},
                     handle="mcp04:noise", status="returned", is_empty=False, step=4, local_epoch=0)
        observe_read(table, name="read_file", path="engine/collision.py",
                     args={"path":"engine/collision.py"}, handle="mcp04:collision",
                     status="returned", is_empty=False, step=1, local_epoch=0)
        a = rank_read_atoms(table.values(), task_text="fix collision bug", tail_text="",
                            edit_paths=("engine/collision.py",))
        b = rank_read_atoms(table.values(), task_text="fix collision bug", tail_text="",
                            edit_paths=("engine/collision.py",))
        self.assertEqual(a, b)
        self.assertEqual(a[1][0].path, "engine/collision.py")


class WorkingSetIntegrationTests(unittest.TestCase):
    def test_two_queries_on_one_file_both_survive_packet(self):
        messages = episode(7)
        for idx, query in [(2, "collision boundary"), (6, "collision velocity")]:
            messages[idx]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                {"path": "engine/collision.py", "query": query})
        packet, receipt = typed_packet(messages, keep_recent=2, budget_tokens=1800)
        text = packet[2]["content"]
        self.assertIn("collision boundary", text)
        self.assertIn("collision velocity", text)
        self.assertEqual(receipt["attention_ranked_reads"], 4)

    def test_negative_evidence_pinned_across_observed_edit(self):
        messages = episode(7)
        for idx in (2, 6):
            messages[idx]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                {"path": "engine/collision.py", "query": "missing collider"})
        for idx in (3, 7):
            messages[idx]["content"] = "0 results"
        packet, receipt = typed_packet(messages, keep_recent=2, budget_tokens=1800)
        text = packet[2]["content"]
        self.assertEqual(receipt["pinned_negative_atoms"], 2)
        self.assertIn("local_edit_epoch=0", text)
        self.assertIn("local_edit_epoch=1", text)
        self.assertIn("not LDT certificates", text)

    def test_negative_evidence_can_fail_closed_under_tight_budget(self):
        messages = episode(7)
        for idx in (3, 7, 9, 12):
            messages[idx]["content"] = "no results"
        with self.assertRaises(WorkingSetError):
            typed_packet(messages, keep_recent=2, budget_tokens=200)

    def test_error_message_does_not_become_negative_evidence(self):
        messages = episode(7)
        messages[3]["content"] = "file not found"
        messages[3]["isError"] = True
        _, receipt = typed_packet(messages, keep_recent=2, budget_tokens=1800)
        self.assertEqual(receipt["pinned_negative_atoms"], 0)


if __name__ == "__main__":
    unittest.main()
