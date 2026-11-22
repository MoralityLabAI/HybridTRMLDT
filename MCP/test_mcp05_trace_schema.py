"""Deterministic schema census contracts for public replay archives."""
from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from MCP.hubrun.trace_schema import inspect_record, inspect_archive


class SchemaCensusTests(unittest.TestCase):
    def test_observed_messages_tool_calls_and_pairs(self):
        row = {"task": {"data": {"name": "task-1"}}, "traces": [{
            "messages": [
                {"role": "user", "content": "SECRET USER REQUEST"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "function": {
                    "name": "read_file", "arguments": '{"path":"SECRET_PATH"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "SECRET RESULT"}],
            "rewards": {"score": {"score": 1.0}},
        }]}
        result = inspect_record(row)
        self.assertEqual(result["tool_call_entries"], 1)
        self.assertEqual(result["tool_result_messages"], 1)
        self.assertEqual(result["roles"]["assistant"], 1)
        self.assertIn("traces[].messages[]:role-sequence", result["paths"])
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(result["first_trace_keys"], ["messages", "rewards"])

    def test_legacy_completion_supported_without_echoing_payload(self):
        record = {"example_id": 9, "completion": [
            {"role": "assistant", "content": "SECRET RESULT"}]}
        result = inspect_record(record)
        self.assertEqual(result["roles"], {"assistant": 1})
        self.assertIn("completion[]:role-sequence", result["paths"])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_gz_archive_has_sha_and_observability_gate(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "a.results.jsonl.gz"
            with gzip.open(archive, "wt", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps({"example_id": i, "completion": [
                        {"role": "assistant", "tool_calls": [
                            {"function": {"name": "read_file", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}]}) + "\n")
            receipt = inspect_archive(archive, limit=2)
            self.assertEqual(receipt["sampled_rows"], 2)
            self.assertEqual(receipt["tool_call_entries"], 2)
            self.assertEqual(receipt["tool_result_messages"], 2)
            self.assertTrue(receipt["tool_actions_observable"])
            self.assertEqual(len(receipt["sha256"]), 64)

    def test_reject_ambiguous_input_and_invalid_limit(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                inspect_archive(path, limit=0)
            with self.assertRaises(ValueError):
                inspect_archive(path.with_suffix(".txt"))


if __name__ == "__main__":
    unittest.main()
