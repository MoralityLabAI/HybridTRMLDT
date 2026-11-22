"""Offline historical audit regression tests with synthetic tool traces."""
from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from MCP.hubrun.trace_metrics import (
    ARCHIVES, archive_rows, build_report, docedit_audit, event_recon_audit,
)


def doc_case(n: int, *, reads: int = 5, stale: int = 1) -> dict:
    return {"example_id": n, "reward": 1.0, "submitted": True,
            "read_page_calls": reads, "stale_sites": stale, "edit_text_calls": 3,
            "submit_calls": 1, "search_doc_calls": 2, "site_recall": 1.0,
            "completion": [{"role": "tool", "content": "PRIVATE TOOL OUTPUT"}]}


def event_case(name="evt-0", *, missing_args=False):
    call = lambda args: {"name": "read_file", **(
        {} if args is None else {"arguments": args})}
    return {"task": {"data": {"name": name}},
            "traces": [{"nodes": [
                {"message": {"role": "assistant", "tool_calls": [
                    call('{"path":"SECRET/path","offset":0}'),
                    call('{"path":"SECRET/path","offset":0}'),
                    call(None if missing_args else '{"path":"SECRET/path","offset":12}'),
                ]}},
                {"message": {"role": "tool", "tool_call_id": "call1", "content": "SECRET"}},
            ]}]}


class AuditTests(unittest.TestCase):
    def test_docedit_uses_environment_counters(self):
        result = docedit_audit([doc_case(0, stale=1), doc_case(1, stale=3)])
        self.assertEqual(result["means"]["stale_sites"], 2.0)
        self.assertEqual(result["means"]["read_page_calls"], 5.0)
        self.assertEqual(result["submitted"], 2)
        self.assertEqual(result["solved_reward_ge_0_999"], 2)
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_missing_stale_metric_reports_unknown(self):
        row = doc_case(0)
        del row["stale_sites"]
        receipt = docedit_audit([row])
        self.assertIsNone(receipt["means"]["stale_sites"])
        self.assertEqual(receipt["fields_missing_by_episode"]["stale_sites"], 1)

    def test_event_same_path_and_exact_repeat_are_distinct(self):
        result = event_recon_audit([event_case()])
        self.assertEqual(result["tool_calls_total"], 3)
        self.assertEqual(result["read_file_calls_total"], 3)
        self.assertEqual(result["repeat_exact_requests_total"], 1)
        self.assertEqual(result["repeat_same_path_total"], 2)
        self.assertEqual(result["read_args_missing_total"], 0)
        self.assertNotIn("SECRET", json.dumps(result))

    def test_unknown_tool_arguments_fail_closed(self):
        result = event_recon_audit([event_case(missing_args=True)])
        self.assertEqual(result["read_args_missing_total"], 1)
        self.assertIsNone(result["repeat_exact_requests_total"])
        self.assertIsNone(result["repeat_same_path_total"])

    def test_duplicate_cases_rejected(self):
        with self.assertRaises(ValueError):
            docedit_audit([doc_case(0), doc_case(0)])
        with self.assertRaises(ValueError):
            event_recon_audit([event_case("task1"), event_case("task1")])

    def test_paired_archive_report_and_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in ARCHIVES:
                if filename.startswith("docedit-"):
                    rows = [doc_case(0, reads=7 if "goblin" in filename else 3)]
                else:
                    rows = [event_case("evt-0")]
                with gzip.open(root / filename, "wt", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
            report = build_report(root)
            self.assertTrue(report["docedit_pair"]["task_ids_matched"])
            self.assertEqual(report["docedit_pair"]["goblin_minus_raw"]["read_page_calls"], 4.0)
            self.assertEqual(len(report["sources"]), 5)
            self.assertEqual(len(report["sources"][ARCHIVES[0]]["source_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
