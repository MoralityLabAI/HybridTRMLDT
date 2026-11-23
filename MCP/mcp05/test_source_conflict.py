"""Receipt scoring and strict action parsing for the public paired probe."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .source_conflict import SCHEMA, cases, parse_action, score_case, verified_pairs


class SourceConflictContracts(unittest.TestCase):
    @staticmethod
    def row(id_, revision):
        return {"task_id": id_, "schema": SCHEMA, "prompt": "Public observation.",
                "observation": {"authorized_revision": revision, "legal_actions": ["quay", "terrace"],
                                "reports": [{"revision": 7, "route": "quay"},
                                            {"revision": 9, "route": "terrace"}]}}

    def test_validated_pairs_leave_grader_fields_out_of_case_manifest(self):
        rows = [self.row("a" * 24, 7), self.row("b" * 24, 9)]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, receipt = root / "tasks.jsonl", root / "controls.json"
            tasks.write_bytes(payload)
            receipt.write_text(json.dumps({
                "schema": SCHEMA, "heldout_sealed": False, "development_roots": 1,
                "public_tasks_sha256": hashlib.sha256(payload).hexdigest(),
            }), encoding="utf-8")
            staged = cases(verified_pairs(tasks, receipt), root)
            self.assertEqual(staged, [{"id": row["task_id"], "suite": SCHEMA, "repo": str(root),
                                       "task": row["prompt"]} for row in rows])
            tasks.write_bytes(payload + b" ")
            with self.assertRaisesRegex(ValueError, "source receipt"):
                verified_pairs(tasks, receipt)

    def test_malformed_response_differs_from_provider_error(self):
        choices = ["quay", "terrace"]
        self.assertEqual(parse_action(' {"action": "quay"} ', choices), "quay")
        self.assertIsNone(parse_action('{"action":"quay","action":"terrace"}', choices))
        self.assertIsNone(parse_action('prose {"action":"quay"}', choices))
        self.assertIsNone(parse_action('{"action":"unknown"}', choices))
        row = self.row("a" * 24, 7)
        preflight = [{"provider_status": 200, "provider_preflight_parity": True}]
        payload = {"inference_calls": 1, "preflight": preflight, "answer": '{"action":"quay"}'}
        self.assertTrue(score_case(row, payload)["correct"])
        self.assertEqual(score_case(row, {**payload, "answer": "quay"})["status"], "malformed")
        self.assertEqual(score_case(row, {**payload, "preflight": []})["status"], "infrastructure-error")
