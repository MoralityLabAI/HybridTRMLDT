"""GDK-02 revision-bound capture store tests, including restart and tamper."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from MCP.GDK.working_set import WorkingSetError, WorkingSetStore


class WorkingSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gdk-store-")
        self.root = Path(self.temp.name)
        self.store = WorkingSetStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, **args):
        params = dict(
            session_id="user/session:1", catalog_epoch="tree-A",
            packet="ranked symbol snippets\nsymbol:foo", metadata={"router": "bm25"},
        )
        params.update(args)
        return self.store.capture(**params)

    def test_restarts_rehydrate_unmodified_packet(self):
        capture = self.capture()
        stored = WorkingSetStore(self.root).load(
            session_id="user/session:1", catalog_epoch="tree-A")
        self.assertEqual(stored["capture"]["packet"], "ranked symbol snippets\nsymbol:foo")
        self.assertEqual(stored["index"]["generation"], 1)
        self.assertEqual(stored["index"]["last_handle"], capture["handle"])
        self.assertEqual(stored["capture"]["metadata"], {"router": "bm25"})

    def test_idempotent_capture_and_epoch_invalidation(self):
        a = self.capture()
        b = self.capture()
        self.assertEqual(a["handle"], b["handle"])
        self.assertEqual(b["active_count"], 1)
        c = self.capture(catalog_epoch="tree-B", packet="new packet")
        self.assertNotEqual(a["handle"], c["handle"])
        state = self.store.load(session_id="user/session:1", catalog_epoch="tree-B")
        self.assertEqual(state["index"]["active_handles"], [c["handle"]])
        with self.assertRaisesRegex(WorkingSetError, "stale"):
            self.store.load(session_id="user/session:1", catalog_epoch="tree-A")
        old = self.root / "blobs" / (a["handle"].split(":")[1] + ".json")
        self.assertTrue(old.exists(), "historic blob survives invalidation")

    def test_different_sessions_separate_and_path_safe(self):
        self.capture(session_id="../../session/a")
        self.capture(session_id="../../session/b")
        a = self.store.load(session_id="../../session/a", catalog_epoch="tree-A")
        b = self.store.load(session_id="../../session/b", catalog_epoch="tree-A")
        self.assertNotEqual(a["capture"]["session_id"], b["capture"]["session_id"])
        self.assertFalse((self.root.parent / "session").exists())

    def test_tamper_with_blob_detected(self):
        info = self.capture()
        path = self.root / "blobs" / f"{info['handle'][7:]}.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(WorkingSetError, "checksum"):
            self.store.load(session_id="user/session:1", catalog_epoch="tree-A")
        with self.assertRaisesRegex(WorkingSetError, "checksum"):
            self.capture()

    def test_tampered_session_head_detected(self):
        self.capture()
        current = list((self.root / "sessions").glob("*/current.json"))[0]
        obj = json.loads(current.read_text())
        obj["session_id"] = "other"
        current.write_text(json.dumps(obj))
        with self.assertRaisesRegex(WorkingSetError, "identity"):
            self.store.load(session_id="user/session:1", catalog_epoch="tree-A")

    def test_writer_lock_blocks_second_writer(self):
        session_dir, _ = self.store._paths("user/session:1")
        session_dir.mkdir(parents=True)
        (session_dir / ".write.lock").mkdir()
        with self.assertRaisesRegex(WorkingSetError, "already has a writer"):
            self.capture()
        (session_dir / ".write.lock").rmdir()

    def test_refuses_missing_epoch_or_empty_packet(self):
        with self.assertRaises(WorkingSetError):
            self.capture(catalog_epoch="")
        with self.assertRaises(WorkingSetError):
            self.capture(packet="")

    def test_cli_capture_load(self):
        script = [sys.executable, "-m", "MCP.GDK.working_set"]
        params = ["--root", str(self.root), "--session-id", "gdk-cli",
                  "--catalog-epoch", "revision-1"]
        capture = subprocess.run(
            script + ["capture"] + params,
            input=json.dumps({"packet": "HELLO", "metadata": {"tool_calls": 2}}),
            text=True, capture_output=True,
        )
        self.assertEqual(capture.returncode, 0, capture.stderr)
        outcome = json.loads(capture.stdout)
        self.assertEqual(outcome["active_count"], 1)
        reload = subprocess.run(script + ["load"] + params, capture_output=True, text=True)
        self.assertEqual(reload.returncode, 0, reload.stderr)
        self.assertEqual(json.loads(reload.stdout)["capture"]["packet"], "HELLO")


if __name__ == "__main__":
    unittest.main()
